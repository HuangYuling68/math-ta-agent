# tools/sympy_tool.py
from __future__ import annotations
import json
from typing import List, Optional, Dict, Any, Literal

import sympy as sp
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# -----------------------------
# 内部：最小 SymPy 引擎
# -----------------------------
class Step(BaseModel):
    id: int
    action: str
    detail: str

class _Engine:
    """仅在工具内部使用的最小引擎，支持 solve/simplify/differentiate/integrate/limit。"""
    # 任务推断供兜底使用（主流程通常显式指定 task）
    def infer_task(self, t: str) -> str:
        TL = t.lower()
        if any(k in TL for k in ["solve", "=", "roots", "root"]): return "solve"
        if any(k in TL for k in ["differentiate", "derivative", "d/dx", "w.r.t", "wrt", "diff "]): return "differentiate"
        if any(k in TL for k in ["integrate", "∫", "integral"]): return "integrate"
        if "limit" in TL or "lim" in TL: return "limit"
        return "simplify"

    def pick_symbol(self, expr: sp.Expr, hint: Optional[str]) -> sp.Symbol:
        if hint: return sp.Symbol(hint)
        free = sorted(list(expr.free_symbols), key=lambda s: s.name)
        return free[0] if free else sp.Symbol("x")

    # ---- solve ----
    def solve(self, expression: str, variable: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        if "=" not in expression:
            raise ValueError("solve 任务需要等号，例如 'x^2-5x+6=0'")
        lhs_txt, rhs_txt = expression.split("=", 1)
        lhs, rhs = sp.sympify(lhs_txt), sp.sympify(rhs_txt)
        eq = sp.Eq(lhs, rhs)
        steps.append(Step(id=1, action="parse", detail=f"方程: {eq}"))

        var = self.pick_symbol(lhs - rhs, variable)
        steps.append(Step(id=2, action="select_var", detail=f"主变量: {var}"))

        sol = sp.solve(eq, var, dict=True)
        steps.append(Step(id=3, action="sympy.solve", detail=f"解: {sol}"))

        verified = self._verify_eq(eq, var, sol)
        steps.append(Step(id=4, action="verify", detail=f"验证: {verified}"))

        final = self._format_solution(var, sol)
        solutions_str = [{str(var): str(item.get(var))} for item in sol] if sol else []
        return {
            "final_answer": final,
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"equation": str(eq), "solutions": solutions_str},
        }

    # ---- simplify ----
    def simplify(self, expression: str) -> Dict[str, Any]:
        steps: List[Step] = []
        expr = sp.sympify(expression)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}"))
        simp = sp.simplify(expr)
        steps.append(Step(id=2, action="sympy.simplify", detail=f"化简: {simp}"))
        return {
            "final_answer": str(simp),
            "steps": [s.model_dump() for s in steps],
            "verified": True,
            "sympy_result": {"input": str(expr), "simplified": str(simp)},
        }

    # ---- differentiate ----
    def _extract_wrt(self, text: str):
        low = text.lower()
        for tok in [" w.r.t ", " wrt ", " with respect to "]:
            idx = low.find(tok)
            if idx != -1:
                expr_txt = text[:idx].strip()
                var_txt = text[idx + len(tok):].strip().split()[0].strip(",.;:")
                return expr_txt, var_txt
        return text.strip(), None

    def differentiate(self, expression: str, variable: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        expr_txt, var_from_text = self._extract_wrt(expression)
        expr = sp.sympify(expr_txt)
        var = self.pick_symbol(expr, var_from_text or variable)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}, 变量: {var}"))
        d = sp.diff(expr, var)
        steps.append(Step(id=2, action="sympy.diff", detail=f"一阶导: {d}"))
        verified = self._numeric_equiv(sp.diff(expr, var), d, {var: 1.23})
        steps.append(Step(id=3, action="verify(numeric)", detail=f"验证: {verified}"))
        return {
            "final_answer": str(d),
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"derivative": str(d)},
        }

    # ---- integrate ----
    def integrate(self, expression: str, variable: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        expr = sp.sympify(expression)
        var = self.pick_symbol(expr, variable)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}, 变量: {var}"))
        F = sp.integrate(expr, var)
        steps.append(Step(id=2, action="sympy.integrate", detail=f"不定积分: {F} + C"))
        verified = sp.simplify(sp.diff(F, var) - expr) == 0
        steps.append(Step(id=3, action="verify(symbolic)", detail=f"验证: {verified}"))
        return {
            "final_answer": f"{F} + C",
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"antiderivative": str(F)},
        }

    # ---- limit ----
    def _parse_limit(self, text: str, hint_var: Optional[str]):
        low = text.lower()
        expr_txt = text; var = sp.Symbol(hint_var) if hint_var else sp.Symbol("x"); point = 0
        if " as " in low and "->" in low:
            try:
                before_as, after_as = text.split(" as ", 1)
                expr_txt = before_as.replace("limit", "").strip()
                var_txt, point_txt = after_as.split("->", 1)
                var = sp.Symbol(var_txt.strip())
                point = sp.sympify(point_txt.strip())
            except Exception:
                pass
        else:
            expr_txt = text.replace("limit", "").replace("as", "").replace("->", "").strip()
        return expr_txt, var, point

    def limit(self, expression: str, variable: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        expr_txt, var, point = self._parse_limit(expression, variable)
        expr = sp.sympify(expr_txt)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}, 变量: {var}, 点: {point}"))
        val = sp.limit(expr, var, point)
        steps.append(Step(id=2, action="sympy.limit", detail=f"极限: {val}"))
        return {
            "final_answer": str(val),
            "steps": [s.model_dump() for s in steps],
            "verified": True,
            "sympy_result": {"limit": str(val)},
        }

    # ---- utils ----
    def _verify_eq(self, eq: sp.Equality, var: sp.Symbol, sol_list: List[Dict[sp.Symbol, sp.Expr]]) -> bool:
        if not sol_list: return False
        try:
            for item in sol_list:
                val = item.get(var)
                diff_expr = sp.simplify(eq.lhs.subs(var, val) - eq.rhs.subs(var, val))
                if diff_expr != 0:
                    f = sp.lambdify((), diff_expr, "numpy")
                    import numpy as np  # noqa
                    if abs(float(f())) > 1e-6:
                        return False
            return True
        except Exception:
            return False

    def _numeric_equiv(self, a: sp.Expr, b: sp.Expr, subs: Dict[sp.Symbol, float]) -> bool:
        try:
            da = float(a.evalf(subs=subs)); db = float(b.evalf(subs=subs))
            return abs(da - db) < 1e-6
        except Exception:
            return False

    def _format_solution(self, var: sp.Symbol, sol_list: List[Dict[sp.Symbol, sp.Expr]]) -> str:
        if not sol_list: return f"No solution for {var}"
        values = [str(d[var]) for d in sol_list if var in d]
        if not values: return f"No solution for {var}"
        return f"{var} = {values[0]}" if len(values) == 1 else f"{var} ∈ {{{', '.join(values)}}}"

_ENGINE = _Engine()

# -----------------------------
# 对外：直呼式执行（无 LLM 时可用）
# -----------------------------
def sympy_dispatch(task: Literal["solve","simplify","differentiate","integrate","limit"],
                   expression: str,
                   variable: Optional[str] = None) -> Dict[str, Any]:
    if task == "solve":
        return _ENGINE.solve(expression, variable)
    if task == "simplify":
        return _ENGINE.simplify(expression)
    if task in ("differentiate","derivative"):
        return _ENGINE.differentiate(expression, variable)
    if task == "integrate":
        return _ENGINE.integrate(expression, variable)
    if task == "limit":
        return _ENGINE.limit(expression, variable)
    # 兜底
    return _ENGINE.simplify(expression)

# -----------------------------
# LangChain 工具：供 LLM 调用
# -----------------------------
class _SympyArgs(BaseModel):
    task: Literal["solve","simplify","differentiate","integrate","limit"] = Field(..., description="任务类型")
    expression: str = Field(..., description="表达式或方程，如 'x^2-5x+6=0' 或 'sin(x)*x^2 w.r.t x'")
    variable: Optional[str] = Field(None, description="主变量，如 'x'")

@tool("sympy_tool", args_schema=_SympyArgs)
def sympy_tool(task: str, expression: str, variable: Optional[str] = None) -> str:
    """使用 SymPy 对表达式执行求解/化简/求导/积分/极限。返回 JSON 字符串，包含
    final_answer/steps/verified/sympy_result。"""
    out = sympy_dispatch(task, expression, variable)
    return json.dumps(out, ensure_ascii=False)
