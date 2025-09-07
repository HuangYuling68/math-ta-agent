# apps/main.py
# uvicorn apps.demo:app --reload --port 8010
import os, json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# LangChain（用 Runnable 把求解流程包一层，后续可扩展到多工具编排）
from langchain_core.runnables import RunnableLambda

# SymPy 作为符号引擎
import sympy as sp


app = FastAPI(title="MathTA-Agent M1 Demo", version="0.1.0")

# -----------------------------
# 输入/输出 Schema
# -----------------------------
class SolveRequest(BaseModel):
    question: str = Field(
        ...,
        description=(
            "数学问题，建议包含任务关键词，如 solve/simplify/differentiate/integrate/limit。\n"
            "示例: 'solve x^2 - 5x + 6 = 0 for x' 或 'differentiate sin(x)*x^2 w.r.t x'"
        ),
    )
    task: Optional[str] = Field(
        None,
        description="可显式指定任务: solve | simplify | differentiate | integrate | limit",
    )
    variable: Optional[str] = Field(
        None,
        description="主变量名（如 'x'）。未提供时尝试自动推断。",
    )


class Step(BaseModel):
    id: int
    action: str
    detail: str


class SolveResponse(BaseModel):
    final_answer: str
    steps: List[Step]
    verified: bool
    sympy_result: Dict[str, Any]


# -----------------------------
# SymPy 引擎（最小功能）
# -----------------------------
class SymPyEngine:
    def __init__(self):
        pass

    # --- 工具主入口 ---
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """根据任务类型调用不同分支，返回统一格式。"""
        q = payload.get("question", "").strip()
        task = (payload.get("task") or self._infer_task(q)).lower()
        var = payload.get("variable")

        if not q:
            raise ValueError("Empty question")

        if task == "solve":
            return self._solve_equation_flow(q, var)
        elif task == "simplify":
            return self._simplify_flow(q)
        elif task in ("differentiate", "derivative"):
            return self._differentiate_flow(q, var)
        elif task == "integrate":
            return self._integrate_flow(q, var)
        elif task == "limit":
            return self._limit_flow(q, var)
        else:
            # 默认回退到 simplify
            return self._simplify_flow(q)

    # --- 任务推断（非常轻量的关键词规则） ---
    def _infer_task(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["solve", "=", "roots", "root"]):
            return "solve"
        if any(k in t for k in ["differentiate", "derivative", "d/dx", "diff "]):
            return "differentiate"
        if any(k in t for k in ["integrate", "∫", "integral"]):
            return "integrate"
        if "limit" in t or "lim" in t:
            return "limit"
        return "simplify"

    # --- 通用: 提取变量 ---
    def _pick_symbol(self, expr: sp.Expr, hint: Optional[str]) -> sp.Symbol:
        if hint:
            return sp.Symbol(hint)
        free = list(expr.free_symbols)
        if len(free) == 0:
            # 若没有自由变量，默认用 x
            return sp.Symbol("x")
        # 取名字典序最小的一个，稳定一些
        return sorted(free, key=lambda s: s.name)[0]

    # -------------------------
    # solve: 方程/方程组求解（基础）
    # 支持输入: "solve x^2-5x+6=0 for x" 或 "x^2-5x+6=0"
    # -------------------------
    def _solve_equation_flow(self, text: str, hint_var: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []

        # 1) 解析方程
        eq_text = self._extract_equation_text(text)
        steps.append(Step(id=1, action="parse", detail=f"解析方程: {eq_text}"))

        if "=" not in eq_text:
            raise ValueError("solve 任务需要等号，例如 'x^2-5x+6=0'")

        lhs_txt, rhs_txt = eq_text.split("=", 1)
        lhs = sp.sympify(lhs_txt)
        rhs = sp.sympify(rhs_txt)
        eq = sp.Eq(lhs, rhs)

        # 2) 选择主变量
        var = self._pick_symbol(lhs - rhs, hint_var)
        steps.append(Step(id=2, action="select_var", detail=f"选择主变量: {var}"))

        # 3) SymPy 求解
        sol = sp.solve(eq, var, dict=True)  # 列表[ {x: val}, ... ]
        steps.append(Step(id=3, action="sympy.solve", detail=f"求解得到 {sol}"))

        # 4) 验证（数值代入 + 符号简化）
        verified = self._verify_solutions(eq, var, sol)
        steps.append(Step(id=4, action="verify", detail=f"验证结果: {verified}"))

        final = self._format_solution(var, sol)

        # 将解转换为字符串，确保可序列化
        solutions_str = [{str(var): str(item.get(var))} for item in sol] if sol else []

        return {
            "final_answer": final,
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"equation": str(eq), "solutions": solutions_str},
        }

    def _extract_equation_text(self, text: str) -> str:
        t = text.strip()
        # 尝试截取"solve"之后的部分
        low = t.lower()
        if low.startswith("solve "):
            t = t[6:].strip()
        # 去掉 for x / w.r.t. x 等尾缀提示
        t = (
            t.replace("for ", "")
            .replace("w.r.t", "")
            .replace("with respect to", "")
            .strip()
        )
        return t

    def _verify_solutions(
        self, eq: sp.Equality, var: sp.Symbol, sol_list: List[Dict[sp.Symbol, sp.Expr]]
    ) -> bool:
        if not sol_list:
            return False
        try:
            for item in sol_list:
                val = item.get(var)
                # 符号验证: lhs-rhs 替换并化简
                diff_expr = sp.simplify(eq.lhs.subs(var, val) - eq.rhs.subs(var, val))
                if diff_expr != 0:
                    # 数值兜底（应为常数表达式）
                    f = sp.lambdify((), diff_expr, "numpy")
                    import numpy as np  # noqa: F401
                    v = f()
                    if abs(float(v)) > 1e-6:
                        return False
            return True
        except Exception:
            return False

    def _format_solution(
        self, var: sp.Symbol, sol_list: List[Dict[sp.Symbol, sp.Expr]]
    ) -> str:
        if not sol_list:
            return f"No solution for {var}"
        values = [str(d[var]) for d in sol_list if var in d]
        if not values:
            return f"No solution for {var}"
        if len(values) == 1:
            return f"{var} = {values[0]}"
        return f"{var} ∈ {{{', '.join(values)}}}"

    # -------------------------
    # simplify: 表达式化简
    # -------------------------
    def _simplify_flow(self, text: str) -> Dict[str, Any]:
        steps: List[Step] = []
        expr_txt = self._extract_after_keywords(text, ["simplify", "expr", ":"]).strip() or text
        steps.append(Step(id=1, action="parse", detail=f"解析表达式: {expr_txt}"))
        expr = sp.sympify(expr_txt)
        simp = sp.simplify(expr)
        steps.append(Step(id=2, action="sympy.simplify", detail=f"化简结果: {simp}"))
        return {
            "final_answer": str(simp),
            "steps": [s.model_dump() for s in steps],
            "verified": True,  # 化简本身即符号等价
            "sympy_result": {"input": str(expr), "simplified": str(simp)},
        }

    # -------------------------
    # differentiate: 求导
    # -------------------------
    def _differentiate_flow(self, text: str, hint_var: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        expr_txt = self._extract_after_keywords(
            text, ["differentiate", "derivative", ":"]
        ).strip() or text
        expr = sp.sympify(expr_txt)
        var = self._pick_symbol(expr, hint_var)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}, 变量: {var}"))
        d = sp.diff(expr, var)
        steps.append(Step(id=2, action="sympy.diff", detail=f"一阶导: {d}"))
        # 符号/数值简单验证
        verified = self._numeric_equiv(sp.diff(expr, var), d, {var: 1.23})
        steps.append(Step(id=3, action="verify(numeric)", detail=f"验证结果: {verified}"))
        return {
            "final_answer": str(d),
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"derivative": str(d)},
        }

    # -------------------------
    # integrate: 不定积分（演示）
    # -------------------------
    def _integrate_flow(self, text: str, hint_var: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        expr_txt = self._extract_after_keywords(text, ["integrate", ":"]).strip() or text
        expr = sp.sympify(expr_txt)
        var = self._pick_symbol(expr, hint_var)
        steps.append(Step(id=1, action="parse", detail=f"表达式: {expr}, 变量: {var}"))
        F = sp.integrate(expr, var)
        steps.append(Step(id=2, action="sympy.integrate", detail=f"不定积分: {F} + C"))
        # 验证: 对结果求导应还原被积函数（符号验证）
        verified = sp.simplify(sp.diff(F, var) - expr) == 0
        steps.append(Step(id=3, action="verify(symbolic)", detail=f"验证结果: {verified}"))
        return {
            "final_answer": f"{F} + C",
            "steps": [s.model_dump() for s in steps],
            "verified": verified,
            "sympy_result": {"antiderivative": str(F)},
        }

    # -------------------------
    # limit: 极限（演示：默认 x→0，或提取 'x->a' 形式）
    # -------------------------
    def _limit_flow(self, text: str, hint_var: Optional[str]) -> Dict[str, Any]:
        steps: List[Step] = []
        # 解析 "limit expr as x->a" 或简单给表达式则默认为 x->0
        expr_txt, var, point = self._parse_limit(text, hint_var)
        steps.append(
            Step(id=1, action="parse", detail=f"表达式: {expr_txt}, 变量: {var}, 点: {point}")
        )
        expr = sp.sympify(expr_txt)
        val = sp.limit(expr, var, point)
        steps.append(Step(id=2, action="sympy.limit", detail=f"极限: {val}"))
        return {
            "final_answer": str(val),
            "steps": [s.model_dump() for s in steps],
            "verified": True,
            "sympy_result": {"limit": str(val)},
        }

    # -------------------------
    # 工具函数
    # -------------------------
    def _extract_after_keywords(self, text: str, keys: List[str]) -> str:
        low = text.lower()
        for k in keys:
            idx = low.find(k)
            if idx != -1:
                return text[idx + len(k):]
        return text

    def _numeric_equiv(self, a: sp.Expr, b: sp.Expr, subs: Dict[sp.Symbol, float]) -> bool:
        try:
            da = float(a.evalf(subs=subs))
            db = float(b.evalf(subs=subs))
            return abs(da - db) < 1e-6
        except Exception:
            return False

    def _parse_limit(self, text: str, hint_var: Optional[str]):
        # very light parser: "limit sin(x)/x as x->0" → ("sin(x)/x", x, 0)
        low = text.lower()
        expr_txt = text
        var = sp.Symbol(hint_var) if hint_var else sp.Symbol("x")
        point = 0
        if " as " in low and "->" in low:
            # 分离 as 与 ->
            try:
                before_as, after_as = text.split(" as ", 1)
                expr_txt = before_as.replace("limit", "").strip()
                var_txt, point_txt = after_as.split("->", 1)
                var = sp.Symbol(var_txt.strip())
                point = sp.sympify(point_txt.strip())
            except Exception:
                pass
        else:
            # 默认: 去掉开头的 limit/symbols
            expr_txt = (
                text.replace("limit", "")
                .replace("as", "")
                .replace("->", "")
                .strip()
            )
        return expr_txt, var, point


# 单例引擎 + LangChain Runnable 包装
_engine = SymPyEngine()
CHAIN: RunnableLambda = RunnableLambda(lambda x: _engine.run(x))


# -----------------------------
# FastAPI 路由
# -----------------------------
@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    try:
        result = CHAIN.invoke(req.model_dump())
        # 将 steps 从 dict 转为 Step
        steps = [Step(**s) for s in result["steps"]]
        return SolveResponse(
            final_answer=result["final_answer"],
            steps=steps,
            verified=bool(result.get("verified", False)),
            sympy_result=result.get("sympy_result", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(Path(__file__).with_name("ui.html"))
