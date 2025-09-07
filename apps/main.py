# apps/main.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

# 导入工具：后续新增 RAG / Wikipedia 工具时也在此引入并加入 TOOLS
from .tools.sympy_tool import sympy_tool, sympy_dispatch

# =========================================
# 环境变量 & LLM 实例
# =========================================
"""
在项目根目录创建 .env（可选）：
OPENAI_API_KEY=sk-xxxx
# OPENAI_BASE_URL=https://api.xxx.com/v1   # 若使用兼容端点
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
MAX_AGENT_STEPS=4
"""
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "4"))

# 供 LLM 调用的工具集合（将来加 RAG / Wikipedia 直接塞这里）
TOOLS = {
    "sympy_tool": sympy_tool,
    # "rag_tool": rag_tool,
    # "wiki_tool": wiki_tool,
}

LLM_PLANNER = None
LLM_VALIDATOR = None
if OPENAI_API_KEY:
    LLM_PLANNER = ChatOpenAI(
        model=LLM_MODEL, temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None,
    ).bind_tools(list(TOOLS.values()))
    # 校验专用（可与 PLANNER 复用，单独留出便于将来策略差异化）
    LLM_VALIDATOR = ChatOpenAI(
        model=LLM_MODEL, temperature=0.0,
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None,
    )
    print("Sucessfully set LLM Tools.")

# =========================================
# FastAPI
# =========================================
app = FastAPI(title="MathTA-Agent · Planner+Tools Loop", version="0.3.0")

# ---------- I/O Schema ----------
class Step(BaseModel):
    id: int
    action: str
    detail: str

class SolveRequest(BaseModel):
    question: str = Field(..., description="自然语言题目或指令。")

class SolveResponse(BaseModel):
    final_answer: str
    steps: List[Step]
    verified: bool
    sympy_result: Dict[str, Any] = {}
    explanation: Optional[str] = None

# =========================================
# Agent Planner System Prompt
# =========================================
SYSTEM_PLANNER = SystemMessage(
    content=(
        "你是严谨的数学助教与任务规划器。工作流：\n"
        "1) 阅读用户问题，判断是否需要调用工具。\n"
        "2) 如需符号操作（求解/化简/求导/积分/极限），调用 `sympy_tool`，参数：{task, expression, variable}。\n"
        "3) 工具返回后，基于结果继续推理，可再次选择调用工具或直接给出最终答案。\n"
        "4) 如无必要调用工具，直接回答。\n"
        "5) 回答用中文，可包含 LaTeX。\n"
        "终止条件：当你有把握给出最终答案且无需更多工具时，直接回答，不再发起工具调用。"
    )
)

# =========================================
# 验证器：优先使用工具的 verified；若无，则请求 LLM 进行形式校验
# =========================================
def llm_validate(question: str, final_answer: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """返回 dict: {'pass': bool, 'reason': str}"""
    # 有工具 verified 直接复用
    if isinstance(evidence, dict) and "verified" in evidence:
        return {"pass": bool(evidence["verified"]), "reason": "tool-verified"}

    # 无 LLM 时无法进一步验证
    if not LLM_VALIDATOR:
        return {"pass": False, "reason": "no-llm-validator"}

    prompt = (
        "请对下述问答进行形式校验：如果答案可由基础代数/微积分恒等式推导（无需额外背景），"
        "则判定 PASS；否则 FAIL。输出 JSON：{'pass':true/false,'reason':'...'}。\n"
        f"题目：{question}\n答案：{final_answer}\n已知工具证据（可能为空）：{json.dumps(evidence, ensure_ascii=False)}"
    )
    try:
        msg = [SystemMessage("只输出 JSON，不要多余文本。"), HumanMessage(prompt)]
        rsp = LLM_VALIDATOR.invoke(msg).content.strip()
        # 兼容不规范输出
        start = rsp.find("{"); end = rsp.rfind("}")
        obj = json.loads(rsp[start:end+1]) if start != -1 else {"pass": False, "reason": "invalid-json"}
        obj["pass"] = bool(obj.get("pass", False))
        obj["reason"] = str(obj.get("reason", ""))
        return obj
    except Exception as e:
        return {"pass": False, "reason": f"validator-error: {e}"}

# =========================================
# 简易降级：无 LLM 时的本地直呼路径
# =========================================
def simple_fallback(question: str) -> Dict[str, Any]:
    """没有 API Key 时的兜底：从问题中粗略推断 task 与 expression，并直呼 SymPy 工具。"""
    ql = question.strip()
    low = ql.lower()
    task = "simplify"
    expr = ql
    var = None

    # 轻规则
    if any(k in low for k in ["solve", "="]): task = "solve"; expr = ql.replace("solve", "", 1).strip()
    elif any(k in low for k in ["differentiate", "derivative", "d/dx", "w.r.t", "wrt", "diff "]):
        task = "differentiate"
        expr = ql.replace("differentiate", "").replace("derivative", "")
    elif any(k in low for k in ["integrate", "∫", "integral"]):
        task = "integrate"; expr = ql.replace("integrate", "")
    elif "limit" in low or "lim" in low:
        task = "limit"; expr = ql.replace("limit", "")

    tool_obj = sympy_dispatch(task.strip(), expr.strip(), var)
    final = tool_obj.get("final_answer", "")
    verified = bool(tool_obj.get("verified", False))
    return {
        "final_answer": final,
        "steps": tool_obj.get("steps", []),
        "verified": verified,
        "sympy_result": tool_obj.get("sympy_result", {}),
        "explanation": None,
    }

# =========================================
# Agent 循环：LLM 任务拆解 → 工具调用（可多轮）→ 终止 → LLM 校验
# =========================================
def agent_loop(question: str, max_steps: int = MAX_AGENT_STEPS) -> Dict[str, Any]:
    if not LLM_PLANNER:
        return simple_fallback(question)

    msgs: List[Any] = [SYSTEM_PLANNER, HumanMessage(question)]
    trace_steps: List[Dict[str, Any]] = []
    last_tool_obj: Dict[str, Any] | None = None
    final_answer: Optional[str] = None

    for _ in range(max_steps):
        ai: AIMessage = LLM_PLANNER.invoke(msgs)

        # 有工具调用 → 执行并把结果回传给 LLM
        if getattr(ai, "tool_calls", None):
            tc = ai.tool_calls[0]  # 简化：每步处理第一个调用
            name = tc["name"]; args = tc["args"]
            tool = TOOLS.get(name)
            if tool is None:
                # 未注册工具，直接中止
                final_answer = "（系统错误：未注册的工具）"
                break

            tool_json = tool.invoke(args)  # 运行工具，返回 JSON 字符串
            msgs += [ai, ToolMessage(tool_json, tool_call_id=tc["id"])]

            # 记录轨迹
            short_out = tool_json if len(tool_json) < 300 else tool_json[:300] + " …"
            trace_steps.append({"id": len(trace_steps)+1, "action": f"tool:{name}", "detail": f"args={args}; out={short_out}"})

            # 保存最后一次的结构化结果，便于后续校验
            try:
                last_tool_obj = json.loads(tool_json)
            except Exception:
                last_tool_obj = None

            # 工具返回后继续下一轮（由 LLM 决策是否结束/继续）
            continue

        # 无工具调用 → 认为 LLM 已产出最终答案
        final_answer = ai.content
        trace_steps.append({"id": len(trace_steps)+1, "action": "answer", "detail": "LLM 提供最终答案"})
        break

    # 如果循环结束仍无文本答案，但拿到了工具结果，则请 LLM 汇总一个最终答案
    if final_answer is None and last_tool_obj is not None:
        synth_prompt = "请基于以上工具结果，给出最终中文答案（可含 LaTeX）。"
        ai2 = LLM_PLANNER.invoke(msgs + [HumanMessage(synth_prompt)])
        final_answer = ai2.content
        trace_steps.append({"id": len(trace_steps)+1, "action": "synthesize", "detail": "LLM 基于工具结果生成最终答案"})

    if final_answer is None:
        final_answer = "未能生成答案。"

    # 校验（优先使用工具 verified；否则用 LLM 验证）
    val = llm_validate(question, final_answer, last_tool_obj or {})
    verified = bool(val.get("pass", False))

    # 汇总 sympy_result（若存在）
    sympy_result = {}
    if isinstance(last_tool_obj, dict) and "sympy_result" in last_tool_obj:
        sympy_result = last_tool_obj["sympy_result"]

    return {
        "final_answer": final_answer,
        "steps": trace_steps,
        "verified": verified,
        "sympy_result": sympy_result,
        "explanation": final_answer,  # 这里直接把最终文本也当作讲解；未来可单独生成 explanation
    }

# =========================================
# API 路由
# =========================================
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        result = agent_loop(req.question)
        steps = [Step(**s) for s in result.get("steps", [])]
        return SolveResponse(
            final_answer=result.get("final_answer", ""),
            steps=steps,
            verified=bool(result.get("verified", False)),
            sympy_result=result.get("sympy_result", {}),
            explanation=result.get("explanation"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    mode = "planner+tools" if LLM_PLANNER else "sympy-fallback"
    return {"status": "ok", "version": "0.3.0", "mode": mode}

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("ui.html"))

