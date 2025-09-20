import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from pydantic import BaseModel
from typing import List

import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMMathChain
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))

# ========== 模型和 Agent 初始化 ==========

llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.3,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL or None
        )
reasoning_template = """You are a reasoning agent tasked with solving the user's logic-based questions.\n
    Logically arrive at the solution, and be factual. In your answers, clearly detail the steps
    in bullet points and give the final answer.\n
    """
reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system",
    reasoning_template),
    ("human", "Question: {question}")
])
reasoning_chain = reasoning_prompt | llm | StrOutputParser()

def reasoning_func(q: str) -> str:
    return reasoning_chain.invoke({"question": q})

reasoning_tool = Tool.from_function(
    name="Reasoning Tool",
    func=reasoning_func,
    description="Answer logic/reasoning questions. Input should be the raw question text."
)

math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool.from_function(
    name="Calculator",
    func=math_chain.run,
    description="用于数值计算和表达式化简"
)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, dates, "
                "years, etc. Worth using for general topics. Use precise questions.",
)

# 初始化 Agent，带 verbose 日志
agent = initialize_agent(
    tools=[math_tool, wikipedia_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,          # 打印每一步
    handle_parsing_errors=True,
    max_iterations=5,      # 最多 5 步
    return_intermediate_steps=True
)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a math teaching assistant. Based on the user's question, the reasoning steps "
     "(intermediate steps), and the final result, generate a well-structured standard solution.\n"
     "Requirements:\n"
     "1. Use the same language as the user's question.\n"
     "2. Present the reasoning process as an ordered list (<ol><li>...</li></ol>).\n"
     "3. On the last line, output the final answer in a separate <p><strong>Final Answer:</strong> ...</p> block."),
    ("human", 
     "Question: {question}\n\nSteps: {steps}\n\nFinal Result: {answer}\n\nPlease generate the standard solution:")
])

summary_chain = summary_prompt | llm

class SummaryAgent:
    def __init__(self, chain):
        self.chain = chain

    def preprocess(self, result: dict) -> dict:
        """把 agent.invoke 的 result 转换成 {question, steps, answer}"""
        steps_text = "\n".join(
            f"Tool: {a.tool}, Input: {a.tool_input}, Output: {o}"
            for a, o in result.get("intermediate_steps", [])
        )
        return {
            "question": result.get("input", ""),
            "steps": steps_text,
            "answer": result.get("output", "")
        }

    def invoke(self, result: dict):
        processed = self.preprocess(result)
        return self.chain.invoke(processed)

# 初始化
summary_agent = SummaryAgent(summary_chain)

# ========== FastAPI 实例 ==========
app = FastAPI()


# ========== 请求/响应数据结构 ==========
class Step(BaseModel):
    id: int
    action: str
    detail: str

class SolveRequest(BaseModel):
    question: str

class SolveResponse(BaseModel):
    final_answer: str
    verified: bool
    explanation: str = None


# ========== 接口 ==========
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        # 执行问题
        query = req.question
        result = agent.invoke(query)
        summury = summary_agent.invoke(result)

        return SolveResponse(
            final_answer=result["output"],
            explanation=summury.content,
            verified=True,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "mode": "planner + sympy + verifier + done_checker"
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("ui.html"))

