## math-ta-agent

An agentic math tutor that fuses Wikipedia retrieval, domain RAG, and SymPy symbolic reasoning
to deliver step-by-step, verifiable solutions.

```
uvicorn main:app --reload --port 8010
```

             ┌──────────────────────────┐
             │   🌐 User Question Input │
             └────────────┬─────────────┘
                          │
                          ▼
                ┌────────────────┐
                │ 🤖 PlannerAgent │ ──┬─▶ 判断是否使用工具？
                └────┬───────────┘   │
                     │               ▼
                     │     ┌────────────────────┐
                     └────▶│ 🧰 Tool Agents 调用 │
                           └────┬──────┬────────┘
                                │      │
         ┌─────────────────────┘      └─────────────┐
         ▼                                          ▼
┌────────────────────┐                   ┌────────────────────┐
│🧮 SymPyToolAgent    │                   │📚 RagToolAgent      │
└────────────────────┘                   └────────────────────┘
         ▼                                          ▼
    工具返回结果                                检索文档段落

                 ┌────────────────────┐
                 │ 🧠 推理Agent（可选） │  ← 可组合多个推理模块
                 └────────┬───────────┘
                          ▼
                ┌────────────────┐
                │ ✅ VerifyAgent  │ ← 用 LLM 判定是否可信
                └────────┬───────┘
                         ▼
                 ┌───────────────┐
                 │ ✅ Final Answer│
                 └───────────────┘
