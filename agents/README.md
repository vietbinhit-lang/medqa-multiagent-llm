# Agents

Implemented (see agents/orchestrator.py for how they are wired together):

- llm_client.py - thin OpenAI-compatible wrapper; only base_url changes between a local model and a real API.
- parsing.py - shared helpers to turn free-form LLM text into {"answer": "A", "explanation": "..."}.
- medical_reasoning_agent.py - primary reasoning agent that proposes an answer and explanation, optionally grounded in RAG context and long-term memory hints.
- critic_agent.py - second, independent agent that reviews the reasoning agent's proposal and gives its own judgement (may agree or disagree). This is what makes V2/V3/V4 an actual multi-agent workflow.
- verifier_agent.py - third agent, the final arbiter between the reasoning agent and the critic. Used in V3; skipped in V4, where the orchestrator falls back to a simple rule (prefer the critic's answer if it gave a different, valid one) instead of a third LLM call.
- orchestrator.py - the Orchestrator class. The same class implements V0-V4; behaviour is driven entirely by the "components" block of the active config (configs/*.yaml) - see its docstring for the exact agent/RAG/memory combination per variant.

## Serving a local LLM

Agents call an OpenAI-compatible endpoint, so the same code works against a local model or a real API - only base_url changes.

Option A, vLLM (recommended, supports batching):

    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

Option B, Ollama:

    ollama serve
    ollama run qwen2.5:7b-instruct

Then point configs/*.yaml -> model.base_url at the running server, e.g. http://localhost:8000/v1.

## Verified

A mocked-LLM smoke test confirmed the call count and answer-selection logic per variant:

- V0 (reasoning only): 1 LLM call.
- V3 (full, with verifier): 3 LLM calls (reasoning, critic, verifier).
- V4 (full, no verifier): 2 LLM calls (reasoning, critic), final answer falls back to the critic's answer when it disagrees.

## Open questions to confirm with the instructor

- Exact scope of long-term memory: across questions in a run, or within one multi-agent conversation.
- Whether the verifier is one of the required >=3 agents, or a separate optional component on top of 3 others.
- Which medical corpus to use for RAG (see rag/README.md).
