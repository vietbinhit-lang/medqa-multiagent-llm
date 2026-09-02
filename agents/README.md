# Agents

Planned components:

- orchestrator.py - coordinates the multi-agent workflow (>= 3 agents) for a single MedQA question.
- medical_reasoning_agent.py - primary reasoning agent that proposes an answer and explanation.
- verifier_agent.py - checks and critiques the reasoning agent's answer before it is finalized (used in V3, disabled in V4).

## Serving a local LLM

Agents call an OpenAI-compatible endpoint, so the same code works against a local model or a real API - only base_url changes.

Option A, vLLM (recommended, supports batching):

    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000

Option B, Ollama:

    ollama serve
    ollama run qwen2.5:7b-instruct

Then point configs/*.yaml -> model.base_url at the running server, e.g. http://localhost:8000/v1.

## Open questions to confirm with the instructor

- Exact scope of long-term memory: across questions in a run, or within one multi-agent conversation.
- Whether the verifier is one of the required agents or a separate optional component.
- Which medical corpus to use for RAG (see rag/README.md).

