"""Primary reasoning agent: proposes an answer and explanation for a MedQA
question, optionally grounded in retrieved passages and prior agent turns.
"""

from agents.llm_client import LLMClient
from agents.parsing import format_options, parse_answer

SYSTEM_PROMPT = (
    "You are a careful medical reasoning assistant answering USMLE-style "
    "multiple-choice questions. Think step by step, then commit to exactly "
    "one option. Always end your reply with a final line in the exact "
    "format: ANSWER: <letter>"
)


class MedicalReasoningAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def propose(self, question: dict, context_passages=None, memory_turns=None) -> dict:
        prompt_parts = [
            f"Question:\n{question['question']}\n",
            "Options:",
            format_options(question["options"]),
        ]

        if context_passages:
            joined = "\n---\n".join(context_passages)
            prompt_parts.append(f"\nRelevant reference material:\n{joined}")

        prompt_parts.append(
            "\nReason step by step, then finish with a line in the exact format: ANSWER: <letter>"
        )
        user_prompt = "\n".join(prompt_parts)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory_turns:
            messages.extend(memory_turns)
        messages.append({"role": "user", "content": user_prompt})

        raw = self.llm.chat(messages)
        result = parse_answer(raw)
        result["raw"] = raw
        return result

