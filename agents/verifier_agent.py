"""Verifier agent: the final arbiter. Given the reasoning agent's proposal
and the critic's independent review (which may disagree), decides the
answer the system actually submits. Used in V3; skipped in V4, where the
orchestrator falls back to a simple agreement rule instead of a third LLM
call - see agents/orchestrator.py.
"""

from agents.llm_client import LLMClient
from agents.parsing import format_options, parse_answer

SYSTEM_PROMPT = (
    "You are the senior physician making the final call on a USMLE-style "
    "question. Two colleagues have given their answers, which may disagree. "
    "Weigh both opinions and decide the single best answer. Always end your "
    "reply with a final line in the exact format: ANSWER: <letter>"
)


class VerifierAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def decide(self, question: dict, proposal: dict, review: dict) -> dict:
        user_prompt = (
            f"Question:\n{question['question']}\n\n"
            f"Options:\n{format_options(question['options'])}\n\n"
            f"Colleague 1 answer: {proposal.get('answer')}\n"
            f"Colleague 1 reasoning:\n{proposal.get('explanation')}\n\n"
            f"Colleague 2 answer: {review.get('answer')}\n"
            f"Colleague 2 reasoning:\n{review.get('explanation')}\n\n"
            "Decide the final answer, then finish with a line in the exact format: ANSWER: <letter>"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.chat(messages)
        result = parse_answer(raw)
        result["raw"] = raw
        return result
