"""Critic agent: independently reviews the reasoning agent's proposed answer,
looking for reasoning errors, and gives its own independent judgement (which
may agree or disagree). This is the second of the required >=3 agents and is
what makes V2/V3/V4 an actual multi-agent workflow rather than one model
talking to itself once.
"""

from agents.llm_client import LLMClient
from agents.parsing import format_options, parse_answer

SYSTEM_PROMPT = (
    "You are a skeptical medical reviewer. You are given a USMLE-style "
    "question and another doctor's proposed answer with their reasoning. "
    "Check their reasoning for errors, consider whether a different option "
    "fits better, and give your own independent judgement. Always end your "
    "reply with a final line in the exact format: ANSWER: <letter>"
)


class CriticAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def review(self, question: dict, proposal: dict, context_passages=None) -> dict:
        prompt_parts = [
            f"Question:\n{question['question']}\n",
            "Options:",
            format_options(question["options"]),
            f"\nProposed answer: {proposal.get('answer')}",
            f"Proposed reasoning:\n{proposal.get('explanation')}",
        ]
        if context_passages:
            joined = "\n---\n".join(context_passages)
            prompt_parts.append(f"\nRelevant reference material:\n{joined}")
        prompt_parts.append(
            "\nGive your independent review, then finish with a line in the exact format: ANSWER: <letter>"
        )
        user_prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.chat(messages)
        result = parse_answer(raw)
        result["raw"] = raw
        return result
