"""Shared parsing helpers for turning free-form LLM output into a structured
{"answer": "A", "explanation": "..."} prediction.

MedQA is 4-way multiple choice (A-D). Agents are prompted to end their reply
with a strict "ANSWER: <letter>" line so parsing stays simple; this module is
the one place that has to deal with an LLM not following that instruction.
"""

import re

VALID_OPTIONS = ("A", "B", "C", "D")


def parse_answer(text: str) -> dict:
    """Extract {"answer": letter_or_None, "explanation": text} from raw LLM output."""
    match = re.search(r"ANSWER:\s*([A-D])", text, re.IGNORECASE)
    answer = match.group(1).upper() if match else None
    if answer not in VALID_OPTIONS:
        answer = None
    explanation = re.sub(r"ANSWER:\s*[A-D]\s*$", "", text, flags=re.IGNORECASE).strip()
    return {"answer": answer, "explanation": explanation}


def format_options(options: dict) -> str:
    return "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))

