"""Coordinates the full multi-agent workflow for one MedQA question.

Behaviour is driven entirely by the "components" block of the active config
(configs/v0_baseline.yaml ... configs/v3_full.yaml), so the same Orchestrator
class implements every variant V0-V4:

  V0  reasoning agent only, no RAG, no memory
  V1  reasoning agent + RAG
  V2  reasoning + critic + verifier, no memory
  V3  reasoning + critic + verifier + RAG + memory (full system)
  V4  reasoning + critic, no verifier, + RAG + memory
"""

from typing import Optional

from agents.critic_agent import CriticAgent
from agents.llm_client import LLMClient
from agents.medical_reasoning_agent import MedicalReasoningAgent
from agents.verifier_agent import VerifierAgent
from memory.memory import LongTermMemory, ShortTermMemory
from rag.retriever import Retriever


class Orchestrator:
    def __init__(
        self,
        config: dict,
        retriever: Optional[Retriever] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        self.config = config
        self.components = config["components"]

        llm = LLMClient(config["model"])
        self.reasoning_agent = MedicalReasoningAgent(llm)
        self.critic_agent = CriticAgent(llm) if self.components.get("multi_agent") else None
        self.verifier_agent = VerifierAgent(llm) if self.components.get("verifier") else None

        self.retriever = retriever if self.components.get("rag") else None
        self.long_term_memory = long_term_memory if self.components.get("long_term_memory") else None

    def run(self, question: dict) -> dict:
        short_term = ShortTermMemory() if self.components.get("short_term_memory") else None

        context_passages = None
        if self.retriever is not None:
            context_passages = self.retriever.retrieve(question["question"])

        memory_turns = None
        if self.long_term_memory is not None:
            hints = self.long_term_memory.search(question["question"])
            if hints:
                memory_turns = [{"role": "system", "content": f"Notes from past questions: {hints}"}]

        proposal = self.reasoning_agent.propose(question, context_passages, memory_turns)
        if short_term:
            short_term.add("assistant", proposal.get("raw", ""))

        trace = {"proposal": proposal}
        final = proposal

        if self.critic_agent is not None:
            review = self.critic_agent.review(question, proposal, context_passages)
            trace["review"] = review
            if short_term:
                short_term.add("assistant", review.get("raw", ""))

            if self.verifier_agent is not None:
                final = self.verifier_agent.decide(question, proposal, review)
                trace["verifier"] = final
            else:
                # V4: no verifier - fall back to a simple rule instead of a
                # third LLM call: keep the reasoning agent's answer unless
                # the critic gave a different, valid, independent answer.
                final = review if review.get("answer") else proposal

        if self.long_term_memory is not None:
            self.long_term_memory.add(
                {"question": question["question"][:200], "final_answer": final.get("answer")}
            )

        return {
            "id": question.get("id"),
            "answer": final.get("answer"),
            "explanation": final.get("explanation"),
            "trace": trace,
        }
