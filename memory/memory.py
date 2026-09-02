"""Short-term and long-term memory for the multi-agent workflow.

Short-term memory: scoped to a single question's multi-agent conversation.
Long-term memory: persists across questions (e.g. past mistakes, retrieved facts
worth reusing) - confirm the exact scope with your instructor before relying on
it for the official evaluation run (see agents/README.md, open questions).
"""

from collections import deque
from typing import Any, Deque, List


class ShortTermMemory:
    """Holds the running conversation/state for one question."""

    def __init__(self, max_turns: int = 20):
        self.turns: Deque[dict] = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def as_list(self) -> List[dict]:
        return list(self.turns)

    def clear(self) -> None:
        self.turns.clear()


class LongTermMemory:
    """Persists facts/lessons across questions within an evaluation run."""

    def __init__(self):
        self._store: List[Any] = []

    def add(self, item: Any) -> None:
        self._store.append(item)

    def search(self, query: str, top_k: int = 3) -> List[Any]:
        # TODO: replace with embedding-based similarity search once corpus/format is decided.
        return self._store[-top_k:]

