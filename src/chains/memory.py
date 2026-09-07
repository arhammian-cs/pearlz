"""Bounded, per-session conversational memory.

Only a bounded window of prior turns is ever formatted into the condensation
prompt -- the full history is never concatenated into every LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    """Simple bounded turn history for a single chat session."""

    history_window_turns: int
    turns: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(ConversationTurn(question=question, answer=answer))
        if self.history_window_turns > 0:
            self.turns = self.turns[-self.history_window_turns :]

    def reset(self) -> None:
        self.turns.clear()

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def format_for_condensation(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)
