from __future__ import annotations

from datetime import datetime
from difflib import unified_diff
from typing import Dict, List

from openai import OpenAI


class ExemplarStore:
    """In-memory store of high-confidence coding exemplars keyed by category."""

    def __init__(self):
        self._exemplars: Dict[int, List[dict]] = {}

    def add_exemplar(
        self,
        text: str,
        code: int,
        reasoning: str,
        confidence: float,
        agent_agreement: bool,
    ) -> None:
        exemplar = {
            "text": text,
            "code": code,
            "reasoning": reasoning,
            "confidence": confidence,
            "agent_agreement": agent_agreement,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._exemplars.setdefault(code, []).append(exemplar)

    def get_exemplars(self, category: int | None = None, k: int = 3) -> List[dict]:
        if category is None:
            exemplars = [item for items in self._exemplars.values() for item in items]
        else:
            exemplars = list(self._exemplars.get(category, []))

        ranked = sorted(
            exemplars,
            key=lambda ex: ex["confidence"] * (1.5 if ex["agent_agreement"] else 1.0),
            reverse=True,
        )
        return ranked[:k]

    def get_exemplars_for_prompt(self, k_per_category: int = 2) -> str:
        if not self._exemplars:
            return ""

        lines: List[str] = []
        for category in sorted(self._exemplars.keys()):
            top_examples = self.get_exemplars(category=category, k=k_per_category)
            for exemplar in top_examples:
                lines.append(
                    f'EXEMPLAR (Category {category}): "{exemplar["text"]}" → Code {exemplar["code"]}'
                )
                lines.append(f"Reasoning: {exemplar['reasoning']}")
        return "\n".join(lines)

    def prune(self, max_per_category: int = 5) -> None:
        for category in list(self._exemplars.keys()):
            self._exemplars[category] = self.get_exemplars(category=category, k=max_per_category)


class CodingMemos:
    """In-memory store of coding memos for ambiguities and edge cases."""

    def __init__(self):
        self._memos: List[dict] = []

    def add_memo(self, agent_id: str, text_id: str, memo_text: str, category: int) -> None:
        self._memos.append(
            {
                "agent_id": agent_id,
                "text_id": text_id,
                "memo_text": memo_text,
                "category": category,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    def get_memos_for_category(self, category: int) -> List[dict]:
        return [memo for memo in self._memos if memo["category"] == category]

    def get_recent_memos(self, k: int = 10) -> List[dict]:
        return self._memos[-k:]

    def summarize(self, client: OpenAI, model: str) -> str:
        if len(self._memos) < 3:
            return "\n".join(
                [f"- {memo['text_id']} ({memo['agent_id']}): {memo['memo_text']}" for memo in self._memos]
            )

        memo_block = "\n".join(
            [
                f"- {memo['text_id']} | Agent {memo['agent_id']} | Category {memo['category']}: {memo['memo_text']}"
                for memo in self._memos
            ]
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the coding memos into 3-5 concise, actionable insights.",
                },
                {"role": "user", "content": memo_block},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


class CodebookHistory:
    """Tracks in-memory versions of codebook updates across chunks."""

    def __init__(self):
        self._versions: List[dict] = []
        self._current_index: int | None = None

    def add_version(self, codebook_text: str, rationale: str, chunk_id: int) -> None:
        version = {
            "version_id": len(self._versions),
            "codebook_text": codebook_text,
            "rationale": rationale,
            "chunk_id": chunk_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._versions.append(version)
        self._current_index = version["version_id"]

    def get_current(self) -> str:
        if self._current_index is None:
            return ""
        return self._versions[self._current_index]["codebook_text"]

    def get_diff(self, v1: int, v2: int) -> str:
        if v1 >= len(self._versions) or v2 >= len(self._versions) or v1 < 0 or v2 < 0:
            return ""

        from_text = self._versions[v1]["codebook_text"].splitlines(keepends=True)
        to_text = self._versions[v2]["codebook_text"].splitlines(keepends=True)
        diff = unified_diff(from_text, to_text, fromfile=f"version_{v1}", tofile=f"version_{v2}")
        return "".join(diff)

    def get_evolution_summary(self) -> str:
        if not self._versions:
            return ""
        return "\n".join(
            [
                f"Version {version['version_id']} (chunk {version['chunk_id']}): {version['rationale']}"
                for version in self._versions
            ]
        )

    def rollback(self, version_id: int) -> None:
        if 0 <= version_id < len(self._versions):
            self._current_index = version_id
