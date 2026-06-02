"""Four-type memory store with provenance, decay, and causal retrieval.

Stores:
  - identity: stable operating principles (rarely changes)
  - preferences: user constraints and style preferences
  - project_state: files, decisions, open risks, milestones (frequently updated)
  - skills: reusable procedures from successful episodes

Retrieval prefers causal relevance over semantic similarity alone:
items are scored by recency × confidence × access_pattern.
"""

import json
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from causal_harness.memory.schema import (
    MemoryType,
    MemoryItem,
    memory_item_to_dict,
    memory_item_from_dict,
)


class MemoryStore:
    """File-backed typed memory with decay-aware retrieval.

    Each memory type is stored in its own JSON file. The store is
    append-many — items can be added, updated, or superseded but
    old versions remain in the file (marked superseded) for audit.

    Usage:
        store = MemoryStore("sessions/memory")
        store.add(MemoryItem(memory_id="1", memory_type=MemoryType.PREFERENCES,
                             content="Prefer short responses", confidence=0.9))
        results = store.retrieve("response style", memory_type=MemoryType.PREFERENCES)
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._caches: dict[MemoryType, dict[str, MemoryItem]] = {
            t: {} for t in MemoryType
        }
        self._load_all()

    def add(self, item: MemoryItem) -> str:
        """Add a new memory item. Returns the memory_id."""
        if not item.memory_id:
            item.memory_id = uuid.uuid4().hex[:12]
        cache = self._caches[item.memory_type]
        cache[item.memory_id] = item
        self._save_type(item.memory_type)
        return item.memory_id

    def update(self, memory_id: str, memory_type: MemoryType, **fields) -> MemoryItem | None:
        """Update fields on an existing memory item. Returns None if not found."""
        cache = self._caches[memory_type]
        item = cache.get(memory_id)
        if item is None:
            return None
        for key, value in fields.items():
            if hasattr(item, key):
                setattr(item, key, value)
        item.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_type(memory_type)
        return item

    def supersede(self, memory_id: str, memory_type: MemoryType,
                  replacement: MemoryItem) -> str | None:
        """Mark an item as replaced by a new one. The old item stays for audit."""
        cache = self._caches[memory_type]
        old = cache.get(memory_id)
        if old is None:
            return None
        new_id = self.add(replacement)
        old.supersede(new_id)
        self._save_type(memory_type)
        return new_id

    def get(self, memory_id: str, memory_type: MemoryType) -> MemoryItem | None:
        cache = self._caches[memory_type]
        item = cache.get(memory_id)
        if item:
            item.touch()
        return item

    def retrieve(
        self,
        query: str = "",
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        active_only: bool = True,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Retrieve memory items scored by relevance.

        Scoring: recency × confidence × keyword_match.
        This is intentionally simple — causal relevance scoring is
        done by the controller, not the store.

        Args:
            query: Free-text keyword search in content + tags.
            memory_type: Filter to one type, or None for all.
            tags: Filter to items with any of these tags.
            active_only: If True, only return non-superseded items.
            limit: Max items to return.
        """
        candidates: list[MemoryItem] = []
        types = [memory_type] if memory_type else list(MemoryType)

        for mt in types:
            for item in self._caches[mt].values():
                if active_only and not item.is_active():
                    continue
                if tags and not any(t in item.tags for t in tags):
                    continue
                if query and not self._matches_query(item, query):
                    continue
                candidates.append(item)

        scored = [(self._score(item, query), item) for item in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def retrieve_by_source(self, event_id: str) -> list[MemoryItem]:
        """Find all memory items derived from a given event."""
        results = []
        for cache in self._caches.values():
            for item in cache.values():
                if event_id in item.source_events:
                    results.append(item)
        return results

    def list_active(self, memory_type: MemoryType) -> list[MemoryItem]:
        return [item for item in self._caches[memory_type].values() if item.is_active()]

    def decay_stale(self, memory_type: MemoryType, max_age_days: int = 30) -> int:
        """Reduce confidence of items not accessed recently. Returns count of decayed."""
        count = 0
        now = datetime.now(timezone.utc)
        for item in self._caches[memory_type].values():
            if not item.is_active():
                continue
            if item.last_accessed:
                last = datetime.fromisoformat(item.last_accessed)
                age = (now - last).days
                if age > max_age_days:
                    decay = min(0.5, age / (max_age_days * 2))
                    item.confidence = max(0.1, item.confidence - decay)
                    count += 1
        if count:
            self._save_type(memory_type)
        return count

    # -- file I/O --

    def _file_path(self, memory_type: MemoryType) -> Path:
        return self.store_dir / f"{memory_type.value}.json"

    def _save_type(self, memory_type: MemoryType) -> None:
        cache = self._caches[memory_type]
        data = [memory_item_to_dict(item) for item in cache.values()]
        path = self._file_path(memory_type)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on same filesystem

    def _load_all(self) -> None:
        for mt in MemoryType:
            self._load_type(mt)

    def _load_type(self, memory_type: MemoryType) -> None:
        path = self._file_path(memory_type)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cache = self._caches[memory_type]
            for raw in data:
                item = memory_item_from_dict(raw)
                cache[item.memory_id] = item
        except (json.JSONDecodeError, KeyError):
            pass

    # -- scoring --

    @staticmethod
    def _score(item: MemoryItem, query: str) -> float:
        # Recency: how recently was this accessed? (0–1)
        recency = 0.5
        if item.last_accessed:
            try:
                last = datetime.fromisoformat(item.last_accessed)
                now = datetime.now(timezone.utc)
                age_days = (now - last).days
                recency = max(0.0, 1.0 - age_days / 60)  # linear decay over 60 days
            except ValueError:
                pass

        # Confidence is direct
        conf = item.confidence

        # Keyword match
        keyword_score = 0.0
        if query:
            q_lower = query.lower()
            if q_lower in item.content.lower():
                keyword_score = 1.0
            # Partial match in tags
            for tag in item.tags:
                if q_lower in tag.lower():
                    keyword_score = max(keyword_score, 0.7)

        return 0.3 * recency + 0.4 * conf + 0.3 * keyword_score

    @staticmethod
    def _matches_query(item: MemoryItem, query: str) -> bool:
        q = query.lower()
        if q in item.content.lower():
            return True
        for tag in item.tags:
            if q in tag.lower():
                return True
        return False
