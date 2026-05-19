import asyncio
from typing import Dict, List, Optional
from MLAgentBench.schema import WorkerState, WhiteboardEntry


class Whiteboard:
    """Async-safe shared state for all worker agents."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._states: Dict[str, WorkerState] = {}
        self._entries: Dict[str, List[WhiteboardEntry]] = {}

    async def update_worker_state(self, worker_id: str, state: WorkerState) -> None:
        async with self._lock:
            self._states[worker_id] = state
            if worker_id not in self._entries:
                self._entries[worker_id] = []

    async def append_entry(self, worker_id: str, entry: WhiteboardEntry) -> None:
        async with self._lock:
            if worker_id not in self._entries:
                self._entries[worker_id] = []
            self._entries[worker_id].append(entry)

    def read_all(self) -> Dict[str, WorkerState]:
        return dict(self._states)

    def snapshot(self, worker_id: str) -> List[WhiteboardEntry]:
        return list(self._entries.get(worker_id, []))

    def all_snapshots(self) -> Dict[str, List[WhiteboardEntry]]:
        return {wid: list(entries) for wid, entries in self._entries.items()}
