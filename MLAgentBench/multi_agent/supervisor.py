import asyncio
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.schema import WorkerStatus


class SupervisorAgent:
    """Async supervisor: polls Whiteboard, fires upgrade_event on stagnation."""

    def __init__(
        self,
        whiteboard: Whiteboard,
        upgrade_event: asyncio.Event,
        stagnation_window: int = 5,
        poll_interval: float = 10.0,
    ):
        self._wb = whiteboard
        self._upgrade_event = upgrade_event
        self._stagnation_window = stagnation_window
        self._poll_interval = poll_interval

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            self._check()

    def _check(self) -> None:
        states = self._wb.read_all()
        for wid, state in states.items():
            if state.status != WorkerStatus.RUNNING:
                continue
            if self._is_stagnant(state):
                self._upgrade_event.set()
                return

    def _is_stagnant(self, state) -> bool:
        actions = state.last_actions
        if len(actions) < self._stagnation_window:
            return False
        window = actions[-self._stagnation_window:]
        return len(set(window)) == 1
