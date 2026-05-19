import re
from typing import Optional
from MLAgentBench.multi_agent.whiteboard import Whiteboard


_LOSS_PATTERNS = [
    re.compile(r"val(?:idation)?[_\s]?loss[:\s=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"eval[_\s]?loss[:\s=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"(?<![a-z])loss[:\s=]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def extract_eval_loss(output: str) -> Optional[float]:
    """Return the lowest loss value found in output text, or None."""
    found = []
    for pattern in _LOSS_PATTERNS:
        for match in pattern.finditer(output):
            try:
                found.append(float(match.group(1)))
            except ValueError:
                pass
    return min(found) if found else None


def select_best_worker(whiteboard: Whiteboard) -> Optional[str]:
    """Return the worker_id with the lowest best_eval_loss."""
    states = whiteboard.read_all()
    candidates = {
        wid: state.best_eval_loss
        for wid, state in states.items()
        if state.best_eval_loss is not None
    }
    if not candidates:
        return None
    return min(candidates, key=candidates.__getitem__)
