import dataclasses
from dataclasses import dataclass
from argparse import Namespace
import json
from typing import Any, Dict, List

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        #if it is a function, use its string name
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif hasattr(o, '__call__'):
            return o.__name__
        elif isinstance(o, Namespace):
            return vars(o)

        return super().default(o)

class TooLongPromptError(Exception):
    pass
class LLMError(Exception):
    pass

class EnvException(Exception):
    def __init__(self, message):
        self.message = message 
    def __str__(self):
        return self.message

@dataclass(frozen=True)
class ActionInfo:
    name: str
    description: str
    usage: dict
    return_value: str
    function: str
    is_primitive: bool = False

@dataclass(frozen=True)
class Action:
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class Step:
    action: Action
    observation: str  # What was returned
    timestamp: float  # When the action was taken


@dataclass(frozen=True)
class Trace:
    steps: List[Step]
    low_level_steps: List[Step]
    action_infos: Dict[str, ActionInfo]
    task_description: str


from enum import Enum
from typing import Optional


class WorkerStatus(Enum):
    RUNNING = "running"
    STAGNANT = "stagnant"
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass
class WorkerState:
    worker_id: str
    model: str
    status: WorkerStatus = WorkerStatus.RUNNING
    current_step: int = 0
    best_eval_loss: Optional[float] = None
    last_actions: List[str] = dataclasses.field(default_factory=list)
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclass
class WhiteboardEntry:
    worker_id: str
    step: int
    action: str
    observation: str
    eval_loss: Optional[float]
    timestamp: float
