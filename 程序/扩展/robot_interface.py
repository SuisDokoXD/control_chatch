from dataclasses import dataclass
from typing import Protocol
from .detection import DetectionResult

@dataclass(frozen=True)
class RobotReply:
    seq: int
    ok: bool
    code: str = "OK"
    message: str = ""

class RobotInterface(Protocol):
    def send_sort(self, seq: int, target: DetectionResult) -> RobotReply: ...
    def return_to_observation(self) -> RobotReply: ...

class FakeRobot:
    def __init__(self, fail_next=False):
        self.fail_next = fail_next
        self.commands = []

    def send_sort(self, seq, target):
        self.commands.append((seq, target.object_id))
        if self.fail_next:
            self.fail_next = False
            return RobotReply(seq, False, "SIMULATED_FAILURE", "fake robot failure")
        return RobotReply(seq, True)

    def return_to_observation(self):
        return RobotReply(0, True, "OBSERVATION_READY")
