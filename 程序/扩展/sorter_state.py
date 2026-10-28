from enum import Enum
from .robot_interface import RobotInterface

class SorterState(str, Enum):
    OBSERVE = "observe"
    COMMAND = "command"
    WAIT_ACK = "wait_ack"
    VERIFY_REMOVED = "verify_removed"
    COUNTED = "counted"
    ERROR = "error"

class SorterController:
    def __init__(self, robot: RobotInterface):
        self.robot = robot
        self.state = SorterState.OBSERVE
        self.next_seq = 1
        self.counted_sequences = set()

    def execute_one(self, target, still_present_after_grab=False):
        if not target.valid or target.tray_xy_mm is None:
            target.reject("not_ready_for_grab")
            self.state = SorterState.ERROR
            return False
        seq = self.next_seq
        self.next_seq += 1
        self.state = SorterState.COMMAND
        reply = self.robot.send_sort(seq, target)
        self.state = SorterState.WAIT_ACK
        if not reply.ok:
            self.state = SorterState.ERROR
            return False
        self.robot.return_to_observation()
        self.state = SorterState.VERIFY_REMOVED
        if still_present_after_grab or seq in self.counted_sequences:
            self.state = SorterState.ERROR
            return False
        self.counted_sequences.add(seq)
        self.state = SorterState.COUNTED
        return True
