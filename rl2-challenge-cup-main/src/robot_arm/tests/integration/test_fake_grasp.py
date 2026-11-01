from robot_arm.manipulation import (
    Arx5Kinematics,
    ExecutionResult,
    GraspPlanner,
    Pose,
    SafetyError,
    SafetyGuard,
    Target,
)
from robot_arm.manipulation.state_machine import GraspStateMachine
from robot_arm.simulation import FixedTargetProvider


class FakeArm:
    def __init__(self):
        self.joints = (0.0, 0.645, 1.040, -0.770, 0.0, 0.0)
        self.holding = False
        self.executed = []
        self.connect_count = self.close_count = self.recover_count = 0
        self.hold_count = 0

    def connect(self):
        self.connect_count += 1

    def self_check(self):
        return True, ""

    def current_joints(self):
        return self.joints

    def home(self):
        pass

    def search_pose(self):
        pass

    def execute(self, command):
        self.executed.append(command.name)
        self.joints = command.joints
        if command.name == "close":
            self.holding = True
        if command.name == "open":
            self.holding = False
        return ExecutionResult(True)

    def has_object(self):
        return self.holding

    def recover(self, plan):
        self.recover_count += 1

    def hold_enabled(self):
        self.hold_count += 1

    def close(self):
        self.close_count += 1


def test_fake_target_to_fake_arm_closed_loop():
    kinematics = Arx5Kinematics()
    machine = GraspStateMachine(
        FakeArm(),
        FixedTargetProvider((0.3, 0.0, 0.2)),
        GraspPlanner(kinematics),
        SafetyGuard(kinematics),
    )
    result = machine.run()
    assert result.success
    assert result.history[-1] == "DONE"


def test_real_mode_holds_enabled_after_done_until_close():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    result = GraspStateMachine(
        arm,
        FixedTargetProvider((0.3, 0.0, 0.2)),
        GraspPlanner(kinematics),
        SafetyGuard(kinematics),
        hold_after_done=True,
    ).run()
    assert result.success
    assert arm.hold_count == arm.close_count == 1


def test_external_runtime_reuses_connected_arm_without_closing():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    arm.connect()
    result = GraspStateMachine(
        arm,
        FixedTargetProvider((0.3, 0.0, 0.2)),
        GraspPlanner(kinematics),
        SafetyGuard(kinematics),
        manage_resources=False,
    ).run()
    assert result.success
    assert arm.connect_count == 1
    assert arm.close_count == 0


def test_cancellation_recovers_without_closing_external_arm():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    cancel = lambda: "close" in arm.executed
    result = GraspStateMachine(
        arm,
        FixedTargetProvider((0.3, 0.0, 0.2)),
        GraspPlanner(kinematics),
        SafetyGuard(kinematics),
        manage_resources=False,
        cancel_requested=cancel,
    ).run()
    assert result.cancelled and not result.fatal
    assert result.history[-2:] == ("RECOVERY", "CANCELLED")
    assert arm.recover_count == 1
    assert arm.close_count == 0


class FailedArm(FakeArm):
    def execute(self, command):
        result = super().execute(command)
        if command.name == "close":
            self.holding = False
        return result


def test_failure_enters_recovery():
    kinematics = Arx5Kinematics()
    result = GraspStateMachine(
        FailedArm(), FixedTargetProvider((0.3, 0.0, 0.2)), GraspPlanner(kinematics), SafetyGuard(kinematics)
    ).run()
    assert not result.success
    assert result.history[-2:] == ("RECOVERY", "FAILED")


class DroppedArm(FakeArm):
    def execute(self, command):
        result = super().execute(command)
        if command.name == "lift":
            self.holding = False
        return result


def test_object_dropped_after_lift_enters_recovery():
    kinematics = Arx5Kinematics()
    result = GraspStateMachine(
        DroppedArm(), FixedTargetProvider((0.3, 0.0, 0.2)), GraspPlanner(kinematics), SafetyGuard(kinematics)
    ).run()
    assert not result.success
    assert result.history[-2:] == ("RECOVERY", "FAILED")
    assert "抬升后未检测到持物" in result.error
    assert "PLACE" not in result.history


class RecordingSafety(SafetyGuard):
    def __init__(self, kinematics, arm=None):
        super().__init__(kinematics)
        self.arm = arm
        self.starts = []
        self.used_latest = []

    def validate_path(self, start_or_commands, command=None):
        if command is not None:
            self.starts.append(tuple(start_or_commands))
            if self.arm is not None:
                self.used_latest.append(tuple(start_or_commands) == self.arm.feedbacks[-1])
        return super().validate_path(start_or_commands, command)


class DriftArm(FakeArm):
    def __init__(self):
        super().__init__()
        self.drift = 0.001
        self.feedbacks = []

    def current_joints(self):
        joints = list(self.joints)
        joints[0] += self.drift
        self.drift += 0.001
        feedback = tuple(joints)
        self.feedbacks.append(feedback)
        return feedback


def test_each_segment_uses_latest_joint_feedback_for_path_validation():
    kinematics = Arx5Kinematics()
    arm = DriftArm()
    safety = RecordingSafety(kinematics, arm)
    result = GraspStateMachine(
        arm, FixedTargetProvider((0.3, 0.0, 0.2)), GraspPlanner(kinematics), safety
    ).run()
    assert result.success
    assert len(safety.starts) == 10
    assert all(safety.used_latest)


class RejectLivePath(RecordingSafety):
    def validate_path(self, start_or_commands, command=None):
        if command is not None:
            raise SafetyError("实时起点路径不安全")
        return super().validate_path(start_or_commands, command)


def test_live_path_rejection_prevents_motion():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    result = GraspStateMachine(
        arm, FixedTargetProvider((0.3, 0.0, 0.2)), GraspPlanner(kinematics), RejectLivePath(kinematics)
    ).run()
    assert not result.success
    assert "实时起点路径不安全" in result.error
    assert arm.executed == []
    assert result.history[-2:] == ("RECOVERY", "FAILED")


class RetryTargetProvider:
    def __init__(self):
        self.calls = 0

    def lock(self):
        self.calls += 1
        z = 0.07 if self.calls == 1 else 0.20
        return Target(f"target-{self.calls}", Pose.from_xyz((0.3, 0.0, z)))

    def close(self):
        pass


def test_recoverable_target_error_returns_home_and_reuses_connection():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    provider = RetryTargetProvider()
    result = GraspStateMachine(
        arm, provider, GraspPlanner(kinematics), SafetyGuard(kinematics), max_retries=1
    ).run()
    assert result.success
    assert provider.calls == 2
    assert arm.connect_count == arm.close_count == 1
    assert arm.recover_count == 1
    assert result.history.count("HOME") == result.history.count("SEARCH_POSE") == 2
    assert "RECOVERY" in result.history


def test_continuous_retry_still_reuses_one_connection():
    kinematics = Arx5Kinematics()
    arm = FakeArm()
    provider = RetryTargetProvider()
    result = GraspStateMachine(
        arm, provider, GraspPlanner(kinematics), SafetyGuard(kinematics), max_retries=None
    ).run()
    assert result.success
    assert provider.calls == 2
    assert arm.connect_count == arm.close_count == 1


class HardwareFaultArm(FakeArm):
    def self_check(self):
        return False, "CAN 反馈中断"


def test_hardware_fault_is_not_automatically_retried():
    kinematics = Arx5Kinematics()
    arm = HardwareFaultArm()
    result = GraspStateMachine(
        arm,
        FixedTargetProvider((0.3, 0.0, 0.2)),
        GraspPlanner(kinematics),
        SafetyGuard(kinematics),
        max_retries=2,
    ).run()
    assert not result.success
    assert "CAN 反馈中断" in result.error
    assert "HOME" not in result.history
    assert result.fatal is True
    assert arm.connect_count == arm.close_count == 1
    assert arm.recover_count == 0
