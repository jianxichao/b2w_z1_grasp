"""抓取流程只编排接口，不关心相机、MuJoCo 或 CAN 的实现。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .planner import GraspPlanner, PlanningError
from .safety import SafetyError, SafetyGuard
from .types import ArmBackend, HardwareFault, MotionCommand, PickPlan, Target


class TargetProvider(Protocol):
    def lock(self) -> Target: ...
    def close(self) -> None: ...


class State(str, Enum):
    CHECK = "CHECK"
    HOME = "HOME"
    SEARCH_POSE = "SEARCH_POSE"
    LOCK_TARGET = "LOCK_TARGET"
    PLAN = "PLAN"
    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    CLOSE = "CLOSE"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    TRANSIT = "TRANSIT"
    PLACE = "PLACE"
    OPEN = "OPEN"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETREAT = "RETREAT"
    RECOVERY = "RECOVERY"
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RunResult:
    success: bool
    history: tuple[str, ...]
    error: str = ""
    fatal: bool = False
    cancelled: bool = False


class AttemptError(RuntimeError):
    """A target or grasp failure that may be retried without reconnecting CAN."""


class TaskCancelled(RuntimeError):
    """A cooperative cancellation checked between safe motion segments."""


class GraspStateMachine:
    """执行一次锁定目标后的抓取；运动过程中不更新本次目标。"""

    def __init__(
        self,
        arm: ArmBackend,
        targets: TargetProvider,
        planner: GraspPlanner,
        safety: SafetyGuard,
        on_state: Callable[[State], None] | None = None,
        max_retries: int | None = 2,
        hold_after_done: bool = False,
        manage_resources: bool = True,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.arm, self.targets = arm, targets
        self.planner, self.safety = planner, safety
        self.on_state = on_state
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.max_retries = max_retries
        self.hold_after_done = hold_after_done
        self.manage_resources = manage_resources
        self.cancel_requested = cancel_requested
        self._history: list[str] = []
        self._active_plan: PickPlan | None = None

    def _state(self, state: State) -> None:
        if (
            self.cancel_requested is not None
            and state not in {State.RECOVERY, State.CANCELLED, State.FAILED}
            and self.cancel_requested()
        ):
            raise TaskCancelled("任务已取消")
        self._history.append(state.value)
        if self.on_state:
            self.on_state(state)

    def _execute(self, state: State, plan: PickPlan, name: str) -> None:
        self._state(state)
        command = plan.command(name)
        self.safety.validate_path(self.arm.current_joints(), command)
        result = self.arm.execute(command)
        if not result.reached or result.cancelled:
            if result.cancelled:
                raise HardwareFault(result.reason or f"{name} 已取消")
            raise AttemptError(result.reason or f"{name} 未到位")

    def _go_named_pose(self, state: State, command: MotionCommand, action: Callable[[], object]) -> None:
        self._state(state)
        self.safety.validate_path(self.arm.current_joints(), command)
        result = action()
        if result is not None and not result.reached:
            if result.cancelled:
                raise HardwareFault(result.reason or f"{command.name} 已取消")
            raise AttemptError(result.reason or f"{command.name} 未到位")

    def _require_object(self, stage: str) -> None:
        if not self.arm.has_object():
            raise AttemptError(f"{stage}未检测到持物")

    def _prepare_retry(self, plan: PickPlan | None) -> None:
        self._state(State.RECOVERY)
        prepare = getattr(self.targets, "prepare_retry", None)
        if prepare is not None:
            prepare()
        self.arm.recover(plan)

    def _attempt(self) -> None:
        self._active_plan = None
        planner_settings = self.planner.settings
        self._go_named_pose(
            State.HOME,
            MotionCommand("home", planner_settings.home_joints, planner_settings.open_gripper, 1.0),
            self.arm.home,
        )
        self._go_named_pose(
            State.SEARCH_POSE,
            MotionCommand("search", planner_settings.search_joints, planner_settings.open_gripper, 1.0),
            self.arm.search_pose,
        )
        self._state(State.LOCK_TARGET)
        target = self.targets.lock()
        try:
            self.safety.validate_target(target)
            self._state(State.PLAN)
            plan = self.planner.plan(target, self.arm.current_joints())
            self.safety.validate_path(plan.commands)
            self._active_plan = plan
        except (PlanningError, SafetyError) as exc:
            raise AttemptError(str(exc)) from exc

        self._execute(State.PREGRASP, plan, "pregrasp")
        self._execute(State.GRASP, plan, "grasp")
        self._execute(State.CLOSE, plan, "close")
        self._state(State.VERIFY_GRASP)
        self._require_object("闭合后")
        self._execute(State.LIFT, plan, "lift")
        self._require_object("抬升后")
        self._execute(State.TRANSIT, plan, "transit")
        self._require_object("移动到放置区上方后")
        self._execute(State.PLACE, plan, "place")
        self._require_object("放置点张开前")
        self._execute(State.OPEN, plan, "open")
        self._state(State.VERIFY_PLACE)
        if self.arm.has_object():
            raise AttemptError("夹爪仍检测到物体，放置失败")
        self._execute(State.RETREAT, plan, "retreat")

    def run(self) -> RunResult:
        self._history = []
        try:
            self._state(State.CHECK)
            if self.manage_resources:
                self.arm.connect()
            ok, reason = self.arm.self_check()
            if not ok:
                raise HardwareFault(reason or "机械臂自检失败")

            attempt = 0
            while True:
                try:
                    self._attempt()
                    break
                except (AttemptError, TimeoutError, PlanningError, SafetyError) as exc:
                    if self.max_retries is not None and attempt >= self.max_retries:
                        raise AttemptError(f"已重试 {self.max_retries} 次：{exc}") from exc
                    self._prepare_retry(self._active_plan)
                    attempt += 1
            self._state(State.DONE)
            if self.hold_after_done:
                self.arm.hold_enabled()
            return RunResult(True, tuple(self._history))
        except TaskCancelled as exc:
            self._state(State.RECOVERY)
            try:
                self.arm.recover(self._active_plan)
            except Exception:
                pass
            self._state(State.CANCELLED)
            return RunResult(False, tuple(self._history), str(exc), cancelled=True)
        except (RuntimeError, ValueError, TimeoutError, PlanningError, SafetyError) as exc:
            self._state(State.RECOVERY)
            if not isinstance(exc, HardwareFault):
                try:
                    self.arm.recover(self._active_plan)
                except Exception:
                    pass
            self._state(State.FAILED)
            return RunResult(
                False, tuple(self._history), str(exc), fatal=isinstance(exc, HardwareFault)
            )
        finally:
            if self.manage_resources:
                # 先停止视觉线程，再关闭 CAN。
                try:
                    self.targets.close()
                finally:
                    self.arm.close()
