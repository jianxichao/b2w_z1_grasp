"""Tier-2 frontier fixes (路径规划分析与优化.md 第二梯队 C3/C4).

C3: a goal send whose future never resolves (dead action server) used to
leave ``_goal_pending`` latched True, which blocks ``_exploration_step`` and
is invisible to the progress watchdog (no ``_active_goal`` yet).  A dedicated
``goal_send_timeout_s`` watchdog now clears it and replans without counting
a failure.

C4: the narrow-passage threshold (``path_clearance_m``) is now a declared
node parameter forwarded into FrontierExplorer.
"""

from types import SimpleNamespace

from rclpy.time import Time

from rl2_nav.frontier_explorer import FrontierExplorer
from rl2_nav.frontier_explorer_node import FrontierExplorerNode


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warn(self, *args, **kwargs):
        del args, kwargs

    def error(self, *args, **kwargs):
        del args, kwargs


def _make_frontier_pending(now_ns, **overrides):
    node = object.__new__(FrontierExplorerNode)
    node._goal_pending = True
    node._goal_sent_time = Time(nanoseconds=100_000_000_000)
    node._goal_last_progress_time = Time(nanoseconds=100_000_000_000)
    node._goal_best_distance_m = 1.0
    node._goal_generation = 5
    node._active_goal_generation = 5
    node._active_goal = None
    node._active_candidate = SimpleNamespace(x=1.0, y=2.0)
    node._consecutive_failures = 0
    node._replan_requested = False
    values = {"goal_send_timeout_s": 10.0}
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node.get_logger = lambda: _Logger()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: Time(nanoseconds=now_ns)
    )
    events = []
    node._publish_event = events.append
    node._events = events
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


# ---------------------------------------------------------------------------
# C3: pending-send watchdog
# ---------------------------------------------------------------------------


def test_stale_pending_send_is_cleared_without_counting_failure() -> None:
    # Sent at t=100s, watchdog fires at t=111s (> 10s timeout).
    node = _make_frontier_pending(now_ns=111_000_000_000)

    node._check_goal_watchdog()

    assert node._goal_pending is False
    assert node._active_goal is None
    assert node._active_candidate is None
    assert node._goal_sent_time is None
    assert node._active_goal_generation is None
    # A late response for the stale send must be ignored.
    assert node._goal_generation == 6
    # Server outage is not the goal's fault: no fuse acceleration.
    assert node._consecutive_failures == 0
    assert node._events == ["frontier_goal_send_stale"]
    assert node._replan_requested is True


def test_fresh_pending_send_is_left_alone() -> None:
    # Sent at t=100s, checked at t=105s (< 10s timeout).
    node = _make_frontier_pending(now_ns=105_000_000_000)

    node._check_goal_watchdog()

    assert node._goal_pending is True
    assert node._events == []
    assert node._replan_requested is False


def test_pending_watchdog_disabled_with_zero_timeout() -> None:
    node = _make_frontier_pending(now_ns=1_000_000_000_000)
    node.get_parameter = lambda name: SimpleNamespace(value=0.0)

    node._check_goal_watchdog()

    assert node._goal_pending is True


# ---------------------------------------------------------------------------
# C4: path_clearance_m passthrough
# ---------------------------------------------------------------------------


def test_frontier_explorer_accepts_path_clearance() -> None:
    explorer = FrontierExplorer(path_clearance_m=0.20)

    assert explorer.path_clearance_m == 0.20
