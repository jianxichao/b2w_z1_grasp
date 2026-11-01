"""No physical sockets, SDK ownership or hardware movement."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import threading
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import wrist_search_io as io


@pytest.fixture
def wrist(tmp_path,monkeypatch):
    w=object.__new__(io.Wrist)
    w.scanning=threading.Event();w.scanning.set();w.done=threading.Event()
    w.calls=[];w.error=None;w.running=True;w.output=tmp_path
    w.should_stop=lambda:True;w.camera_transform=np.eye(4);w.model=object()
    w.gripper=-.5;w.target=np.full(6,.2);w.center=w.target.copy()
    w.snapshot=lambda:(w.target.copy(),np.zeros(6))
    w.sdk=NS(ArmFSMState=NS(JOINTCTRL=2))
    w.arm=NS(feedbackHealthy=lambda:True,getCurrentState=lambda:2,
        requestJointHold=lambda:w.calls.append('hold'),enableJointTracking=lambda:w.calls.append('track'),
        requestPassive=lambda:w.calls.append('passive'))
    w.worker=threading.Thread(target=lambda:w.done.wait());w.worker.start()
    def plan(model,q,camera,settings):
        assert not w.worker.is_alive()  # No simultaneous command writers.
        return dict(start_q_rad=q.tolist(),goal_q_rad=[0]*6)
    monkeypatch.setattr(io,'plan_startup_reset',plan)
    w.control=lambda:w.calls.append('hold_worker')
    return w


def test_exit_reset_joins_scan_worker_and_preserves_sdk_and_gripper(wrist):
    w=wrist;original=w.should_stop
    def move(plan,settings):
        assert not w.should_stop() and not w.scanning.is_set()
        assert settings['log_prefix']=='exit_reset'
        w.calls.append('reset');w.target=np.array(plan['goal_q_rad'])
    w.move_to_observation=move
    w.reset_for_exit({},lambda:False)
    assert w.calls==['hold','track','reset'] and w.running
    assert w.gripper==-.5 and np.max(abs(w.target))==0
    assert w.should_stop is original and (w.output/'exit_reset_plan.json').exists()


def test_failed_exit_reset_restarts_hold_without_closing_sdk(wrist):
    w=wrist
    def fail(*args):raise RuntimeError('tracking error')
    w.move_to_observation=fail
    with pytest.raises(RuntimeError,match='tracking error'):w.reset_for_exit({},lambda:False)
    w.worker.join(timeout=1.)
    assert w.running and 'hold_worker' in w.calls and 'tracking error' in w.error
    assert not w.done.is_set() and not w.scanning.is_set()


def test_unhealthy_arm_never_gets_tracking_reenabled_for_exit(wrist):
    w=wrist;w.arm.feedbackHealthy=lambda:False
    with pytest.raises(RuntimeError,match='healthy JOINTCTRL'):w.reset_for_exit({},lambda:False)
    w.worker.join(timeout=1.)
    assert 'track' not in w.calls and 'passive' in w.calls and w.running
