import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from b2w_perception.wrist_search_core import SettleTracker
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from wrist_search_io import Wrist


def test_slow_position_drift_and_velocity_are_distinguished():
    tracker=SettleTracker();goal=np.zeros(6)
    q=goal.copy();q[1]=.03
    m=tracker.update(0,q,np.zeros(6),goal)
    assert not m['position_ok'] and m['speed_ok'] and m['worst_position_joint']==2
    dq=np.zeros(6);dq[5]=.04
    m=tracker.update(1,goal,dq,goal)
    assert m['position_ok'] and not m['speed_ok'] and m['worst_speed_joint']==6


def test_stable_time_requires_both_conditions_and_resets_on_a_speed_spike():
    tracker=SettleTracker();zero=np.zeros(6)
    assert not tracker.update(0,zero,zero,zero)['settled']
    assert not tracker.update(.3,zero,zero,zero)['settled']
    dq=zero.copy();dq[0]=.04
    assert tracker.update(.35,zero,dq,zero)['stable_for_sec']==0
    assert not tracker.update(.5,zero,zero,zero)['settled']
    assert tracker.update(.91,zero,zero,zero)['settled']


@pytest.fixture
def wrist(tmp_path,monkeypatch):
    clock=SimpleNamespace(t=0.)
    monkeypatch.setattr('wrist_search_io.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_search_io.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    w=object.__new__(Wrist);w.output=tmp_path;w.gripper=-.5
    w.pump=lambda:None;w.should_stop=lambda:False
    w.sdk=SimpleNamespace(ArmFSMState=SimpleNamespace(JOINTCTRL=2))
    w.arm=SimpleNamespace(feedbackHealthy=lambda:True,getCurrentState=lambda:2,
                          setArmCmd=lambda *args:None,setGripperCmd=lambda *args:None)
    def fk(q,index):
        t=np.eye(4);t[:3,3]=np.asarray(q)[:3]*.1;return t
    w.model=SimpleNamespace(forwardKinematics=fk)
    return w,clock


def test_timeout_records_measured_reason_instead_of_generic_failure(wrist):
    w,clock=wrist
    q=np.zeros(6);q[1]=.04
    w.snapshot=lambda:(q,np.zeros(6))
    with pytest.raises(RuntimeError,match='J2 error=0.0400rad'):
        w.wait_observation_settle(np.zeros(6),{'settle_timeout_sec':.5})
    status=json.loads((w.output/'observation_status.json').read_text())
    assert status['phase']=='settle_timeout'
    assert status['max_joint_error_rad']==pytest.approx(.04)
    assert status['end_position_error_m']==pytest.approx(.004)
    assert status['speed_ok'] and not status['position_ok']


def test_late_settling_succeeds_without_loosening_joint_tolerances(wrist):
    w,clock=wrist
    def snapshot():
        q=np.zeros(6)
        if clock.t<4.5:q[1]=.03
        return q,np.zeros(6)
    w.snapshot=snapshot
    w.wait_observation_settle(np.zeros(6),{'settle_timeout_sec':10.})
    status=json.loads((w.output/'observation_status.json').read_text())
    assert status['phase']=='settled' and status['settled']
    assert status['elapsed_sec']>4.5
    assert status['position_tolerance_rad']==.025
    assert status['speed_tolerance_rad_s']==.025


def test_relaxed_arrival_config_accepts_moderate_error_but_still_rejects_large_error():
    tracker=SettleTracker(position_tolerance=.08,speed_tolerance=.06,stable_sec=.3)
    zero=np.zeros(6);q=zero.copy();q[1]=.05
    dq=zero.copy();dq[4]=.04
    assert not tracker.update(0,q,dq,zero)['settled']
    assert tracker.update(.31,q,dq,zero)['settled']
    q[1]=.10
    assert not tracker.update(.5,q,dq,zero)['position_ok']


def test_position_window_accepts_static_encoders_despite_noisy_reported_speed():
    tracker=SettleTracker(position_tolerance=.08,speed_tolerance=.06,stable_sec=.3,
                          position_window_sec=.6)
    zero=np.zeros(6)
    for i in range(60):
        q=np.full(6,.0067+.0002*np.sin(i))
        m=tracker.update(i*.02,q,np.full(6,.10),zero)
        if i<30:assert not m['settled']
    assert m['settled'] and m['position_window_stable'] and not m['speed_ok']
    assert m['settle_basis']=='position_window'


@pytest.mark.parametrize('case',['moving','oscillating','off_target','feedback_gap','sparse'])
def test_position_window_rejects_motion_bad_position_or_insufficient_feedback(case):
    tracker=SettleTracker(position_tolerance=.08,speed_tolerance=.06,stable_sec=.3,
                          position_window_sec=.6)
    zero=np.zeros(6)
    for i in range(60):
        t=i*.02;q=zero.copy()
        if case=='moving':q[0]=t*.10
        if case=='oscillating':q[0]=.02*np.sin(t*30)
        if case=='off_target':q[0]=.10
        if case=='feedback_gap':t=i*.2
        if case=='sparse':t=i*.14
        m=tracker.update(t,q,np.full(6,.10),zero)
        assert not m['settled']


def test_position_window_is_invalidated_when_encoder_moves_after_settling():
    tracker=SettleTracker(position_tolerance=.08,speed_tolerance=.06,stable_sec=.3,
                          position_window_sec=.6)
    zero=np.zeros(6);dq=np.full(6,.1)
    for i in range(60):m=tracker.update(i*.02,zero,dq,zero)
    assert m['settled']
    q=zero.copy();q[0]=.02
    m=tracker.update(1.2,q,dq,zero)
    assert not m['settled'] and m['stable_for_sec']==0


def test_reset_joint_motion_keeps_gripper_angle_and_writes_separate_status(wrist):
    w,clock=wrist
    w.target=np.full(6,.05)
    w.snapshot=lambda:(w.target.copy(),np.zeros(6))
    angles=[]
    w.arm.setArmCmd=lambda q,*unused:setattr(w,'target',np.asarray(q).copy())
    w.arm.setGripperCmd=lambda q,*unused:angles.append(q)
    w.move_to_observation({'start_q_rad':[.05]*6,'goal_q_rad':[0.]*6},
        {'joint_speed_rad_s':.08,'joint_position_tolerance_rad':.08,
         'joint_speed_tolerance_rad_s':.06,'settle_stable_sec':.3,
         'label':'机械臂启动复位','log_prefix':'reset'})
    assert angles and all(angle==w.gripper for angle in angles)
    assert np.max(np.abs(w.target))<1e-8
    status=json.loads((w.output/'reset_status.json').read_text())
    assert status['settled']
    assert not (w.output/'observation_status.json').exists()
