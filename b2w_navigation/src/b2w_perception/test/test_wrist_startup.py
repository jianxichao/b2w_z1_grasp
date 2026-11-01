"""Read-only startup checks; no SDK instance or physical UDP socket."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from wrist_search_io import Wrist


@pytest.fixture
def startup(tmp_path,monkeypatch):
    clock=NS(t=0.)
    monkeypatch.setattr('wrist_search_io.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_search_io.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    w=object.__new__(Wrist);w.output=tmp_path;w.cfg={}
    w.pump=lambda:None;w.should_stop=lambda:False
    # Expose feedback only: issuing hold/tracking/motion during observation must fail.
    w.arm=NS(feedbackHealthy=lambda:True,getCurrentState=lambda:2)
    return w,clock


def test_static_positions_with_speed_spikes_pass_before_any_motion_enable(startup):
    w,clock=startup
    w.snapshot=lambda:(np.full(6,.006+.0001*np.sin(clock.t*200)),np.full(6,.10))
    q=w.wait_startup_still(2)
    assert .6<=clock.t<1. and np.max(abs(q))<.007
    report=json.loads((w.output/'startup_status.json').read_text())
    assert report['startup_still'] and report['max_joint_speed_rad_s']==.1


def test_real_motion_is_rejected_even_if_reported_velocity_is_zero(startup):
    w,clock=startup
    w.snapshot=lambda:(np.full(6,clock.t*.05),np.zeros(6))
    with pytest.raises(RuntimeError,match='no hold/motion enabled'):w.wait_startup_still(2)
    report=json.loads((w.output/'startup_status.json').read_text())
    assert report['phase']=='startup_timeout' and not report['startup_still']


def test_motion_that_stops_is_observed_again_before_passing(startup):
    w,clock=startup
    w.snapshot=lambda:(np.full(6,min(clock.t,1.)*.1),np.full(6,.08))
    assert w.wait_startup_still(2)==pytest.approx([.1]*6)
    assert clock.t>1.5


@pytest.mark.parametrize('fault',['feedback','state','nonfinite','cancel'])
def test_bad_feedback_or_cancel_aborts_without_control_commands(startup,fault):
    w,clock=startup;w.snapshot=lambda:(np.zeros(6),np.zeros(6))
    if fault=='feedback':w.arm.feedbackHealthy=lambda:False
    if fault=='state':w.arm.getCurrentState=lambda:1
    if fault=='nonfinite':w.snapshot=lambda:(np.full(6,float('nan')),np.zeros(6))
    if fault=='cancel':w.should_stop=lambda:True
    with pytest.raises((ValueError,RuntimeError)):w.wait_startup_still(2)


def test_repeated_scheduler_gaps_do_not_qualify_as_a_full_position_window(startup):
    w,clock=startup;w.snapshot=lambda:(np.zeros(6),np.zeros(6))
    w.pump=lambda:setattr(clock,'t',clock.t+.2)
    with pytest.raises(RuntimeError,match='startup position'):w.wait_startup_still(2)


def test_startup_cancellation_reports_exact_external_reason_without_enabling_motion(startup):
    w,clock=startup
    w.snapshot=lambda:(np.zeros(6),np.zeros(6))
    w.stop_reason=lambda:'Base is moving during arm preparation (speed_metric=0.0410)'
    with pytest.raises(RuntimeError,match='speed_metric=0.0410'):
        w.wait_startup_still(2)
    report=json.loads((w.output/'startup_status.json').read_text())
    assert report['phase']=='startup_canceled' and 'speed_metric=0.0410' in report['reason']
    assert report['hold_motion_enabled'] is False


def test_startup_resets_gripper_to_zero_before_search_without_angle_arrival_gate(startup):
    w,clock=startup;w.cfg={'startup_gripper_reset':True};w.gripper=-.8
    joints=np.array([.1,.2,-.3,.1,0.,0.]);arm_commands=[];claw_commands=[]
    w.snapshot=lambda:(joints.copy(),np.zeros(6))
    w.sdk=NS(ArmFSMState=NS(JOINTCTRL=2))
    w.arm.gripperQ=-.8
    w.arm.setArmCmd=lambda q,*args:arm_commands.append(q.copy())
    velocities=[]
    def command(q,v,*args):claw_commands.append(q);velocities.append(v)
    w.arm.setGripperCmd=command
    w.reset_startup_gripper()
    assert w.gripper==w.arm.gripperQ==0.
    assert claw_commands[0]==-.8 and claw_commands[-1]==0.
    assert clock.t>=2.1
    assert all(np.allclose(q,joints) for q in arm_commands)
    assert max(velocities)>.0 and velocities[-1]==0.


def test_startup_gripper_reset_can_be_canceled_before_commands(startup):
    w,clock=startup;w.gripper=-.8;w.snapshot=lambda:(np.zeros(6),np.zeros(6))
    w.should_stop=lambda:True
    with pytest.raises(RuntimeError,match='stop during startup gripper reset'):w.reset_startup_gripper()
