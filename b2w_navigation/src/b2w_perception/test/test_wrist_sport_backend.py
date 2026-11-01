import json
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from wrist_sport_approach import WristSportApproach
from b2w_interfaces.srv import SportLocalGoal
from std_msgs.msg import UInt64
from unitree_api.msg import Request


@pytest.fixture
def backend(tmp_path):
    b=object.__new__(WristSportApproach);b.cfg={};b.side='left';b.offset=.5
    b.t_base_arm=np.eye(4);b.t_end_camera=np.eye(4)
    b.request_type=SportLocalGoal;b.UInt64=UInt64;b.Request=Request
    b.active=False;b.command_id=None;b.status=None;b.status_at=0.;b.pending=None
    b.ensure_controller=lambda:None
    b.disabled=0;b.disable_other_motion=lambda:setattr(b,'disabled',b.disabled+1)
    b.requests=[];b.heartbeats=[];b.stops=[]
    b.pending_reply=Future()
    b.client=NS(service_is_ready=lambda:True,call_async=lambda req:b.requests.append(req) or b.pending_reply)
    b.keepalive=NS(publish=b.heartbeats.append);b.stop_pub=NS(publish=b.stops.append)
    b.stop_client=NS(service_is_ready=lambda:True,call_async=lambda req:None)
    app=NS(stop_requested=False,exit_requested=False,odom_at=time.monotonic(),gates={},output=tmp_path,
        cfg={'frame_timeout_sec':2.},SetBool=NS(Request=lambda:NS(data=False)),events=[],
        policy=NS(state='TARGET_LOCALIZING'))
    app.publish=lambda event,**kw:app.events.append(event)
    app.wrist=NS(error=None,arm=NS(feedbackHealthy=lambda:True),snapshot=lambda:(np.zeros(6),np.zeros(6)),
                 model=NS(forwardKinematics=lambda *unused:np.eye(4)))
    def stop(reason,state):b.cancel();app.policy.state=state;app.events.append(reason)
    app.stop=stop;b.app=app
    b.item={'captured_at':time.monotonic(),'camera_xyz_m':[1.,.1,.2]}
    return b


def test_backend_needs_no_map_tf_or_nav2_and_waits_for_position_acceptance(backend):
    b=backend;b.begin(b.item,np.zeros(6))
    assert b.disabled==1 and b.app.policy.state=='SPORT_POSITION_ACCEPTING'
    assert b.requests[0].x==1. and b.requests[0].y==.6
    b.pending_reply.set_result(NS(success=True,command_id=42))
    b.status={'command_id':42,'state':'MOVING'};b.status_at=time.monotonic()
    b.tick()
    assert b.app.policy.state=='SPORT_POSITION_MOVING' and b.heartbeats[0].data==42
    assert (b.app.output/'sport_side_goal.json').exists()


def test_first_detection_freezes_world_target_before_base_moves(backend):
    b=backend;b.app.body_pose=np.eye(4);b.app.body_pose[:3,3]=[10.,20.,.6]
    b.app.body_pose_at=time.monotonic();b.app.grasp=object()
    b.begin(b.item,np.zeros(6))
    assert b.target_world==pytest.approx([11.,20.1,.8])
    b.app.body_pose[:3,3]=[11.,20.5,.3]
    assert b.target_world==pytest.approx([11.,20.1,.8])
    report=json.loads((b.app.output/'sport_side_goal.json').read_text())
    assert report['target_world_xyz_m']==pytest.approx([11.,20.1,.8])


@pytest.mark.parametrize('stand_down',[False,True])
def test_crouch_backshift_compensation_only_moves_forward_and_preserves_side_offset(backend,stand_down):
    b=backend;b.offset=1.;b.cfg={'stand_down_forward_compensation_m':.07}
    b.status={'stand_down_on_arrival':stand_down}
    b.begin(b.item,np.zeros(6))
    assert b.ideal_goal_base==pytest.approx([1.,1.1])
    assert b.goal_base==pytest.approx([1.07 if stand_down else 1.,1.1])
    assert b.requests[0].yaw==0.


def test_late_service_acceptance_after_stop_only_disables_bridge(backend):
    b=backend;b.begin(b.item,np.zeros(6));b.cancel()
    b.pending_reply.set_result(NS(success=True,command_id=42));b.tick()
    assert not b.active and all(r.header.identity.api_id==1003 for r in b.stops)
    assert not b.heartbeats


def test_wrong_lease_status_cannot_report_arrival(backend):
    b=backend;b.begin(b.item,np.zeros(6));b.pending_reply.set_result(NS(success=True,command_id=42))
    b.status={'command_id':41,'state':'ARRIVED'};b.status_at=time.monotonic();b.tick()
    assert b.app.policy.state=='SPORT_POSITION_ACCEPTING' and b.active
    b.status={'command_id':42,'state':'ARRIVED'};b.tick()
    assert b.app.policy.state=='ARRIVED' and not b.active


def test_firmware_rejection_keeps_failure_visible_and_cancel_stops(backend):
    b=backend;b.begin(b.item,np.zeros(6));b.pending_reply.set_result(NS(success=True,command_id=42))
    b.status={'command_id':42,'state':'FAULT','info':'API 1008 rejected'};b.status_at=time.monotonic()
    with pytest.raises(RuntimeError,match='1008 rejected'):b.tick()
    b.cancel()
    assert not b.active and len(b.stops)==3


def test_standalone_handoff_does_not_wait_for_absent_nav2_services(backend):
    b=backend;app=b.app;calls=[]
    app.node=NS(get_node_names=lambda:['b2w_sport_cmd_bridge'])
    def ready_client(ready):
        return NS(service_is_ready=lambda:ready,
            wait_for_service=lambda **kw:pytest.fail('blocking service wait starves odometry'),
            call_async=lambda req:calls.append(req) or NS())
    app.clients={'/b2w_sport_cmd_bridge/enable':ready_client(True),
        '/cmd_vel_safety_node/enable':ready_client(False),
        '/frontier_explorer_node/enable':ready_client(False)}
    app.wait=lambda future:NS(success=True)
    app.cancel=NS(service_is_ready=lambda:False)
    WristSportApproach.disable_other_motion(b)
    assert len(calls)==1 and calls[0].data is False


def test_service_discovery_keeps_odometry_callbacks_running(backend,monkeypatch):
    b=backend;clock=NS(t=0.)
    monkeypatch.setattr('wrist_sport_approach.time.monotonic',lambda:clock.t)
    b.app.node=None;b.app.odom_at=-1.
    def spin(node,timeout_sec):
        clock.t+=timeout_sec;b.app.odom_at=clock.t
    b.app.ros=NS(spin_once=spin)
    client=NS(service_is_ready=lambda:clock.t>=.1)
    assert b.wait_ready(client,.3)
    assert clock.t-b.app.odom_at==0. and clock.t<=.12


def test_service_wait_can_be_interrupted_by_ctrl_c(backend,monkeypatch):
    b=backend;clock=NS(t=0.)
    monkeypatch.setattr('wrist_sport_approach.time.monotonic',lambda:clock.t)
    b.app.node=None
    def spin(node,timeout_sec):
        clock.t+=timeout_sec;b.app.exit_requested=True
    b.app.ros=NS(spin_once=spin)
    with pytest.raises(RuntimeError,match='interrupted'):b.wait_ready(NS(service_is_ready=lambda:False),.3)
    assert not b.requests


def test_nav2_stop_and_cancel_failures_do_not_block_sport_submission(backend):
    b=backend;app=b.app
    app.node=NS(get_node_names=lambda:pytest.fail('Nav2 node inspection must not be required'))
    bridge_future=NS();ignored_future=NS()
    def client(future):return NS(service_is_ready=lambda:True,call_async=lambda req:future)
    app.clients={'/b2w_sport_cmd_bridge/enable':client(bridge_future),
                 '/cmd_vel_safety_node/enable':client(ignored_future),
                 '/frontier_explorer_node/enable':client(ignored_future)}
    app.SetBool=NS(Request=lambda:NS(data=False));app.CancelGoal=NS(Request=lambda:NS())
    app.cancel=client(ignored_future)
    def wait(future):
        assert future is bridge_future
        return NS(success=True)
    app.wait=wait
    WristSportApproach.disable_other_motion(b)


def test_position_trace_records_streaming_feedback_without_issuing_commands(backend):
    b=backend;b.active=True;b.command_id=42
    data={'command_id':42,'state':'MOVING','move_command_count':10,'measured_speed_m_s':0.}
    b.on_status(NS(data=json.dumps(data)))
    trace=json.loads((b.app.output/'sport_position_trace.jsonl').read_text())
    assert trace['move_command_count']==10 and trace['measured_speed_m_s']==0.
    assert 'received_monotonic_s' in trace and not b.requests
    b.on_status(NS(data=json.dumps(dict(data,command_id=43))))
    assert len((b.app.output/'sport_position_trace.jsonl').read_text().splitlines())==1


@pytest.mark.parametrize('version',[None,4,5])
def test_controller_capability_handshake_rejects_stale_bridge(backend,monkeypatch,version):
    b=backend;clock=NS(t=1.)
    monkeypatch.setattr('wrist_sport_approach.time.monotonic',lambda:clock.t)
    def spin(*unused,**kw):
        clock.t+=kw['timeout_sec']
        b.status={'controller_version':version,'linear_deadzone_bias_m_s':.2,
                  'angular_deadzone_bias_rad_s':.2} if version else {'state':'IDLE'}
    b.app.ros=NS(spin_once=spin);b.app.node=None
    if version and version>=5:
        WristSportApproach.ensure_controller(b)
        assert clock.t<1.1
    else:
        with pytest.raises(RuntimeError,match='deadzone compensation is not loaded'):
            WristSportApproach.ensure_controller(b)
    assert not b.requests and not b.stops


def test_arrival_with_standdown_does_not_send_another_stop(backend):
    b=backend;b.begin(b.item,np.zeros(6));b.pending_reply.set_result(NS(success=True,command_id=42))
    b.status={'command_id':42,'state':'ARRIVED','stand_down_acknowledged':True};b.status_at=time.monotonic()
    b.app.stop=lambda *unused:pytest.fail('StopMove must not overwrite the completed StandDown')
    b.tick()
    assert b.app.policy.state=='ARRIVED' and not b.active and not b.stops


def test_arrival_after_three_second_wait_starts_handoff_without_reply_or_stop(backend):
    b=backend;b.begin(b.item,np.zeros(6));b.pending_reply.set_result(NS(success=True,command_id=42))
    b.status={'command_id':42,'state':'ARRIVED','stand_down_acknowledged':False,
              'stand_down_wait_completed':True,'physical_posture_verified':False}
    b.status_at=time.monotonic()
    b.app.stop=lambda *unused:pytest.fail('Do not overwrite StandDown after timed arrival')
    b.tick()
    assert b.app.policy.state=='ARRIVED' and not b.active and not b.stops
