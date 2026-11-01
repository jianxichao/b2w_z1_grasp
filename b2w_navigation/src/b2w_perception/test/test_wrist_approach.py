import json
import math
import sys
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from b2w_perception.wrist_approach_core import map_target,transform_from_pose,rigid_transform

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from wrist_approach import WristApproach


def test_transform_order_includes_body_height_and_rotating_arm_camera():
    base=transform_from_pose([10,20,.5],[0,0,math.sin(math.pi/4),math.cos(math.pi/4)])
    mount=np.eye(4);mount[:3,3]=[.078,0,.09602]
    arm=np.eye(4);arm[0,3]=.2
    camera=transform_from_pose([.04,0,0],[0,math.sin(math.pi/4),0,math.cos(math.pi/4)])
    result=map_target([[0,0,2]]*3,base,mount,arm,camera)
    assert result==pytest.approx([10,22.318,.59602])


@pytest.mark.parametrize('points',[
    [],[[0,0,1],[0,0,1],[0,0,2]],[[0,0,-1]]*3,[[0,0,float('nan')]]*3])
def test_missing_inconsistent_or_nonfinite_depth_is_rejected(points):
    with pytest.raises(ValueError):map_target(points,*[np.eye(4)]*4)


def test_reflection_and_nonrigid_mount_are_rejected():
    t=np.eye(4);t[0,0]=-1
    with pytest.raises(ValueError):rigid_transform(t)


@pytest.fixture
def approach(tmp_path):
    from rclpy.clock import Clock
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import PoseStamped
    a=object.__new__(WristApproach)
    a.cfg=json.loads((ROOT/'config/wrist_search.json').read_text())['approach']
    a.t_base_arm=rigid_transform(a.cfg['T_base_arm']);a.t_end_camera=np.eye(4)
    a.Goal,a.PoseStamped=NavigateToPose.Goal,PoseStamped
    a.active=False;a.handle=None;a.pending=None;a.result=None;a.epoch=0;a.stopped_since=None
    a.last_frame=-1
    a.samples=[dict(point=[3.,0.,.1],q=np.zeros(6),box=[0,0,10,10]) for _ in range(3)]
    a.publisher=NS(publish=lambda m:None)
    a.map=OccupancyGrid();a.map.header.frame_id='map'
    a.map.info.width=a.map.info.height=200;a.map.info.resolution=.1
    a.map.info.origin.position.x=a.map.info.origin.position.y=-10.
    a.map.info.origin.orientation.w=1.;a.map.data=[0]*40000;a.map_at=time.monotonic()
    pose=np.eye(4);pose[2,3]=.5
    a.tf_matrix=lambda frame,stamp=None:pose.copy() if frame!='odom' else np.eye(4)
    app=NS(cfg={'frame_timeout_sec':2.,'confidence':.65,'confirm_frames':3},stop_requested=False,
        policy=NS(state='FOUND',paused_at=time.monotonic()-1),output=tmp_path,events=[],
        odom_at=time.monotonic(),scan_at=time.monotonic(),base_speed=0.,gates={'a':False,'b':False},
        gate_grace_until=0.,node=NS(get_clock=lambda:Clock()))
    fk=np.eye(4);fk[0,3]=.2
    app.wrist=NS(error=None,model=NS(forwardKinematics=lambda *args:fk),
        arm=NS(feedbackHealthy=lambda:True),snapshot=lambda:(np.zeros(6),np.zeros(6)))
    app.item=dict(number=4,captured_at=time.monotonic(),box=[0,0,10,10],confidence=.9,camera_xyz_m=[3,0,.1])
    app.vision=NS(error=None,read=lambda:app.item)
    app.publish=lambda event,**kw:app.events.append(event)
    app.stop_base=lambda:app.events.append('stop_base') or True
    def enable(frontier=True):
        app.events.append(('enable',frontier));app.gates={'a':True,'b':True}
    app.enable_base=enable
    def stop(reason,state='STOPPED'):
        a.cancel();app.stop_base();app.policy.state=state;app.events.append(reason)
    app.stop=stop
    pending=Future();result=Future();calls=[]
    handle=NS(accepted=True,get_result_async=lambda:result,cancel_goal_async=lambda:calls.append('cancel'))
    a.nav=NS(server_is_ready=lambda:True,send_goal_async=lambda goal:pending)
    a.app=app
    return a,app,pending,result,handle,calls,pose


def test_goal_ack_enables_only_gates_then_success_requires_measured_arrival(approach):
    a,app,pending,result,handle,calls,pose=approach
    a.begin(app.item,np.zeros(6))
    assert app.policy.state=='APPROACH_PLANNING'
    assert not any(isinstance(x,tuple) for x in app.events)
    assert (app.output/'approach_target.json').exists()
    assert a.target[2]==pytest.approx(.69602)
    pending.set_result(handle);a.tick()
    assert app.policy.state=='APPROACHING' and ('enable',False) in app.events
    # Replay movement in small increments so localization discontinuity detection remains active.
    start=pose[:2,3].copy()
    for fraction in np.linspace(0,1,25):
        pose[:2,3]=start+(a.goal_xy-start)*fraction
        a.tick()
    result.set_result(NS(status=4));a.tick()
    assert app.policy.state=='APPROACH_SETTLING'
    a.stopped_since=time.monotonic()-1;a.tick()
    assert app.policy.state=='ARRIVED' and not a.active
    assert 'arrival_verified' in app.events


def test_stop_before_late_goal_acceptance_cancels_without_enabling_motion(approach):
    a,app,pending,result,handle,calls,pose=approach
    a.begin(app.item,np.zeros(6));app.stop('operator_stop')
    pending.set_result(handle);a.tick()
    assert calls==['cancel'] and app.policy.state=='STOPPED'
    assert ('enable',False) not in app.events


@pytest.mark.parametrize('case',['no_depth','invalid_depth','stale_map','stale_exposure','stop'])
def test_bad_inputs_never_submit_goal_or_enable_motion(approach,case):
    a,app,pending,result,handle,calls,pose=approach
    if case=='no_depth':app.item['camera_xyz_m']=None
    if case=='invalid_depth':app.item['camera_xyz_m']=[0,0,float('nan')]
    if case=='stale_map':a.map_at=time.monotonic()-10
    if case=='stale_exposure':app.item['captured_at']=time.monotonic()-3
    if case=='stop':app.stop_requested=True
    with pytest.raises((ValueError,RuntimeError)):a.begin(app.item,np.zeros(6))
    assert not a.active and ('enable',False) not in app.events


def test_target_already_near_stays_stopped_without_detour(approach):
    a,app,pending,result,handle,calls,pose=approach
    app.item['camera_xyz_m']=[.6,0,.1]
    a.begin(app.item,np.zeros(6))
    assert app.policy.state=='ARRIVED' and not a.active
    assert ('enable',False) not in app.events


@pytest.mark.parametrize('fault',['nav_failure','camera','odom','arm_drift','tf_jump','timeout','external_gate','false_success'])
def test_approach_failures_are_detected_and_cancelable(approach,fault):
    a,app,pending,result,handle,calls,pose=approach
    a.begin(app.item,np.zeros(6));pending.set_result(handle);a.tick()
    if fault=='nav_failure':result.set_result(NS(status=6))
    if fault=='camera':app.vision.error='camera disconnected'
    if fault=='odom':app.odom_at=time.monotonic()-2
    if fault=='arm_drift':app.wrist.snapshot=lambda:(np.full(6,.1),np.zeros(6))
    if fault=='tf_jump':pose[0,3]=1.
    if fault=='timeout':a.started-=100
    if fault=='external_gate':app.gates['a']=False
    if fault=='false_success':
        result.set_result(NS(status=4));a.tick();a.stopped_since=time.monotonic()-1
    with pytest.raises(RuntimeError):a.tick()
    app.stop('fault','FAULT')
    assert app.policy.state=='FAULT' and not a.active and calls==['cancel']


def test_tf_history_uses_exposure_time_and_body_base_height():
    from tf2_ros import Buffer
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    a=object.__new__(WristApproach)
    a.cfg={'map_frame':'map','tf_timeout_sec':.5}
    a.app=NS(cfg={'frame_timeout_sec':2.},node=NS(get_clock=lambda:NS(now=lambda:Time(seconds=10))))
    a.buffer=Buffer()
    def add(parent,child,sec,xyz,static=False):
        msg=TransformStamped();msg.header.frame_id=parent;msg.child_frame_id=child
        msg.header.stamp=Time(seconds=sec).to_msg()
        msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z=map(float,xyz)
        msg.transform.rotation.w=1.
        (a.buffer.set_transform_static if static else a.buffer.set_transform)(msg,'test')
    add('map','odom',0,[0,0,0],True)
    add('odom','base_footprint',9,[0,0,0]);add('odom','base_footprint',10,[1,0,0])
    add('base_footprint','base_link',0,[0,0,.5],True)
    historical=a.tf_matrix('base_link',Time(seconds=9,nanoseconds=500000000))
    assert historical[:3,3]==pytest.approx([.5,0,.5])
    assert a.tf_matrix('base_footprint')[:3,3]==pytest.approx([1,0,0])
    a.app.node.get_clock=lambda:NS(now=lambda:Time(seconds=13))
    with pytest.raises(RuntimeError,match='stale'):a.tf_matrix('base_link')


def test_stop_after_acceptance_but_before_tick_cancels_accepted_handle(approach):
    a,app,pending,result,handle,calls,pose=approach
    a.begin(app.item,np.zeros(6));pending.set_result(handle)
    assert a.handle is None  # Executor received acceptance; tick has not consumed it yet.
    app.stop('operator_stop')
    assert calls==['cancel'] and ('enable',False) not in app.events


@pytest.mark.parametrize('case',['unknown_map','occupied_map','rotated_map'])
def test_map_contents_do_not_pre_reject_direct_standoff_goal(approach,case):
    a,app,pending,result,handle,calls,pose=approach
    if case=='unknown_map':a.map.data=[-1]*40000
    if case=='occupied_map':a.map.data=[100]*40000
    if case=='rotated_map':a.map.info.origin.orientation.z=.1
    a.begin(app.item,np.zeros(6))
    assert app.policy.state=='APPROACH_PLANNING' and a.active
    vector=a.target[:2]-pose[:2,3]
    expected=a.target[:2]-a.cfg['standoff_m']*vector/np.linalg.norm(vector)
    assert a.goal_xy==pytest.approx(expected)
    assert np.linalg.norm(a.target[:2]-a.goal_xy)==pytest.approx(a.cfg['standoff_m'])
    # A submitted goal is not permission to drive: acceptance still owns gate enable.
    assert ('enable',False) not in app.events
    report=json.loads((app.output/'approach_target.json').read_text())
    assert report['goal_selection']=='direct_standoff'


@pytest.fixture
def delayed_tf(monkeypatch):
    from tf2_ros import Buffer
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    clock=NS(t=0.)
    monkeypatch.setattr('wrist_approach.time.monotonic',lambda:clock.t)
    a=object.__new__(WristApproach)
    a.cfg={'map_frame':'map','tf_timeout_sec':.5,'tf_lookup_wait_sec':.3}
    a.app=NS(cfg={'frame_timeout_sec':2.},stop_requested=False,exit_requested=False,
        release_requested=False,node=NS(get_clock=lambda:NS(now=lambda:Time(seconds=10,nanoseconds=20000000))))
    a.buffer=Buffer()
    def add(ns,x):
        msg=TransformStamped();msg.header.frame_id='map';msg.child_frame_id='base_link'
        msg.header.stamp=Time(nanoseconds=ns).to_msg()
        msg.transform.translation.x=float(x);msg.transform.rotation.w=1.
        a.buffer.set_transform(msg,'test')
    add(10_000_000_000,0.)
    return a,clock,add,Time(seconds=10,nanoseconds=17500000)


def test_historical_tf_waits_and_spins_for_a_17ms_late_sample(delayed_tf):
    a,clock,add,stamp=delayed_tf
    def spin(node,timeout_sec):
        clock.t+=timeout_sec
        add(10_020_000_000,2.)
    a.app.ros=NS(spin_once=spin)
    t=a.tf_matrix('base_link',stamp)
    assert t[0,3]==pytest.approx(1.75)  # History interpolation, not latest-pose substitution.
    assert 0<clock.t<=.3


def test_missing_tf_times_out_without_falling_back_to_old_latest(delayed_tf):
    a,clock,add,stamp=delayed_tf
    a.app.ros=NS(spin_once=lambda node,timeout_sec:setattr(clock,'t',clock.t+timeout_sec))
    with pytest.raises(RuntimeError,match='TF unavailable after 0.30s'):a.tf_matrix('base_link',stamp)
    assert clock.t==pytest.approx(.3)


@pytest.mark.parametrize('flag',['stop_requested','exit_requested'])
def test_operator_can_interrupt_tf_wait(delayed_tf,flag):
    a,clock,add,stamp=delayed_tf
    def spin(node,timeout_sec):
        clock.t+=timeout_sec;setattr(a.app,flag,True)
    a.app.ros=NS(spin_once=spin)
    with pytest.raises(RuntimeError,match='interrupted'):a.tf_matrix('base_link',stamp)
    assert clock.t<.3


@pytest.mark.parametrize('status,name',[(5,'CANCELED'),(6,'ABORTED')])
def test_nav2_failure_persists_actual_status_before_fault(approach,status,name):
    a,app,pending,result,handle,calls,pose=approach
    a.begin(app.item,np.zeros(6));pending.set_result(handle);a.tick()
    result.set_result(NS(status=status,result=NS()))
    with pytest.raises(RuntimeError,match=name):a.tick()
    report=json.loads((app.output/'nav2_result.json').read_text())
    assert report['status']==status and report['status_name']==name
    assert 'error_code' not in report and 'nav2_result' in app.events
    app.stop('fault','FAULT')
    assert calls==['cancel'] and not a.active
