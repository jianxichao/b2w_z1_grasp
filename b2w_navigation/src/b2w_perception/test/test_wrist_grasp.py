"""Offline post-arrival verification: no ArmInterface, sockets or robot commands."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'b2w_navigation/scripts'))
from wrist_grasp import WristGrasp
from wrist_search_io import Wrist
from b2w_perception.wrist_search_core import plan_observation,plan_startup_reset
from b2w_perception.wrist_approach_core import sport_body_transform,world_target_to_arm,transform_from_pose


@pytest.fixture
def configs():
    return (json.loads((ROOT/'b2w_navigation/config/wrist_search.json').read_text()),
            json.loads((ROOT/'unitree_z1/yolo/grasp_demo/config.json').read_text()))


def frame(number=1,stamp=10.1):
    return dict(number=number,captured_at=stamp,confidence=.9,box=[20,20,80,60],
                camera_xyz_m=[0.,0.,.4],annotated=np.zeros((100,100,3),np.uint8))


@pytest.mark.parametrize('observations',[
    [(i,True) for i in (1,2)],
    [(i,i in (1,4,6)) for i in range(1,7)],
    [(i,True) for i in (1,1,2)],
])
def test_single_view_search_requires_two_of_three_votes_and_ignores_duplicate_frames(configs,monkeypatch,observations,tmp_path):
    cfg,g=configs;clock=NS(t=1.);moves=[];frames=iter(observations);seen=[]
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_grasp.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    monkeypatch.setattr('wrist_grasp.plan_observation',lambda *args,**kw:dict(goal_q_rad=[0.]*6))
    w=NS(target=np.zeros(6),camera_transform=np.eye(4),gripper=-.5,model=object(),scanning=threading.Event())
    w.snapshot=lambda:(w.target.copy(),np.zeros(6))
    w.arm=NS(setArmCmd=lambda *args:None,setGripperCmd=lambda *args:None)
    w.move_to_observation=lambda plan,settings:moves.append(settings)
    def read():
        number,hit=next(frames);seen.append(number)
        item=frame(number,clock.t-.001)
        if not hit:item.update(box=None,confidence=0.)
        return item
    app=NS(wrist=w,cfg=cfg,output=tmp_path,approach=NS(side='left'),vision=NS(read=read),policy=NS(state='ARRIVED'))
    app.publish=lambda *args,**kw:None
    s=WristGrasp(app,cfg['post_grasp'],g)
    s.guard=lambda:setattr(clock,'t',clock.t+.05)
    s.estimated_target_arm=lambda:np.array([.1,-.5,.3])
    s.plan_hint_view=lambda hint:(dict(goal_q_rad=[0.]*6),dict(s.settings('aim',turn=True),yaw_deg=-80.,pitch_candidates_deg=[0.]),0.)
    s.confirm=lambda:'fresh_depth'
    assert s.search()=='fresh_depth'
    assert len(moves)==1 and len(moves[0]['pitch_candidates_deg'])==1
    assert moves[0]['joint_speed_rad_s']==.2
    assert moves[0]['joint_position_tolerance_rad']==.08
    assert s.vote_policy.vote_count==2 and s.vote_policy.vote_window==3
    assert seen==[number for number,hit in observations]
    assert not w.scanning.is_set()


def test_after_vote_one_fresh_depth_frame_is_enough(configs,monkeypatch):
    cfg,g=configs;clock=NS(t=1.);count=[]
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    w=NS(target=np.zeros(6),camera_transform=np.eye(4),gripper=-.5,
         model=NS(forwardKinematics=lambda *args:np.eye(4)))
    w.snapshot=lambda:(w.target.copy(),np.zeros(6))
    w.wait_observation_settle=lambda *args:None
    def read():
        count.append(1)
        return frame(42,clock.t-.001)
    app=NS(wrist=w,cfg=cfg,vision=NS(read=read))
    s=WristGrasp(app,cfg['post_grasp'],g)
    s.guard=lambda:setattr(clock,'t',clock.t+.05)
    item,q,point=s.confirm()
    assert count==[1] and item['number']==42
    assert point==pytest.approx([0.,0.,.4])


def test_turn_and_grasp_arrival_accept_configured_joint_tolerances(configs):
    from b2w_perception.wrist_search_core import SettleTracker
    cfg,g=configs;s=WristGrasp(NS(cfg=cfg),cfg['post_grasp'],g)
    settings=s.settings('turn',turn=True)
    tracker=SettleTracker(settings['joint_position_tolerance_rad'],
                          settings['joint_speed_tolerance_rad_s'],settings['settle_stable_sec'])
    for t in (0.,.1,.21):
        result=tracker.update(t,np.full(6,.06),np.full(6,.08),np.zeros(6))
    assert result['settled']
    assert s.settings('grasp')['joint_position_tolerance_rad']==.08
    assert s.settings('grasp')['joint_speed_rad_s']==.1


def test_world_target_hint_accounts_for_translation_turn_and_crouch():
    mount=np.eye(4);mount[:3,3]=[.078,0,.09602]
    initial=transform_from_pose([10.,20.,.6],[0.,0.,math.sin(.3),math.cos(.3)])
    initial_target=np.array([1.5,.2,-.4])
    world=(initial@np.r_[initial_target,1.])[:3]
    current=transform_from_pose([10.4,20.7,.3],[0.,0.,math.sin(.5),math.cos(.5)])
    hint=world_target_to_arm(world,current,mount)
    np.testing.assert_allclose((current@mount@np.r_[hint,1.])[:3],world)
    assert hint[2]==pytest.approx(-.19602)
    assert not np.allclose(hint,initial_target)


def test_sport_body_pose_preserves_height_and_imu_pitch_without_body_height_field():
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler('xyz',[.1,-.2,.3]).as_quat()
    msg=NS(error_code=0,position=[1.,2.,.28],body_height=0.,
           imu_state=NS(quaternion=[q[3],q[0],q[1],q[2]]))
    t=sport_body_transform(msg)
    assert t[2,3]==pytest.approx(.28)
    np.testing.assert_allclose(t[:3,:3],Rotation.from_quat(q).as_matrix())


@pytest.mark.parametrize('bad',['missing','stale'])
def test_old_target_is_never_used_with_missing_or_stale_body_pose(configs,bad,monkeypatch):
    cfg,g=configs;clock=NS(t=10.)
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_grasp.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    app=NS(approach=NS(target_world=None if bad=='missing' else np.zeros(3),t_base_arm=np.eye(4)),
           body_pose=np.eye(4),body_pose_at=time.monotonic()-1.,vision=NS(read=lambda:frame(1,clock.t-.001)))
    grasp=WristGrasp(app,cfg['post_grasp'],g);grasp.guard=lambda:None
    with pytest.raises(RuntimeError):grasp.estimated_target_arm()


@pytest.fixture
def model():
    sys.path.insert(0,str(ROOT/'unitree_z1/z1_sdk/lib'))
    import unitree_arm_interface as sdk
    return sdk.Z1Model(np.zeros(3),0.,np.zeros(3),np.eye(3))


@pytest.mark.parametrize('side,yaw',[('left',-90.),('right',90.)])
def test_real_sdk_single_side_view_and_reset_have_verified_paths(configs,model,side,yaw):
    cfg,g=configs;camera=np.array(g['calibration']['T_end_camera'])
    plan=plan_observation(model,np.array(cfg['reset']['target_q_rad']),camera,cfg['observation'])
    q=np.array(plan['goal_q_rad'])
    w=NS(model=model,camera_transform=camera,snapshot=lambda:(q.copy(),np.zeros(6)))
    sequence=WristGrasp(NS(wrist=w,cfg=cfg),cfg['post_grasp'],g)
    sequence.guard=lambda:None
    p,settings,error=sequence.plan_hint_view(np.array([.1,-.7 if side=='left' else .7,.1]))
    q=np.array(p['goal_q_rad'])
    np.testing.assert_allclose(model.forwardKinematics(q,6),np.array(p['T_arm_end']),atol=.003)
    assert error<1.
    assert len(settings['pitch_candidates_deg'])==1
    assert p['sweep']['optical_sweep_deg']==pytest.approx(12.,abs=.01)
    assert plan_startup_reset(model,q,camera,cfg['reset'])['motor_zero_calibration'] is False


@pytest.mark.parametrize('side,sign',[('left',-1.),('right',1.)])
@pytest.mark.parametrize('height',[.12,.2,.3])
def test_real_sdk_demo_grasp_paths_are_reachable_on_both_sides(configs,model,side,sign,height):
    cfg,g=configs;camera=np.array(g['calibration']['T_end_camera'])
    sys.path.insert(0,str(ROOT/'unitree_z1/yolo/grasp_demo'))
    plan=plan_observation(model,np.array(cfg['reset']['target_q_rad']),camera,
         dict(cfg['observation'],target_end_xyz_m=[0.,sign*.2,.3],yaw_deg=sign*90.),4,math.radians(10))
    q=np.array(plan['goal_q_rad']);w=NS(model=model,camera_transform=camera)
    s=WristGrasp(NS(wrist=w,cfg=cfg),cfg['post_grasp'],g)
    s.guard=lambda:None
    point=np.array([0.,sign*.45,height]);item=dict(box=[270,210,370,270],meta=dict(fx=600.,fy=600.))
    grip=s.solve(item,q,point)
    actual=np.asarray(model.forwardKinematics(grip,6))@np.array(g['calibration']['T_end_tcp'])
    assert np.linalg.norm(actual[:3,3]-point)<=.040001
    assert grip.shape==(6,)


def test_real_sdk_finds_verified_candidate_for_reported_center_beyond_local_radius(configs,model):
    cfg,g=configs;sys.path.insert(0,str(ROOT/'unitree_z1/yolo/grasp_demo'))
    w=NS(model=model,camera_transform=np.array(g['calibration']['T_end_camera']))
    s=WristGrasp(NS(wrist=w,cfg=cfg),cfg['post_grasp'],g);s.guard=lambda:None
    q=np.array([-1.7683319,.8240073,-.5463686,-.0054576,-.0427096,-.0013983])
    point=np.array([-.44618657,-.61995836,.00398187])
    grip=s.solve(dict(box=[270,210,370,270],meta=dict(fx=600.,fy=600.)),q,point)
    tcp=np.array(model.forwardKinematics(grip,6))@np.array(g['calibration']['T_end_tcp'])
    assert np.linalg.norm(tcp[:3,3]-point)>.04
    assert np.isfinite(grip).all()


@pytest.mark.parametrize('moved',[False,True])
def test_slow_planning_refreshes_same_view_before_opening(sequence,monkeypatch,moved):
    s=sequence;clock=NS(t=10.)
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    point=np.array([0.,-.4,.2]);q=np.zeros(6)
    s.search=lambda:(frame(stamp=clock.t),q,point)
    def solve(*args):clock.t+=3.;return np.full(6,.2)
    s.solve=solve
    s.confirm=lambda:(frame(stamp=clock.t),q,point+np.array([.03 if moved else 0.,0,0]))
    if moved:
        with pytest.raises(RuntimeError,match='moved during IK planning'):s.run()
        assert 'open' not in s.events
    else:
        s.run()
        assert s.events.index('GRASP_REFRESHING')<s.events.index('open')


@pytest.fixture
def sequence(configs,tmp_path,monkeypatch):
    cfg,g=configs;events=[]
    cfg['post_grasp']['visual_servo']['enabled']=False
    @contextmanager
    def ownership():
        events.append('take_control')
        try:yield
        finally:events.append('resume_hold')
    w=NS(pump=lambda:None,should_stop=lambda:False,exclusive_commands=ownership,
         model=NS(forwardKinematics=lambda q,link:np.eye(4)),camera_transform=np.eye(4),gripper=-.5)
    app=NS(wrist=w,cfg=cfg,output=tmp_path,policy=NS(state='ARRIVED'))
    app.publish=lambda event,**kw:events.append(app.policy.state)
    s=WristGrasp(app,cfg['post_grasp'],g);s.guard=lambda:None
    target=(frame(stamp=time.monotonic()),np.zeros(6),np.array([0.,-.4,.2]))
    s.search=lambda:target;s.confirm=lambda:target
    s.solve=lambda *args:np.full(6,.2)
    s.move=lambda goal,label,**kw:events.append(label)
    s.gripper=lambda target,closing=False:events.append('close' if closing else 'open') or False
    monkeypatch.setattr('wrist_grasp.validate_observation_path',lambda *args,**kw:[])
    s.events=events
    return s


def test_full_sequence_grips_directly_then_resets_once(sequence):
    s=sequence;s.run();s.run()
    assert s.app.policy.state=='GRASP_DONE' and s.started and not s.running
    assert s.events.index('GRASP_APPROACHING')<s.events.index('close')<s.events.index('合爪归位')
    assert 'GRASP_PREAPPROACH' not in s.events and 'GRASP_RECHECKING' not in s.events
    assert 'GRASP_RETREATING' not in s.events
    assert s.events.count('take_control')==1 and s.events[-1]=='resume_hold'
    assert (s.app.output/'grasp_target.json').exists()


@pytest.mark.parametrize('fault',['depth_missing','approach_failed','unreachable'])
def test_failed_direct_grasp_never_closes_or_auto_resets(sequence,fault):
    s=sequence
    if fault=='depth_missing':
        def missing():raise RuntimeError('no valid depth')
        s.search=missing
    if fault=='approach_failed':
        def move(*args,**kw):raise RuntimeError('tracking failed')
        s.move=move
    if fault=='unreachable':
        def solve(*args):raise ValueError('no IK')
        s.solve=solve
    with pytest.raises((ValueError,RuntimeError)):s.run()
    assert 'close' not in s.events and '合爪归位' not in s.events
    assert s.events[-1]=='resume_hold' and not s.running


@pytest.mark.parametrize('fault',['operator','exit','odom','moving_base','gate','arm','camera','joints'])
def test_sequence_watchdogs_interrupt_without_issuing_commands(configs,fault):
    cfg,g=configs;now=time.monotonic()
    w=NS(error=None,target=np.zeros(6),snapshot=lambda:(np.zeros(6),np.zeros(6)),
         arm=NS(feedbackHealthy=lambda:True,getCurrentState=lambda:2),sdk=NS(ArmFSMState=NS(JOINTCTRL=2)))
    app=NS(wrist=w,cfg=cfg,stop_requested=False,exit_requested=False,release_requested=False,
           odom_at=now,base_speed=0.,gates={},vision=NS(error=None,read=lambda:frame(stamp=now)))
    if fault=='operator':app.stop_requested=True
    if fault=='exit':app.exit_requested=True
    if fault=='odom':app.odom_at=now-1.
    if fault=='moving_base':app.base_speed=.04
    if fault=='gate':app.gates={'bridge':True}
    if fault=='arm':w.arm.feedbackHealthy=lambda:False
    if fault=='camera':app.vision.error='camera disconnected'
    if fault=='joints':w.snapshot=lambda:(np.full(6,.2),np.zeros(6))
    assert WristGrasp(app,cfg['post_grasp'],g).interrupted()


def test_exclusive_commands_join_scan_worker_and_keep_same_sdk_alive():
    w=object.__new__(Wrist);w.done=threading.Event();w.scanning=threading.Event()
    w.scanning.set();w.error=None;w.running=True
    w.target=np.zeros(6);w.snapshot=lambda:(np.full(6,.1),np.zeros(6))
    w.sdk=NS(ArmFSMState=NS(JOINTCTRL=2))
    w.arm=NS(feedbackHealthy=lambda:True,getCurrentState=lambda:2)
    old_arm=w.arm;w.worker=threading.Thread(target=lambda:w.done.wait());w.worker.start()
    original=w.worker;resumed=threading.Event();w.control=resumed.set
    with pytest.raises(RuntimeError,match='cancel'):
        with w.exclusive_commands():
            assert not original.is_alive() and not w.scanning.is_set()
            assert w.arm is old_arm and w.running
            raise RuntimeError('cancel')
    w.worker.join(timeout=1.)
    assert resumed.is_set() and w.arm is old_arm and w.running
    assert w.target==pytest.approx(np.full(6,.1)) and not w.done.is_set()


@pytest.mark.parametrize('mode',['normal','contact','open_stalled','partial_open','invalid_feedback','invalid_open_feedback'])
def test_gripper_ramp_and_contact_do_not_report_verified_grasp(configs,monkeypatch,mode):
    cfg,g=configs;clock=NS(t=0.);commands=[];velocities=[];events=[];measured=NS(q=-.5)
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_grasp.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    def send(q,v,*args):
        commands.append((clock.t,q))
        velocities.append(v)
        if mode=='normal':measured.q=q
        if mode=='partial_open':measured.q=max(q,-1.1)
    w=NS(target=np.zeros(6),gripper=-.5,arm=NS(
        lowstate=NS(getGripperQ=lambda:float('nan') if mode in ('invalid_feedback','invalid_open_feedback') else measured.q),
        setArmCmd=lambda *args:None,setGripperCmd=send))
    app=NS(wrist=w,publish=lambda event,**fields:events.append((event,fields)))
    s=WristGrasp(app,cfg['post_grasp'],g);s.guard=lambda:None
    if mode=='open_stalled':measured.q=.005;w.gripper=0.
    if mode=='invalid_feedback':
        with pytest.raises(RuntimeError,match='Invalid measured'):s.gripper(0.,closing=True)
        assert not commands
    elif mode in ('open_stalled','partial_open','invalid_open_feedback'):
        assert s.gripper(-math.pi/2) is False
        assert s.gripper_status['gripper_result']=='open_command_completed'
        assert s.gripper_status['gripper_open_check_enabled'] is False
        assert w.gripper==pytest.approx(-math.pi/2)
        assert clock.t<cfg['post_grasp']['gripper_timeout_sec']
        assert min(velocities)<0. and velocities[-1]==0.
        if mode=='invalid_open_feedback':assert s.gripper_status['gripper_actual_rad'] is None
    else:
        contact=s.gripper(0.,closing=True)
        assert contact is (mode=='contact') and w.gripper==0.
        for (t0,q0),(t1,q1) in zip(commands,commands[1:]):
            assert abs(q1-q0)<=cfg['post_grasp']['gripper_speed_rad_s']*(t1-t0)+1e-9
        if contact:assert events[-1][1]['grasp_verified'] is False


def test_target_directed_view_handles_a_target_close_to_arm_base(configs):
    cfg,g=configs
    sys.path.insert(0,str(ROOT/'unitree_z1/z1_sdk/lib'))
    import unitree_arm_interface as sdk
    model=sdk.Z1Model(np.zeros(3),0.,np.zeros(3),np.eye(3))
    camera=np.array(g['calibration']['T_end_camera'])
    initial=plan_observation(model,np.array(cfg['reset']['target_q_rad']),camera,cfg['observation'])
    start=np.array(initial['goal_q_rad'])
    w=NS(model=model,camera_transform=camera,snapshot=lambda:(start.copy(),np.zeros(6)))
    grasp=WristGrasp(NS(cfg=cfg,wrist=w),cfg['post_grasp'],g);grasp.guard=lambda:None
    hint=np.array([-.03,-.05,-.1])
    plan,settings,error=grasp.plan_hint_view(hint)
    assert error<=5.
    assert plan['goal_q_rad'][0]<0.
    assert len(plan['path_q_rad'])==151


def test_real_close_target_is_not_rejected_by_camera_distance(configs):
    cfg,g=configs
    sys.path.insert(0,str(ROOT/'unitree_z1/z1_sdk/lib'))
    import unitree_arm_interface as sdk
    model=sdk.Z1Model(np.zeros(3),0.,np.zeros(3),np.eye(3))
    camera=np.array(g['calibration']['T_end_camera'])
    # Replay the measured arm joints and transformed sponge position from the fault.
    start=np.array([-.001274383976124227,.823243260383606,-.5447165966033936,
                    -.011705614626407623,.0016991798765957355,-.0010796877322718501])
    hint=np.array([-.039736814609558344,-.06061803241929179,.2782463502763335])
    wrist=NS(model=model,camera_transform=camera,snapshot=lambda:(start.copy(),np.zeros(6)))
    grasp=WristGrasp(NS(cfg=cfg,wrist=wrist),cfg['post_grasp'],g);grasp.guard=lambda:None
    plan,_,error=grasp.plan_hint_view(hint)
    actual_camera=np.asarray(model.forwardKinematics(np.array(plan['goal_q_rad']),6))@camera
    distance=np.linalg.norm(hint-actual_camera[:3,3])
    assert 0.<distance<.1
    assert error<=5.
    assert len(plan['path_q_rad'])==151
    assert plan['sweep']['amplitude_deg']==pytest.approx(6.)


@pytest.mark.parametrize('visible',[False,True])
def test_arrival_aim_uses_new_pose_and_prefers_new_visible_depth(configs,monkeypatch,visible):
    cfg,g=configs;clock=NS(t=100.);polls=[]
    monkeypatch.setattr('wrist_grasp.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_grasp.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    mount=np.eye(4);mount[:3,3]=[.078,0.,.09602]
    world=np.array([1.,.5,.1]);new_body=transform_from_pose([.8,.2,.3],[0.,0.,math.sin(.2),math.cos(.2)])
    end=np.eye(4);end[:3,3]=[.2,0.,.3]
    wrist=NS(model=NS(forwardKinematics=lambda *args:end),camera_transform=np.eye(4),
             snapshot=lambda:(np.zeros(6),np.zeros(6)))
    app=NS(cfg=cfg,wrist=wrist,body_pose=np.eye(4),body_pose_at=99.99,
           approach=NS(target_world=world.copy(),t_base_arm=mount))
    def read():
        item=frame(7,clock.t-.001)
        if not visible:item.update(box=None,confidence=0.,camera_xyz_m=None)
        return item
    app.vision=NS(read=read)
    def pump():
        polls.append(True);clock.t+=.02
        if len(polls)>1:app.body_pose=new_body;app.body_pose_at=clock.t
    grasp=WristGrasp(app,cfg['post_grasp'],g);grasp.guard=pump
    hint=grasp.estimated_target_arm()
    assert len(polls)==2  # The cached pre-arrival pose was not accepted.
    if visible:
        np.testing.assert_allclose(hint,[.2,0.,.7])
        np.testing.assert_allclose(app.approach.target_world,(new_body@mount@np.r_[hint,1.])[:3])
        assert grasp.target_update['target_update_source']=='post_arrival_rgbd'
    else:
        np.testing.assert_allclose(hint,world_target_to_arm(world,new_body,mount))
        assert grasp.target_update['target_update_source']=='post_arrival_sport_pose'
    assert grasp.target_update['body_pose_received_monotonic_s']>grasp.target_update['pose_requested_monotonic_s']


@pytest.mark.parametrize('hint',[
    [-.039736814609558344,-.06061803241929179,.2782463502763335],
    [-.039736814609558344,.06061803241929179,.2782463502763335],
    [.2,-.5,.0],[.2,.5,.0],
])
def test_observation_arm_and_lens_turn_toward_target_side(configs,hint):
    cfg,g=configs
    sys.path.insert(0,str(ROOT/'unitree_z1/z1_sdk/lib'))
    import unitree_arm_interface as sdk
    model=sdk.Z1Model(np.zeros(3),0.,np.zeros(3),np.eye(3))
    camera=np.array(g['calibration']['T_end_camera']);hint=np.array(hint)
    start=np.array([.002407169435173273,.8234438896179199,-.5443035960197449,
                    -.008088961243629456,-.004129949025809765,.002354071708396077])
    wrist=NS(model=model,camera_transform=camera,snapshot=lambda:(start.copy(),np.zeros(6)))
    grasp=WristGrasp(NS(cfg=cfg,wrist=wrist),cfg['post_grasp'],g);grasp.guard=lambda:None
    plan,settings,error=grasp.plan_hint_view(hint)
    goal=np.array(plan['goal_q_rad']);lens=np.asarray(model.forwardKinematics(goal,6))@camera
    assert goal[0]*hint[1]>=0.  # Right-hand targets must not select a left J1 turn.
    assert lens[1,2]*hint[1]>0.
    assert error<=5.
    assert lens[:2,2]@hint[:2]>=0.  # Do not look back inward after overshooting.
    assert len(plan['path_q_rad'])==151


def test_portrait_camera_maps_ground_sponge_down_with_arm_already_at_zero_roll(configs,model):
    cfg,g=configs
    import unitree_arm_interface as sdk
    # Match the gripper-enabled SDK end frame used by the running ArmInterface.
    model=sdk.Z1Model(np.array([.0382,0.,0.]),0.,np.zeros(3),np.eye(3))
    camera=np.array(g['calibration']['T_end_camera'])
    # Saved initial exposure: floor is on the image's left, while J6 is ~0.13deg.
    q=np.array([.0020708744414150715,.8231016993522644,-.5446870923042297,
                -.009381040930747986,-.003970650490373373,.0022714724764227867])
    point=np.array([-.5058490810180812,-.06658244229636207,1.846000087680295])
    base_camera=np.array(cfg['approach']['T_base_arm'])@np.array(model.forwardKinematics(q,6))@camera
    target=(base_camera@np.r_[point,1.])[:3]
    np.testing.assert_allclose(target,[1.8727087145717967,.06377497069479247,-.6133800204026177],atol=.002)
    assert base_camera[2,0]>.9  # Image right is physical up; image left is down.
    assert abs(q[5])<.01
    np.testing.assert_allclose(camera[:3,3],[-.04,0.,-.04])
