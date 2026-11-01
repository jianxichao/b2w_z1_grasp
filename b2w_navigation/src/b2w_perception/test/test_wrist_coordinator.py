from pathlib import Path
from types import SimpleNamespace
import sys,time
import numpy as np
import pytest
from b2w_perception.wrist_search_core import SearchPolicy

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from search_with_wrist import Coordinator


@pytest.fixture
def app(tmp_path):
    node=object.__new__(Coordinator)
    node.cfg={'frame_timeout_sec':2.,'confidence':.65,'approach':{'depth_wait_timeout_sec':10.}}
    node.base_speed=0.
    node.explore=False;node.output=tmp_path
    node.policy=SearchPolicy();node.policy.start(time.monotonic())
    node.stop_requested=node.release_requested=False
    node.limit_event=None;node.still_since=None;node.last_status=time.monotonic()
    node.events=[]
    node.wrist=SimpleNamespace(error=None,paused=0,resumed=0)
    node.wrist.pause=lambda:setattr(node.wrist,'paused',node.wrist.paused+1)
    node.wrist.resume=lambda:setattr(node.wrist,'resumed',node.wrist.resumed+1)
    node.wrist.snapshot=lambda:(np.zeros(6),np.zeros(6))
    node.publish=lambda event,**kw:node.events.append(event)
    node.stop_base=lambda:node.events.append('stop_base') or True
    node.enable_base=lambda:node.events.append('enable_base')
    node.item=dict(number=1,captured_at=time.monotonic(),received_at=time.monotonic(),
                   box=None,confidence=0.,camera_xyz_m=None,annotated=np.zeros((10,10,3),np.uint8))
    node.vision=SimpleNamespace(error=None,read=lambda:node.item)
    return node


def test_single_candidate_does_not_interrupt_search_before_five_votes(app):
    app.item.update(box=[10,10,50,70],confidence=.9)
    app.step()
    assert app.policy.state=='SEARCHING' and app.policy.vote_count==1
    assert app.wrist.paused==0 and 'stop_base' not in app.events
    assert not (app.output/'found.json').exists()


def test_confirmed_can_latches_hold_and_never_calls_approach(app):
    app.item.update(box=[10,10,50,70],confidence=.9)
    app.step()
    for seq in (2,3,4,5):
        app.still_since=time.monotonic()-1
        app.item.update(number=seq,captured_at=time.monotonic(),received_at=time.monotonic())
        app.step()
    assert app.policy.state=='FOUND'
    assert app.wrist.resumed==0
    assert 'enable_base' not in app.events
    import json
    found=json.loads((app.output/'found.json').read_text())
    assert not found['approach_enabled'] and not found['grasp_enabled']
    assert not found['map_target_published']
    app.item.update(number=5,box=None)
    app.step()
    assert app.policy.state=='FOUND'


@pytest.mark.parametrize('fault',['vision','arm','old_exposure'])
def test_sensor_fault_cancels_both_motions_and_never_automatically_resumes(app,fault):
    if fault=='vision':app.vision.error='camera disconnected'
    elif fault=='arm':app.wrist.error='arm feedback lost'
    else:app.item['captured_at']=time.monotonic()-3  # New inference of old data is still stale.
    app.step()
    assert app.policy.state=='FAULT'
    assert app.wrist.paused==1 and app.events[0]=='stop_base'
    app.vision.error=app.wrist.error=None
    app.item['captured_at']=time.monotonic()
    app.step()
    assert app.policy.state=='FAULT' and 'enable_base' not in app.events


def test_release_during_search_is_rejected_but_stop_keeps_sdk_owner(app):
    response=SimpleNamespace(success=False,message='')
    app.on_release(None,response)
    assert not response.success and not app.release_requested
    app.on_stop(None,response)
    assert response.success and app.stop_requested
    app.step()
    assert app.policy.state=='STOPPED'
    app.on_release(None,response)
    assert response.success and app.release_requested


def test_start_enters_search_without_an_extra_verification_phase(app,monkeypatch):
    app.item.update(box=[10,10,50,70],confidence=.9,captured_at=100.)
    app.ros=SimpleNamespace(spin_once=lambda *args,**kw:None)
    app.node=None
    app.wrist.report={}
    ticks=iter(np.arange(100.,110.,.1))
    monkeypatch.setattr('search_with_wrist.time.monotonic',lambda:float(next(ticks)))
    app.start()
    assert app.policy.state=='SEARCHING'
    assert 'enable_base' in app.events and app.wrist.resumed==1
    assert 'search_started' in app.events


def test_arm_feedback_fault_clears_old_enable_command_instead_of_rearming():
    import threading
    from wrist_search_io import Wrist
    wrist=object.__new__(Wrist)
    wrist.done=threading.Event();wrist.scanning=threading.Event();wrist.scanning.set()
    calls=[]
    wrist.sdk=SimpleNamespace(ArmFSMState=SimpleNamespace(JOINTCTRL=2))
    wrist.arm=SimpleNamespace(feedbackHealthy=lambda:False,getCurrentState=lambda:2,
                              requestPassive=lambda:calls.append('passive'),
                              requestJointHold=lambda:calls.append('hold'))
    wrist.control()
    assert calls==['passive']
    assert wrist.error is not None and not wrist.scanning.is_set()


@pytest.mark.parametrize('fault',['udp','motor_connection','motor_error'])
def test_communication_loss_suspends_without_passive_but_motor_fault_keeps_protection(tmp_path,fault):
    import threading,json
    from wrist_search_io import Wrist
    wrist=object.__new__(Wrist);wrist.output=tmp_path
    wrist.scanning=threading.Event();wrist.scanning.set()
    wrist.sdk=SimpleNamespace(ArmFSMState=SimpleNamespace(JOINTCTRL=2))
    calls=[]
    wrist.arm=SimpleNamespace(feedbackHealthy=lambda:False,getCurrentState=lambda:2,
        feedbackStatus=lambda:dict(udp_disconnected=fault=='udp',frame_received=True,positions_finite=True,
            controller_state=2,motor_faults=[] if fault=='udp' else [dict(joint=2,error=1 if fault=='motor_error' else 0,connection=1 if fault=='motor_connection' else 0)]),
        suspendTracking=lambda:calls.append('suspend'),requestPassive=lambda:calls.append('passive'))
    wrist.latch_fault('transport or motor fault')
    assert calls==(['passive'] if fault=='motor_error' else ['suspend'])
    assert not wrist.scanning.is_set()
    status=json.loads((tmp_path/'arm_fault_status.json').read_text())
    assert status['udp_disconnected'] is (fault=='udp')


def test_release_is_rejected_during_observation_preparation(app):
    app.policy.state='PREPARING'
    response=SimpleNamespace(success=False,message='')
    app.on_release(None,response)
    assert not response.success and not app.release_requested


def test_post_grasp_starts_only_after_arrival_and_only_once(app):
    app.grasp=SimpleNamespace(started=False,calls=0)
    def run():
        app.grasp.calls+=1;app.grasp.started=True;app.policy.state='GRASP_DONE'
    app.grasp.run=run
    app.policy.state='SPORT_POSITION_MOVING';app.step()
    assert app.grasp.calls==0
    app.policy.state='ARRIVED';app.step();app.step()
    assert app.grasp.calls==1 and app.policy.state=='GRASP_DONE'


def test_post_grasp_is_not_started_after_operator_stop(app):
    app.policy.state='ARRIVED';app.stop_requested=True
    app.grasp=SimpleNamespace(started=False,run=lambda:pytest.fail('Stopped approach must not start grasp'))
    app.step()
    assert app.policy.state=='STOPPED'


def test_completed_grasp_can_release_control_explicitly(app):
    app.policy.state='GRASP_DONE';response=SimpleNamespace(success=False,message='')
    app.on_release(None,response)
    assert response.success and app.release_requested


def test_sport_body_pose_callback_rejects_repeated_and_invalid_feedback(app):
    msg=SimpleNamespace(error_code=0,body_height=0.,position=[1.,2.,.6],
        stamp=SimpleNamespace(sec=1,nanosec=2),imu_state=SimpleNamespace(quaternion=[1.,0.,0.,0.]))
    app.on_sport_body(msg)
    assert app.body_pose[:3,3]==pytest.approx([1.,2.,.6])
    first=app.body_pose_at;msg.position=[2.,3.,.3]
    app.on_sport_body(msg)
    assert app.body_pose_at==first and app.body_pose[2,3]==.6
    msg.stamp.nanosec=3;msg.error_code=1
    app.on_sport_body(msg)
    assert app.body_pose is None


def test_preparation_stop_callback_works_before_sdk_owner_is_attached(app):
    app.policy.state='PREPARING';app.wrist=None
    response=SimpleNamespace(success=False,message='')
    app.on_stop(None,response)
    assert response.success and app.stop_requested


def test_stop_during_arm_preparation_sends_no_motion_then_latches_measured_hold():
    import threading
    from wrist_search_io import Wrist
    wrist=object.__new__(Wrist)
    wrist.center=np.zeros(6);wrist.target=np.zeros(6);wrist.gripper=-.5
    wrist.should_stop=lambda:True;wrist.pump=lambda:None
    wrist.sdk=SimpleNamespace(ArmFSMState=SimpleNamespace(JOINTCTRL=2))
    calls=[];wrist.scanning=threading.Event()
    wrist.snapshot=lambda:(np.zeros(6),np.zeros(6))
    wrist.arm=SimpleNamespace(feedbackHealthy=lambda:True,getCurrentState=lambda:2,
        requestJointHold=lambda:calls.append('hold'),requestPassive=lambda:calls.append('passive'))
    with pytest.raises(RuntimeError,match='Operator/sensor stop'):
        wrist.move_to_observation({'start_q_rad':[0.]*6,'goal_q_rad':[.1]*6},{'joint_speed_rad_s':.1})
    wrist.latch_fault('Interrupted')
    assert calls==['hold'] and wrist.error=='Interrupted'


def test_status_distinguishes_requested_exploration_from_enabled_motion(app):
    import threading,json
    app.explore=True;app.gates={'sport':False,'safety':False}
    app.wrist.scanning=threading.Event();app.wrist.running=True
    app.wrist.arm=SimpleNamespace(feedbackHealthy=lambda:True)
    messages=[]
    app.String=lambda **kw:SimpleNamespace(**kw)
    app.publisher=SimpleNamespace(publish=messages.append)
    Coordinator.publish(app,'heartbeat',arm_error=None)
    data=json.loads(messages[0].data)
    assert data['explore'] is True
    assert data['base_output_enabled'] is False
    assert data['wrist_scanning'] is False
    assert data['arm_control_alive'] is True


def test_approach_mode_votes_on_rgb_then_waits_for_depth_without_fault(app):
    calls=[]
    app.approach=SimpleNamespace(record=lambda *a:None,cancel=lambda:None,tick=lambda:None,
                                begin=lambda *a:calls.append('begin'))
    app.item.update(box=[10,10,50,70],confidence=.9,camera_xyz_m=None)
    app.step()
    assert app.policy.state=='SEARCHING' and 'stop_base' not in app.events
    for seq in (2,3,4,5):
        app.still_since=time.monotonic()-1
        app.item.update(number=seq,captured_at=time.monotonic())
        app.step()
    assert app.policy.state=='TARGET_LOCALIZING' and not calls


def test_voted_target_handoff_blocks_release_and_never_resumes_frontier(app):
    calls=[]
    def stop_base():
        if app.policy.state=='TARGET_LOCALIZING':
            response=SimpleNamespace(success=False,message='')
            app.on_release(None,response)
            assert not response.success
        app.events.append('stop_base');return True
    app.stop_base=stop_base
    def begin(*args):
        calls.append('begin');app.policy.state='APPROACH_PLANNING'
    app.approach=SimpleNamespace(record=lambda *a:calls.append('record'),cancel=lambda:None,
        tick=lambda:None,begin=begin,target=np.array([2,0,.2]))
    app.item.update(box=[10,10,50,70],confidence=.9,camera_xyz_m=[0,0,2])
    app.step()
    for seq in (2,3,4,5):
        app.still_since=time.monotonic()-1
        app.item.update(number=seq,captured_at=time.monotonic())
        app.step()
    assert app.policy.state=='TARGET_LOCALIZING'
    app.item.update(number=6,captured_at=time.monotonic())
    app.step()
    assert calls.count('record')==0 and calls.count('begin')==1
    assert app.policy.state=='APPROACH_PLANNING'
    assert 'enable_base' not in app.events and app.wrist.resumed==0
    import json
    assert json.loads((app.output/'found.json').read_text())['map_target_published']


def test_misses_and_noisy_joint_speed_do_not_block_five_rgb_votes(app):
    app.wrist.snapshot=lambda:(np.zeros(6),np.full(6,.2))
    for seq in range(1,11):
        hit=seq in (1,3,5,7,10)
        app.item.update(number=seq,captured_at=time.monotonic(),
                        box=[10,10,50,70] if hit else None,confidence=.9 if hit else 0.)
        app.step()
    assert app.policy.state=='FOUND' and app.policy.vote_count==5
    assert app.events.count('stop_base')==1


def test_vote_can_wait_for_one_valid_depth_without_three_frame_cache(app):
    calls=[]
    def begin(*args):calls.append('begin');app.policy.state='APPROACH_PLANNING'
    app.approach=SimpleNamespace(cancel=lambda:None,tick=lambda:None,begin=begin,target=np.array([2,0,.2]))
    for seq in range(1,6):
        app.item.update(number=seq,captured_at=time.monotonic(),box=[10,10,50,70],confidence=.9)
        app.step()
    assert app.policy.state=='TARGET_LOCALIZING' and not calls
    app.item.update(number=6,captured_at=time.monotonic(),box=None,confidence=0.)
    app.step()
    assert app.policy.state=='TARGET_LOCALIZING' and not calls
    app.item.update(number=7,captured_at=time.monotonic(),box=[10,10,50,70],confidence=.9,camera_xyz_m=[0,0,2])
    app.step()
    assert app.policy.state=='APPROACH_PLANNING' and calls==['begin']


def test_no_depth_timeout_holds_found_instead_of_fault(app):
    app.approach=SimpleNamespace(cancel=lambda:None,tick=lambda:None,begin=lambda *a:pytest.fail('must not approach'))
    for seq in range(1,6):
        app.item.update(number=seq,captured_at=time.monotonic(),box=[10,10,50,70],confidence=.9)
        app.step()
    app.localization_started=time.monotonic()-11
    app.item.update(number=6,captured_at=time.monotonic())
    app.step()
    assert app.policy.state=='FOUND' and 'target_found_but_depth_unavailable' in app.events


def test_fault_reason_remains_latched_when_can_reappears(app):
    app.vision.error='camera disconnected';app.step()
    assert app.policy.state=='FAULT' and app.fault_reason=='camera disconnected'
    app.vision.error=None
    for seq in range(1,11):
        app.item.update(number=seq,captured_at=time.monotonic(),box=[10,10,50,70],confidence=.9)
        app.step()
    assert app.policy.state=='FAULT' and app.policy.vote_count==0
    assert 'enable_base' not in app.events


def test_no_localization_if_base_stop_was_not_acknowledged(app):
    app.approach=SimpleNamespace(cancel=lambda:None,tick=lambda:None,begin=lambda *a:pytest.fail('must not approach'))
    app.stop_base=lambda:False
    for seq in range(1,6):
        app.item.update(number=seq,captured_at=time.monotonic(),box=[10,10,50,70],confidence=.9)
        app.step()
    assert app.policy.state=='FAULT' and 'Base stop' in app.fault_reason


def test_ctrl_c_stops_base_then_resets_then_allows_exit(app):
    app.cfg['reset']={};app.exit_requested=True
    calls=[]
    app.wrist.reset_for_exit=lambda settings,cancel:calls.append(('reset',cancel()))
    app.step()
    assert calls==[('reset',False)] and app.events[0]=='stop_base'
    assert app.policy.state=='EXIT_READY' and app.release_requested
    assert 'exit_reset_complete' in app.events and not app.exiting


def test_reset_failure_keeps_process_and_sdk_hold(app):
    app.cfg['reset']={};app.exit_requested=True
    def fail(*args):raise RuntimeError('joint feedback unhealthy')
    app.wrist.reset_for_exit=fail
    app.step()
    assert app.policy.state=='FAULT' and not app.release_requested
    assert 'joint feedback unhealthy' in app.fault_reason
    assert not app.exiting and app.wrist.paused>=1


def test_second_ctrl_c_aborts_reset_and_does_not_exit(app):
    app.cfg['reset']={};app.exit_requested=True
    def interrupt(settings,cancel):
        app.request_exit()
        assert cancel()
        raise RuntimeError('exit reset interrupted')
    app.wrist.reset_for_exit=interrupt
    app.step()
    assert app.policy.state=='FAULT' and not app.release_requested


def test_exit_reset_does_not_require_a_healthy_camera(app):
    app.cfg['reset']={};app.exit_requested=True;app.vision.error='camera disconnected'
    app.wrist.reset_for_exit=lambda settings,cancel:None if not cancel() else pytest.fail('camera should not block reset')
    app.step()
    assert app.policy.state=='EXIT_READY' and app.release_requested


def test_exit_base_stop_failure_never_starts_arm_reset(app):
    app.cfg['reset']={};app.exit_requested=True
    app.stop_base=lambda:False
    app.wrist.reset_for_exit=lambda *a:pytest.fail('must not move arm')
    app.step()
    assert app.policy.state=='FAULT' and not app.release_requested


def test_sport_position_ctrl_c_waits_for_base_to_stop_before_arm_reset(app):
    app.cfg['reset']={};app.exit_requested=True;app.odom_at=time.monotonic();app.base_speed=.1
    app.approach=SimpleNamespace(backend='sport_velocity_goal',cancel=lambda:None)
    app.node=None
    def spin(*unused,**kw):app.base_speed=0.;app.odom_at=time.monotonic()
    app.ros=SimpleNamespace(spin_once=spin)
    app.wrist.reset_for_exit=lambda settings,cancel:None if not cancel() else pytest.fail('base still moving')
    app.step()
    assert app.policy.state=='EXIT_READY' and app.release_requested


def test_sport_backend_configuration_does_not_hide_active_exploration_gate(app):
    import json,threading
    app.explore=True;app.gates={'sport':True,'safety':True}
    app.approach=SimpleNamespace(backend='sport_velocity_goal',active=False,command_id=None)
    app.wrist.scanning=threading.Event();app.wrist.running=True
    app.wrist.arm=SimpleNamespace(feedbackHealthy=lambda:True)
    messages=[];app.String=lambda **kw:SimpleNamespace(**kw)
    app.publisher=SimpleNamespace(publish=messages.append)
    Coordinator.publish(app,'heartbeat')
    data=json.loads(messages[0].data)
    assert data['base_output_enabled'] and data['base_command_source']=='nav2_velocity'
    assert not data['sport_position_active']


@pytest.mark.parametrize('missing',[None,'scan','odom'])
def test_exploration_preparation_waits_for_subscription_data_and_names_missing_topic(app,monkeypatch,missing):
    clock=SimpleNamespace(t=10.)
    monkeypatch.setattr('search_with_wrist.time.monotonic',lambda:clock.t)
    app.node=None;app.explore=True;app.odom_at=app.scan_at=0.;app.base_speed=float('inf')
    app.item['captured_at']=clock.t
    def spin(*unused,**kw):
        clock.t+=kw['timeout_sec'];app.item['captured_at']=clock.t
        if clock.t>=10.2:
            if missing!='odom':app.odom_at=clock.t;app.base_speed=0.
            if missing!='scan':app.scan_at=clock.t
    app.ros=SimpleNamespace(spin_once=spin)
    if missing:
        topic='/scan' if missing=='scan' else '/b2w/sport/odom'
        with pytest.raises(RuntimeError,match=topic) as error:app.prepare_observation()
        assert '--explore requires arm_camera_search.launch.py' in str(error.value)
        assert clock.t>=15. and 'preparing_observation_base_stopped' not in app.events
    else:
        app.prepare_observation()
        assert 10.7<=clock.t<11.
        assert app.events[-1]=='preparing_observation_base_stopped'
    assert app.wrist.resumed==0


def test_exploration_preparation_operator_stop_does_not_wait_for_sensor_timeout(app,monkeypatch):
    clock=SimpleNamespace(t=10.)
    monkeypatch.setattr('search_with_wrist.time.monotonic',lambda:clock.t)
    app.node=None;app.explore=True;app.odom_at=app.scan_at=0.;app.item['captured_at']=clock.t
    def spin(*unused,**kw):clock.t+=.02;app.exit_requested=True
    app.ros=SimpleNamespace(spin_once=spin)
    with pytest.raises(RuntimeError,match='Operator canceled'):app.prepare_observation()
    assert clock.t<10.1
