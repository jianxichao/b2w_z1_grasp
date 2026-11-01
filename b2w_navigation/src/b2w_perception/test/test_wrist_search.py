import math
from pathlib import Path
import sys
import numpy as np
import pytest
from b2w_perception.wrist_search_core import SearchPolicy,scan_offset,validate_sweep

BOX=[10,10,50,70]


def test_no_can_keeps_searching_until_bounded_time_limit():
    p=SearchPolicy(max_search_sec=10);p.start(100)
    for i in range(5):
        assert p.observe(i,101+i,101+i,None,0,True) is None
        assert p.state=='SEARCHING'
    assert p.tick(110)=='STOP' and p.state=='LIMIT'


def test_five_votes_within_ten_frames_accept_intermittent_misses_without_velocity_gate():
    p=SearchPolicy();p.start(100)
    sequence=[True,False,True,False,True,False,True,False,False,True]
    for i,hit in enumerate(sequence,1):
        result=p.observe(i,100+i,100+i,BOX if hit else None,.8 if hit else 0,False)
        assert result==('FOUND' if i==10 else None)
    assert p.vote_count==5 and p.state=='FOUND'
    assert p.observe(11,111,111,None,0,False) is None
    assert p.tick(300) is None


def test_repeated_or_out_of_order_frame_cannot_add_votes():
    p=SearchPolicy();p.start(100)
    p.observe(5,101,101,BOX,.9,False)
    for frame in (5,4,3):assert p.observe(frame,101,102,BOX,.9,True) is None
    assert p.vote_count==1 and len(p.votes)==1


def test_window_rolls_forward_instead_of_resetting_after_ten_frames():
    p=SearchPolicy();p.start(100)
    for i in range(1,11):
        hit=i in (2,4,6,8)
        assert p.observe(i,100+i,100+i,BOX if hit else None,.9 if hit else 0,False) is None
    assert p.state=='SEARCHING' and p.vote_count==4 and len(p.votes)==10
    assert p.observe(11,111,111,BOX,.9,False)=='FOUND'
    assert p.vote_count==5 and len(p.votes)==10


def test_votes_can_confirm_before_ten_frames_and_misses_do_not_reset():
    p=SearchPolicy();p.start(100)
    for i in range(1,7):
        hit=i!=3
        result=p.observe(i,100+i,100+i,BOX if hit else None,.9 if hit else 0,False)
    assert result=='FOUND' and p.vote_count==5 and len(p.votes)==6


def test_fault_state_is_latched_even_if_new_can_frames_arrive():
    p=SearchPolicy();p.start(100);p.state='FAULT'
    for i in range(1,11):assert p.observe(i,100+i,100+i,BOX,.9,True) is None
    assert p.state=='FAULT' and p.vote_count==0


def test_old_or_future_exposure_does_not_pause_a_search():
    p=SearchPolicy();p.start(100)
    assert p.observe(1,101,104,BOX,.9,True) is None
    assert p.observe(2,105,104,BOX,.9,True) is None
    assert p.state=='SEARCHING'


@pytest.mark.parametrize('speed',[.03,.09,.12,.15])
def test_scan_is_bounded_and_velocity_never_exceeds_requested_limit(speed):
    amplitude=math.radians(15)
    t=np.arange(0,130,.002)
    values=np.array([scan_offset(x,amplitude,speed) for x in t])
    assert values[0]==0
    assert np.max(np.abs(values))<=amplitude
    assert np.max(np.abs(np.diff(values)/.002))<=speed+1e-5
    assert values.min()<-.99*amplitude and values.max()>.99*amplitude


@pytest.fixture
def model():
    # Model construction only: does not create an ArmInterface or UDP socket.
    root=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(root/'unitree_z1/z1_sdk/lib'))
    import unitree_arm_interface as sdk
    return sdk.Z1Model(np.zeros(3),0.,np.zeros(3),np.eye(3))


def test_real_model_accepts_raised_j5_scan_and_rejects_j6_image_roll(model):
    q=np.array([0.,1.2,-1.3,-.4,0.,0.])
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    report=validate_sweep(model,q,camera,joint=4)
    assert report['optical_sweep_deg']==pytest.approx(30,abs=.001)
    with pytest.raises(ValueError,match='rolls the image'):
        validate_sweep(model,q,camera,joint=5)


def test_real_model_refuses_limits_before_any_sweep(model):
    q=np.array([0.,1.2,-1.3,-.4,float(model.getJointQMax()[4])-.01,0.])
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    with pytest.raises(ValueError,match='joint limit'):
        validate_sweep(model,q,camera)


def test_home_near_limits_can_prepare_a_full_sweep_observation_pose(model):
    from b2w_perception.wrist_search_core import plan_observation
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    start=np.array([0.,0.,-.005,-.074,0.,0.])
    with pytest.raises(ValueError,match='joint limit'):
        validate_sweep(model,start,camera)
    plan=plan_observation(model,start,camera,{'target_end_xyz_m':[.2,0,.3],'search_radius_m':.04})
    end=np.asarray(model.forwardKinematics(np.array(plan['goal_q_rad']),6))
    np.testing.assert_allclose(end[:3,3],[.2,0,.3],atol=.003)
    assert plan['sweep']['optical_sweep_deg']==pytest.approx(30,abs=.001)
    assert len(plan['path_q_rad'])==151
    np.testing.assert_allclose(plan['path_q_rad'][0],start)


def test_small_initial_encoder_limit_error_can_only_improve_during_preparation(model):
    from b2w_perception.wrist_search_core import plan_observation
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    start=np.array([0.,-.005,-.005,-.074,0.,0.])
    plan=plan_observation(model,start,camera,{'search_radius_m':.04})
    q=np.array(plan['path_q_rad'])
    assert np.all(np.diff(q[:,1])>=-1e-8)
    assert q[-1,1]>.03


def test_preparation_rejects_large_initial_limit_violation(model):
    from b2w_perception.wrist_search_core import validate_observation_path
    camera=np.eye(4)
    start=np.array([0.,-.1,-.005,-.074,0.,0.])
    goal=np.array([0.,1.,-.5,-.2,0.,0.])
    with pytest.raises(ValueError,match='Initial joints exceed'):
        validate_observation_path(model,start,goal,camera)


def test_startup_reset_accepts_home_boundary_then_replans_a_safe_observation(model):
    from b2w_perception.wrist_search_core import plan_startup_reset,plan_observation
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    start=np.array([0.,.98,-.51,-.21,.1,0.])
    home=[0.,0.,-.005,-.074,0.,0.]
    reset=plan_startup_reset(model,start,camera,{'target_q_rad':home})
    np.testing.assert_allclose(reset['path_q_rad'][-1],home,atol=1e-8)
    assert reset['motor_zero_calibration'] is False
    observation=plan_observation(model,np.array(home),camera,{'search_radius_m':.04})
    np.testing.assert_allclose(observation['path_q_rad'][0],home)
    assert observation['sweep']['optical_sweep_deg']==pytest.approx(30,abs=.001)


def test_reset_rejects_configured_joint_target_outside_model_limits(model):
    from b2w_perception.wrist_search_core import plan_startup_reset
    start=np.array([0.,.98,-.51,-.21,0.,0.])
    with pytest.raises(ValueError,match='joint limits'):
        plan_startup_reset(model,start,np.eye(4),{'target_q_rad':[0.,-1.,0.,0.,0.,0.]})


def test_unreachable_observation_searches_beyond_four_cm(model):
    from b2w_perception.wrist_search_core import plan_observation
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    start=np.array([0.,1.2,-1.3,-.4,0.,0.])
    target=np.array([.9,0.,.3])
    plan=plan_observation(model,start,camera,{'target_end_xyz_m':target.tolist()})
    assert plan['search_mode']=='nearest_reachable'
    assert plan['position_offset_m']>.04
    np.testing.assert_allclose(model.forwardKinematics(np.array(plan['goal_q_rad']),6),plan['T_arm_end'],atol=.003)
    assert len(plan['path_q_rad'])==151
    assert plan['sweep']['optical_sweep_deg']>=15.


def test_observation_workspace_search_can_be_canceled(model):
    from b2w_perception.wrist_search_core import plan_observation
    calls=[]
    def cancel():
        calls.append(True)
        if len(calls)>=3:raise RuntimeError('operator canceled workspace search')
    with pytest.raises(RuntimeError,match='operator canceled workspace search'):
        plan_observation(model,np.array([0.,1.2,-1.3,-.4,0.,0.]),np.eye(4),
            {'target_end_xyz_m':[.9,0.,.3],'search_radius_m':0.,'planning_poll':cancel})


def test_observation_fallback_does_not_accept_an_invalid_path(model,monkeypatch):
    from b2w_perception.wrist_search_core import plan_observation
    def reject(*args,**kwargs):raise ValueError('blocked path')
    monkeypatch.setattr('b2w_perception.wrist_search_core.validate_observation_path',reject)
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    with pytest.raises(ValueError,match='No usable observation pose'):
        plan_observation(model,np.array([0.,1.2,-1.3,-.4,0.,0.]),camera,{'target_end_xyz_m':[.9,0.,.3]})
