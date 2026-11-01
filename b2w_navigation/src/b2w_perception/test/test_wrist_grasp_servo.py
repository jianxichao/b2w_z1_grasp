"""Visual approach tests: simulated feedback/clock only, no SDK controller or UDP."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'b2w_navigation/scripts'))
from wrist_grasp import WristGrasp
from wrist_grasp_servo import measured_target,segment_goal,final_segment_allowed
from b2w_perception.wrist_search_core import validate_observation_path,plan_startup_reset


class Model:
    def forwardKinematics(self,q,link):
        pose=np.eye(4);pose[:3,3]=q[:3];return pose
    def getJointQMin(self):return np.full(6,-2.)
    def getJointQMax(self):return np.full(6,2.)


@pytest.fixture
def servo(tmp_path,monkeypatch):
    clock=NS(t=10.);state=NS(q=np.zeros(6),number=0,visible=True,drop_once=False,point=np.array([.2,0.,.2]))
    monkeypatch.setattr('wrist_grasp_servo.time.monotonic',lambda:clock.t)
    monkeypatch.setattr('wrist_grasp_servo.time.sleep',lambda dt:setattr(clock,'t',clock.t+dt))
    cfg=json.loads((ROOT/'b2w_navigation/config/wrist_search.json').read_text())
    grasp=json.loads((ROOT/'unitree_z1/yolo/grasp_demo/config.json').read_text());grasp['calibration']['T_end_tcp']=np.eye(4).tolist()
    w=NS(model=Model(),camera_transform=np.eye(4),target=state.q.copy(),gripper=0.)
    w.snapshot=lambda:(state.q.copy(),np.zeros(6))
    w.wait_observation_settle=lambda *a:setattr(clock,'t',clock.t+.03)
    w.arm=NS(setArmCmd=lambda q,*a:None,setGripperCmd=lambda *a:None)
    def read():
        state.number+=1
        return dict(number=state.number,captured_at=clock.t-.0001,box=[0,0,10,10] if state.visible else None,
            confidence=.9 if state.visible else 0.,camera_xyz_m=(state.point-state.q[:3]).tolist(),
            meta=dict(ppx=5.,ppy=5.,fx=100.,fy=100.))
    events=[];moves=[]
    app=NS(wrist=w,cfg=cfg,vision=NS(read=read),policy=NS(state='GRASP_APPROACHING'),output=tmp_path)
    app.publish=lambda event,**kw:events.append((app.policy.state,kw))
    s=WristGrasp(app,cfg['post_grasp'],grasp)
    s.guard=lambda:setattr(clock,'t',clock.t+.01)
    s.solve=lambda item,q,point:np.r_[point,np.zeros(3)]
    def move(q,label,**kw):
        if state.drop_once:
            state.drop_once=False;state.visible=False
            try:s._motion_watch()
            finally:state.visible=True
        moves.append(q.copy());state.q=q.copy();w.target=q.copy()
        clock.t+=.1
    s.move=move
    return s,state,clock,events,moves


def test_segmented_approach_updates_until_short_final_move(servo):
    s,state,clock,events,moves=servo
    s.visual_approach({},state.q,state.point.copy(),np.r_[state.point,np.zeros(3)])
    assert len(moves)>2
    positions=[np.zeros(3)]+[q[:3] for q in moves]
    assert max(np.linalg.norm(np.diff(positions,axis=0),axis=1))<=.030001
    np.testing.assert_allclose(state.q[:3],state.point)
    assert any(state=='GRASP_FINAL_APPROACH' for state,_ in events)
    assert any('frame_number' in data for _,data in events)


def test_motion_loss_continues_last_goal_without_wait_or_reacquisition(servo):
    s,state,clock,events,moves=servo;state.drop_once=True
    holds=[];s.app.wrist.arm.setArmCmd=lambda q,*a:holds.append(q.copy())
    s.visual_approach({},state.q,state.point.copy(),np.r_[state.point,np.zeros(3)])
    assert any(data.get('target_lost_continue') for _,data in events)
    assert not any(name=='GRASP_TARGET_LOST' for name,_ in events)
    assert len(moves)==2
    np.testing.assert_allclose(state.q[:3],state.point)
    assert s._motion_watch is None


def test_persistent_detection_loss_immediately_finishes_last_valid_goal(servo):
    s,state,clock,events,moves=servo;state.visible=False
    s.visual_approach({},state.q,state.point.copy(),np.r_[state.point,np.zeros(3)])
    assert len(moves)==1 and clock.t<10.3
    np.testing.assert_allclose(state.q[:3],state.point)


def test_wrong_target_jump_is_not_followed(servo):
    s,state,clock,events,moves=servo;state.point+=.5
    s.visual_approach({},state.q,np.array([.2,0.,.2]),np.r_[.2,0.,.2,0.,0.,0.])
    assert len(moves)==1
    np.testing.assert_allclose(state.q[:3],[.2,0.,.2])


def test_update_correction_is_capped(servo):
    s,state,clock,events,moves=servo
    from wrist_grasp_servo import validate_servo_config
    s.servo_cfg=validate_servo_config({})
    reference=state.point+np.array([.05,0.,0.])
    item,q,point=s.update_held_target(reference)
    assert np.linalg.norm(point-reference)==pytest.approx(.02)


def test_depth_loss_uses_last_height_but_absent_box_does_not():
    camera=np.eye(4);camera[:3,:3]=np.diag([1.,-1.,-1.]);camera[2,3]=.5
    item=dict(box=[40,40,60,60],confidence=.9,camera_xyz_m=None,
              meta=dict(ppx=50.,ppy=50.,fx=100.,fy=100.))
    point,source=measured_target(item,camera,np.array([0.,0.,.02]),.65,True)
    np.testing.assert_allclose(point,[0.,0.,.02]);assert source=='height_plane'
    item['box']=None
    assert measured_target(item,camera,np.array([0.,0.,.02]),.65,True) is None


def test_final_blind_finish_rejects_far_goal_even_when_current_pose_is_close():
    model=Model();tool=np.eye(4);start=np.zeros(6);point=np.array([.01,0.,0.])
    assert final_segment_allowed(model,start,np.r_[point,np.zeros(3)],tool,point,.03,.2)
    assert not final_segment_allowed(model,start,np.array([.15,0,0,0,0,0]),tool,point,.03,.2)


def test_stalled_unreachable_grasp_stops_instead_of_closing(servo):
    s,state,clock,events,moves=servo
    s.solve=lambda *args:state.q.copy()
    with pytest.raises(RuntimeError,match='Cannot reach sponge center'):
        s.visual_approach({},state.q,state.point.copy(),state.q.copy())
    assert not moves


def test_camera_or_control_fault_propagates_without_motion(servo):
    s,state,clock,events,moves=servo
    def fail():raise RuntimeError('camera disconnected')
    s.guard=fail
    with pytest.raises(RuntimeError,match='camera disconnected'):
        s.visual_approach({},state.q,state.point.copy(),state.q.copy())
    assert not moves


def test_physical_floor_allows_under_arm_base_but_not_under_ground():
    class HeightModel(Model):
        def forwardKinematics(self,q,link):
            pose=np.eye(4);pose[2,3]=.1-q[0];return pose
    model=HeightModel();start=np.zeros(6);goal=np.array([.2,0,0,0,0,0]);plane=[0,0,1,.3]
    with pytest.raises(ValueError):validate_observation_path(model,start,goal,np.eye(4))
    assert len(validate_observation_path(model,start,goal,np.eye(4),floor_plane=plane))==151
    plan=plan_startup_reset(model,goal,np.eye(4),dict(target_q_rad=start.tolist(),floor_plane=plane))
    assert len(plan['path_q_rad'])==151
    with pytest.raises(ValueError):
        validate_observation_path(model,start,np.array([.41,0,0,0,0,0]),np.eye(4),floor_plane=plane)


def test_last_three_cm_can_complete_when_target_is_occluded(servo):
    s,state,clock,events,moves=servo;original=s.move
    def move(q,label,**kwargs):
        original(q,label,**kwargs)
        if np.linalg.norm(state.q[:3]-state.point)<=.03:state.visible=False
    s.move=move
    s.visual_approach({},state.q,state.point.copy(),np.r_[state.point,np.zeros(3)])
    assert not state.visible
    np.testing.assert_allclose(state.q[:3],state.point)
    assert any(data.get('final_blind') for _,data in events)


def test_camera_fault_still_stops_the_final_blind_segment(servo):
    s,state,clock,events,moves=servo;original=s.guard
    def guard():
        original()
        if np.linalg.norm(state.q[:3]-state.point)<=.03:raise RuntimeError('camera disconnected')
    s.guard=guard
    with pytest.raises(RuntimeError,match='camera disconnected'):
        s.visual_approach({},state.q,state.point.copy(),np.r_[state.point,np.zeros(3)])
    assert np.linalg.norm(state.q[:3]-state.point)>1e-6


def test_old_frame_is_not_used_for_correction_and_does_not_trigger_waiting(servo):
    s,state,clock,events,moves=servo
    from wrist_grasp_servo import validate_servo_config
    s.servo_cfg=validate_servo_config({})
    read=s.app.vision.read
    def duplicate():
        result=read();result['number']=1;result['captured_at']=0.;return result
    s.app.vision.read=duplicate
    item,q,point=s.update_held_target(state.point.copy())
    assert item is None and clock.t<10.3
    np.testing.assert_allclose(point,state.point)
    assert not moves


@pytest.mark.parametrize('lost',[False,True])
def test_full_grasp_closes_and_resets_only_after_visual_approach_success(servo,lost):
    from contextlib import contextmanager
    s,state,clock,events,moves=servo;actions=[];w=s.app.wrist
    @contextmanager
    def control():yield
    w.exclusive_commands=control;w.pump=lambda:None;w.should_stop=lambda:False
    def search():return s.app.vision.read(),state.q.copy(),state.point.copy()
    s.search=search;s.confirm=search
    s.gripper=lambda target,closing=False:actions.append('close' if closing else 'open') or False
    s.app.exit_requested=False
    state.visible=not lost
    s.run()
    assert actions==['open','close'] and s.app.policy.state=='GRASP_DONE'
    np.testing.assert_allclose(state.q,s.app.cfg['reset']['target_q_rad'])
    if lost:assert any(data.get('target_lost_continue') for _,data in events)
    assert not s.running


def test_loss_during_slow_initial_planning_does_not_wait_for_new_detection(servo):
    from contextlib import contextmanager
    s,state,clock,events,moves=servo;w=s.app.wrist;actions=[]
    @contextmanager
    def control():yield
    w.exclusive_commands=control;w.pump=lambda:None;w.should_stop=lambda:False
    s.search=lambda:(s.app.vision.read(),state.q.copy(),state.point.copy())
    original=s.solve
    def solve(*args):
        goal=original(*args);clock.t+=3.;state.visible=False;return goal
    s.solve=solve
    s.gripper=lambda target,closing=False:actions.append('close' if closing else 'open') or False
    s.run()
    assert actions==['open','close'] and s.app.policy.state=='GRASP_DONE'
    assert clock.t<14.
