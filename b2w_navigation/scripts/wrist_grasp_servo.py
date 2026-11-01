"""Segmented visual correction; images used for 3-D updates only while held still."""
import json
import math
import time
import numpy as np
from wrist_search_io import send_gripper_command

DEFAULTS=dict(enabled=True,step_m=.03,max_joint_step_rad=.20,max_correction_m=.02,
              association_distance_m=.08,final_blind_distance_m=.03,
              max_steps=60,max_duration_sec=180.,use_ground_projection=True)


def validate_servo_config(cfg):
    values=dict(DEFAULTS,**cfg)
    for key in ('enabled','use_ground_projection'):
        if not isinstance(values[key],bool):raise ValueError('Invalid visual_servo.'+key)
    bounds=dict(step_m=(.005,.05),max_joint_step_rad=(.05,.3),max_correction_m=(.001,.05),
                association_distance_m=(.02,.2),
                final_blind_distance_m=(.005,.04),max_steps=(1,100),max_duration_sec=(10.,300.))
    for key,(low,high) in bounds.items():
        if not isinstance(values[key],(int,float)) or not math.isfinite(values[key]) or not low<=values[key]<=high:
            raise ValueError('Invalid visual_servo.'+key)
    if not isinstance(values['max_steps'],int):raise ValueError('visual_servo.max_steps must be an integer')
    return values


def measured_target(item,camera,reference,confidence,projection):
    """Match the existing target; a box without depth can intersect its last height plane."""
    if item.get('box') is None or item.get('confidence',0)<confidence:return None
    point=item.get('camera_xyz_m')
    if point is not None and np.asarray(point).shape==(3,) and np.isfinite(point).all() and point[2]>0:
        return (camera@np.r_[point,1.])[:3],'depth'
    if not projection:return None
    meta=item['meta'];x1,y1,x2,y2=item['box']
    ray=camera[:3,:3]@np.array([((x1+x2)/2-meta['ppx'])/meta['fx'],
                               ((y1+y2)/2-meta['ppy'])/meta['fy'],1.])
    if abs(ray[2])<1e-6:return None
    distance=(reference[2]-camera[2,3])/ray[2]
    if distance<=0:return None
    return camera[:3,3]+distance*ray,'height_plane'


def segment_goal(model,start,goal,tool,step,joint_step):
    """Bound translation along the actual joint path as well as joint displacement."""
    origin=(np.asarray(model.forwardKinematics(start,6))@tool)[:3,3]
    chosen=start.copy()
    for fraction in np.linspace(0.,1.,201)[1:]:
        candidate=start+fraction*(goal-start)
        tcp=(np.asarray(model.forwardKinematics(candidate,6))@tool)[:3,3]
        if np.linalg.norm(tcp-origin)>step or np.max(np.abs(candidate-start))>joint_step:break
        chosen=candidate
    if np.max(np.abs(chosen-start))<1e-8:
        raise RuntimeError('No bounded visual approach step available')
    return chosen


def final_segment_allowed(model,start,goal,tool,point,distance,joint_step):
    if np.max(np.abs(goal-start))>joint_step:return False
    points=[(np.asarray(model.forwardKinematics(start+u*(goal-start),6))@tool)[:3,3]
            for u in np.linspace(0.,1.,31)]
    return (np.linalg.norm(points[0]-point)<=distance and np.linalg.norm(points[-1]-point)<=distance
            and float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)))<=distance)


class VisualApproach:
    def set_grasp_floor(self):
        # Sport body Z represents height above the floor. Express world Z=0
        # in arm coordinates, including current body roll/pitch and the mount.
        app=self.app
        if getattr(app,'body_pose',None) is None:return
        transform=np.asarray(app.body_pose)@np.asarray(app.approach.t_base_arm)
        app.wrist.path_floor_plane=np.r_[transform[2,:3],transform[2,3]]

    def hold_measured(self):
        w=self.app.wrist
        w.target=w.snapshot()[0].copy()
        w.arm.setArmCmd(w.target,np.zeros(6),np.zeros(6));send_gripper_command(w.arm,w.gripper)

    def servo_trace(self,**fields):
        self.app.publish('grasp_visual_update',**fields)
        with (self.app.output/'grasp_visual_trace.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(monotonic_s=time.monotonic(),**fields))+'\n')

    def update_held_target(self,reference):
        app=self.app;w=app.wrist;cfg=self.servo_cfg
        self.stage('GRASP_UPDATING')
        self.guard();seen=app.vision.read()
        if seen is None or seen.get('box') is None or seen.get('confidence',0)<app.cfg['confidence']:
            return None,w.snapshot()[0],reference
        since=time.monotonic()
        self.hold_measured()
        held=w.snapshot()[0].copy()
        w.wait_observation_settle(held,self.settings('停稳更新抓取位置',turn=True))
        self.guard();item=app.vision.read();q=w.snapshot()[0]
        sample=None
        if (item is not None and since<item['captured_at']<=time.monotonic()
                and np.max(np.abs(q-held))<=.01):
            camera=np.asarray(w.model.forwardKinematics(q,6))@w.camera_transform
            sample=measured_target(item,camera,reference,app.cfg['confidence'],cfg['use_ground_projection'])
        if sample is None or np.linalg.norm(sample[0]-reference)>cfg['association_distance_m']:
            # No reacquisition wait or vote: caller finishes the last valid goal.
            return None,q,reference
        delta=sample[0]-reference
        delta*=min(1.,cfg['max_correction_m']/max(np.linalg.norm(delta),1e-9))
        updated=reference+delta
        self.servo_trace(source=sample[1],frame_number=item['number'],
            measured_arm_xyz_m=sample[0].tolist(),updated_arm_xyz_m=updated.tolist(),
            correction_m=float(np.linalg.norm(delta)))
        return item,q,updated

    def finish_cached(self,grip_q,point):
        self._motion_watch=None
        self.stage('GRASP_CACHED_APPROACH')
        self.servo_trace(source='cached_target',target_lost_continue=True,
            updated_arm_xyz_m=point.tolist(),grasp_q_rad=np.asarray(grip_q).tolist())
        self.move(grip_q,'目标丢失，按最后有效抓取点继续')

    def visual_approach(self,item,q,point,grip_q):
        self.servo_cfg=validate_servo_config(self.cfg.get('visual_servo',{}));cfg=self.servo_cfg
        if not cfg['enabled']:
            self.move(grip_q,'沿框短边姿态接近海绵');return
        w=self.app.wrist;tool=np.asarray(self.grasp_cfg['calibration']['T_end_tcp'])
        begun=time.monotonic();self._target_missing=False;previous=None;stalled=0
        for step in range(cfg['max_steps']):
            self.guard()
            if self._target_missing:
                self.finish_cached(grip_q,point);return
            if time.monotonic()-begun>cfg['max_duration_sec']:break
            # Close-range occlusion locks the last goal; sensor/control watchdogs stay active.
            tcp=(np.asarray(w.model.forwardKinematics(w.snapshot()[0],6))@tool)[:3,3]
            if step>0 and final_segment_allowed(w.model,w.snapshot()[0],grip_q,tool,
                    point,cfg['final_blind_distance_m'],cfg['max_joint_step_rad']):
                self.stage('GRASP_FINAL_APPROACH')
                self.servo_trace(final_blind=True,remaining_m=float(np.linalg.norm(tcp-point)))
                self.move(grip_q,'最后短距离接近');return
            fresh_item,q,updated=self.update_held_target(point)
            if fresh_item is None:
                self.finish_cached(grip_q,point);return
            item,point=fresh_item,updated
            self.stage('GRASP_PLANNING')
            grip_q=self.solve(item,q,point)
            # Optimization can take time. Refresh at the same held pose, then
            # rebuild the goal if the target estimate changed materially.
            if time.monotonic()-item['captured_at']>self.app.cfg['frame_timeout_sec']:
                fresh_item,q,updated=self.update_held_target(point)
                if fresh_item is None:
                    self.finish_cached(grip_q,point);return
                item=fresh_item
                if np.linalg.norm(updated-point)>.003:
                    point=updated;grip_q=self.solve(item,q,point)
            current=w.snapshot()[0]
            tcp=(np.asarray(w.model.forwardKinematics(current,6))@tool)[:3,3]
            selected=(np.asarray(w.model.forwardKinematics(grip_q,6))@tool)[:3,3]
            (self.app.output/'grasp_target.json').write_text(json.dumps(dict(
                camera_xyz_m=item.get('camera_xyz_m'),surface_arm_xyz_m=point.tolist(),
                grasp_q_rad=grip_q.tolist(),selected_tcp_arm_xyz_m=selected.tolist(),
                center_offset_m=float(np.linalg.norm(selected-point)),frame_number=item['number']),indent=2)+'\n')
            if final_segment_allowed(w.model,current,grip_q,tool,point,
                    cfg['final_blind_distance_m'],cfg['max_joint_step_rad']):
                self.stage('GRASP_FINAL_APPROACH');self.move(grip_q,'最后短距离接近');return
            if previous is not None and np.max(np.abs(current-previous))<.002:stalled+=1
            else:stalled=0
            if stalled>=3 or np.max(np.abs(grip_q-current))<.005:
                raise RuntimeError(f'Cannot reach sponge center; remaining={np.linalg.norm(tcp-point):.3f}m; grasp held')
            previous=current.copy()
            goal=segment_goal(w.model,current,grip_q,tool,cfg['step_m'],cfg['max_joint_step_rad'])
            self.stage('GRASP_APPROACHING',visual_step=step+1)
            last_frame=[item['number']]
            def watch():
                seen=self.app.vision.read()
                if seen is not None and seen['number']>last_frame[0]:
                    last_frame[0]=seen['number']
                    if seen.get('box') is None or seen.get('confidence',0)<self.app.cfg['confidence']:
                        self._target_missing=True
            self._motion_watch=watch
            try:self.move(goal,'分段接近海绵')
            finally:self._motion_watch=None
            if self._target_missing:
                self.finish_cached(grip_q,point);return
        self.hold_measured()
        raise RuntimeError('Visual approach budget exhausted; held position, no close')
