"""Post-arrival side search, fresh RGB-D grasp and closed-gripper reset.

Runs in the existing arm/camera session. ROS and operator input are pumped on
every wait; the scan worker relinquishes command ownership for the sequence.
"""
import json
import math
import sys
import time

import cv2
import numpy as np

from wrist_search_io import ROOT,send_gripper_command,arm_feedback_snapshot
from wrist_grasp_servo import VisualApproach,validate_servo_config
from b2w_perception.wrist_search_core import (
    SearchPolicy, plan_observation, nearest_observation, plan_startup_reset, scan_offset, validate_observation_path,
)
from b2w_perception.wrist_approach_core import world_target_to_arm


def joint_plan(wrist, goal, reset=False):
    start=wrist.snapshot()[0]
    path=validate_observation_path(wrist.model,start,goal,wrist.camera_transform,
                                   goal_margin=0. if reset else .03,
                                   floor_plane=getattr(wrist,'path_floor_plane',None))
    return dict(start_q_rad=start.tolist(),goal_q_rad=np.asarray(goal).tolist(),path_q_rad=path)


class WristGrasp(VisualApproach):
    def __init__(self,app,cfg,grasp_cfg,view=False):
        self.app,self.cfg,self.grasp_cfg,self.view=app,cfg,grasp_cfg,view
        self.started=False;self.running=False;self.search_deadline=None
        self.last_heartbeat=0.;self.last_stop_reason=None;self.vote_policy=None;self.gripper_status={}
        self.target_update={}
        self._motion_watch=None
        self.validate()

    def validate(self):
        cfg=self.cfg
        bounds={'side_offset_m':(.0,1.0),'max_search_sec':(30.,600.),
                'scan_amplitude_deg':(1.,20.),'scan_speed_rad_s':(.01,.15),
                'view_radius_m':(.15,.3),'view_height_m':(.2,.45),
                'depth_wait_timeout_sec':(2.,20.),'joint_speed_rad_s':(.02,.15),
                'joint_position_tolerance_rad':(.025,.10),
                'turn_speed_rad_s':(.02,.25),
                'turn_joint_position_tolerance_rad':(.025,.10),
                'turn_joint_speed_tolerance_rad_s':(.025,.15),
                'turn_settle_stable_sec':(.1,1.),'turn_settle_position_window_sec':(.2,1.),
                'turn_settle_position_range_rad':(.001,.01),
                'gripper_speed_rad_s':(.05,.5),'gripper_timeout_sec':(4.,15.),
                'gripper_open_wait_sec':(0.,3.)}
        for key,(lo,hi) in bounds.items():
            value=cfg[key]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError('Invalid post_grasp.'+key)
        g=self.grasp_cfg['grasp']
        validate_servo_config(cfg.get('visual_servo',{}))
        if not -math.pi/2<=g['open_rad']<g['close_rad']<=0:
            raise ValueError('Invalid gripper opening/closing angles')

    def stage(self,state,**fields):
        self.app.policy.state=state
        self.app.publish('post_arrival_grasp',**fields)

    def interrupted(self):
        app=self.app;w=app.wrist;now=time.monotonic()
        reason=None
        if app.stop_requested or app.exit_requested:reason='Operator canceled grasp'
        elif app.release_requested:reason='Arm control release requested'
        elif now-app.odom_at>.5 or app.base_speed>=.03:reason='Base odometry stale or base moving during grasp'
        elif any(app.gates.values()):reason='Base velocity gate reopened during grasp'
        elif w.error:reason=w.error
        elif not w.arm.feedbackHealthy() or w.arm.getCurrentState()!=w.sdk.ArmFSMState.JOINTCTRL:
            reason='Arm feedback/state fault during grasp: '+json.dumps(arm_feedback_snapshot(w.arm))
        elif (not np.isfinite(w.snapshot()[0]).all() or
                np.max(np.abs(w.snapshot()[0]-w.target))>.12):
            reason='Arm joint tracking error during grasp'
        elif app.vision.error:reason=app.vision.error
        else:
            item=app.vision.read()
            if item is None or not 0<=now-item['captured_at']<=app.cfg['frame_timeout_sec']:
                reason='Camera heartbeat stale during grasp'
        if self.search_deadline is not None and now>self.search_deadline:
            reason=reason or 'Post-arrival wrist search timed out'
        self.last_stop_reason=reason
        return reason is not None

    def pump(self):
        app=self.app
        app.ros.spin_once(app.node,timeout_sec=0.)
        if self.view:
            item=app.vision.read()
            if item is not None:
                cv2.imshow('Z1 camera search',item['annotated'])
                if cv2.waitKey(1)&0xff==ord('q'):app.stop_requested=True
        if time.monotonic()-self.last_heartbeat>1.:
            app.publish('grasp_heartbeat');self.last_heartbeat=time.monotonic()
        if self._motion_watch is not None:self._motion_watch()

    def guard(self):
        self.pump()
        if self.interrupted():raise RuntimeError(self.last_stop_reason)

    def settings(self,label,turn=False):
        settings=dict(self.app.cfg['observation'])
        settings.update(joint_speed_rad_s=self.cfg['joint_speed_rad_s'],label=label,
                        log_prefix='grasp',joint_position_tolerance_rad=self.cfg['joint_position_tolerance_rad'])
        if turn:
            settings.update(joint_speed_rad_s=self.cfg['turn_speed_rad_s'],joint_speed_limit_rad_s=.25,
                joint_position_tolerance_rad=self.cfg['turn_joint_position_tolerance_rad'],
                joint_speed_tolerance_rad_s=self.cfg['turn_joint_speed_tolerance_rad_s'],
                settle_stable_sec=self.cfg['turn_settle_stable_sec'],
                settle_position_window_sec=self.cfg['turn_settle_position_window_sec'],
                settle_position_range_rad=self.cfg['turn_settle_position_range_rad'])
        return settings

    def move(self,goal,label,reset=False):
        self.guard()
        w=self.app.wrist
        plan=joint_plan(w,goal,reset)
        self.guard()  # Planning may take time; dispatch ROS feedback again first.
        w.move_to_observation(plan,self.settings(label))

    def confirm(self):
        """After the RGB vote, take one fresh depth exposure at the held wrist pose."""
        app=self.app;w=app.wrist
        q=w.snapshot()[0];w.target=q.copy()
        w.wait_observation_settle(q,self.settings('停腕定位',turn=True))
        since=time.monotonic();q=w.snapshot()[0];last=-1
        while time.monotonic()-since<self.cfg['depth_wait_timeout_sec']:
            self.guard();item=app.vision.read();current=w.snapshot()[0]
            if item is not None and item['number']>last:
                last=item['number'];point=item['camera_xyz_m']
                if (since<item['captured_at']<=time.monotonic() and
                        time.monotonic()-item['captured_at']<=app.cfg['frame_timeout_sec'] and
                        item['box'] is not None and item['confidence']>=app.cfg['confidence'] and
                        point is not None and np.asarray(point).shape==(3,) and
                        np.isfinite(point).all() and point[2]>0 and np.max(np.abs(current-q))<=.01):
                    camera=np.asarray(w.model.forwardKinematics(current,6))@w.camera_transform
                    return item,current,(camera@np.r_[point,1.])[:3]
            w.arm.setArmCmd(w.target,np.zeros(6),np.zeros(6))
            send_gripper_command(w.arm,w.gripper)
            time.sleep(.02)
        raise RuntimeError('Sponge vote passed but no fresh valid depth at held wrist pose')

    def estimated_target_arm(self):
        app=self.app
        target=getattr(app.approach,'target_world',None)
        if target is None:raise RuntimeError('Frozen first-detection world target is missing')
        # Receive feedback after arrival, rather than accepting a cached pose
        # just because its callback ran less than 0.5s ago.
        requested=time.monotonic();item=None
        while time.monotonic()-requested<1.:
            self.guard();item=app.vision.read()
            if (app.body_pose is not None and requested<app.body_pose_at<=time.monotonic()
                    and item is not None and requested<item['captured_at']<=time.monotonic()):break
            time.sleep(.01)
        else:raise RuntimeError('Fresh post-arrival Sport pose and camera frame required for target-directed search')
        body=np.asarray(app.body_pose).copy()
        hint=world_target_to_arm(target,body,app.approach.t_base_arm)
        source='post_arrival_sport_pose'
        point=item.get('camera_xyz_m')
        if (item.get('box') is not None and item.get('confidence',0)>=app.cfg['confidence']
                and point is not None and np.asarray(point).shape==(3,)
                and np.isfinite(point).all() and point[2]>0):
            # If the sponge is already visible, prefer a new measurement for
            # aiming. This is not grasp confirmation; the 2-of-3 vote follows.
            q=app.wrist.snapshot()[0]
            camera=np.asarray(app.wrist.model.forwardKinematics(q,6))@app.wrist.camera_transform
            hint=(camera@np.r_[point,1.])[:3]
            target=(body@app.approach.t_base_arm@np.r_[hint,1.])[:3]
            app.approach.target_world=target.copy()
            source='post_arrival_rgbd'
        self.target_update=dict(target_update_source=source,pose_requested_monotonic_s=requested,
            body_pose_received_monotonic_s=app.body_pose_at,T_sport_body=body.tolist(),
            T_base_arm=app.approach.t_base_arm.tolist(),target_world_xyz_m=np.asarray(target).tolist(),
            target_frame_number=item['number'],target_frame_captured_monotonic_s=item['captured_at'],
            updated_target_base_xyz_m=(app.approach.t_base_arm@np.r_[hint,1.])[:3].tolist())
        return hint

    def plan_hint_view(self,hint):
        """Aim one reachable observation pose at the estimate; do not move to an old grasp point."""
        app=self.app;w=app.wrist;start=w.snapshot()[0]
        yaw=math.atan2(hint[1],hint[0])
        # Stay on the body's side of a nearby target. A fixed 20cm reach can
        # overshoot a target only 7cm from the arm and force the lens to look back.
        radius=min(self.cfg['view_radius_m'],.5*float(np.linalg.norm(hint[:2])))
        end_position=np.array([radius*math.cos(yaw),radius*math.sin(yaw),self.cfg['view_height_m']])
        ray=hint-end_position
        pitch=math.atan2(-ray[2],math.hypot(ray[0],ray[1]))
        settings=self.settings('朝停车后更新的海绵位置观察',turn=True)
        settings['target_end_xyz_m']=end_position.tolist()
        settings['aim_target_arm_xyz_m']=np.asarray(hint).tolist()
        settings['target_bearing_rad']=yaw
        settings['planning_poll']=self.guard
        amplitude=math.radians(self.cfg['scan_amplitude_deg'])
        # Refine lens offset in IK only; dispatch a single final physical move.
        for _ in range(4):
            self.guard()
            settings.update(yaw_deg=math.degrees(yaw),pitch_candidates_deg=[math.degrees(pitch)])
            plan=plan_observation(w.model,start,w.camera_transform,settings,4,amplitude)
            camera=np.asarray(w.model.forwardKinematics(np.asarray(plan['goal_q_rad']),6))@w.camera_transform
            ray=hint-camera[:3,3]
            direction=ray/max(np.linalg.norm(ray),1e-9)
            error=math.degrees(math.acos(float(np.clip(np.dot(camera[:3,2],direction),-1,1))))
            if error<1. or (plan.get('search_mode')=='nearest_reachable' and error<=5.):break
            yaw=math.atan2(ray[1],ray[0]);pitch=math.atan2(-ray[2],math.hypot(ray[0],ray[1]))
        if error>5.:
            end=np.asarray(w.model.forwardKinematics(np.asarray(plan['goal_q_rad']),6))
            plan=nearest_observation(w.model,start,w.camera_transform,settings,4,amplitude,
                                    end_position,end[:3,:3],plan.get('ik_attempts',0),'Camera aim refinement failed')
            error=plan['aim_error_deg']
        return plan,settings,error

    def search(self):
        app=self.app;w=app.wrist
        self.search_deadline=time.monotonic()+self.cfg['max_search_sec']
        amplitude=math.radians(self.cfg['scan_amplitude_deg']);speed=self.cfg['scan_speed_rad_s']
        self.guard()
        hint=self.estimated_target_arm()
        self.stage('GRASP_TURNING',estimated_target_arm_xyz_m=hint.tolist(),
                   search_mode='updated_target_hint',**self.target_update)
        (app.output/'grasp_search_hint.json').write_text(json.dumps(dict(
            state='PLANNING',estimated_target_arm_xyz_m=hint.tolist(),**self.target_update),indent=2)+'\n')
        plan,settings,error=self.plan_hint_view(hint)
        report=dict(estimated_target_arm_xyz_m=hint.tolist(),yaw_deg=settings['yaw_deg'],
                    pitch_deg=settings['pitch_candidates_deg'][0],aim_error_deg=error,
                    scan_amplitude_deg=self.cfg['scan_amplitude_deg'],scan_speed_rad_s=speed)
        report.update(self.target_update)
        selected_camera=None if 'T_arm_end' not in plan else np.asarray(plan['T_arm_end'])@w.camera_transform
        report.update(target_bearing_deg=math.degrees(settings.get('target_bearing_rad',math.atan2(hint[1],hint[0]))),
                      selected_base_joint_deg=math.degrees(plan['goal_q_rad'][0]),
                      selected_camera_yaw_deg=None if selected_camera is None else math.degrees(
                          math.atan2(selected_camera[1,2],selected_camera[0,2])))
        report.update(selected_observation_end_xyz_m=plan.get('target_end_xyz_m'),
                      observation_search_mode=plan.get('search_mode','local_ik'),
                      observation_position_offset_m=plan.get('position_offset_m'),
                      camera_target_distance_m=plan.get('camera_target_distance_m'))
        (app.output/'grasp_observation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        (app.output/'grasp_search_hint.json').write_text(json.dumps(dict(state='PLANNED',**report),indent=2)+'\n')
        self.guard();w.move_to_observation(plan,settings)
        center=w.snapshot()[0].copy();begun=time.monotonic()
        self.vote_policy=SearchPolicy(app.cfg['vote_required_frames'],app.cfg['confidence'],
                                     app.cfg['vote_window_frames'],self.cfg['max_search_sec'])
        self.vote_policy.start(begun)
        self.stage('GRASP_SEARCHING',**report,scan_joint=5,
                   grasp_vote_required=self.vote_policy.required_frames,grasp_vote_window=self.vote_policy.vote_window)
        w.scanning.set()
        try:
            while True:
                self.guard();elapsed=time.monotonic()-begun
                w.target=center.copy();w.target[4]+=scan_offset(elapsed,amplitude,speed)
                measured=w.snapshot()[0]
                if not np.isfinite(measured).all() or np.max(np.abs(measured-w.target))>.12:
                    raise RuntimeError('Post-arrival scan tracking error')
                w.arm.setArmCmd(w.target,np.zeros(6),np.zeros(6))
                send_gripper_command(w.arm,w.gripper)
                item=app.vision.read()
                if item is not None and item['captured_at']>begun:
                    action=self.vote_policy.observe(item['number'],item['captured_at'],time.monotonic(),
                                                    item['box'],item['confidence'])
                    if action=='FOUND':
                        w.scanning.clear();w.target=w.snapshot()[0].copy()
                        self.stage('GRASP_LOCALIZING',grasp_vote_hits=self.vote_policy.vote_count,
                                   grasp_vote_frames=len(self.vote_policy.votes))
                        self.search_deadline=None
                        return self.confirm()
                time.sleep(.02)
        finally:w.scanning.clear()

    def solve(self,item,q,point):
        """Reuse demo_self's short-side orientation and nearby IK, with checked paths."""
        from demo_self import nearby_target,short_side_rotation
        w=self.app.wrist
        camera=np.asarray(w.model.forwardKinematics(q,6))@w.camera_transform
        tool=np.asarray(self.grasp_cfg['calibration']['T_end_tcp'])
        rotation=short_side_rotation(item['box'],item['meta'],camera,
                                     np.asarray(w.model.forwardKinematics(q,6))@tool)
        def validate_candidate(solved):
            self.guard()
            plane=getattr(w,'path_floor_plane',None)
            validate_observation_path(w.model,q,solved,w.camera_transform,floor_plane=plane)
            plan_startup_reset(w.model,solved,w.camera_transform,dict(self.app.cfg['reset'],floor_plane=plane))
        _,grip_q=nearby_target(w.model,q,point,tool,rotation,camera[:3,2],validate_candidate,self.guard)
        grip_q=np.asarray(grip_q)
        return grip_q

    def gripper(self,target,closing=False):
        w=self.app.wrist
        self.guard();start=float(w.gripper);begun=time.monotonic()
        if closing:
            start=float(w.arm.lowstate.getGripperQ())
            if not math.isfinite(start) or not -math.pi/2-.15<=start<=.15:
                raise RuntimeError('Invalid measured gripper angle')
        duration=max(.1,abs(target-start)/self.cfg['gripper_speed_rad_s'])
        last_saved=-float('inf')
        def report(result=None):
            nonlocal last_saved
            measured=actual if math.isfinite(actual) else None
            self.gripper_status=dict(gripper_action='closing' if closing else 'opening',
                gripper_target_rad=float(target),gripper_actual_rad=measured,
                gripper_error_rad=None if measured is None else float(target-measured),
                gripper_elapsed_sec=elapsed,gripper_command_speed_rad_s=velocity,
                gripper_open_check_enabled=False)
            if hasattr(w.arm,'commandSnapshot'):
                self.gripper_status['sdk_command']=dict(w.arm.commandSnapshot())
            if result:self.gripper_status['gripper_result']=result
            output=getattr(self.app,'output',None)
            if output is not None and (elapsed-last_saved>=.2 or result):
                (output/'gripper_status.json').write_text(json.dumps(self.gripper_status,indent=2)+'\n')
                last_saved=elapsed
        while True:
            self.guard();elapsed=time.monotonic()-begun
            w.gripper=start+(target-start)*min(1.,elapsed/duration)
            w.arm.setArmCmd(w.target,np.zeros(6),np.zeros(6))
            velocity=(target-start)/duration if elapsed<duration else 0.
            send_gripper_command(w.arm,w.gripper,velocity)
            actual=float(w.arm.lowstate.getGripperQ())
            if closing and not math.isfinite(actual):raise RuntimeError('Non-finite gripper feedback')
            report()
            if not closing and elapsed>=duration+self.cfg['gripper_open_wait_sec']:
                w.gripper=float(target);report('open_command_completed')
                return False
            if closing and elapsed>=duration and abs(actual-target)<.15:
                w.gripper=float(target);report('closed');return False
            if closing and elapsed>=self.cfg['gripper_timeout_sec']:
                w.gripper=float(target);report('contact_inferred')
                self.app.publish('gripper_contact_inferred',grasp_verified=False,actual_gripper_rad=actual)
                return True
            time.sleep(.02)

    def run(self):
        if self.started:return
        self.started=True;self.running=True
        app=self.app;w=app.wrist
        old_pump,old_stop=w.pump,w.should_stop
        old_reason=getattr(w,'stop_reason',None)
        (app.output/'grasp_result.json').write_text(json.dumps(dict(
            state='RUNNING',started_monotonic_s=time.monotonic()),indent=2)+'\n')
        (app.output/'grasp_visual_trace.jsonl').write_text('')
        sys.path.insert(0,str(ROOT.parent/'unitree_z1/yolo/grasp_demo'))
        try:
            with w.exclusive_commands():
                w.pump,w.should_stop=self.pump,self.interrupted
                w.stop_reason=lambda:self.last_stop_reason
                self.guard()
                item,q,point=self.search()
                self.set_grasp_floor()
                self.stage('GRASP_PLANNING')
                # Record this attempt before IK; a failure must not leave a previous target looking current.
                (app.output/'grasp_target.json').write_text(json.dumps(dict(
                    state='PLANNING',camera_xyz_m=item['camera_xyz_m'],surface_arm_xyz_m=point.tolist(),
                    frame_number=item['number']),indent=2)+'\n')
                grip_q=self.solve(item,q,point)
                self.guard()
                if time.monotonic()-item['captured_at']>app.cfg['frame_timeout_sec']:
                    # Joint-space projection can take longer than one frame timeout.
                    # Refresh this same held view, without another scan or RGB vote.
                    self.stage('GRASP_REFRESHING')
                    self.servo_cfg=validate_servo_config(self.cfg.get('visual_servo',{}))
                    if self.servo_cfg['enabled']:
                        fresh_item,fresh_q,fresh_point=self.update_held_target(point)
                        if fresh_item is not None:
                            changed=np.linalg.norm(fresh_point-point)>.003
                            item,q,point=fresh_item,fresh_q,fresh_point
                            if changed:grip_q=self.solve(item,q,point)
                    else:
                        fresh_item,fresh_q,fresh_point=self.confirm()
                        if np.max(np.abs(fresh_q-q))>.01 or np.linalg.norm(fresh_point-point)>.02:
                            raise RuntimeError('Target or held arm moved during IK planning; grasp canceled')
                        item,q,point=fresh_item,fresh_q,fresh_point
                    self.guard()
                selected_tcp=(np.asarray(w.model.forwardKinematics(grip_q,6))@
                              np.asarray(self.grasp_cfg['calibration']['T_end_tcp']))[:3,3]
                center_offset=float(np.linalg.norm(selected_tcp-point))
                app.publish('grasp_candidate_selected',selected_tcp_arm_xyz_m=selected_tcp.tolist(),
                            grasp_center_offset_m=center_offset)
                (app.output/'grasp_target.json').write_text(json.dumps(dict(
                    camera_xyz_m=item['camera_xyz_m'],surface_arm_xyz_m=point.tolist(),
                    grasp_q_rad=grip_q.tolist(),
                    selected_tcp_arm_xyz_m=selected_tcp.tolist(),
                    center_offset_m=center_offset,
                    frame_number=item['number']),indent=2)+'\n')
                self.stage('GRASP_OPENING');self.gripper(self.grasp_cfg['grasp']['open_rad'])
                self.stage('GRASP_APPROACHING');self.visual_approach(item,q,point,grip_q)
                self.stage('GRASP_CLOSING')
                contact=self.gripper(self.grasp_cfg['grasp']['close_rad'],closing=True)
                self.stage('GRASP_RESETTING')
                self.move(np.asarray(app.cfg['reset']['target_q_rad']),'合爪归位',reset=True)
                self.stage('GRASP_DONE',grasp_verified=False,contact_inferred=contact,
                           reset_q_rad=app.cfg['reset']['target_q_rad'],gripper_held_closed=True)
                (app.output/'grasp_result.json').write_text(json.dumps(dict(
                    state='GRASP_DONE',grasp_verified=False,contact_inferred=contact,
                    reset_q_rad=app.cfg['reset']['target_q_rad'],
                    gripper_held_closed=True),indent=2)+'\n')
        except Exception as exc:
            (app.output/'grasp_result.json').write_text(json.dumps(dict(
                state='FAULT',failed_stage=app.policy.state,reason=str(exc),
                grasp_verified=False),indent=2)+'\n')
            raise
        finally:
            self.search_deadline=None;self.running=False
            w.pump,w.should_stop=old_pump,old_stop
            w.stop_reason=old_reason
