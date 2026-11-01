"""Nav2 adapter for wrist search. The Coordinator remains the sole motion-gate owner."""
import json
import math
import time
import numpy as np
from b2w_perception.wrist_approach_core import (
    rigid_transform,transform_from_pose,map_target,yaw_of,angle_error)


class WristApproach:
    def __init__(self,app,cfg,t_end_camera):
        from tf2_ros import Buffer,TransformListener
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import OccupancyGrid
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import qos_profile_sensor_data
        self.app,self.cfg=app,cfg
        self.t_base_arm=rigid_transform(cfg['T_base_arm'])
        self.t_end_camera=rigid_transform(t_end_camera)
        if cfg['base_frame']!='base_link' or cfg['map_frame']!='map':
            raise ValueError('This stack requires base_link mounting and map navigation frames')
        values=[cfg[k] for k in ('standoff_m','timeout_sec','arrival_tolerance_m',
                                'max_target_distance_m','map_timeout_sec','tf_timeout_sec','depth_wait_timeout_sec')]
        if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Invalid approach limits')
        tf_wait=float(cfg.get('tf_lookup_wait_sec',.3))
        if not math.isfinite(tf_wait) or not 0<=tf_wait<=1.:
            raise ValueError('TF lookup wait must be 0..1 seconds')
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,app.node)
        self.nav=ActionClient(app.node,NavigateToPose,'/navigate_to_pose')
        self.Goal,self.PoseStamped=NavigateToPose.Goal,PoseStamped
        self.publisher=app.node.create_publisher(PoseStamped,'/wrist_search/target_pose',10)
        self.map=None;self.map_at=0.
        app.node.create_subscription(OccupancyGrid,'/navigation_map',self.on_map,qos_profile_sensor_data)
        self.active=False;self.handle=None
        self.pending=None;self.result=None;self.epoch=0;self.stopped_since=None

    def on_map(self,msg):
        self.map=msg;self.map_at=time.monotonic()

    def tf_matrix(self,source,stamp=None):
        from rclpy.time import Time
        from tf2_ros import TransformException
        # Historical camera timestamps can be a few milliseconds ahead of the
        # latest odom/SLAM sample. Spin this same executor while waiting so TF
        # callbacks can fill the buffer; a blocking TF timeout would starve them.
        wait_sec=float(self.cfg.get('tf_lookup_wait_sec',.3)) if stamp is not None else 0.
        deadline=time.monotonic()+wait_sec
        while True:
            if (getattr(self.app,'stop_requested',False) or getattr(self.app,'exit_requested',False)
                    or getattr(self.app,'release_requested',False)):
                raise RuntimeError('TF lookup interrupted by operator stop/exit')
            try:
                tf=self.buffer.lookup_transform(self.cfg['map_frame'],source,stamp if stamp is not None else Time())
                break
            except TransformException as exc:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise RuntimeError(f'TF unavailable after {wait_sec:.2f}s ({source} -> '
                                       f"{self.cfg['map_frame']}): {exc}") from exc
                self.app.ros.spin_once(self.app.node,timeout_sec=min(.02,remaining))
        stamp_ns=tf.header.stamp.sec*10**9+tf.header.stamp.nanosec
        age=(self.app.node.get_clock().now().nanoseconds-stamp_ns)/1e9
        age_limit=self.cfg['tf_timeout_sec']+(self.app.cfg['frame_timeout_sec'] if stamp is not None else 0.)
        # SLAM may date map->odom slightly ahead for its transform tolerance.
        if age<-.5 or age>age_limit:
            raise RuntimeError('Map TF is stale: '+source)
        p,q=tf.transform.translation,tf.transform.rotation
        return transform_from_pose([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])

    def fresh_map(self):
        if self.map is None or time.monotonic()-self.map_at>self.cfg['map_timeout_sec']:
            raise RuntimeError('Navigation map missing or stale')
        m=self.map
        if (m.header.frame_id!=self.cfg['map_frame'] or m.info.resolution<=0 or
                len(m.data)!=m.info.width*m.info.height):
            raise RuntimeError('Invalid navigation map/frame')
        return m

    def begin(self,item,q):
        from rclpy.time import Time
        from b2w_nav.mission_supervisor_core import direct_approach_goal_xy
        if self.app.stop_requested:raise RuntimeError('Approach canceled by operator')
        if item['camera_xyz_m'] is None:raise ValueError('Approach requires a valid depth position')
        age=time.monotonic()-item['captured_at']
        if not 0<=age<=self.app.cfg['frame_timeout_sec']:raise RuntimeError('Confirmed exposure is stale')
        # captured_at is the local acquisition-start time; use TF history, not a later moving pose.
        stamp=Time(nanoseconds=self.app.node.get_clock().now().nanoseconds-int(age*1e9))
        t=self.tf_matrix(self.cfg['base_frame'],stamp)
        fk=np.asarray(self.app.wrist.model.forwardKinematics(q,6))
        self.target=map_target([item['camera_xyz_m']],t,self.t_base_arm,fk,
                               self.t_end_camera)
        current=self.tf_matrix('base_footprint')
        distance=float(np.linalg.norm(self.target[:2]-current[:2,3]))
        if not .15<=distance<=self.cfg['max_target_distance_m']:
            raise RuntimeError('Target horizontal distance is outside approach limits')
        self.hold_q=np.array(q);self.started=time.monotonic();self.stopped_since=None
        self.goal_xy=None;self.handle=None;self.pending=None;self.result=None
        self.last_pose=current;self.last_pose_at=self.started
        self.last_map_odom=self.tf_matrix('odom')
        report=dict(target_map_xyz_m=self.target.tolist(),T_map_base=t.tolist(),
                    T_base_arm=self.t_base_arm.tolist(),T_arm_end=fk.tolist(),
                    T_end_camera=self.t_end_camera.tolist(),q_confirm_rad=q.tolist(),
                    camera_points_m=[item['camera_xyz_m']],
                    standoff_m=self.cfg['standoff_m'],grasp_enabled=False)
        # A target already inside the stopping radius must not trigger a reverse/sideways detour.
        if distance<=self.cfg['standoff_m']+self.cfg['arrival_tolerance_m']:
            report['already_near']=True
            self.save(report)
            self.app.stop('target_already_near_no_motion','ARRIVED')
            return
        self.fresh_map()
        # Select the stop position geometrically; Nav2 owns path and collision checks.
        self.goal_xy=np.array(direct_approach_goal_xy(
            current[0,3],current[1,3],self.target[0],self.target[1],self.cfg['standoff_m']))
        self.goal_yaw=math.atan2(self.target[1]-self.goal_xy[1],self.target[0]-self.goal_xy[0])
        if not self.nav.server_is_ready():raise RuntimeError('Nav2 navigate_to_pose action is unavailable')
        # Re-establish exclusive ownership: Frontier stays disabled throughout the approach.
        if not self.app.stop_base():raise RuntimeError('Base stop not acknowledged before approach')
        if self.app.stop_requested:raise RuntimeError('Approach canceled before goal submission')
        goal=self.Goal();goal.pose=self.pose(*self.goal_xy,self.goal_yaw)
        report.update(goal_map_xy_m=self.goal_xy.tolist(),goal_yaw_rad=self.goal_yaw,already_near=False,
                      goal_selection='direct_standoff')
        self.save(report)
        self.active=True;self.epoch+=1;epoch=self.epoch
        self.app.policy.state='APPROACH_PLANNING'
        self.pending=self.nav.send_goal_async(goal)
        # If stopped while a goal request is in flight, cancel any late acceptance.
        def late_accept(future):
            try:
                handle=future.result()
                if handle and handle.accepted and (not self.active or self.epoch!=epoch):handle.cancel_goal_async()
            except Exception:pass
        self.pending.add_done_callback(late_accept)
        self.app.publish('approach_goal_submitted',**report)

    def pose(self,x,y,yaw):
        p=self.PoseStamped();p.header.frame_id=self.cfg['map_frame']
        p.header.stamp=self.app.node.get_clock().now().to_msg()
        p.pose.position.x=float(x);p.pose.position.y=float(y)
        p.pose.orientation.z=math.sin(yaw/2);p.pose.orientation.w=math.cos(yaw/2)
        return p

    def save(self,report):
        (self.app.output/'approach_target.json').write_text(json.dumps(report,indent=2)+'\n')
        p=self.pose(self.target[0],self.target[1],0.);p.pose.position.z=float(self.target[2])
        self.publisher.publish(p)

    def cancel(self):
        self.active=False;self.epoch+=1
        handle,self.handle=self.handle,None
        if handle is None and self.pending is not None and self.pending.done():
            try:handle=self.pending.result()
            except Exception:pass
        if handle is not None and handle.accepted:
            # The Coordinator also closes both gates and issues a global Nav2 cancel.
            # A local cancellation failure must never prevent those stop requests.
            try:handle.cancel_goal_async()
            except Exception:pass

    def tick(self):
        if not self.active:return
        now=time.monotonic();app=self.app;item=app.vision.read()
        if app.stop_requested:raise RuntimeError('Operator canceled approach')
        if now-self.started>self.cfg['timeout_sec']:raise RuntimeError('Approach timeout')
        if app.wrist.error or app.vision.error:raise RuntimeError(app.wrist.error or app.vision.error)
        if item is None or not 0<=now-item['captured_at']<=app.cfg['frame_timeout_sec']:
            raise RuntimeError('Camera stale during approach')
        if now-app.odom_at>.5 or now-app.scan_at>1.:raise RuntimeError('Navigation sensors stale during approach')
        if not app.wrist.arm.feedbackHealthy():raise RuntimeError('Arm feedback lost during approach')
        q,_=app.wrist.snapshot()
        if not np.isfinite(q).all() or np.max(np.abs(q-self.hold_q))>.08:
            raise RuntimeError('Held arm moved during approach')
        self.fresh_map()
        pose=self.tf_matrix('base_footprint');anchor=self.tf_matrix('odom')
        dt=max(0.,now-self.last_pose_at)
        if (np.linalg.norm(pose[:2,3]-self.last_pose[:2,3])>.25+.3*dt or
                abs(angle_error(yaw_of(pose),yaw_of(self.last_pose)))>.3+.6*dt or
                np.linalg.norm(anchor[:3,3]-self.last_map_odom[:3,3])>.25 or
                abs(angle_error(yaw_of(anchor),yaw_of(self.last_map_odom)))>.25):
            raise RuntimeError('Localization jumped during approach')
        self.last_pose,self.last_pose_at,self.last_map_odom=pose,now,anchor
        state=app.policy.state
        if state=='APPROACH_PLANNING' and self.pending.done():
            self.handle=self.pending.result()
            if self.handle is None or not self.handle.accepted:raise RuntimeError('Nav2 rejected approach goal')
            self.result=self.handle.get_result_async()
            app.enable_base(frontier=False)
            app.policy.state='APPROACHING'
            app.publish('approach_started',goal_map_xy_m=self.goal_xy.tolist())
        elif state=='APPROACHING':
            if now>app.gate_grace_until and (len(app.gates)<2 or not all(app.gates.values())):
                raise RuntimeError('Motion gate closed during approach')
            if self.result.done():
                result=self.result.result()
                status=None if result is None else int(result.status)
                status_name={4:'SUCCEEDED',5:'CANCELED',6:'ABORTED'}.get(status,'UNKNOWN')
                details=dict(status=status,status_name=status_name,monotonic_s=now,
                             goal_map_xy_m=self.goal_xy.tolist(),target_map_xyz_m=self.target.tolist())
                payload=getattr(result,'result',None)
                # Humble's NavigateToPose result contains only Empty. Preserve
                # richer fields if supplied by another version, without inventing a cause.
                for name in ('error_code','error_msg'):
                    value=getattr(payload,name,None)
                    if value is not None:details[name]=value
                (app.output/'nav2_result.json').write_text(json.dumps(details,indent=2)+'\n')
                app.publish('nav2_result',**details)
                if status!=4:
                    raise RuntimeError(f'Nav2 approach {status_name} (status={status}); '
                                       'see output/wrist_search/nav2_result.json and planner/controller logs')
                if not app.stop_base():raise RuntimeError('Arrival stop not acknowledged')
                app.policy.state='APPROACH_SETTLING'
        elif state=='APPROACH_SETTLING':
            if app.base_speed<.03:
                self.stopped_since=self.stopped_since or now
                if now-self.stopped_since>=.5:
                    distance=float(np.linalg.norm(self.target[:2]-pose[:2,3]))
                    if (np.linalg.norm(pose[:2,3]-self.goal_xy)>self.cfg['arrival_tolerance_m'] or
                            abs(distance-self.cfg['standoff_m'])>self.cfg['arrival_tolerance_m'] or
                            abs(angle_error(yaw_of(pose),self.goal_yaw))>.35):
                        raise RuntimeError('Nav2 success but measured arrival is outside tolerance')
                    app.stop('approach_arrived','ARRIVED')
                    app.publish('arrival_verified',target_distance_m=distance,grasp_enabled=False)
            else:self.stopped_since=None
