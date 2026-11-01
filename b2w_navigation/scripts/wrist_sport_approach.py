"""Wrist target -> robot-local side goal -> software Move velocity controller, without Nav2 approach."""
import json
import time
import numpy as np
from b2w_perception.wrist_approach_core import rigid_transform,local_side_goal


class WristSportApproach:
    backend='sport_velocity_goal'

    def __init__(self,app,cfg,t_end_camera):
        from b2w_interfaces.srv import SportLocalGoal
        from std_msgs.msg import String,UInt64
        from unitree_api.msg import Request
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
        self.app,self.cfg=app,cfg
        self.t_base_arm=rigid_transform(cfg['T_base_arm']);self.t_end_camera=rigid_transform(t_end_camera)
        self.side=cfg.get('side','left');self.offset=cfg.get('side_offset_m',.5)
        self.request_type=SportLocalGoal;self.UInt64=UInt64
        self.client=app.node.create_client(SportLocalGoal,'/b2w_sport_cmd_bridge/move_to_local_goal')
        self.stop_client=app.clients['/b2w_sport_cmd_bridge/enable']
        self.keepalive=app.node.create_publisher(UInt64,'/b2w_sport_cmd_bridge/position_keepalive',10)
        self.Request=Request
        self.stop_pub=app.node.create_publisher(Request,'/api/sport/request',10)
        qos=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        app.node.create_subscription(String,'/b2w_sport_cmd_bridge/position_status',self.on_status,qos)
        self.active=False;self.command_id=None;self.pending=None;self.status=None;self.status_at=0.
        self.target_world=None

    def on_status(self,msg):
        try:
            data=json.loads(msg.data)
            if self.command_id is None or int(data.get('command_id',-1))==self.command_id:
                self.status=data;self.status_at=time.monotonic()
                if self.active:
                    with (self.app.output/'sport_position_trace.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(received_monotonic_s=self.status_at,**data))+'\n')
        except (ValueError,TypeError):pass

    def wait_ready(self,client,timeout):
        """Service discovery must not block odometry/status callbacks on this executor."""
        end=time.monotonic()+timeout
        while not client.service_is_ready():
            if self.app.stop_requested or getattr(self.app,'exit_requested',False):
                raise RuntimeError('Service wait interrupted by operator stop/exit')
            remaining=end-time.monotonic()
            if remaining<=0:return False
            self.app.ros.spin_once(self.app.node,timeout_sec=min(.02,remaining))
        return True

    def disable_other_motion(self):
        app=self.app
        if app.stop_requested or getattr(app,'exit_requested',False):raise RuntimeError('Position approach canceled')
        self.ensure_controller()
        # Disabled bridge is silent. Never enable Twist output during position control.
        required='/b2w_sport_cmd_bridge/enable'
        client=app.clients[required]
        if not self.wait_ready(client,.3):raise RuntimeError('Sport bridge stop service unavailable')
        req=app.SetBool.Request();req.data=False
        response=app.wait(client.call_async(req))
        if response is None or not response.success:raise RuntimeError('Sport bridge stop not acknowledged')
        # Best-effort cancellation only: no Nav2 presence, lifecycle, map, footprint
        # or stop acknowledgement is a prerequisite for Sport position control.
        for name,optional in app.clients.items():
            if name!=required and optional.service_is_ready():
                request=app.SetBool.Request();request.data=False;optional.call_async(request)
        if app.cancel.service_is_ready():app.cancel.call_async(app.CancelGoal.Request())

    def ensure_controller(self):
        """Reject an old bridge before arming a goal with silently missing compensation."""
        end=time.monotonic()+1.5
        while time.monotonic()<end:
            if self.app.stop_requested or getattr(self.app,'exit_requested',False):
                raise RuntimeError('Controller discovery interrupted by operator')
            if (self.status and self.status.get('controller_version',0)>=5 and
                    'linear_deadzone_bias_m_s' in self.status and
                    'angular_deadzone_bias_rad_s' in self.status):
                if not getattr(self,'_controller_reported',False):
                    self.app.publish('sport_controller_ready',controller_version=self.status['controller_version'],
                        linear_deadzone_bias_m_s=self.status['linear_deadzone_bias_m_s'],
                        angular_deadzone_bias_rad_s=self.status['angular_deadzone_bias_rad_s'])
                    self._controller_reported=True
                return
            self.app.ros.spin_once(self.app.node,timeout_sec=.02)
        raise RuntimeError('Sport bridge is old or unavailable: arrival/longitudinal alignment update or deadzone compensation is not loaded. '
            'Restart sport_position_bringup.launch.py and then restart this search program.')

    def prepare_base(self):
        self.app.touched_base=True
        self.disable_other_motion()
        end=time.monotonic()+5.
        while time.monotonic()<end:
            self.app.ros.spin_once(self.app.node,timeout_sec=.02)
            if self.app.stop_requested or getattr(self.app,'exit_requested',False):raise RuntimeError('Startup canceled')
            if time.monotonic()-self.app.odom_at<.5 and self.app.base_speed<.03:return
        raise RuntimeError('Fresh stationary Sport odometry required before arm preparation')

    def begin(self,item,q):
        app=self.app
        if not 0<=time.monotonic()-item['captured_at']<=app.cfg['frame_timeout_sec']:
            raise RuntimeError('Side-goal depth exposure is stale')
        self.disable_other_motion()
        fk=np.asarray(app.wrist.model.forwardKinematics(q,6))
        self.target_base,self.goal_base,self.goal_yaw=local_side_goal(item['camera_xyz_m'],
            self.t_base_arm,fk,self.t_end_camera,
            self.side,self.offset)
        self.ideal_goal_base=self.goal_base.copy()
        compensation=float(self.cfg.get('stand_down_forward_compensation_m',0.))
        if not np.isfinite(compensation) or not 0<=compensation<=.2:
            raise ValueError('Invalid StandDown forward compensation')
        self.forward_compensation=compensation if (self.status or {}).get('stand_down_on_arrival',False) else 0.
        self.goal_base[0]+=self.forward_compensation
        self.target_world=None
        body=getattr(app,'body_pose',None)
        if body is not None and 0<=time.monotonic()-app.body_pose_at<=.5:
            self.target_world=(rigid_transform(body)@np.r_[self.target_base,1.])[:3]
        if getattr(app,'grasp',None) is not None and self.target_world is None:
            raise RuntimeError('Fresh Sport body pose required to preserve target through approach and crouch')
        if np.linalg.norm(self.goal_base)>self.cfg.get('sport_max_distance_m',3.5):
            raise RuntimeError('Side goal exceeds Local goal distance limit')
        self.hold_q=np.array(q);self.active=True;self.command_id=None;self.status=None
        self.started=time.monotonic();self.last_keepalive=-float('inf')
        report=dict(backend=self.backend,frame='robot_local_at_command_start',side=self.side,
            side_offset_m=self.offset,target_base_xyz_m=self.target_base.tolist(),
            target_world_xyz_m=None if self.target_world is None else self.target_world.tolist(),
            ideal_goal_base_xy_m=self.ideal_goal_base.tolist(),
            stand_down_forward_compensation_m=self.forward_compensation,
            goal_base_xy_m=self.goal_base.tolist(),goal_yaw_rad=self.goal_yaw,
            arrival_heading_mode='keep_initial_heading',
            q_confirm_rad=q.tolist(),T_base_arm=self.t_base_arm.tolist(),
            T_arm_end=fk.tolist(),camera_xyz_m=item['camera_xyz_m'],
            T_end_camera=self.t_end_camera.tolist(),automatic_obstacle_avoidance=False)
        (app.output/'sport_side_goal.json').write_text(json.dumps(report,indent=2)+'\n')
        if not self.wait_ready(self.client,1.):
            self.active=False;raise RuntimeError('move_to_local_goal service missing; restart updated Sport bridge/launch')
        if app.stop_requested or getattr(app,'exit_requested',False):
            self.active=False;raise RuntimeError('Local goal canceled before submission')
        request=self.request_type.Request()
        request.x,request.y=map(float,self.goal_base);request.yaw=float(self.goal_yaw)
        app.policy.state='SPORT_POSITION_ACCEPTING'
        self.pending=self.client.call_async(request)
        self.pending.add_done_callback(self.late_accept)
        app.publish('sport_side_goal_submitted',**report)

    def late_accept(self,future):
        try:
            result=future.result()
            if result and result.success and not self.active:
                req=self.app.SetBool.Request();req.data=False
                self.stop_client.call_async(req)
        except Exception:pass

    def cancel(self):
        owned=self.active;self.active=False
        if owned:
            self.emergency_stop()
            req=self.app.SetBool.Request();req.data=False
            # Bridge also stops on keepalive loss if this service is unavailable.
            if self.stop_client.service_is_ready():self.stop_client.call_async(req)

    def emergency_stop(self):
        for _ in range(3):
            req=self.Request();req.header.identity.id=time.monotonic_ns()
            req.header.identity.api_id=1003;req.parameter='{}';self.stop_pub.publish(req)

    def tick(self):
        if not self.active:return
        app=self.app;now=time.monotonic()
        if app.stop_requested or getattr(app,'exit_requested',False):raise RuntimeError('Operator canceled Local goal')
        if app.wrist.error:raise RuntimeError(app.wrist.error)
        if not app.wrist.arm.feedbackHealthy():raise RuntimeError('Arm feedback lost during Local goal')
        q,_=app.wrist.snapshot()
        if not np.isfinite(q).all() or np.max(np.abs(q-self.hold_q))>.08:
            raise RuntimeError('Held arm moved during Local goal')
        if now-app.odom_at>.5:raise RuntimeError('Sport odometry stale during Local goal')
        if any(app.gates.values()):raise RuntimeError('Velocity gate reopened during Local goal')
        if now-self.started>125.:raise RuntimeError('Local goal result timeout')
        if now-self.started>1. and now-self.status_at>1.:
            self.emergency_stop();raise RuntimeError('Position bridge status stale; StopMove requested')
        if self.command_id is None:
            if not self.pending.done():
                if now-self.started>2.:raise RuntimeError('Local goal service timed out')
                return
            result=self.pending.result()
            if not result or not result.success:raise RuntimeError('Local goal not queued: '+str(getattr(result,'message','no response')))
            self.command_id=int(result.command_id)
        if now-self.last_keepalive>=.1:
            self.keepalive.publish(self.UInt64(data=self.command_id));self.last_keepalive=now
        status=self.status
        if status and int(status.get('command_id',-1))==self.command_id:
            state=status.get('state')
            if state in ('FAULT','CANCELED'):
                (app.output/'sport_position_result.json').write_text(json.dumps(status,indent=2)+'\n')
                raise RuntimeError('Local goal '+state+': '+str(status.get('info')))
            if state=='MOVING' and app.policy.state!='SPORT_POSITION_MOVING':
                app.policy.state='SPORT_POSITION_MOVING';app.publish('sport_position_started',command_id=self.command_id,control_api=1008)
            if state=='ARRIVED':
                (app.output/'sport_position_result.json').write_text(json.dumps(status,indent=2)+'\n')
                self.active=False
                if status.get('stand_down_wait_completed') or status.get('stand_down_acknowledged'):
                    # Bridge already stopped and finished StandDown; another StopMove can override posture.
                    app.policy.state='ARRIVED'
                else:app.stop('sport_side_arrived','ARRIVED')
                app.publish('side_arrival_verified',side=self.side,offset_m=self.offset,
                    grasp_enabled=getattr(app,'grasp',None) is not None,
                    stand_down_acknowledged=status.get('stand_down_acknowledged',False),
                    physical_posture_verified=status.get('physical_posture_verified',False),
                    stand_down_wait_completed=status.get('stand_down_wait_completed',False))
