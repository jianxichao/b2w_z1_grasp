"""Robot-local point controller using only Sport Move(1008) and StopMove(1003)."""
import json
import math


def wrap(a):return math.atan2(math.sin(a),math.cos(a))


def clamp(v,limit):return max(-limit,min(limit,v))


def compensate_deadzone(value,bias,limit):
    """Signed feedforward for requested motion; exact stop remains exact zero."""
    if value==0.:return 0.
    return clamp(value+math.copysign(bias,value),limit)


def sport_pose(msg):
    q=[float(v) for v in msg.imu_state.quaternion]
    x,y=[float(v) for v in msg.position[:2]]
    norm=math.sqrt(sum(v*v for v in q))
    if (msg.error_code or not .9<=norm<=1.1 or
            not all(math.isfinite(v) for v in [x,y,*q,*msg.velocity,msg.yaw_speed])):
        raise ValueError('Invalid Sport pose/state')
    w,qx,qy,qz=[v/norm for v in q]
    return (x,y,math.atan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz)),
            math.hypot(msg.velocity[0],msg.velocity[1]),abs(msg.yaw_speed))


class PositionSession:
    def __init__(self,send,publish,clock,heartbeat_timeout=.5,state_timeout=.5,
                 max_distance=3.5,timeout=120.,max_speed=.5,
                 linear_speed=.30,angular_speed=.20,linear_accel=.15,angular_accel=.4,
                 linear_bias=.20,angular_bias=.20,position_tolerance=.12,yaw_tolerance=.15,
                 stand_down_on_arrival=False,longitudinal_tolerance=None):
        limits=(heartbeat_timeout,state_timeout,max_distance,timeout,max_speed,linear_speed,
                angular_speed,linear_accel,angular_accel)
        if not all(math.isfinite(v) and v>0 for v in limits):raise ValueError('Invalid local goal limits')
        if linear_speed>.30 or angular_speed>.3:raise ValueError('Local goal command limits exceed configured test envelope')
        if not all(math.isfinite(v) and v>=0 for v in (linear_bias,angular_bias)):
            raise ValueError('Deadzone biases must be finite and nonnegative')
        self.send,self.publish,self.clock=send,publish,clock
        self.heartbeat_timeout,self.state_timeout=heartbeat_timeout,state_timeout
        self.max_distance,self.timeout,self.max_speed=max_distance,timeout,max_speed
        self.linear_speed,self.angular_speed=linear_speed,angular_speed
        self.linear_accel,self.angular_accel=linear_accel,angular_accel
        self.linear_bias,self.angular_bias=linear_bias,angular_bias
        if not (.05<=position_tolerance<=.5 and .05<=yaw_tolerance<=math.pi/2):
            raise ValueError('Invalid arrival tolerances')
        self.position_tolerance,self.yaw_tolerance=position_tolerance,yaw_tolerance
        if longitudinal_tolerance is not None and not (
                math.isfinite(longitudinal_tolerance) and .02<=longitudinal_tolerance<=position_tolerance):
            raise ValueError('Invalid longitudinal arrival tolerance')
        self.longitudinal_tolerance=longitudinal_tolerance
        self.longitudinal_error=None
        self.stand_down_on_arrival=bool(stand_down_on_arrival)
        self.stand_down_wait_completed=False
        self.posture_request_id=None;self.stand_down_acknowledged=False
        self.active=False;self.state='IDLE';self.phase='IDLE';self.command_id=0;self.info=''
        self.pose=None;self.pose_at=-float('inf');self.last_stamp=None
        self.goal=None;self.still_since=None;self.move_requests=set()
        self.command=(0.,0.);self.distance_error=None;self.yaw_error=None;self.local_pose=None
        self.move_count=0;self.last_move_at=None

    def receive(self,msg):
        try:
            pose=sport_pose(msg);stamp=(int(msg.stamp.sec),int(msg.stamp.nanosec))
            if stamp!=(0,0) and stamp==self.last_stamp:return
            self.last_stamp=stamp;self.pose=pose;self.pose_at=self.clock()
        except (ValueError,TypeError,IndexError):self.pose=None

    def status(self):
        data=dict(active=self.active,state=self.state,controller_phase=self.phase,
            command_id=self.command_id,info=self.info,control_api=1008,controller_version=5,
            commanded_vx_m_s=self.command[0],commanded_yaw_rate_rad_s=self.command[1],
            distance_error_m=self.distance_error,yaw_error_rad=self.yaw_error,
            move_command_count=self.move_count,
            last_move_age_sec=None if self.last_move_at is None else self.clock()-self.last_move_at,
            measured_speed_m_s=None if self.pose is None else self.pose[3],
            measured_yaw_speed_rad_s=None if self.pose is None else self.pose[4],
            linear_deadzone_bias_m_s=self.linear_bias,angular_deadzone_bias_rad_s=self.angular_bias,
            arrival_position_tolerance_m=self.position_tolerance,arrival_yaw_tolerance_rad=self.yaw_tolerance,
            arrival_longitudinal_tolerance_m=(self.position_tolerance if self.longitudinal_tolerance is None
                                              else self.longitudinal_tolerance),
            longitudinal_error_m=self.longitudinal_error,
            stand_down_on_arrival=self.stand_down_on_arrival,
            stand_down_acknowledged=self.stand_down_acknowledged,
            physical_posture_verified=False,stand_down_wait_completed=self.stand_down_wait_completed,
            stand_down_wait_sec=3.0)
        if self.goal is not None:
            data.update(goal_local_xy_yaw=list(self.goal),anchor_sport_world_xy_yaw=list(self.anchor),
                        measured_local_xy_yaw=self.local_pose)
        self.publish(data)

    def start(self,x,y,yaw,command_id):
        if self.active:raise ValueError('A local goal already owns the Sport API')
        if (not all(math.isfinite(v) for v in (x,y,yaw)) or math.hypot(x,y)>self.max_distance or abs(yaw)>math.pi):
            raise ValueError('Invalid local goal')
        now=self.clock()
        if self.pose is None or now-self.pose_at>self.state_timeout:raise ValueError('Sport state missing/stale')
        if self.pose[3]>.03 or self.pose[4]>.03:raise ValueError('Base must stop before local goal')
        self.anchor=self.pose[:3];self.goal=(float(x),float(y),float(yaw))
        self.command_id=int(command_id);self.active=True;self.started=now;self.heartbeat_at=now
        self.last_tick=now;self.progress_at=now;self.progress_metric=None;self.still_since=None
        self.command=(0.,0.);self.local_pose=None;self.distance_error=None;self.yaw_error=None
        self.longitudinal_error=None
        self.move_requests.clear();self.position_reached=False
        self.move_count=0;self.last_move_at=None
        self.posture_request_id=None;self.stand_down_acknowledged=False
        self.stand_down_wait_completed=False
        self.state='MOVING';self.phase='TURN_TO_GOAL'
        self.info='Local odometry controller armed; no motion command sent until control tick'
        self.status()

    def heartbeat(self,command_id):
        if self.active and int(command_id)==self.command_id:self.heartbeat_at=self.clock()

    def finish(self,state,info,send_stop=True):
        self.active=False;self.state=state;self.phase=state;self.info=str(info);self.command=(0.,0.)
        self.move_requests.clear()
        if send_stop:
            for _ in range(3):self.send(1003,'{}')
        self.status()

    def cancel(self,reason,send_stop=True):
        if self.active:self.finish('CANCELED',reason,send_stop)

    def response(self,request_id,api_id,success,code,info):
        if (self.active and self.phase=='STAND_DOWN_WAIT' and api_id==1005 and
                request_id==self.posture_request_id):
            if success:self.stand_down_acknowledged=True;self.status()
            else:self.finish('FAULT',f'StandDown API 1005 rejected: code={code}, info={info}')
            return
        # Move is a streaming API; success replies are optional and are not arrival feedback.
        if not self.active or api_id!=1008 or request_id not in self.move_requests:return
        self.move_requests.discard(request_id)
        if not success:self.finish('FAULT',f'Sport Move API 1008 rejected: code={code}, info={info}')

    def tick_arrival_posture(self,now):
        if self.phase=='STOPPING_AT_GOAL':
            if self.pose[3]<.05 and self.pose[4]<.05:
                self.still_since=self.still_since or now
            else:self.still_since=None
            if self.still_since is not None and now-self.still_since>=.3:
                self.phase='STAND_DOWN_WAIT';self.posture_started=now;self.still_since=None
                self.posture_request_id=self.send(1005,'{}')
                self.info='StandDown sent once; waiting 3 seconds and stationary feedback (reply optional)'
            elif now-self.arrival_stopped_at>5.:
                self.finish('FAULT','Base did not stop before StandDown');return
        elif self.phase=='STAND_DOWN_WAIT':
            if self.pose[3]<.03 and self.pose[4]<.03:
                self.still_since=self.still_since or now
            else:self.still_since=None
            if (now-self.posture_started>=3. and self.still_since is not None and
                    now-self.still_since>=.3):
                self.stand_down_wait_completed=True
                self.finish('ARRIVED','StandDown 3-second wait completed; base stationary; API reply optional',send_stop=False)
                return
            if now-self.posture_started>8.:
                self.finish('FAULT','Base did not settle after StandDown 3-second wait');return
        self.status()

    def tick(self,other_motion_enabled=False):
        if not self.active:return
        now=self.clock();dt=now-self.last_tick;reason=None
        if other_motion_enabled:reason='Another velocity controller was enabled during local goal'
        elif now-self.heartbeat_at>self.heartbeat_timeout:reason='Local goal client heartbeat expired'
        elif self.pose is None or now-self.pose_at>self.state_timeout:reason='Sport state missing/stale during local goal'
        elif now-self.started>self.timeout:reason='Local goal timed out'
        elif self.pose[3]>self.max_speed:reason='Measured speed exceeds local goal monitoring limit'
        elif dt>.5:reason='Local goal control scheduler stalled'
        if reason:self.finish('FAULT',reason);return
        if dt<=0:return
        self.last_tick=now
        ax,ay,ayaw=self.anchor;x,y,yaw,speed,yaw_speed=self.pose
        dx,dy=x-ax,y-ay;c,s=math.cos(ayaw),math.sin(ayaw)
        lx,ly=c*dx+s*dy,-s*dx+c*dy;heading=wrap(yaw-ayaw)
        self.local_pose=[lx,ly,heading]
        ex,ey=self.goal[0]-lx,self.goal[1]-ly;distance=math.hypot(ex,ey)
        final_error=wrap(self.goal[2]-heading)
        self.distance_error=distance;self.yaw_error=final_error
        self.longitudinal_error=math.cos(self.goal[2])*ex+math.sin(self.goal[2])*ey
        if self.phase in ('STOPPING_AT_GOAL','STAND_DOWN_WAIT'):
            self.tick_arrival_posture(now);return
        long_tolerance=self.position_tolerance if self.longitudinal_tolerance is None else self.longitudinal_tolerance
        within_position=(distance<=self.position_tolerance and abs(self.longitudinal_error)<=long_tolerance)
        if self.stand_down_on_arrival:
            # A loose radius alone stops short of the target. Align longitudinally,
            # and recheck position while turning before latching a final stop.
            self.position_reached=within_position
        elif within_position and distance<=self.position_tolerance*5/6:
            self.position_reached=True
        elif not within_position:self.position_reached=False
        if self.stand_down_on_arrival and self.position_reached and abs(final_error)<=self.yaw_tolerance:
            self.phase='STOPPING_AT_GOAL';self.command=(0.,0.);self.move_requests.clear()
            self.arrival_stopped_at=now;self.still_since=None
            for _ in range(3):self.send(1003,'{}')
            self.info='Arrival tolerance reached; position and heading corrections latched off'
            self.status();return
        previous_phase=self.phase
        if self.position_reached:
            desired_v=0.
            desired_w=clamp(final_error,self.angular_speed) if abs(final_error)>self.yaw_tolerance else 0.
            self.phase='FACE_TARGET' if abs(final_error)>self.yaw_tolerance else 'SETTLING'
            metric=abs(final_error)
        else:
            bearing_error=wrap(math.atan2(ey,ex)-heading)
            if self.phase not in ('DRIVE','TURN_TO_GOAL'):self.phase='TURN_TO_GOAL'
            if self.phase=='TURN_TO_GOAL' and abs(bearing_error)<=.12:self.phase='DRIVE'
            if self.phase=='DRIVE' and abs(bearing_error)>.35:self.phase='TURN_TO_GOAL'
            # Avoid amplifying tiny tracking noise into repeated biased turns.
            desired_w=clamp(bearing_error,self.angular_speed) if abs(bearing_error)>.02 else 0.
            desired_v=min(self.linear_speed,.5*distance)*max(0.,math.cos(bearing_error)) if self.phase=='DRIVE' else 0.
            metric=distance if self.phase=='DRIVE' else abs(bearing_error)
        if self.phase!=previous_phase:
            self.progress_at=now;self.progress_metric=metric
        if self.progress_metric is None or metric<self.progress_metric-.02:
            self.progress_metric=metric;self.progress_at=now
        if self.phase!='SETTLING' and now-self.progress_at>10.:
            self.finish('FAULT','No measured progress during '+self.phase);return
        if within_position and abs(final_error)<=self.yaw_tolerance and speed<.03 and yaw_speed<.03:
            self.still_since=self.still_since or now
            if now-self.still_since>=.3:
                self.finish('ARRIVED','Software local-goal arrival verified');return
        else:self.still_since=None
        desired_v=compensate_deadzone(desired_v,self.linear_bias,self.linear_speed)
        desired_w=compensate_deadzone(desired_w,self.angular_bias,self.angular_speed)
        step=min(dt,.2)
        vx=self.command[0]+clamp(desired_v-self.command[0],self.linear_accel*step)
        wz=self.command[1]+clamp(desired_w-self.command[1],self.angular_accel*step)
        self.command=(vx,wz)
        request_id=self.send(1008,json.dumps({'x':vx,'y':0.,'z':wz}))
        self.move_count+=1;self.last_move_at=now
        self.move_requests.add(request_id)
        if len(self.move_requests)>32:self.move_requests.remove(min(self.move_requests))
        self.info='Software local-goal tracking via Move(1008); no automatic obstacle avoidance'
        self.status()
