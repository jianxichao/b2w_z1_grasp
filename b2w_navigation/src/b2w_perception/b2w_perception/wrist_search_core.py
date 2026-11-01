"""ROS/SDK-free first-stage search policy. No approach or grasp states exist."""
from dataclasses import dataclass, field
import math
import time
import numpy as np


def iou(a, b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(np.prod(wh))
    return intersection/max(1e-9,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


@dataclass
class SearchPolicy:
    required_frames: int = 5
    confidence: float = .65
    vote_window: int = 10
    max_search_sec: float = 180.0
    state: str = 'IDLE'
    started: float = 0.0
    paused_at: float = 0.0
    last_box: object = None
    last_frame: int = -1
    votes: list = field(default_factory=list)

    def __post_init__(self):
        if not 1<=self.required_frames<=self.vote_window<=30:
            raise ValueError('Require 1 <= votes required <= vote window <= 30')

    @property
    def vote_count(self):return sum(self.votes)

    def start(self, now):
        self.state,self.started='SEARCHING',now
        self.votes.clear();self.last_frame=-1;self.last_box=None

    def observe(self, frame, captured_at, now, box, confidence, stationary=None):
        # A rolling RGB vote; no stationary, consecutive-hit or depth requirement.
        if self.state!='SEARCHING' or frame<=self.last_frame:return None
        if not 0 <= now-captured_at <= 2.0:return None
        self.last_frame=frame
        good=box is not None and math.isfinite(confidence) and confidence>=self.confidence
        self.votes.append(bool(good));self.votes=self.votes[-self.vote_window:]
        if good:self.last_box=box
        if self.vote_count>=self.required_frames:
            self.state,self.paused_at='FOUND',now
            return 'FOUND'
        return None

    def tick(self, now):
        if self.state=='SEARCHING' and now-self.started>=self.max_search_sec:
            self.state='LIMIT'
            return 'STOP'
        return None


def scan_offset(elapsed, amplitude, speed):
    if not (0<amplitude<=math.radians(30) and 0<speed<=.15):
        raise ValueError('Scan amplitude must be (0,30deg], speed (0,0.15rad/s]')
    # Starts at the current pose with zero velocity; long-run speed is bounded.
    ramp=min(1.,max(0.,elapsed)/2.)
    blend=ramp*ramp*(3-2*ramp)
    return amplitude*math.sin(speed/amplitude*max(0.,elapsed)*.5)*blend


def validate_sweep(model, center, camera_transform, joint=4, amplitude=math.radians(15)):
    if not math.isfinite(amplitude) or not 0<amplitude<=math.radians(30):
        raise ValueError('Sweep amplitude must be (0,30deg]')
    center=np.asarray(center,float)
    camera_transform=np.asarray(camera_transform,float)
    if center.shape!=(6,) or not np.isfinite(center).all() or joint not in (4,5):
        raise ValueError('Expected six joint angles and wrist joint index 4 or 5')
    lo,hi=np.asarray(model.getJointQMin()),np.asarray(model.getJointQMax())
    reference=np.asarray(model.forwardKinematics(center,6))
    ref_camera=reference@camera_transform
    directions=[]
    for offset in np.linspace(-amplitude,amplitude,61):
        q=center.copy();q[joint]+=offset
        if np.any(q<lo+.03) or np.any(q>hi-.03):
            bad=np.flatnonzero((q<lo+.03)|(q>hi-.03))
            detail=', '.join(f'J{i+1}={q[i]:.4f} outside [{lo[i]+.03:.4f},{hi[i]-.03:.4f}]rad' for i in bad)
            raise ValueError('Sweep crosses a joint limit/margin: '+detail)
        end=np.asarray(model.forwardKinematics(q,6));camera=end@camera_transform
        if camera[2,3]<.15:
            raise ValueError('Camera is not raised: prepare an observation pose first')
        if np.linalg.norm(end[:3,3]-reference[:3,3])>.06 or np.linalg.norm(camera[:3,3]-ref_camera[:3,3])>.08:
            raise ValueError('Sweep translates the end/camera too far')
        for link in range(1,7):
            point=np.asarray(model.forwardKinematics(q,link))[:3,3]
            if not np.isfinite(point).all() or np.linalg.norm(point[:2])>.76 or point[2]<-.02:
                raise ValueError('Sweep link leaves the configured arm envelope')
        directions.append(camera[:3,2])
    angle=math.acos(float(np.clip(np.dot(directions[0],directions[-1]),-1,1)))
    if angle<amplitude:
        raise ValueError('This joint mostly rolls the image instead of changing view; use J5')
    return {'joint':joint+1,'amplitude_deg':math.degrees(amplitude),
            'optical_sweep_deg':math.degrees(angle),'camera_center_arm_m':ref_camera[:3,3].tolist()}


def validate_observation_path(model, start, goal, camera_transform, tolerance=.02, goal_margin=.03,floor_plane=None):
    """Check actual joint interpolation, allowing only small initial feedback error.

    This is a sampled kinematic envelope, not a mesh collision detector.
    Existing small limit excursions may only shrink on the way into the limits.
    """
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    lo,hi=np.asarray(model.getJointQMin()),np.asarray(model.getJointQMax())
    if start.shape!=(6,) or goal.shape!=(6,) or not np.isfinite([start,goal]).all():
        raise ValueError('Invalid observation joint path')
    if np.any(start<lo-tolerance) or np.any(start>hi+tolerance):
        raise ValueError('Initial joints exceed model limits by more than feedback tolerance')
    if not 0<=goal_margin<=.10:raise ValueError('Invalid goal joint margin')
    if np.any(goal<lo+goal_margin-1e-8) or np.any(goal>hi-goal_margin+1e-8):
        raise ValueError('Observation/reset target leaves its allowed joint limits/margin')
    plane=None if floor_plane is None else np.asarray(floor_plane,float)
    if plane is not None:
        if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-9:
            raise ValueError('Invalid physical floor plane')
        plane=plane/np.linalg.norm(plane[:3])
    samples=[]
    for u in np.linspace(0,1,151):
        q=start+(goal-start)*u
        lower=np.minimum(lo,start);upper=np.maximum(hi,start)
        if np.any(q<lower-1e-7) or np.any(q>upper+1e-7):
            raise ValueError('Observation path worsens a joint limit excursion')
        end=np.asarray(model.forwardKinematics(q,6))
        camera=end@camera_transform
        camera_height=camera[2,3] if plane is None else float(plane@np.r_[camera[:3,3],1.])
        if not np.isfinite(camera).all() or camera_height<.03:
            raise ValueError('Observation path places the camera below its arm-base clearance')
        for link in range(1,7):
            p=np.asarray(model.forwardKinematics(q,link))[:3,3]
            below=p[2]<-.02 if plane is None else float(plane@np.r_[p,1.])<.01
            if not np.isfinite(p).all() or below or p[2]>.90 or np.linalg.norm(p[:2])>.76:
                raise ValueError('Observation path leaves the arm envelope')
        samples.append(q.tolist())
    return samples


def validate_observation_direction(goal,pose,camera_transform,settings):
    """Keep the arm base and camera facing the target side, without folding back."""
    bearing=settings.get('target_bearing_rad')
    if bearing is None:return
    error=math.atan2(math.sin(goal[0]-bearing),math.cos(goal[0]-bearing))
    if abs(error)>math.pi/3+1e-7:
        raise ValueError('Observation arm base turns away from target side')
    if math.sin(goal[0])*math.sin(bearing)<-1e-6:
        raise ValueError('Observation arm base rotates to the opposite side')
    direction=np.array([math.cos(bearing),math.sin(bearing)])
    camera=np.asarray(pose)@camera_transform
    if float(camera[:2,2]@direction)<-1e-6:
        raise ValueError('Observation camera looks back toward the opposite side')


def plan_observation(model,start,camera_transform,settings,joint=4,amplitude=math.radians(15)):
    """Select a verified IK pose near the requested SDK end-frame position."""
    start=np.asarray(start,float);camera_transform=np.asarray(camera_transform,float)
    target=np.asarray(settings.get('target_end_xyz_m',[.20,0.,.30]),float)
    radius=float(settings.get('search_radius_m',.04))
    if target.shape!=(3,) or not np.isfinite(target).all() or not 0<=radius<=.08:
        raise ValueError('Invalid observation position/neighborhood')
    offsets=[(x,y,z) for x in (0.,-.02,.02,-.04,.04)
             for y in (0.,-.02,.02) for z in (0.,-.02,.02,-.04,.04)
             if np.linalg.norm([x,y,z])<=radius+1e-9]
    offsets.sort(key=lambda p:np.linalg.norm(p))
    pitches=settings.get('pitch_candidates_deg',[15.,5.,25.,0.,-10.])
    if not pitches or not np.isfinite(pitches).all():raise ValueError('Invalid observation pitches')
    yaw=math.radians(float(settings.get('yaw_deg',0.)))
    if not math.isfinite(yaw):raise ValueError('Invalid observation yaw')
    cy,sy=math.cos(yaw),math.sin(yaw)
    attempts=0;last_error='No IK solution'
    for offset in offsets:
        if settings.get('planning_poll') is not None:settings['planning_poll']()
        for degrees in pitches:
            pitch=math.radians(float(degrees));c,s=math.cos(pitch),math.sin(pitch)
            pose=np.eye(4);pose[:3,:3]=[[cy*c,-sy,cy*s],[sy*c,cy,sy*s],[-s,0,c]]
            pose[:3,3]=target+offset
            attempts+=1
            ok,solved=model.inverseKinematics(pose,start,True)
            if not ok:continue
            goal=np.asarray(solved,float)
            if not np.allclose(model.forwardKinematics(goal,6),pose,atol=.003,rtol=0):
                last_error='IK forward verification failed';continue
            try:
                validate_observation_direction(goal,pose,camera_transform,settings)
                sweep=validate_sweep(model,goal,camera_transform,joint,amplitude)
                path=validate_observation_path(model,start,goal,camera_transform)
            except ValueError as exc:
                last_error=str(exc);continue
            return dict(start_q_rad=start.tolist(),goal_q_rad=goal.tolist(),
                        target_end_xyz_m=pose[:3,3].tolist(),pitch_deg=float(degrees),
                        T_arm_end=pose.tolist(),path_q_rad=path,sweep=sweep,ik_attempts=attempts)
    return nearest_observation(model,start,camera_transform,settings,joint,amplitude,
                               target,pose[:3,:3],attempts,last_error)


def nearest_observation(model,start,camera_transform,settings,joint,amplitude,
                        target,rotation,attempts,last_error):
    """Project an unreachable observation point onto usable joint configurations.

    No Cartesian displacement cap. This is a multistart numerical search, not a
    guarantee of a globally nearest solution or a replacement for path checks.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    lower=np.asarray(model.getJointQMin(),float)+.030001
    upper=np.asarray(model.getJointQMax(),float)-.030001
    lower[joint]+=amplitude;upper[joint]-=amplitude
    bearing=settings.get('target_bearing_rad')
    if bearing is not None:
        lower[0]=max(lower[0],bearing-math.pi/3)
        upper[0]=min(upper[0],bearing+math.pi/3)
        if math.sin(bearing)<-1e-6:upper[0]=min(upper[0],0.)
        elif math.sin(bearing)>1e-6:lower[0]=max(lower[0],0.)
        if lower[0]>=upper[0]:raise ValueError('Target direction is outside arm base rotation range')
    aim=settings.get('aim_target_arm_xyz_m')
    if aim is not None:
        aim=np.asarray(aim,float)
        if aim.shape!=(3,) or not np.isfinite(aim).all():raise ValueError('Invalid observation aim target')
    initial=np.clip(start,lower,upper)
    seeds=[initial,(lower+upper)/2]
    for index in (0,1,2,3):
        for delta in (-.8,.8):
            seed=initial.copy();seed[index]+=delta;seeds.append(np.clip(seed,lower,upper))
    seeds.extend(np.random.default_rng(0).uniform(lower,upper,size=(12,6)))
    feasible=[];last_poll=-float('inf')
    rejected=dict(ik=0,fk=0,path=0,aim=0)
    def poll():
        nonlocal last_poll
        if settings.get('planning_poll') is not None and time.monotonic()-last_poll>=.05:
            settings['planning_poll']();last_poll=time.monotonic()
    def checked(candidate):
        nonlocal attempts,last_error
        poll();attempts+=1
        pose=np.asarray(model.forwardKinematics(candidate,6))
        ok,verified=model.inverseKinematics(pose,candidate,True)
        if not ok:
            rejected['ik']+=1;last_error='No IK solution for workspace candidate';return None
        verified=np.asarray(verified,float)
        if not np.allclose(model.forwardKinematics(verified,6),pose,atol=.003,rtol=0):
            rejected['fk']+=1;last_error='Workspace IK forward verification failed';return None
        try:
            validate_observation_direction(verified,pose,camera_transform,settings)
            sweep=validate_sweep(model,verified,camera_transform,joint,amplitude)
            path=validate_observation_path(model,start,verified,camera_transform)
        except ValueError as exc:
            rejected['path']+=1;last_error=str(exc);return None
        return verified,pose,sweep,path
    def keep(result):
        nonlocal last_error
        if result is None:return
        q,pose,sweep,path=result
        camera=pose@camera_transform
        error=0.
        if aim is not None:
            ray=aim-camera[:3,3];distance=np.linalg.norm(ray)
            # Only an exactly coincident point has no defined viewing direction.
            if distance<=1e-9:
                rejected['aim']+=1;last_error='Camera and target coincide; viewing direction undefined';return
            error=math.degrees(math.acos(float(np.clip(np.dot(camera[:3,2],ray/distance),-1,1))))
            if error>5.:
                rejected['aim']+=1;last_error=f'Camera aim error {error:.2f}deg exceeds 5deg';return
        offset=float(np.linalg.norm(pose[:3,3]-target))
        feasible.append((offset,dict(start_q_rad=start.tolist(),goal_q_rad=q.tolist(),
            target_end_xyz_m=pose[:3,3].tolist(),requested_end_xyz_m=target.tolist(),
            position_offset_m=offset,aim_error_deg=error,search_mode='nearest_reachable',
            camera_target_distance_m=None if aim is None else float(distance),
            pitch_deg=math.degrees(math.atan2(-pose[2,0],math.hypot(pose[0,0],pose[1,0]))),
            T_arm_end=pose.tolist(),path_q_rad=path,sweep=sweep)))
    baseline=checked(initial);keep(baseline)
    def residual(q):
        poll();pose=np.asarray(model.forwardKinematics(q,6));camera=pose@camera_transform
        rotation_error=Rotation.from_matrix(rotation.T@pose[:3,:3]).as_rotvec()
        if aim is None:return np.r_[pose[:3,3]-target,.15*rotation_error]
        ray=aim-camera[:3,3];distance=np.linalg.norm(ray)
        outward=0. if bearing is None else float(camera[:2,2]@np.array([math.cos(bearing),math.sin(bearing)]))
        return np.r_[pose[:3,3]-target,2.*(camera[:3,2]-ray/max(distance,1e-9)),
                     .001*rotation_error,4.*min(0.,outward)]
    for seed in seeds:
        poll()
        solution=least_squares(residual,seed,bounds=(lower,upper),max_nfev=120,
                               ftol=1e-7,xtol=1e-7,gtol=1e-7)
        result=checked(solution.x)
        keep(result)
        if result is None and baseline is not None:
            # Recover a usable point when the optimizer touches a path/envelope boundary.
            low,high=0.,1.;usable=baseline
            for _ in range(10):
                fraction=(low+high)/2
                candidate=checked(baseline[0]+fraction*(solution.x-baseline[0]))
                if candidate is None:high=fraction
                else:low=fraction;usable=candidate
            keep(usable)
    if feasible:
        plan=min(feasible,key=lambda item:item[0])[1];plan['ik_attempts']=attempts
        return plan
    raise ValueError('No usable observation pose after local and joint-workspace search; '
                     f'target={target.tolist()}, attempts={attempts}, rejected={rejected}, last rejection: {last_error}')


@dataclass
class SettleTracker:
    """Report which measured joint condition prevents observation arrival."""
    position_tolerance: float = .025
    speed_tolerance: float = .025
    stable_sec: float = .4
    stable_since: object = None
    position_window_sec: float = 0.
    position_range_tolerance: float = .003
    _samples: list = field(default_factory=list,init=False,repr=False)

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (
                self.position_tolerance,self.speed_tolerance,self.stable_sec)):
            raise ValueError('Settle tolerances and stable time must be finite and positive')
        if (not math.isfinite(self.position_window_sec) or self.position_window_sec<0 or
                not math.isfinite(self.position_range_tolerance) or self.position_range_tolerance<=0):
            raise ValueError('Invalid position stability window/range')

    def update(self,now,q,dq,goal):
        q,dq,goal=np.asarray(q,float),np.asarray(dq,float),np.asarray(goal,float)
        if any(v.shape!=(6,) or not np.isfinite(v).all() for v in (q,dq,goal)):
            raise ValueError('Non-finite or malformed joint feedback during settling')
        error=np.abs(q-goal);velocity=np.abs(dq)
        position_ok=bool(np.max(error)<self.position_tolerance)
        speed_ok=bool(np.max(velocity)<self.speed_tolerance)
        # Encoder positions provide an independent arrival check when reported dq is noisy.
        # Require a full, densely sampled window inside the target tolerance; a real
        # displacement or feedback gap invalidates this alternative.
        position_stable=False;window_span=0.;position_range=None
        if self.position_window_sec>0:
            if (not position_ok or (self._samples and
                    (now<=self._samples[-1][0] or now-self._samples[-1][0]>.15))):
                self._samples.clear()
            if position_ok:
                self._samples.append((now,q.copy()))
                while len(self._samples)>1 and now-self._samples[1][0]>=self.position_window_sec:
                    self._samples.pop(0)
                window_span=now-self._samples[0][0]
                position_range=float(np.max(np.ptp([s[1] for s in self._samples],axis=0)))
                position_stable=bool(len(self._samples)>=10 and window_span>=self.position_window_sec
                    and position_range<self.position_range_tolerance
                    and position_range/max(window_span,1e-9)<self.speed_tolerance)
        if position_ok and (speed_ok or position_stable):
            if self.stable_since is None:self.stable_since=now
        else:self.stable_since=None
        stable_for=0. if self.stable_since is None else max(0.,now-self.stable_since)
        return dict(q_measured_rad=q.tolist(),q_goal_rad=goal.tolist(),dq_measured_rad_s=dq.tolist(),
                    joint_error_rad=(q-goal).tolist(),max_joint_error_rad=float(np.max(error)),
                    max_joint_error_deg=math.degrees(float(np.max(error))),
                    worst_position_joint=int(np.argmax(error))+1,
                    max_joint_speed_rad_s=float(np.max(velocity)),
                    worst_speed_joint=int(np.argmax(velocity))+1,
                    position_ok=position_ok,speed_ok=speed_ok,stable_for_sec=stable_for,
                    position_window_stable=position_stable,position_window_span_sec=window_span,
                    position_window_range_rad=position_range,
                    position_window_required_sec=self.position_window_sec,
                    position_range_tolerance_rad=self.position_range_tolerance,
                    settle_basis=('reported_speed' if speed_ok else
                                  'position_window' if position_stable else 'waiting'),
                    position_tolerance_rad=self.position_tolerance,speed_tolerance_rad_s=self.speed_tolerance,
                    stable_required_sec=self.stable_sec,settled=bool(stable_for>=self.stable_sec))


def plan_startup_reset(model,start,camera_transform,settings):
    """Return to configured startFlat joints, without motor zero calibration."""
    goal=np.asarray(settings['target_q_rad'],float)
    path=validate_observation_path(model,start,goal,np.asarray(camera_transform,float),goal_margin=0.,
                                   floor_plane=settings.get('floor_plane'))
    pose=np.asarray(model.forwardKinematics(goal,6))
    return dict(stage='reset',start_q_rad=np.asarray(start,float).tolist(),goal_q_rad=goal.tolist(),
                target_end_xyz_m=pose[:3,3].tolist(),T_arm_end=pose.tolist(),path_q_rad=path,
                source=settings.get('source','configured startFlat joints'),motor_zero_calibration=False)
