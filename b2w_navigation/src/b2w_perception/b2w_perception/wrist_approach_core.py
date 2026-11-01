"""Geometry for a frozen, depth-confirmed wrist target. No hardware or ROS calls."""
import math
import numpy as np


def rigid_transform(value):
    t=np.asarray(value,float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or
            not np.allclose(t[3],[0,0,0,1],atol=1e-7) or
            not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return t


def transform_from_pose(position,quaternion):
    p=np.asarray(position,float);q=np.asarray(quaternion,float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Invalid pose')
    norm=np.linalg.norm(q)
    if abs(norm-1.)>.01:raise ValueError('Invalid quaternion norm')
    x,y,z,w=q/norm
    t=np.eye(4);t[:3,3]=p
    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t


def map_target(camera_points,t_map_base,t_base_arm,t_arm_end,t_end_camera,max_spread=.08):
    points=np.asarray(camera_points,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<1 or not np.isfinite(points).all():
        raise ValueError('Approach requires a fresh valid depth position')
    center=np.median(points,axis=0)
    if np.any(points[:,2]<=0) or np.max(np.linalg.norm(points-center,axis=1))>max_spread:
        raise ValueError('Depth target is inconsistent across confirmation frames')
    chain=(rigid_transform(t_map_base)@rigid_transform(t_base_arm)@
           rigid_transform(t_arm_end)@rigid_transform(t_end_camera))
    return (chain@np.r_[center,1.])[:3]


def yaw_of(t):
    return math.atan2(t[1,0],t[0,0])


def angle_error(a,b):
    return math.atan2(math.sin(a-b),math.cos(a-b))


def local_side_goal(camera_point,t_base_arm,t_arm_end,t_end_camera,side='left',offset=.5):
    """Offset sideways and retain the initial body heading (local yaw zero)."""
    if side not in ('left','right') or not math.isfinite(offset) or not .1<=offset<=1.:
        raise ValueError('Invalid side/offset')
    target=map_target([camera_point],np.eye(4),t_base_arm,t_arm_end,t_end_camera)
    goal=target[:2].copy();goal[1]+=offset if side=='left' else -offset
    yaw=0.  # Keep the target alongside the body for the subsequent arm grasp.
    return target,goal,yaw


def sport_body_transform(msg):
    """Sport world -> current body pose, including height and IMU roll/pitch."""
    if msg.error_code:raise ValueError('Sport body feedback reports an error')
    p=np.asarray(msg.position,float).copy()
    height=float(msg.body_height)
    if math.isfinite(height) and .05<height<1.5:p[2]=height
    if p.shape!=(3,) or not .05<p[2]<1.5:raise ValueError('Invalid Sport body height')
    w,x,y,z=msg.imu_state.quaternion
    return transform_from_pose(p,[x,y,z,w])


def world_target_to_arm(target_world,t_world_body,t_base_arm):
    point=np.asarray(target_world,float)
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Invalid frozen world target')
    return np.linalg.solve(rigid_transform(t_world_body)@rigid_transform(t_base_arm),np.r_[point,1.])[:3]
