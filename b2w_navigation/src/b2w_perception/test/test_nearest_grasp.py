"""Nearest-point selection with a synthetic IK model; no SDK client or motion."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'unitree_z1/yolo/grasp_demo'))
from demo_self import nearby_target


class CartesianModel:
    def __init__(self,accept):self.accept=accept;self.attempts=[]
    def getJointQMin(self):return np.full(6,-10.)
    def getJointQMax(self):return np.full(6,10.)
    def forwardKinematics(self,q,link):
        t=np.eye(4);t[:3,3]=q[:3];t[:3,:3]=Rotation.from_rotvec(q[3:]).as_matrix()
        return t
    def inverseKinematics(self,t,q,check):
        self.attempts.append(t[:3,3].copy())
        return self.accept(t[:3,3]),np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_rotvec()]


def test_center_is_chosen_without_any_forced_penetration():
    m=CartesianModel(lambda point:True);center=np.array([.3,0,.3])
    _,q=nearby_target(m,np.zeros(6),center,np.eye(4),np.eye(3),[1,0,0])
    np.testing.assert_allclose(q[:3],center)
    assert len(m.attempts)==1


@pytest.mark.parametrize('sign',[1.,-1.])
def test_nearest_sampled_reachable_point_can_lie_on_either_side_of_center(sign):
    center=np.array([.3,0,.3])
    m=CartesianModel(lambda point:sign*(point[0]-center[0])>=.015)
    _,q=nearby_target(m,np.zeros(6),center,np.eye(4),np.eye(3),[1,0,0])
    assert q[:3]==pytest.approx(center+np.array([sign*.015,0,0]),abs=3e-5)


def test_invalid_path_candidate_is_skipped_for_the_next_nearest_candidate():
    center=np.array([.3,0,.3]);m=CartesianModel(lambda point:True)
    def validate(q):
        if not q[0]>center[0]+.005:raise ValueError('Rejected path')
    _,q=nearby_target(m,np.zeros(6),center,np.eye(4),np.eye(3),[1,0,0],validate)
    assert np.linalg.norm(q[:3]-center)==pytest.approx(.005,abs=3e-5)


def test_unreachable_target_still_stops_with_center_and_rejection_diagnostics():
    m=CartesianModel(lambda point:False)
    with pytest.raises(RuntimeError,match='center=.*rejected='):
        nearby_target(m,np.zeros(6),np.array([.3,0,.3]),np.eye(4),np.eye(3),[1,0,0])


def test_joint_space_search_finds_reachable_point_beyond_four_centimeters():
    class LimitedCartesianModel(CartesianModel):
        def getJointQMin(self):return np.array([.38,-.1,.2,-1.,-1.,-1.])
        def getJointQMax(self):return np.array([.8,.1,.5,1.,1.,1.])
    model=LimitedCartesianModel(lambda point:point[0]>=.38)
    _,q=nearby_target(model,np.array([.5,0,.3,0,0,0]),np.array([.3,0,.3]),
                      np.eye(4),np.eye(3),[1,0,0])
    assert q[:3]==pytest.approx([.38,0,.3],abs=1e-4)
    assert np.linalg.norm(q[:3]-[.3,0,.3])>.04


def test_expanding_search_remains_cancelable():
    model=CartesianModel(lambda point:False)
    def canceled():raise RuntimeError('operator canceled')
    with pytest.raises(RuntimeError,match='operator canceled'):
        nearby_target(model,np.zeros(6),np.array([.3,0,.3]),np.eye(4),np.eye(3),[1,0,0],poll=canceled)
    assert model.attempts==[]
