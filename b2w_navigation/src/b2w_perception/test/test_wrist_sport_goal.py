import numpy as np
import pytest
from b2w_perception.wrist_approach_core import local_side_goal


@pytest.mark.parametrize('side,dy',[('left',.5),('right',-.5)])
def test_body_side_offset_retains_heading_and_target_is_alongside(side,dy):
    mount=np.eye(4);mount[:3,3]=[.078,0,.09602]
    arm=np.eye(4);arm[:3,3]=[.2,0,.3]
    target,goal,heading=local_side_goal([1.,.1,.2],mount,arm,np.eye(4),side,.5)
    assert target==pytest.approx([1.278,.1,.59602])
    assert goal==pytest.approx([target[0],target[1]+dy])
    assert heading==pytest.approx(0.)
    # With this heading the target has zero forward displacement and lies
    # directly to the right/left of the stopped robot.
    delta=target[:2]-goal
    assert delta==pytest.approx([0.,-dy])
    assert np.linalg.norm(goal-target[:2])==pytest.approx(.5)


def test_real_hand_eye_rotation_is_used_before_side_offset():
    camera=np.array([[0,0,1,-.04],[-1,0,0,0],[0,-1,0,-.04],[0,0,0,1]])
    target,goal,yaw=local_side_goal([.1,.2,1.],np.eye(4),np.eye(4),camera,'left',.5)
    assert target==pytest.approx([.96,-.1,-.24])
    assert goal==pytest.approx([.96,.4])
