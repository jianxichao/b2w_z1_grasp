import math
from types import SimpleNamespace
from unitree_go.msg import SportModeState
from b2w_hardware.sport_odom_node import SportOdom


class Recorder:
    def __init__(self): self.messages = []
    def publish(self, msg): self.messages.append(msg)
    def sendTransform(self, msg): self.messages.append(msg)


def make_node():
    node = object.__new__(SportOdom)
    node.origin = node.last_stamp = node.last_received = None
    node.pub, node.tf = Recorder(), Recorder()
    from builtin_interfaces.msg import Time
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=100)))
    return node


def state(tick, x, y, yaw=0):
    msg = SportModeState()
    msg.stamp.sec = tick
    msg.position = [float(x),float(y),0.0]
    msg.imu_state.quaternion = [math.cos(yaw/2),0.0,0.0,math.sin(yaw/2)]
    return msg


def test_origin_heading_rotates_world_displacement_into_odom():
    node = make_node()
    node.receive(state(1, 3, 4, math.pi/2))
    node.receive(state(2, 3, 5, math.pi/2))
    msg = node.pub.messages[-1]
    assert abs(msg.pose.pose.position.x-1) < 1e-6
    assert abs(msg.pose.pose.position.y) < 1e-6
    assert abs(msg.pose.pose.orientation.z) < 1e-6
    assert msg.header.frame_id == 'odom' and msg.child_frame_id == 'base_footprint'


def test_repeated_tick_and_faults_cannot_refresh_odom():
    node = make_node()
    node.receive(state(1, 0, 0))
    received = node.last_received
    node.receive(state(1, 1, 0))
    bad = state(2, 1, 0)
    bad.error_code = 1
    node.receive(bad)
    node.receive(state(3, float('nan'), 0))
    assert len(node.pub.messages) == 1
    assert node.last_received == received


def test_invalid_imu_quaternion_does_not_create_a_tf():
    node = make_node()
    msg = state(1, 0, 0)
    msg.imu_state.quaternion = [0.0]*4
    node.receive(msg)
    assert node.origin is None and not node.tf.messages


def test_body_tf_uses_feedback_height_and_keeps_official_mounts_in_body_frame():
    node = make_node()
    msg = state(1, 0, 0, math.pi/2)
    msg.position[2] = 0.4
    msg.body_height = 0.0  # Actual B2W firmware reports this field as zero.
    node.receive(msg)
    body = node.tf.messages[-1]
    assert body.header.frame_id == 'base_footprint'
    assert body.child_frame_id == 'base_link'
    assert abs(body.transform.translation.z-0.4) < 1e-6
    assert abs(body.transform.rotation.z) < 1e-6
    assert abs(body.transform.rotation.w-1.0) < 1e-6


def test_body_tf_removes_world_yaw_but_retains_body_pitch():
    node = make_node()
    msg = state(1, 0, 0)
    yaw, pitch = math.pi/2, .2
    cy,sy,cp,sp = math.cos(yaw/2),math.sin(yaw/2),math.cos(pitch/2),math.sin(pitch/2)
    msg.imu_state.quaternion = [cy*cp,-sy*sp,cy*sp,sy*cp]
    msg.body_height = .4
    node.receive(msg)
    q = node.tf.messages[-1].transform.rotation
    assert abs(q.y-math.sin(pitch/2)) < 1e-6
    assert abs(q.w-math.cos(pitch/2)) < 1e-6
    assert abs(q.z) < 1e-6 and abs(q.x) < 1e-6
