from types import SimpleNamespace
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2
from b2w_hardware.pointcloud_restamp_node import PointCloudRestampNode


def test_raw_sensor_clock_is_rebased_without_claiming_map_coordinates():
    node = object.__new__(PointCloudRestampNode)
    node._frame_id = ''
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda:Time(sec=100)))
    outputs = []
    node._publisher = SimpleNamespace(publish=outputs.append)
    cloud = PointCloud2()
    cloud.header.stamp.sec = 120  # Real robot and PC differ by about 20 seconds.
    cloud.header.frame_id = 'rslidar'
    cloud.data = bytes([1,2,3,4])
    node._on_cloud(cloud)
    assert outputs[0].header.stamp.sec == 100
    assert outputs[0].header.frame_id == 'rslidar'
    assert bytes(outputs[0].data) == bytes([1,2,3,4])
