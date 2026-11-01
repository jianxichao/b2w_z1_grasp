"""Integration against a simulated Sport response stream; always isolated DDS."""
import os
import json
import time
import rclpy
import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt64
from unitree_api.msg import Request,Response
from unitree_go.msg import SportModeState
from b2w_interfaces.srv import SportLocalGoal
from b2w_hardware.b2w_sport_cmd_bridge import B2WSportCmdBridge


@pytest.mark.parametrize('posture_reply',[True,False])
def test_real_service_position_packet_and_independent_watchdog(posture_reply):
    assert os.environ.get('ROS_DOMAIN_ID')=='139' and os.environ.get('ROS_LOCALHOST_ONLY')=='1'
    rclpy.init();bridge=B2WSportCmdBridge();mock=rclpy.create_node('mock_sport_position')
    executor=SingleThreadedExecutor();executor.add_node(bridge);executor.add_node(mock)
    packets=[];policies=[];state_pub=mock.create_publisher(SportModeState,'/lf/sportmodestate',10)
    response_pub=mock.create_publisher(Response,'/api/sport/response',10)
    lease_pub=mock.create_publisher(UInt64,'/b2w_sport_cmd_bridge/position_keepalive',10)
    def receive(msg):
        packets.append((int(msg.header.identity.api_id),json.loads(msg.parameter or '{}')))
        policies.append((int(msg.header.identity.api_id),bool(msg.header.policy.noreply)))
        response=Response();response.header.identity=msg.header.identity;response.header.status.code=0
        response.data='{}'
        if posture_reply or int(msg.header.identity.api_id)!=1005:response_pub.publish(response)
    sub=mock.create_subscription(Request,'/api/sport/request',receive,10)
    def publish_state():
        s=SportModeState();s.imu_state.quaternion=[1.,0.,0.,0.]
        t=mock.get_clock().now().to_msg();s.stamp.sec=t.sec;s.stamp.nanosec=t.nanosec;state_pub.publish(s)
    timer=mock.create_timer(.02,publish_state)
    client=mock.create_client(SportLocalGoal,'/b2w_sport_cmd_bridge/move_to_local_goal')
    def spin_for(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:executor.spin_once(timeout_sec=.02)
    try:
        spin_for(.7)
        assert not packets  # Merely launching the bridge never controls the robot.
        req=SportLocalGoal.Request();req.x=1.;req.y=.5;req.yaw=-1.57079632679
        future=client.call_async(req)
        end=time.monotonic()+1
        while not future.done() and time.monotonic()<end:executor.spin_once(timeout_sec=.02)
        result=future.result();assert result.success
        lease=result.command_id
        timer_lease=mock.create_timer(.05,lambda:lease_pub.publish(UInt64(data=lease)))
        spin_for(.5)
        assert bridge._position.state=='MOVING'
        moves=[payload for api,payload in packets if api==1008]
        assert len(moves)>=2
        assert all(set(payload)=={'x','y','z'} and payload['y']==0. for payload in moves)
        assert all(0.<=payload['x']<=.3 and abs(payload['z'])<=.2 for payload in moves)
        assert all(api not in (1015,1036) for api,_ in packets)
        assert all(noreply for api,noreply in policies if api==1008)
        timer_lease.cancel();spin_for(.8)
        assert bridge._position.state=='FAULT' and not bridge._position.active
        assert 1003 in [a for a,_ in packets]
        assert all(not noreply for api,noreply in policies if api==1003)
        # A second, already-near goal exercises real ROS arrival -> stop -> posture ownership.
        start_packet=len(packets)
        req=SportLocalGoal.Request();req.x=.04;req.y=0.;req.yaw=0.
        future=client.call_async(req)
        end=time.monotonic()+1
        while not future.done() and time.monotonic()<end:executor.spin_once(timeout_sec=.02)
        result=future.result();assert result.success
        lease=result.command_id
        timer_lease=mock.create_timer(.05,lambda:lease_pub.publish(UInt64(data=lease)))
        spin_for(5.3)
        assert bridge._position.state=='ARRIVED' and bridge._position.stand_down_wait_completed
        assert bridge._position.stand_down_acknowledged is posture_reply
        arrival_apis=[api for api,_ in packets[start_packet:]]
        assert arrival_apis==[1003,1003,1003,1005]
        timer_lease.cancel()
    finally:
        timer.cancel();executor.remove_node(bridge);executor.remove_node(mock)
        bridge.destroy_node();mock.destroy_node();executor.shutdown();rclpy.shutdown()
