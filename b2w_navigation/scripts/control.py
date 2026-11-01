#!/usr/bin/env python3
"""Explicit staged motion control. Running without a command prints help only."""
import argparse
import sys
import rclpy
from std_srvs.srv import SetBool, Trigger
from preflight import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('enable', 'stop', 'mission-start', 'mapping-start'))
    args = parser.parse_args()
    if args.command in ('enable', 'mission-start'):
        if not check(live=True, target=args.command=='mission-start', motion=True):
            return 1
    rclpy.init()
    node = rclpy.create_node('b2w_manual_control')

    def call(service, data=None, optional=False):
        kind = Trigger if data is None else SetBool
        client = node.create_client(kind, service)
        if not client.wait_for_service(timeout_sec=2.0):
            print(('SKIP' if optional else 'FAIL') + ': service unavailable: ' + service)
            return optional
        request = kind.Request()
        if data is not None:
            request.data = data
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        response = future.result() if future.done() else None
        print(service, ':', response.message if response else 'timeout')
        return response is not None and response.success

    def stop():
        # Attempt every stop even if an upstream node is absent or times out.
        results = [call('/wrist_search/stop', optional=True),
                   call('/mission_supervisor_node/abort', optional=True),
                   call('/frontier_explorer_node/enable', False, optional=True),
                   call('/cmd_vel_safety_node/enable', False),
                   call('/b2w_sport_cmd_bridge/enable', False)]
        return all(results)

    try:
        if args.command == 'stop':
            ok = stop()
        elif args.command == 'enable':
            ok = call('/b2w_sport_cmd_bridge/enable', True)
            if ok:
                ok = call('/cmd_vel_safety_node/enable', True)
            if not ok:
                stop()
        elif args.command == 'mission-start':
            ok = call('/mission_supervisor_node/start')
        else:
            ok = call('/unitree_slam/start_mapping')
        return 0 if ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
