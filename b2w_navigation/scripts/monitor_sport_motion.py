#!/usr/bin/env python3
"""Read-only Sport wire/state recorder. Never creates a command publisher."""
import argparse
import collections
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=45.)
    parser.add_argument('--output',type=Path,default=Path('output/wrist_search/sport_wire_trace.jsonl'))
    args=parser.parse_args()
    if not 0<args.seconds<=120:parser.error('--seconds must be in (0,120]')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from unitree_api.msg import Request,Response
    from unitree_go.msg import SportModeState
    rclpy.init();node=rclpy.create_node('sport_motion_readonly_monitor')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter();last_commands={};reply_counts=collections.Counter();latest={}
    stream=args.output.open('a')
    def write(kind,**fields):
        stream.write(json.dumps(dict(t=time.monotonic(),kind=kind,**fields))+'\n');stream.flush()
    def request(msg):
        api=int(msg.header.identity.api_id);counts[api]+=1;last_commands[api]=msg.parameter
        write('request',api=api,request_id=int(msg.header.identity.id),parameter=msg.parameter,
              noreply=bool(msg.header.policy.noreply),priority=int(msg.header.policy.priority))
    def response(msg):
        api=int(msg.header.identity.api_id);code=int(msg.header.status.code)
        reply_counts[(api,code)]+=1
        write('response',api=api,request_id=int(msg.header.identity.id),code=code,data=msg.data)
    def state(msg):
        latest.update(mode=int(msg.mode),gait_type=int(msg.gait_type),error_code=int(msg.error_code),
            position=list(map(float,msg.position)),velocity=list(map(float,msg.velocity)),
            yaw_speed=float(msg.yaw_speed))
    subscriptions=[node.create_subscription(Request,'/api/sport/request',request,100),
        node.create_subscription(Response,'/api/sport/response',response,100),
        node.create_subscription(SportModeState,'/lf/sportmodestate',state,qos_profile_sensor_data)]
    try:
        started=reported=time.monotonic()
        while time.monotonic()-started<args.seconds:
            rclpy.spin_once(node,timeout_sec=.02)
            if time.monotonic()-reported>=1.:
                report=dict(requests=dict(counts),last_commands=dict(last_commands),
                    replies=[dict(api=api,code=code,count=count) for (api,code),count in reply_counts.items()],
                    sport_state=latest.copy())
                write('summary',**report);print(json.dumps(report),flush=True)
                counts.clear();last_commands.clear();reply_counts.clear();reported=time.monotonic()
    except KeyboardInterrupt:pass
    finally:
        stream.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
