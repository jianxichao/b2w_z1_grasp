#!/usr/bin/env python3
"""Arm-camera search with optional Frontier exploration and Sport/Nav2 approach.

Default is camera preview only. --execute prepares an observation IK pose, scans, and maintains
control after stop/found. Ctrl+C resets the arm and exits. --explore also enables the two base gates + Frontier.
--approach defaults to odometry-based Move velocity control at the sponge's left/right side, then holds. --grasp adds side search, grasp and closed-gripper reset.
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from b2w_perception.wrist_search_core import SearchPolicy, scan_offset
from wrist_search_io import Camera, Wrist, ROOT


class Vision:
    def __init__(self, camera, model, cfg, output):
        self.camera,self.model,self.cfg,self.output=camera,model,cfg,output
        self.latest,self.error=None,None
        self.done,self.ready=threading.Event(),threading.Event()
        self.lock=threading.Lock()
        self.worker=threading.Thread(target=self.run,daemon=True)
        self.worker.start()

    def read(self):
        with self.lock:return self.latest

    def run(self):
        number=0;last_source_frame=-1
        try:
            while not self.done.is_set():
                started=time.monotonic()
                color,depth,meta=self.camera.read()
                capture_sec=time.monotonic()-started
                source_frame=int(meta['frame_number'])
                if source_frame<=last_source_frame:raise RuntimeError('Camera exposure counter repeated or reset')
                last_source_frame=source_frame
                # Camera warm-up happens before motion and may exceed the capture limit.
                if number>0 and capture_sec>self.cfg['max_capture_sec']:
                    raise RuntimeError(f'Camera capture delayed {capture_sec:.2f}s')
                result=self.model.predict(color,classes=self.classes,conf=self.cfg['confidence'],
                                          imgsz=640,device='cpu',verbose=False)[0]
                box,confidence,point=None,0.,None
                if len(result.boxes):
                    index=int(result.boxes.conf.argmax().item())
                    box=result.boxes.xyxy[index].cpu().tolist()
                    confidence=float(result.boxes.conf[index].item())
                    point=depth_point(depth,box,meta,self.cfg['depth'])
                number+=1
                item=dict(number=source_frame,captured_at=time.monotonic() if number==1 else started,
                          received_at=time.monotonic(),box=box,confidence=confidence,
                          camera_xyz_m=point,meta=meta,annotated=result.plot())
                (self.output/'latest.json').write_text(json.dumps({k:v for k,v in item.items() if k!='annotated'},indent=2)+'\n')
                cv2.imwrite(str(self.output/'latest.jpg'),item['annotated'])
                cv2.imwrite(str(self.output/'latest_depth.png'),depth)
                with self.lock:self.latest=item
                self.ready.set()
                time.sleep(.05)
        except Exception as exc:
            self.error=str(exc);self.ready.set()

    @property
    def classes(self):return [i for i,name in self.model.names.items() if name==self.cfg['class_name']]

    def close(self):
        self.done.set();self.camera.close();self.worker.join(timeout=3)


def depth_point(depth,box,meta,cfg):
    x1,y1,x2,y2=box;u,v=(x1+x2)/2,(y1+y2)/2
    hw,hh=(x2-x1)*cfg['roi_fraction']/2,(y2-y1)*cfg['roi_fraction']/2
    patch=depth[max(0,int(v-hh)):min(depth.shape[0],int(v+hh)+1),
                max(0,int(u-hw)):min(depth.shape[1],int(u+hw)+1)].astype(float)*meta['depth_scale_m']
    valid=patch[(patch>=cfg['min_m'])&(patch<=cfg['max_m'])]
    if not patch.size or len(valid)<cfg['min_pixels'] or len(valid)/patch.size<cfg['min_valid_fraction']:
        return None
    z=float(np.median(valid));inliers=valid[np.abs(valid-z)<=cfg['max_spread_m']]
    if len(inliers)<cfg['min_pixels'] or len(inliers)/len(valid)<.7:return None
    z=float(np.median(inliers))
    return [(u-meta['ppx'])*z/meta['fx'],(v-meta['ppy'])*z/meta['fy'],z]


class Coordinator:
    def __init__(self, cfg, wrist, vision, explore, output):
        import rclpy
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
        from rclpy.node import Node
        from std_msgs.msg import String
        from std_srvs.srv import Trigger,SetBool
        from nav_msgs.msg import Odometry
        from unitree_go.msg import SportModeState
        from sensor_msgs.msg import LaserScan
        from action_msgs.srv import CancelGoal
        self.ros=rclpy;self.String=String;self.explore=explore
        self.node=Node('wrist_search')
        self.cfg,self.wrist,self.vision,self.output=cfg,wrist,vision,output
        self.policy=SearchPolicy(cfg['vote_required_frames'],cfg['confidence'],cfg['vote_window_frames'],cfg['max_search_sec'])
        self.policy.start(time.monotonic())
        self.odom_at,self.scan_at,self.base_speed=0.,0.,float('inf')
        self.body_pose=None;self.body_pose_at=0.;self.last_body_stamp=None
        self.gates={};self.stop_requested=False;self.release_requested=False
        self.limit_event=None;self.last_status=0.
        self.touched_base=False;self.gate_grace_until=0.
        self.approach=None;self.grasp=None
        self.fault_reason=None
        self.exit_requested=False;self.exiting=False
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.publisher=self.node.create_publisher(String,'/wrist_search/status',qos)
        self.node.create_subscription(Odometry,'/b2w/sport/odom',self.on_odom,qos_profile_sensor_data)
        self.node.create_subscription(SportModeState,'/lf/sportmodestate',self.on_sport_body,qos_profile_sensor_data)
        self.node.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
        self.node.create_subscription(String,'/task_event',self.on_event,10)
        for topic in ('/b2w_sport_cmd_bridge/status','/cmd_vel_safety_node/status'):
            self.node.create_subscription(String,topic,lambda m,t=topic:self.on_gate(m,t),qos)
        self.node.create_service(Trigger,'~/stop',self.on_stop)
        self.node.create_service(Trigger,'~/release_control',self.on_release)
        self.clients={name:self.node.create_client(SetBool,name) for name in (
            '/b2w_sport_cmd_bridge/enable','/cmd_vel_safety_node/enable','/frontier_explorer_node/enable')}
        self.cancel=self.node.create_client(CancelGoal,'/navigate_to_pose/_action/cancel_goal')
        self.Trigger,self.SetBool,self.CancelGoal=Trigger,SetBool,CancelGoal

    def on_odom(self,msg):
        self.odom_at=time.monotonic()
        self.base_speed=max(math.hypot(msg.twist.twist.linear.x,msg.twist.twist.linear.y),abs(msg.twist.twist.angular.z))

    def on_sport_body(self,msg):
        from b2w_perception.wrist_approach_core import sport_body_transform
        stamp=(int(msg.stamp.sec),int(msg.stamp.nanosec))
        if stamp!=(0,0) and stamp==getattr(self,'last_body_stamp',None):return
        try:
            pose=sport_body_transform(msg)
        except (ValueError,TypeError,IndexError):
            self.body_pose=None;return
        self.body_pose=pose;self.body_pose_at=time.monotonic();self.last_body_stamp=stamp

    def on_scan(self,msg):
        if len(msg.ranges) and any(math.isfinite(x) for x in msg.ranges):self.scan_at=time.monotonic()

    def on_gate(self,msg,topic):
        try:
            data=json.loads(msg.data)
            if 'enabled' in data:self.gates[topic]=bool(data['enabled'])
        except (ValueError,TypeError):pass

    def on_event(self,msg):
        if self.policy.state=='SEARCHING' and msg.data in ('frontier_goal_limit_reached','frontier_failure_limit_reached'):
            self.limit_event=msg.data

    def on_stop(self,request,response):
        if self.wrist is not None:self.wrist.pause()
        self.stop_requested=True
        response.success=True;response.message='Stop queued; arm control will remain alive'
        return response

    def on_release(self,request,response):
        if self.policy.state not in ('STOPPED','FOUND','ARRIVED','GRASP_DONE','LIMIT','FAULT'):
            response.success=False;response.message='Call /wrist_search/stop and wait for preparation/search to stop first'
        else:
            self.release_requested=True;response.success=True
            response.message='Releasing SDK communication; arm must already be supported or secured'
        return response

    def wait(self,future,timeout=1.):
        self.ros.spin_until_future_complete(self.node,future,timeout_sec=timeout)
        if not future.done() or future.exception():return None
        return future.result()

    def stop_base(self):
        sport=getattr(getattr(self,'approach',None),'backend',None)=='sport_velocity_goal'
        if (not self.explore and not sport) or not self.touched_base:return True
        futures=[]
        for name,client in self.clients.items():
            if client.service_is_ready():
                request=self.SetBool.Request();request.data=False
                future=client.call_async(request)
                if not sport or name=='/b2w_sport_cmd_bridge/enable':futures.append(future)
        if self.cancel.service_is_ready():
            future=self.cancel.call_async(self.CancelGoal.Request())
            if not sport:futures.append(future)
        deadline=time.monotonic()+1.
        while any(not f.done() for f in futures) and time.monotonic()<deadline:
            self.ros.spin_once(self.node,timeout_sec=.02)
        ok=len(futures)>=(1 if sport else 3) and all(f.done() and not f.exception() for f in futures)
        for f in futures:
            if f.done() and not f.exception():
                response=f.result()
                if response is None:ok=False
                elif hasattr(response,'success'):ok=ok and response.success
        self.publish('base_stop_requested',acknowledged=bool(ok))
        return ok

    def enable_base(self,frontier=True):
        if self.stop_requested or getattr(self,'exit_requested',False):raise RuntimeError('Operator canceled motion enable')
        if self.wrist.error or self.vision.error:raise RuntimeError(self.wrist.error or self.vision.error)
        item=self.vision.read()
        if item is None or time.monotonic()-item['captured_at']>self.cfg['frame_timeout_sec']:
            raise RuntimeError('Camera is stale before motion enable')
        if not self.explore:return
        self.touched_base=True
        names=['/b2w_sport_cmd_bridge/enable','/cmd_vel_safety_node/enable']
        if frontier:names.append('/frontier_explorer_node/enable')
        for name in names:
            if self.stop_requested or getattr(self,'exit_requested',False):raise RuntimeError('Operator stop during motion enable')
            client=self.clients[name]
            if not client.wait_for_service(timeout_sec=1):raise RuntimeError('Missing service: '+name)
            request=self.SetBool.Request();request.data=True
            response=self.wait(client.call_async(request))
            if response is None or not response.success:raise RuntimeError('Failed to enable: '+name)
        if self.stop_requested or getattr(self,'exit_requested',False):raise RuntimeError('Operator stop during motion enable')
        self.gate_grace_until=time.monotonic()+1.

    def publish(self,event,**fields):
        now=time.monotonic();wrist=self.wrist
        scanning=getattr(wrist,'scanning',None)
        item=self.vision.read()
        camera_alive=(self.vision.error is None and item is not None and
                      0<=now-item['captured_at']<=self.cfg['frame_timeout_sec'])
        arm_alive=False
        if wrist is not None and getattr(wrist,'running',False):
            try:arm_alive=bool(wrist.arm.feedbackHealthy())
            except Exception:pass
        item=dict(state=self.policy.state,event=event,monotonic_s=now,
                  explore=self.explore,base_output_enabled=len(self.gates)==2 and all(self.gates.values()),
                  wrist_scanning=scanning is not None and scanning.is_set(),arm_control_alive=arm_alive,
                  camera_alive=camera_alive,approach_enabled=getattr(self,'approach',None) is not None,
                  grasp_enabled=getattr(self,'grasp',None) is not None)
        gripper_status=getattr(getattr(self,'grasp',None),'gripper_status',None)
        if gripper_status:item.update(gripper_status)
        grasp_vote=getattr(getattr(self,'grasp',None),'vote_policy',None)
        if grasp_vote is not None:
            item.update(grasp_vote_hits=grasp_vote.vote_count,grasp_vote_frames=len(grasp_vote.votes),
                        grasp_vote_required=grasp_vote.required_frames,grasp_vote_window=grasp_vote.vote_window)
        approach=getattr(self,'approach',None)
        if getattr(approach,'backend',None)=='sport_velocity_goal':
            item['approach_backend']='sport_velocity_goal'
            item['sport_position_active']=approach.active
            item['base_output_enabled']=item['base_output_enabled'] or (approach.active and approach.command_id is not None)
            item['base_command_source']=('sport_velocity_goal' if approach.active else
                                         'nav2_velocity' if item['base_output_enabled'] else 'none')
            status=getattr(approach,'status',None)
            if status and status.get('command_id')==approach.command_id:
                for key in ('controller_version','linear_deadzone_bias_m_s','angular_deadzone_bias_rad_s',
                            'arrival_position_tolerance_m','arrival_yaw_tolerance_rad',
                            'arrival_longitudinal_tolerance_m','longitudinal_error_m',
                            'stand_down_on_arrival','stand_down_acknowledged','physical_posture_verified',
                            'stand_down_wait_completed','stand_down_wait_sec',
                            'controller_phase','move_command_count','commanded_vx_m_s',
                            'commanded_yaw_rate_rad_s','measured_speed_m_s',
                            'distance_error_m','measured_local_xy_yaw'):
                    item[key]=status.get(key)
        item.update(vote_hits=self.policy.vote_count,vote_frames=len(self.policy.votes),
                    vote_required=self.policy.required_frames,vote_window=self.policy.vote_window,
                    fault_reason=getattr(self,'fault_reason',None))
        item.update(fields)
        self.publisher.publish(self.String(data=json.dumps(item)))
        with (self.output/'events.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
        print(json.dumps(item,ensure_ascii=False),flush=True)

    def prepare_observation(self):
        self.policy.state='PREPARING'
        if self.explore:
            self.touched_base=True
            if not self.stop_base():raise RuntimeError('Base stop not acknowledged before arm preparation')
            deadline=time.monotonic()+5.;stationary_since=None
            waiting_reason=None
            while time.monotonic()<deadline:
                self.ros.spin_once(self.node,timeout_sec=.02)
                reason=self.preparation_stop_reason(allow_settling=True,include_navigation=False)
                if reason:raise RuntimeError('Arm preparation stopped: '+reason)
                reason=self.preparation_stop_reason(allow_settling=True)
                if reason:
                    stationary_since=None
                    if reason!=waiting_reason:
                        self.publish('preparing_waiting_for_navigation',reason=reason)
                        waiting_reason=reason
                    continue
                if self.base_speed<.03:
                    stationary_since=stationary_since or time.monotonic()
                    if time.monotonic()-stationary_since>.5:break
                else:stationary_since=None
            else:
                reason=self.preparation_stop_reason(allow_settling=True)
                raise RuntimeError('Arm preparation timed out: '+(reason or 'Base did not become stationary')+
                    '. --explore requires arm_camera_search.launch.py (scan + Sport odometry + Frontier/Nav2); '
                    'sport_position_bringup.launch.py supports approach only: remove --explore.')
        elif getattr(getattr(self,'approach',None),'backend',None)=='sport_velocity_goal':
            self.approach.prepare_base()
        self.publish('preparing_observation_base_stopped')

    def preparation_should_stop(self,allow_settling=False):
        return self.preparation_stop_reason(allow_settling) is not None

    def preparation_stop_reason(self,allow_settling=False,include_navigation=True):
        now=time.monotonic();item=self.vision.read()
        if self.stop_requested or getattr(self,'exit_requested',False):return 'Operator canceled preparation'
        if self.vision.error:return 'Camera error: '+str(self.vision.error)
        if item is None:return 'Camera has no frame'
        if now-item['captured_at']>self.cfg['frame_timeout_sec']:
            return f"Camera frame is stale (age={now-item['captured_at']:.3f}s, limit={self.cfg['frame_timeout_sec']:.3f}s)"
        if not include_navigation:return None
        if self.explore:
            missing=[]
            if self.odom_at<=0:missing.append('/b2w/sport/odom not received')
            elif now-self.odom_at>.5:missing.append('/b2w/sport/odom stale (>0.5s)')
            if self.scan_at<=0:missing.append('/scan has no valid finite-range scan')
            elif now-self.scan_at>1.:missing.append('/scan stale (>1.0s)')
            if missing:return '; '.join(missing)
            if not allow_settling and self.base_speed>=.03:return f'Base is moving during arm preparation (speed_metric={self.base_speed:.4f}, limit=0.0300; max of linear speed and yaw rate)'
        elif now-self.odom_at<.5 and self.base_speed>=.03:
            return f'Base is moving during arm preparation (speed_metric={self.base_speed:.4f}, limit=0.0300; max of linear speed and yaw rate)'
        return None

    def start(self):
        if self.stop_requested:raise RuntimeError('Operator canceled search startup')
        if self.wrist.error:raise RuntimeError(self.wrist.error)
        # Gather fresh callbacks before enabling. No arm sweep during setup.
        end=time.monotonic()+1.
        while time.monotonic()<end:self.ros.spin_once(self.node,timeout_sec=.05)
        if self.explore:
            self.touched_base=True
            if not self.stop_base():raise RuntimeError('Could not establish base stop before search')
        self.policy.start(time.monotonic())
        item=self.vision.read()
        if self.vision.error or item is None or time.monotonic()-item['captured_at']>self.cfg['frame_timeout_sec']:
            raise RuntimeError(self.vision.error or 'Camera stale at search start')
        self.enable_base();self.wrist.resume()
        self.publish('search_started',wrist=self.wrist.report)

    def stop(self,reason,state='STOPPED'):
        if state=='FAULT':self.fault_reason=str(reason)
        if getattr(self,'approach',None) is not None:self.approach.cancel()
        self.wrist.pause();self.policy.state=state
        acknowledged=self.stop_base()
        if state=='TARGET_LOCALIZING' and not acknowledged:
            self.policy.state='FAULT';self.fault_reason='Base stop not acknowledged before target localization'
        self.publish(reason,base_stop_acknowledged=acknowledged,arm_error=self.wrist.error)
        return acknowledged

    def request_exit(self):
        if self.wrist is not None:self.wrist.pause()
        if self.exiting:
            self.stop_requested=True  # A second signal interrupts reset, retaining hold.
        else:self.exit_requested=True

    def exit_reset_should_stop(self,allow_settling=False):
        if self.stop_requested:return True
        sport=getattr(getattr(self,'approach',None),'backend',None)=='sport_velocity_goal'
        if self.explore or sport:
            if time.monotonic()-self.odom_at>.5:return True
            if not allow_settling and self.base_speed>=.03:return True
        elif time.monotonic()-getattr(self,'odom_at',0.)<.5 and self.base_speed>=.03:
            return True
        return False

    def reset_and_exit(self):
        self.exiting=True;self.exit_requested=False;self.stop_requested=False
        try:
            if not self.stop('operator_exit_reset','EXIT_RESETTING'):
                raise RuntimeError('Base stop not acknowledged; cannot reset and exit')
            if self.explore or getattr(getattr(self,'approach',None),'backend',None)=='sport_velocity_goal':
                deadline=time.monotonic()+5.
                while True:
                    self.ros.spin_once(self.node,timeout_sec=.02)
                    if self.exit_reset_should_stop(allow_settling=True):
                        raise RuntimeError('Exit reset interrupted or base odom stale')
                    if self.base_speed<.03:break
                    if time.monotonic()>deadline:raise RuntimeError('Base did not stop before exit reset')
            self.wrist.reset_for_exit(self.cfg['reset'],self.exit_reset_should_stop)
            self.policy.state='EXIT_READY'
            self.publish('exit_reset_complete')
            self.release_requested=True
            print('机械臂已复位，正在结束相机和 SDK 通信并退出。',flush=True)
        except Exception as exc:
            self.stop_requested=False
            self.stop('Reset-and-exit failed: '+str(exc),'FAULT')
            print('退出复位未完成，程序继续保持机械臂通信；排除故障后可再按 Ctrl+C 重试。',flush=True)
        finally:self.exiting=False

    def try_approach(self,item,q):
        # One fresh depth frame is enough after the RGB vote. Missing depth is
        # a waiting condition, not an arm/robot fault or another voting stage.
        now=time.monotonic()
        if now-self.localization_started>self.cfg['approach'].get('depth_wait_timeout_sec',10.):
            self.stop('target_found_but_depth_unavailable','FOUND');return
        if (item['box'] is None or item['confidence']<self.cfg['confidence'] or
                item['camera_xyz_m'] is None or item['captured_at']<=self.policy.paused_at or
                self.base_speed>=.03):return
        self.approach.begin(item,q)
        path=self.output/'found.json'
        found=json.loads(path.read_text())
        if getattr(self.approach,'backend',None)=='sport_velocity_goal':
            found.update(map_target_published=False,target_base_xyz_m=self.approach.target_base.tolist(),
                         goal_base_xy_m=self.approach.goal_base.tolist(),backend=self.approach.backend)
        else:found.update(map_target_published=True,target_map_xyz_m=self.approach.target.tolist())
        found.update(camera_xyz_m=item['camera_xyz_m'],q_confirm_rad=q.tolist())
        path.write_text(json.dumps(found,indent=2)+'\n')

    def step(self):
        if getattr(self,'exit_requested',False):
            self.reset_and_exit();return
        now=time.monotonic();item=self.vision.read()
        if self.stop_requested:
            self.stop_requested=False;self.stop('operator_stop')
            print('动作已停止，当前继续保持机械臂通信，因此进程不会退出。\n'
                  '机械臂放稳/支撑后，另一个终端执行：\n'
                  "ros2 service call /wrist_search/release_control std_srvs/srv/Trigger '{}'",flush=True)
        if self.policy.state in ('SEARCHING','TARGET_LOCALIZING'):
            error=self.wrist.error or self.vision.error
            if item is None or now-item['captured_at']>self.cfg['frame_timeout_sec']:error=error or 'Camera heartbeat stale'
            if self.explore and (now-self.odom_at>.5 or now-self.scan_at>1.):error=error or 'Navigation sensors stale'
            if self.explore and self.policy.state=='SEARCHING' and now>self.gate_grace_until and (len(self.gates)<2 or not all(self.gates.values())):
                error=error or 'Motion gate closed externally'
            if error:self.stop(error,'FAULT');return
            if self.limit_event:self.stop(self.limit_event,'LIMIT');return
            q,_=self.wrist.snapshot()
            approach=getattr(self,'approach',None)
            if self.policy.state=='TARGET_LOCALIZING':
                self.try_approach(item,q)
            action=self.policy.observe(item['number'],item['captured_at'],now,item['box'],item['confidence'])
            if action=='FOUND':
                self.stop('sponge_confirmed','TARGET_LOCALIZING' if approach is not None else 'FOUND')
                found=dict(box=item['box'],confidence=item['confidence'],camera_xyz_m=item['camera_xyz_m'],
                           q_confirm_rad=q.tolist(),map_target_published=False,approach_enabled=approach is not None,grasp_enabled=getattr(self,'grasp',None) is not None)
                (self.output/'found.json').write_text(json.dumps(found,indent=2)+'\n')
                cv2.imwrite(str(self.output/'found.jpg'),item['annotated'])
                if approach is not None:
                    self.localization_started=time.monotonic()
                    if self.policy.state=='TARGET_LOCALIZING':self.try_approach(item,q)
            if self.policy.tick(now)=='STOP':self.stop('Search duration limit','LIMIT')
        if getattr(self,'approach',None) is not None:self.approach.tick()
        grasp=getattr(self,'grasp',None)
        if self.policy.state=='ARRIVED' and grasp is not None and not grasp.started:
            grasp.run()
        heartbeat_sec=1. if self.policy.state in ('SEARCHING','TARGET_LOCALIZING','PREPARING','APPROACH_PLANNING','APPROACHING','APPROACH_SETTLING','SPORT_POSITION_ACCEPTING','SPORT_POSITION_MOVING') else 5.
        if now-self.last_status>heartbeat_sec:
            self.publish('heartbeat',arm_error=self.wrist.error)
            self.last_status=now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/wrist_search.json')
    parser.add_argument('--execute',action='store_true',help='Go directly to observation pose and scan; Ctrl+C resets arm and exits')
    parser.add_argument('--explore',action='store_true',help='Also enable Frontier/base motion; requires --execute')
    parser.add_argument('--approach',action='store_true',help='Go to the side of the sponge using software Sport Move control (default); requires --execute')
    parser.add_argument('--grasp',action='store_true',help='After Sport arrival, side/vertical wrist search, grasp, then reset holding closed; requires --execute --approach')
    parser.add_argument('--approach-backend',choices=['sport','nav2'],help='Default: configured software Sport Move backend')
    parser.add_argument('--side',choices=['left','right'],help='Side of sponge in detection-time robot axes; default left')
    parser.add_argument('--view',action='store_true')
    parser.add_argument('--frames',type=int,default=0,help='Preview frame limit (0=continuous)')
    parser.add_argument('--output',type=Path,default=ROOT/'output/wrist_search')
    args=parser.parse_args()
    if args.explore and not args.execute:parser.error('--explore requires --execute')
    if args.approach and not args.execute:parser.error('--approach requires --execute')
    if args.grasp and not (args.execute and args.approach):parser.error('--grasp requires --execute --approach')
    if args.execute and args.frames:parser.error('--frames is for preview only; active arm control remains alive')
    cfg=json.loads(args.config.read_text())
    if cfg['approach'].get('backend')=='sport_move_to_pos':cfg['approach']['backend']='sport_velocity_goal'
    if args.approach_backend:cfg['approach']['backend']='sport_velocity_goal' if args.approach_backend=='sport' else 'nav2'
    if args.side:cfg['approach']['side']=args.side
    if args.grasp:
        if cfg['approach'].get('backend')!='sport_velocity_goal':parser.error('--grasp requires the Sport approach backend')
        cfg['approach']['side_offset_m']=cfg['post_grasp']['side_offset_m']
    if args.approach and cfg['approach'].get('backend','nav2')=='nav2' and not args.explore:
        parser.error('Nav2 approach requires --explore; Sport velocity-goal mode does not')
    if not .5<=cfg['confidence']<1 or not 1<=cfg['vote_required_frames']<=cfg['vote_window_frames']<=30 or cfg['max_search_sec']<=0:
        raise ValueError('Invalid search policy configuration')
    scan_offset(0.,math.radians(cfg['wrist']['amplitude_deg']),cfg['wrist']['speed_rad_s'])
    if not 0<cfg['max_capture_sec']<cfg['frame_timeout_sec']:
        raise ValueError('Invalid frame/verification timeouts')
    args.output.mkdir(parents=True,exist_ok=True)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model_path=(args.config.resolve().parent/cfg['model']).resolve()
    if not model_path.is_file():raise FileNotFoundError(f'Model not found: {model_path}')
    model=YOLO(str(model_path))
    if cfg['class_name'] not in model.names.values():
        raise ValueError(f"Model has no {cfg['class_name']!r} class: {model.names}")
    model.predict(np.zeros((480,640,3),np.uint8),imgsz=640,device='cpu',verbose=False)
    camera=vision=wrist=app=None;ros_started=False
    try:
        camera=Camera(cfg['camera'])
        vision=Vision(camera,model,cfg,args.output)
        if not vision.ready.wait(timeout=30) or vision.error:raise RuntimeError(vision.error or 'Camera startup timed out')
        if args.execute:
            grasp_cfg=json.loads((args.config.resolve().parent/cfg['grasp_config']).resolve().read_text())
            import rclpy
            from rclpy.signals import SignalHandlerOptions
            # Keep ROS/SDK alive while signals request a base stop and arm reset.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO);ros_started=True
            app=Coordinator(cfg,None,vision,args.explore,args.output)
            if args.approach:
                if cfg['approach'].get('backend','nav2')=='sport_velocity_goal':
                    from wrist_sport_approach import WristSportApproach
                    app.approach=WristSportApproach(app,cfg['approach'],grasp_cfg['calibration']['T_end_camera'])
                else:
                    from wrist_approach import WristApproach
                    app.approach=WristApproach(app,cfg['approach'],grasp_cfg['calibration']['T_end_camera'])
            if args.grasp:
                from wrist_grasp import WristGrasp
                app.grasp=WristGrasp(app,cfg['post_grasp'],grasp_cfg,args.view)
            signal.signal(signal.SIGINT,lambda *unused:app.request_exit())
            signal.signal(signal.SIGTERM,lambda *unused:app.request_exit())
            if args.explore:
                end=time.monotonic()+1.
                while time.monotonic()<end:rclpy.spin_once(app.node,timeout_sec=.05)
            app.prepare_observation()
            wrist=Wrist(cfg['wrist'],np.array(grasp_cfg['calibration']['T_end_camera']),
                        prepare=cfg.get('observation'),should_stop=app.preparation_should_stop,
                        pump=lambda:rclpy.spin_once(app.node,timeout_sec=0.),output=args.output,
                        stop_reason=app.preparation_stop_reason)
            app.wrist=wrist
            try:
                if not app.exit_requested:app.start()
            except Exception as exc:app.stop(str(exc),'FAULT')
            while rclpy.ok() and not app.release_requested:
                try:
                    rclpy.spin_once(app.node,timeout_sec=.02)
                    try:app.step()
                    except Exception as exc:app.stop(str(exc),'FAULT')
                    if args.view:
                        cv2.imshow('Z1 camera search',vision.read()['annotated'])
                        if cv2.waitKey(1)&0xff==ord('q'):app.stop_requested=True
                except KeyboardInterrupt:
                    app.request_exit()
        else:
            print('预览：不创建机械臂 SDK，不控制底盘。',flush=True)
            seen=-1;count=0
            while True:
                if vision.error:raise RuntimeError(vision.error)
                item=vision.read()
                if item and item['number']!=seen:
                    seen=item['number'];count+=1
                    print(json.dumps({k:item[k] for k in ('number','box','confidence','camera_xyz_m')},ensure_ascii=False),flush=True)
                    if args.view:cv2.imshow('Z1 camera search',item['annotated'])
                    if args.frames and count>=args.frames:break
                if args.view and cv2.waitKey(1)&0xff==ord('q'):break
                time.sleep(.02)
    except KeyboardInterrupt:pass
    finally:
        if app and app.wrist:
            app.wrist.pause();app.stop_base()
        if wrist:wrist.close()
        if vision:vision.close()
        elif camera:camera.close()
        if app:app.node.destroy_node()
        if ros_started:
            import rclpy
            if rclpy.ok():rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__=='__main__':main()
