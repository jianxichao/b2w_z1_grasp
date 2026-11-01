"""Camera transport and continuous wrist control for stage-one search."""
import getpass,json,math,os,shlex,socket,sys,threading,time
from pathlib import Path
import cv2,numpy as np,paramiko
from b2w_perception.wrist_search_core import scan_offset,validate_sweep,plan_observation,SettleTracker,plan_startup_reset
ROOT=Path(__file__).resolve().parents[1]


def send_gripper_command(arm,target,velocity=0.,torque=0.):
    # Keep SDK public command fields and lowcmd buffer consistent.
    if hasattr(arm,'gripperQ'):
        arm.gripperQ=float(target)
        arm.gripperQd=float(velocity)
        arm.gripperTau=float(torque)
    arm.setGripperCmd(float(target),float(velocity),float(torque))


def arm_feedback_snapshot(arm):
    detail={}
    if hasattr(arm,'feedbackStatus'):detail.update(dict(arm.feedbackStatus()))
    detail['healthy']=bool(arm.feedbackHealthy())
    if 'controller_state' not in detail:detail['controller_state']=int(arm.getCurrentState())
    if hasattr(arm,'commandSnapshot'):detail['command']=dict(arm.commandSnapshot())
    return detail


class Camera:
    def __init__(self,cfg):
        self.ssh=paramiko.SSHClient();self.ssh.load_system_host_keys()
        self.channel=None
        try:
            password=os.environ.get('B2W_CAMERA_PASSWORD') or cfg.get('password')
            if not password:
                password=getpass.getpass('机载相机 SSH 密码（不保存）: ')
            self.ssh.connect(cfg['host'],username=cfg['user'],password=password,
                             timeout=5,auth_timeout=10,look_for_keys=False,allow_agent=False)
            code=Path(__file__).with_name('wrist_camera_agent.py').read_text()
            self.input,self.output,self.errors=self.ssh.exec_command(
                'python3 -u -c '+shlex.quote(code)+' '+shlex.quote(cfg['serial']),timeout=15)
            self.channel=self.output.channel
            self.serial=cfg['serial']
        except BaseException:
            self.close();raise

    def exact(self,count):
        result=bytearray()
        while len(result)<count:
            data=self.output.read(count-len(result))
            if not data:raise EOFError('Remote camera disconnected')
            result.extend(data)
        return bytes(result)

    def read(self):
        self.input.write('READ\n');self.input.flush()
        header=self.output.readline(65536)
        if not header:raise EOFError('Remote camera stopped; verify the selected serial and librealsense bindings')
        meta=json.loads(header)
        if meta['serial']!=self.serial:raise ValueError('Camera identity changed')
        sizes=[int(meta[k]) for k in ('color_bytes','depth_bytes')]
        if any(x<=0 or x>8_000_000 for x in sizes):raise ValueError('Invalid image payload size')
        color,depth=[cv2.imdecode(np.frombuffer(self.exact(x),np.uint8),cv2.IMREAD_UNCHANGED) for x in sizes]
        if color is None or depth is None or depth.dtype!=np.uint16 or color.shape[:2]!=depth.shape:
            raise ValueError('Invalid aligned RGB-D pair')
        if (meta['timestamp_domain']!=meta['color_timestamp_domain'] or
                abs(meta['depth_timestamp_ms']-meta['color_timestamp_ms'])>80):
            raise ValueError('RGB/depth capture timestamps disagree')
        return color,depth,meta

    def close(self):
        if self.channel is not None:
            try:self.input.write('QUIT\n');self.input.flush()
            except Exception:pass
            self.channel.close()
        self.ssh.close()


class Wrist:
    def __init__(self,cfg,camera_transform,prepare=None,should_stop=None,pump=None,output=None,stop_reason=None):
        # Refuse a second SDK owner; never kill an existing holding process.
        probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:probe.bind(('127.0.0.1',8072))
        except OSError as exc:raise RuntimeError('UDP 8072 is occupied; another arm SDK client owns control') from exc
        finally:probe.close()
        sys.path.insert(0,str(ROOT.parent/'unitree_z1/z1_sdk/lib'))
        import unitree_arm_interface as sdk
        if not hasattr(sdk.ArmInterface,'suspendTracking'):
            raise RuntimeError('Z1 Python SDK is old: load the rebuilt feedbackStatus/suspendTracking module')
        self.sdk,self.cfg=sdk,cfg
        self.camera_transform=np.asarray(camera_transform,float)
        self.should_stop=should_stop or (lambda:False)
        self.stop_reason=stop_reason
        self.pump=pump or (lambda:None)
        self.output=output
        self.preparation=None
        self.arm=sdk.ArmInterface(hasGripper=True)
        self.scanning=threading.Event();self.done=threading.Event();self.lock=threading.Lock()
        self.error=None;self.running=False;self.worker=None
        self.arm.setReceiveTimeoutMs(20);self.arm.setWait(False)
        self.arm.loopOn();self.running=True
        try:
            end=time.monotonic()+2
            while not self.arm.feedbackHealthy() and time.monotonic()<end:time.sleep(.02)
            if not self.arm.feedbackHealthy():raise RuntimeError('No healthy Z1 feedback; start z1_ctrl in SDK mode')
            initial_state=self.arm.getCurrentState()
            if initial_state not in (sdk.ArmFSMState.PASSIVE,sdk.ArmFSMState.JOINTCTRL):
                raise RuntimeError('Prepare a stationary raised observation pose before starting')
            q=self.wait_startup_still(initial_state)
            self.center=q.copy();self.target=q.copy()
            self.gripper=float(self.arm.lowstate.getGripperQ())
            self.model=self.arm._ctrlComp.armModel
            try:
                self.plan_current_observation(camera_transform,prepare)
            except ValueError as exc:
                if initial_state==sdk.ArmFSMState.PASSIVE:
                    raise ValueError('Observation planning rejected; no hold/motion activated: '+str(exc))
                self.error='Observation planning rejected: '+str(exc)
                self.report={'rejected':self.error}
            self.arm.requestJointHold()
            deadline=time.monotonic()+3.
            while self.arm.getCurrentState()!=sdk.ArmFSMState.JOINTCTRL:
                if not self.arm.feedbackHealthy() or time.monotonic()>deadline:
                    raise RuntimeError('Could not acknowledge measured joint hold')
                time.sleep(.02)
            self.arm.enableJointTracking()
            if not self.error:
                self.reset_startup_gripper()
            if not self.error:
                try:
                    if self.preparation:
                        self.move_to_observation(self.preparation,prepare)
                    self.center=self.snapshot()[0]
                    self.target=self.center.copy()
                    self.report=validate_sweep(self.model,self.center,camera_transform,
                        cfg['joint']-1,math.radians(cfg['amplitude_deg']))
                    self.report['prepared_end_xyz_m']=np.asarray(self.model.forwardKinematics(self.center,6))[:3,3].tolist()
                except Exception as exc:
                    self.latch_fault('Observation preparation stopped: '+str(exc))
            self.worker=threading.Thread(target=self.control,daemon=True);self.worker.start()
        except BaseException:
            self.close();raise

    def reset_startup_gripper(self):
        """Close to command zero before observation/search; no angle arrival test."""
        if not self.cfg.get('startup_gripper_reset',True):return
        q=self.snapshot()[0].copy();start=float(self.gripper)
        duration=max(.1,abs(start)/.5);begun=time.monotonic()
        print('启动夹爪归0，然后开始机械臂观察与搜索。',flush=True)
        while True:
            self.pump()
            if self.should_stop():raise RuntimeError('Operator/sensor stop during startup gripper reset')
            if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                raise RuntimeError('Arm feedback/state fault during startup gripper reset')
            elapsed=time.monotonic()-begun
            self.gripper=start*(1.-min(1.,elapsed/duration))
            self.arm.setArmCmd(q,np.zeros(6),np.zeros(6))
            velocity=-start/duration if elapsed<duration else 0.
            send_gripper_command(self.arm,self.gripper,velocity)
            if elapsed>=duration+.5:
                self.gripper=0.;return
            time.sleep(.02)

    def wait_startup_still(self,initial_state):
        """Read-only position window before requesting hold or enabling joint tracking."""
        window=float(self.cfg.get('startup_position_window_sec',.6))
        tolerance=float(self.cfg.get('startup_position_range_rad',.003))
        timeout=float(self.cfg.get('startup_still_timeout_sec',3.))
        if not (.2<=window<=2. and 0<tolerance<=.01 and window<timeout<=10.):
            raise ValueError('Invalid startup position observation settings')
        tracker=SettleTracker(position_window_sec=window,position_range_tolerance=tolerance)
        self.settle_log_prefix='startup'
        begun=time.monotonic();last_print=-1.;last_saved=-1.;metrics=None
        while time.monotonic()-begun<timeout:
            self.pump()
            reason_callback=getattr(self,'stop_reason',None)
            reason=reason_callback() if reason_callback is not None else (
                'Operator or sensor stop requested' if self.should_stop() else None)
            if reason is not None:
                if self.output:
                    self.save_settle_status('startup_canceled',dict(metrics or {},
                        reason=str(reason),elapsed_sec=time.monotonic()-begun,hold_motion_enabled=False))
                raise RuntimeError('Startup canceled before arm hold/motion enable: '+str(reason))
            if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=initial_state:
                raise RuntimeError('Arm feedback/state changed during startup observation')
            now=time.monotonic();q,dq=self.snapshot()
            # There is no target pose yet. Only the measured position range matters.
            sample=tracker.update(now,q,dq,q)
            stable=sample['position_window_stable']
            metrics=dict(q_measured_rad=q.tolist(),dq_measured_rad_s=dq.tolist(),
                max_joint_speed_rad_s=sample['max_joint_speed_rad_s'],
                position_window_span_sec=sample['position_window_span_sec'],
                position_window_range_rad=sample['position_window_range_rad'],
                position_window_required_sec=window,position_range_tolerance_rad=tolerance,
                startup_still=stable,elapsed_sec=now-begun,timeout_sec=timeout)
            if self.output and (now-begun-last_saved>=.1 or stable):
                self.save_settle_status('startup_still' if stable else 'startup_waiting',metrics)
                last_saved=now-begun
            if now-begun-last_print>=1. or stable:
                print(f"启动静止检查：位置窗口 {sample['position_window_span_sec']:.2f}/{window:.2f}s，"
                      f"关节角波动 {sample['position_window_range_rad'] or 0.:.5f}/{tolerance:.5f}rad，"
                      f"速度读数 {sample['max_joint_speed_rad_s']:.4f}rad/s（仅记录），"
                      f"{'通过' if stable else '观察中'}",flush=True)
                last_print=now-begun
            if stable:return q.copy()
            time.sleep(.02)
        if self.output and metrics:self.save_settle_status('startup_timeout',metrics)
        raise RuntimeError(f'Arm startup position did not become stable in {timeout:.1f}s; '
                           'no hold/motion enabled; see output/wrist_search/startup_status.json')

    def plan_current_observation(self,camera_transform,prepare):
        q=self.snapshot()[0]
        if prepare and prepare.get('enabled',False):
            def poll():
                self.pump()
                if self.should_stop():raise RuntimeError('Startup observation planning canceled')
            self.preparation=plan_observation(self.model,q,camera_transform,dict(prepare,planning_poll=poll),
                self.cfg['joint']-1,math.radians(self.cfg['amplitude_deg']))
            self.report=self.preparation['sweep']
            if self.output:
                (self.output/'observation_plan.json').write_text(json.dumps(self.preparation,indent=2)+'\n')
            print('观察点 IK 已验证：SDK末端 xyz(m)=',self.preparation['target_end_xyz_m'],
                  'pitch(deg)=',self.preparation['pitch_deg'],flush=True)
        else:
            self.report=validate_sweep(self.model,q,camera_transform,
                self.cfg['joint']-1,math.radians(self.cfg['amplitude_deg']))

    def latch_fault(self,message):
        self.error=str(message);self.scanning.clear()
        detail=arm_feedback_snapshot(self.arm)
        if getattr(self,'output',None):
            (self.output/'arm_fault_status.json').write_text(json.dumps(dict(
                reason=self.error,monotonic_s=time.monotonic(),**detail),indent=2)+'\n')
        hard_motor_fault=any(int(f.get('error',0)) & 0x27 for f in detail.get('motor_faults',[]))
        if hard_motor_fault:
            self.arm.requestPassive()
        elif detail['healthy'] and self.arm.getCurrentState()==self.sdk.ArmFSMState.JOINTCTRL:
            self.arm.requestJointHold()
            self.target=self.snapshot()[0]
        elif hasattr(self.arm,'suspendTracking'):
            # Communication loss is not an instruction to unload the arm.
            # Keep the SDK alive with no state transition or old trajectory replay.
            self.arm.suspendTracking()
        else:
            self.arm.requestPassive()

    def move_to_observation(self,plan,settings):
        start=np.asarray(plan['start_q_rad']);goal=np.asarray(plan['goal_q_rad'])
        speed=float(settings.get('joint_speed_rad_s',.10))
        speed_limit=float(settings.get('joint_speed_limit_rad_s',.15))
        if not 0<speed_limit<=.25 or not 0<speed<=speed_limit:
            raise ValueError('Joint speed exceeds the configured limit (at most 0.25rad/s)')
        duration=max(2.,1.875*float(np.max(np.abs(goal-start)))/speed)
        begun=time.monotonic();last_print=0.
        label=settings.get('label','抬臂到观察点')
        hint=('再次 Ctrl+C 可中断复位并保持。' if settings.get('log_prefix')=='exit_reset'
              else 'Ctrl+C 将取消当前动作并转入退出复位。')
        print(f'底盘保持停止，{label}，预计 {duration:.1f} 秒；{hint}',flush=True)
        while True:
            self.pump()
            if self.should_stop():
                reason=getattr(self,'stop_reason',None)
                raise RuntimeError((reason() if callable(reason) else None) or 'Operator/sensor stop during preparation')
            if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                raise RuntimeError('Arm feedback/state fault during preparation')
            elapsed=time.monotonic()-begun;u=min(1.,elapsed/duration)
            blend=u*u*u*(10+u*(-15+6*u))
            self.target=start+(goal-start)*blend
            q,dq=self.snapshot()
            if not np.isfinite(q).all() or np.max(np.abs(q-self.target))>.12:
                raise RuntimeError('Observation path tracking error exceeds 0.12rad')
            self.arm.setArmCmd(self.target,np.zeros(6),np.zeros(6))
            send_gripper_command(self.arm,self.gripper)
            if elapsed-last_print>=1.:
                print(f'{label}进度 {100*u:.0f}%',flush=True);last_print=elapsed
            if u>=1.:break
            time.sleep(.02)
        self.wait_observation_settle(goal,settings)

    def wait_observation_settle(self,goal,settings):
        self.settle_log_prefix=settings.get('log_prefix','observation')
        label=settings.get('label','观察点')
        tracker=SettleTracker(float(settings.get('joint_position_tolerance_rad',.025)),
                              float(settings.get('joint_speed_tolerance_rad_s',.025)),
                              float(settings.get('settle_stable_sec',.4)),
                              position_window_sec=float(settings.get('settle_position_window_sec',0.)),
                              position_range_tolerance=float(settings.get('settle_position_range_rad',.003)))
        timeout=float(settings.get('settle_timeout_sec',10.))
        if not math.isfinite(timeout) or not .5<=timeout<=30.:
            raise ValueError('Settle timeout must be 0.5..30 seconds')
        begun=time.monotonic();last_print=-1.;last_saved=-1.;metrics=None
        goal_xyz=np.asarray(self.model.forwardKinematics(goal,6))[:3,3]
        while time.monotonic()-begun<timeout:
            self.pump()
            if self.should_stop():
                reason=getattr(self,'stop_reason',None)
                raise RuntimeError((reason() if callable(reason) else None) or 'Preparation interrupted while settling')
            if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                raise RuntimeError('Arm feedback/state fault while settling')
            now=time.monotonic();elapsed=now-begun
            q,dq=self.snapshot();metrics=tracker.update(now,q,dq,goal)
            actual_xyz=np.asarray(self.model.forwardKinematics(q,6))[:3,3]
            metrics.update(elapsed_sec=elapsed,timeout_sec=timeout,
                           actual_end_xyz_m=actual_xyz.tolist(),goal_end_xyz_m=goal_xyz.tolist(),
                           end_position_error_m=float(np.linalg.norm(actual_xyz-goal_xyz)))
            phase='settled' if metrics['settled'] else 'settling'
            if getattr(self,'output',None) and (elapsed-last_saved>=.1 or metrics['settled']):
                self.save_settle_status(phase,metrics);last_saved=elapsed
            if elapsed-last_print>=1. or metrics['settled']:
                print(f"{label}到位检查：角度误差 {metrics['max_joint_error_deg']:.2f}° "
                      f"(J{metrics['worst_position_joint']})，速度 {metrics['max_joint_speed_rad_s']:.4f}rad/s "
                      f"(J{metrics['worst_speed_joint']})，连续稳定 {metrics['stable_for_sec']:.2f}/"
                      f"{tracker.stable_sec:.2f}s，末端位置误差 {metrics['end_position_error_m']*1000:.1f}mm，"
                      f"依据 {metrics['settle_basis']}，位置窗口波动 "
                      f"{math.degrees(metrics['position_window_range_rad'] or 0.):.3f}°",flush=True)
                last_print=elapsed
            if metrics['settled']:return
            self.arm.setArmCmd(goal,np.zeros(6),np.zeros(6))
            send_gripper_command(self.arm,self.gripper)
            time.sleep(.02)
        if metrics is None:raise RuntimeError('Observation settling produced no joint samples')
        if getattr(self,'output',None):self.save_settle_status('settle_timeout',metrics)
        raise RuntimeError(
            f"{label} did not settle in {timeout:.1f}s: "
            f"J{metrics['worst_position_joint']} error={metrics['max_joint_error_rad']:.4f}rad "
            f"(limit {tracker.position_tolerance:.4f}), "
            f"J{metrics['worst_speed_joint']} speed={metrics['max_joint_speed_rad_s']:.4f}rad/s "
            f"(limit {tracker.speed_tolerance:.4f}), "
            f"stable={metrics['stable_for_sec']:.2f}/{tracker.stable_sec:.2f}s; "
            f"see output/wrist_search/{self.settle_log_prefix}_status.json")

    def save_settle_status(self,phase,metrics):
        item=dict(phase=phase,monotonic_s=time.monotonic(),**metrics)
        prefix=getattr(self,'settle_log_prefix','observation')
        (self.output/(prefix+'_status.json')).write_text(json.dumps(item,indent=2)+'\n')
        with (self.output/(prefix+'_settle.jsonl')).open('a') as stream:
            stream.write(json.dumps(item)+'\n')

    def pause(self):self.scanning.clear()

    def exclusive_commands(self):
        """Keep the SDK alive while temporarily transferring the sole command writer."""
        from contextlib import contextmanager

        @contextmanager
        def transfer():
            self.pause();self.done.set()
            if self.worker:
                self.worker.join(timeout=2.)
                if self.worker.is_alive():raise RuntimeError('Arm worker did not relinquish command ownership')
            try:
                if self.error:raise RuntimeError(self.error)
                if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                    raise RuntimeError('Exclusive arm control requires healthy JOINTCTRL')
                self.target=self.snapshot()[0].copy()
                yield
            finally:
                # Resume measured holding through the same SDK instance, never a new client.
                self.scanning.clear()
                if self.arm.feedbackHealthy():self.center=self.target=self.snapshot()[0].copy()
                self.done.clear()
                self.worker=threading.Thread(target=self.control,daemon=True);self.worker.start()
        return transfer()

    def reset_for_exit(self,settings,should_stop):
        """Transfer command ownership from scan worker to a reset in the same SDK session."""
        self.pause();self.done.set()
        if self.worker:
            self.worker.join(timeout=2.)
            if self.worker.is_alive():raise RuntimeError('Arm worker did not stop; exit reset canceled')
        previous_should_stop=self.should_stop
        try:
            self.should_stop=should_stop
            if should_stop():raise RuntimeError('Exit reset canceled or base not stationary')
            if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                raise RuntimeError('Exit reset requires healthy JOINTCTRL feedback')
            q,_=self.snapshot()
            plan=plan_startup_reset(self.model,q,self.camera_transform,
                                   dict(settings,floor_plane=getattr(self,'path_floor_plane',None)))
            if self.output:
                (self.output/'exit_reset_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
            self.target=q.copy()
            self.arm.requestJointHold();self.arm.enableJointTracking()
            self.move_to_observation(plan,dict(settings,label='退出复位',log_prefix='exit_reset'))
            self.center=self.target=np.asarray(plan['goal_q_rad']).copy()
        except Exception as exc:
            self.latch_fault('Exit reset stopped: '+str(exc))
            # A failed reset must retain the SDK and resume the hold watchdog.
            self.done.clear()
            self.worker=threading.Thread(target=self.control,daemon=True);self.worker.start()
            raise
        finally:self.should_stop=previous_should_stop

    def resume(self):
        if self.error:raise RuntimeError(self.error)
        self.scanning.set()

    def snapshot(self):
        return np.array(self.arm.lowstate.getQ()),np.array(self.arm.lowstate.getQd())

    def control(self):
        elapsed=0.;last=time.monotonic();was_scanning=False
        try:
            while not self.done.is_set():
                now=time.monotonic();dt=now-last;last=now
                if dt>.25:raise RuntimeError('Arm control scheduler stalled')
                if not self.arm.feedbackHealthy() or self.arm.getCurrentState()!=self.sdk.ArmFSMState.JOINTCTRL:
                    raise RuntimeError('Arm feedback/state fault; manual recovery required')
                q,dq=self.snapshot()
                scan=self.scanning.is_set()
                if scan:
                    elapsed+=min(dt,.04)
                    self.target=self.center.copy()
                    self.target[self.cfg['joint']-1]+=scan_offset(elapsed,math.radians(self.cfg['amplitude_deg']),self.cfg['speed_rad_s'])
                elif was_scanning:self.target=q.copy()
                was_scanning=scan
                if not np.isfinite(q).all() or np.max(np.abs(q-self.target))>.12:
                    raise RuntimeError('Arm joint tracking error exceeded 0.12rad')
                self.arm.setArmCmd(self.target,np.zeros(6),np.zeros(6))
                send_gripper_command(self.arm,self.gripper)
                time.sleep(.02)
        except Exception as exc:
            self.latch_fault(str(exc))
            # Keep SDK communication alive until explicit release_control.

    def close(self):
        self.done.set()
        if self.worker:self.worker.join(timeout=2)
        if self.running:self.arm.loopOff();self.running=False
