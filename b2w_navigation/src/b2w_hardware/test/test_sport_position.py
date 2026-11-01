import json
import math
from types import SimpleNamespace as NS
import pytest
from b2w_hardware.sport_position import PositionSession, wrap


def state(x=0.,y=0.,yaw=0.,speed=0.,yaw_speed=0.,stamp=0):
    return NS(position=[x,y,.6],velocity=[speed,0.,0.],yaw_speed=yaw_speed,error_code=0,
              imu_state=NS(quaternion=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]),
              stamp=NS(sec=0,nanosec=stamp))


@pytest.fixture
def session():
    clock=NS(t=1.);sent=[];statuses=[]
    def send(api,param):sent.append((api,json.loads(param)));return len(sent)
    p=PositionSession(send,statuses.append,lambda:clock.t)
    p.receive(state())
    return p,clock,sent,statuses


@pytest.mark.parametrize('goal,anchor',[
    ((1.,.6,0.),(0.,0.,0.)),
    ((1.,-.6,0.),(10.,20.,math.pi/2)),
    ((1.,.6,-math.pi/2),(0.,0.,0.)),
    ((1.,-.4,math.pi/2),(10.,20.,math.pi/2)),
    ((-.5,0.,0.),(-3.,4.,-math.pi/2)),
    ((0.,0.,math.pi/2),(0.,0.,0.)),
])
def test_unicycle_reaches_local_goal_without_firmware_position_api(session,goal,anchor):
    p,c,s,status=session;x,y,yaw=anchor;p.receive(state(x,y,yaw));p.start(*goal,91)
    assert not s  # Service acceptance alone never sends a motion command.
    old_v=old_w=0.;dt=.05
    for _ in range(2400):
        c.t+=dt;p.heartbeat(91);p.tick()
        if not p.active:break
        api,cmd=s[-1];assert api==1008 and set(cmd)=={'x','y','z'} and cmd['y']==0.
        v,w=cmd['x'],cmd['z']
        assert 0<=v<=.30 and abs(w)<=.20
        assert abs(v-old_v)<=.15*dt+1e-8 and abs(w-old_w)<=.4*dt+1e-8
        x+=v*math.cos(yaw+w*dt/2)*dt;y+=v*math.sin(yaw+w*dt/2)*dt;yaw=wrap(yaw+w*dt)
        p.receive(state(x,y,yaw,v,w));old_v,old_w=v,w
    assert p.state=='ARRIVED',p.info
    ax,ay,ayaw=anchor
    lx=math.cos(ayaw)*(x-ax)+math.sin(ayaw)*(y-ay)
    ly=-math.sin(ayaw)*(x-ax)+math.cos(ayaw)*(y-ay)
    assert math.hypot(lx-goal[0],ly-goal[1])<=.12
    assert abs(wrap(yaw-ayaw-goal[2]))<=.15
    assert [api for api,_ in s[-3:]]==[1003]*3
    assert all(api in (1008,1003) for api,_ in s)


@pytest.mark.parametrize('fault',['heartbeat','pose','speed','other_controller','timeout','scheduler'])
def test_watchdogs_stop_before_emitting_more_velocity(session,fault):
    p,c,s,status=session;p.start(1,0,0,91)
    c.t+=.6 if fault in ('heartbeat','pose','scheduler') else .1
    if fault=='timeout':c.t+=121.
    if fault!='heartbeat':p.heartbeat(91)
    if fault!='pose':p.receive(state(speed=.6 if fault=='speed' else 0.))
    p.tick(other_motion_enabled=fault=='other_controller')
    assert p.state=='FAULT' and not p.active and [a for a,_ in s]==[1003]*3


def test_no_measured_progress_stops_even_with_fresh_feedback(session):
    p,c,s,status=session;p.start(1,0,0,91)
    for _ in range(110):
        c.t+=.1;p.heartbeat(91);p.receive(state());p.tick()
        if not p.active:break
    assert p.state=='FAULT' and 'No measured progress' in p.info


def test_move_error_stops_and_late_reply_cannot_restart(session):
    p,c,s,status=session;p.start(1,0,0,91);c.t+=.05;p.tick();req=len(s)
    p.response(req,1008,False,3203,'unsupported');p.response(req,1008,True,0,'late')
    assert p.state=='FAULT' and [a for a,_ in s]==[1008,1003,1003,1003]


def test_cancel_during_tracking_ignores_late_reply(session):
    p,c,s,status=session;p.start(1,0,0,91);c.t+=.05;p.tick();req=len(s);p.cancel('stop')
    p.response(req,1008,True,0,'late')
    assert p.state=='CANCELED' and not p.active


def test_old_heartbeat_and_repeated_state_cannot_keep_motion_alive(session):
    p,c,s,status=session;p.receive(state(stamp=1));p.start(1,0,0,91)
    c.t+=.6;p.heartbeat(90);p.receive(state(stamp=1));p.tick()
    assert not p.active


def test_success_reply_does_not_count_as_arrival_and_missing_reply_is_allowed(session):
    p,c,s,status=session;p.start(1,0,0,91)
    for _ in range(3):
        c.t+=.1;p.receive(state());p.heartbeat(91);p.tick()
    p.response(len(s),1008,True,0,'')
    assert p.state=='MOVING' and len(s)==3


def test_motion_telemetry_distinguishes_streaming_commands_from_measured_motion(session):
    p,c,s,status=session;p.start(1,0,0,91)
    for _ in range(20):
        c.t+=.1;p.heartbeat(91);p.receive(state());p.tick()
    assert status[-1]['move_command_count']==20
    assert status[-1]['last_move_age_sec']==0.
    assert status[-1]['commanded_vx_m_s']==pytest.approx(.3)
    assert status[-1]['measured_speed_m_s']==0.
    assert status[-1]['measured_local_xy_yaw']==[0.,0.,0.]
    p.cancel('operator stop');c.t+=1.;p.tick()
    assert status[-1]['move_command_count']==20 and len([a for a,_ in s if a==1008])==20


@pytest.mark.parametrize('value,expected',[(0.,0.),(.05,.25),(-.05,-.25),(.2,.3),(-.2,-.3)])
def test_signed_deadzone_compensation_preserves_stop_direction_and_cap(value,expected):
    from b2w_hardware.sport_position import compensate_deadzone
    assert compensate_deadzone(value,.2,.3)==pytest.approx(expected)


@pytest.mark.parametrize('side',[-1.,1.])
def test_goal_tracking_with_simulated_linear_and_angular_deadzone(side):
    clock=NS(t=1.);packets=[]
    def send(api,param):packets.append((api,json.loads(param)));return len(packets)
    p=PositionSession(send,lambda status:None,lambda:clock.t,angular_speed=.3)
    p.receive(state());p.start(1.,side,side*math.pi/2,91)
    x=y=yaw=0.;dt=.05;old_v=old_w=0.
    for _ in range(2400):
        clock.t+=dt;p.heartbeat(91);p.tick()
        if not p.active:break
        cmd=packets[-1][1];v,w=cmd['x'],cmd['z']
        assert 0<=v<=.3 and abs(w)<=.3 and cmd['y']==0.
        assert abs(v-old_v)<=p.linear_accel*dt+1e-8
        assert abs(w-old_w)<=p.angular_accel*dt+1e-8
        old_v,old_w=v,w
        # Test a plant whose output loses the first 0.2 on both commanded axes.
        actual_v=max(v-.2,0.)
        actual_w=math.copysign(max(abs(w)-.2,0.),w)
        x+=actual_v*math.cos(yaw+actual_w*dt/2)*dt
        y+=actual_v*math.sin(yaw+actual_w*dt/2)*dt;yaw=wrap(yaw+actual_w*dt)
        p.receive(state(x,y,yaw,actual_v,actual_w))
    assert p.state=='ARRIVED',p.info
    assert p.distance_error<=.12 and abs(p.yaw_error)<=.15
    assert [api for api,_ in packets[-3:]]==[1003]*3 and p.command==(0.,0.)


def test_aligned_tracking_does_not_bias_zero_turn_and_arrival_stays_stopped(session):
    p,c,s,status=session;p.start(1.,0.,0.,91)
    for _ in range(5):
        c.t+=.1;p.heartbeat(91);p.receive(state());p.tick()
        assert s[-1][1]['z']==0.
    for _ in range(6):
        c.t+=.1;p.heartbeat(91);p.receive(state(1.,0.,0.));p.tick()
    assert p.state=='ARRIVED' and p.command==(0.,0.)
    assert [api for api,_ in s[-3:]]==[1003]*3


def posture_session(session):
    p,c,s,status=session
    p.stand_down_on_arrival=True;p.position_tolerance=.25;p.yaw_tolerance=.35
    p.start(1.,0.,0.,91)
    c.t+=.1;p.receive(state(.78,0.,-.3));p.tick()
    assert p.phase=='STOPPING_AT_GOAL' and [api for api,_ in s]==[1003]*3
    return p,c,s,status


def test_loose_arrival_latches_no_more_corrections_and_standdown_is_once(session):
    p,c,s,status=posture_session(session)
    for _ in range(5):
        c.t+=.1;p.heartbeat(91);p.receive(state(.72,0.,-.4));p.tick()
    assert p.phase=='STAND_DOWN_WAIT' and s[-1][0]==1005
    p.response(p.posture_request_id,1005,True,0,'')
    for _ in range(45):
        c.t+=.1;p.heartbeat(91);p.receive(state(.72,0.,-.4));p.tick()
    assert p.state=='ARRIVED' and not p.active and p.command==(0.,0.)
    assert [api for api,_ in s]==[1003,1003,1003,1005]
    assert status[-1]['stand_down_acknowledged'] and not status[-1]['physical_posture_verified']


@pytest.mark.parametrize('failure',['reject','settling_timeout','cancel','still_moving'])
def test_posture_failure_or_cancel_cannot_resume_velocity(session,failure):
    p,c,s,status=posture_session(session)
    if failure=='cancel':p.cancel('operator')
    else:
        for _ in range(65 if failure=='still_moving' else 5):
            c.t+=.1;p.heartbeat(91)
            p.receive(state(.78,0.,-.3,speed=.06 if failure=='still_moving' else 0.));p.tick()
        if failure=='reject':p.response(p.posture_request_id,1005,False,3203,'unsupported')
        if failure=='settling_timeout':
            for _ in range(90):
                c.t+=.1;p.heartbeat(91);p.receive(state(.78,0.,-.3,speed=.04));p.tick()
    assert not p.active and p.state==('CANCELED' if failure=='cancel' else 'FAULT')
    assert 1008 not in [api for api,_ in s]
    assert [api for api,_ in s].count(1005)==(0 if failure in ('cancel','still_moving') else 1)
    assert [api for api,_ in s[-3:]]==[1003]*3


@pytest.mark.parametrize('acknowledged',[False,True])
def test_standdown_three_second_wait_starts_from_request_and_does_not_require_reply(session,acknowledged):
    p,c,s,status=posture_session(session)
    while p.phase!='STAND_DOWN_WAIT':
        c.t+=.1;p.heartbeat(91);p.receive(state(.78,0.,-.3));p.tick()
    if acknowledged:p.response(p.posture_request_id,1005,True,0,'')
    while c.t-p.posture_started<2.8:
        c.t+=.1;p.heartbeat(91);p.receive(state(.78,0.,-.3));p.tick()
        assert p.active and p.state=='MOVING' and not p.stand_down_wait_completed
    for _ in range(5):
        c.t+=.1;p.heartbeat(91);p.receive(state(.78,0.,-.3));p.tick()
    assert p.state=='ARRIVED' and p.stand_down_wait_completed
    assert p.stand_down_acknowledged is acknowledged
    assert not status[-1]['physical_posture_verified']
    assert [api for api,_ in s]==[1003,1003,1003,1005]

    p.response(p.posture_request_id,1005,True,0,'late reply')
    p.tick()
    assert [api for api,_ in s]==[1003,1003,1003,1005]


def test_loose_radius_cannot_stop_a_quarter_meter_behind_side_goal(session):
    p,c,s,status=session
    p.stand_down_on_arrival=True;p.position_tolerance=.25;p.longitudinal_tolerance=.06
    p.start(1.93549494,.24883305,0.,91)
    c.t+=.1;p.heartbeat(91);p.receive(state(1.69712478,.21833341,.1475));p.tick()
    assert not p.position_reached and p.phase not in ('STOPPING_AT_GOAL','STAND_DOWN_WAIT')
    assert 1003 not in [api for api,_ in s] and 1005 not in [api for api,_ in s]
    assert p.longitudinal_error==pytest.approx(.23837016)
    c.t+=.1;p.heartbeat(91);p.receive(state(1.89,.2488,0.));p.tick()
    assert p.phase=='STOPPING_AT_GOAL'


def test_position_is_rechecked_while_aligning_heading_before_final_stop(session):
    p,c,s,status=session
    p.stand_down_on_arrival=True;p.position_tolerance=.25;p.longitudinal_tolerance=.06
    p.start(1.,0.,0.,91)
    c.t+=.1;p.heartbeat(91);p.receive(state(.96,0.,.5));p.tick()
    assert p.position_reached
    c.t+=.1;p.heartbeat(91);p.receive(state(.90,0.,0.));p.tick()
    assert not p.position_reached and p.phase!='STOPPING_AT_GOAL'


@pytest.mark.parametrize('side',[-1.,1.])
def test_side_alignment_with_crouch_backshift_simulation(side):
    clock=NS(t=1.);packets=[]
    def send(api,param):packets.append((api,json.loads(param)));return len(packets)
    p=PositionSession(send,lambda status:None,lambda:clock.t,angular_speed=.3,
        linear_accel=.3,position_tolerance=.25,yaw_tolerance=.1,
        longitudinal_tolerance=.06,stand_down_on_arrival=True)
    p.receive(state());ideal_x=1.93549494;p.start(ideal_x+.07,side*.24883305,0.,91)
    x=y=yaw=0.;dt=.05;down_started=None;shift=0.;latched=None
    for _ in range(2400):
        clock.t+=dt;p.heartbeat(91);p.tick()
        if p.phase=='STOPPING_AT_GOAL' and latched is None:latched=p.local_pose.copy()
        if not p.active:break
        api,cmd=packets[-1]
        if api==1008:
            v=max(cmd['x']-.2,0.);w=math.copysign(max(abs(cmd['z'])-.2,0.),cmd['z'])
            x+=v*math.cos(yaw+w*dt/2)*dt;y+=v*math.sin(yaw+w*dt/2)*dt;yaw=wrap(yaw+w*dt)
        else:
            v=w=0.
            if api==1005:
                if down_started is None:down_started=clock.t
                new_shift=.07*min(1.,clock.t-down_started)
                x-=new_shift-shift;shift=new_shift
        p.receive(state(x,y,yaw,v,w))
    assert p.state=='ARRIVED',p.info
    assert abs(ideal_x+.07-latched[0])<=.06
    assert abs(latched[2])<=.1
    assert abs(ideal_x-x)<=.12
    first_down=next(i for i,(api,_) in enumerate(packets) if api==1005)
    assert all(api!=1008 for api,_ in packets[first_down:])
