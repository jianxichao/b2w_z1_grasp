#!/usr/bin/env python3
"""Passively copy existing local Z1 controller->SDK packets; never send packets.

Linux only, Python stdlib only. Live capture requires explicit --capture-existing
and CAP_NET_RAW/root. Does not import/instantiate the SDK or bind its UDP ports.
"""
import argparse
import json
import math
import socket
import struct
import time


def decode_feedback(payload):
    # Local SDK message/arm_common.h: pack(1), little endian x86_64/aarch64.
    # RecvState = head[2], enum(int32), JointState[7], Posture(double[6]).
    # JointState = float T,W,Acc,Pos + two 3-byte Motor_State records.
    if len(payload) != 208 or payload[:2] == b'\0\0':
        raise ValueError('Not a supported 208-byte RecvState')
    state = struct.unpack_from('<i', payload, 2)[0]
    if not 0 <= state <= 16:
        raise ValueError('Invalid FSM value')
    rows = []
    for i in range(6):
        offset = 6 + 22*i
        q = struct.unpack_from('<f', payload, offset+12)[0]
        if not math.isfinite(q):
            raise ValueError('Nonfinite position feedback')
        motors = []
        for j in range(2):
            temp, error, connected = struct.unpack_from('<bBB', payload, offset+16+3*j)
            motors.append(dict(temperature=temp, error=error,
                               connection_state=(connected >> 6) & 3))
        rows.append(dict(number=i+1, name=f'joint{i+1}', actual_rad=q,
                         actual_degree=math.degrees(q), motor_status=motors))
    return dict(source='existing controller RecvState.jointState[0:6].Pos',
                angle_convention='SDK feedback; URDF mapping not verified',
                fsm_value=state, joints=rows,
                target_angles=None, home_configuration=None,
                limitation='Packet receipt does not prove fresh lower-level motor feedback')


def feedback_payload(frame):
    # Linux loopback AF_PACKET has a 14-byte synthetic Ethernet header.
    if len(frame) < 42 or frame[12:14] != b'\x08\x00':
        return None
    ip = frame[14:]
    ihl = (ip[0] & 15)*4
    if ip[0] >> 4 != 4 or ihl < 20 or len(ip) < ihl+8 or ip[9] != 17:
        return None
    if ip[12:20] != socket.inet_aton('127.0.0.1')*2:
        return None
    if struct.unpack_from('!H', ip, 6)[0] & 0x3fff:
        return None  # No fragmented datagrams.
    src, dst, length = struct.unpack_from('!HHH', ip, ihl)
    if (src, dst, length) != (8071, 8072, 216) or len(ip) < ihl+length:
        return None
    return ip[ihl+8:ihl+length]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-existing', action='store_true',
                        help='Explicitly allow passive capture on lo (no transmission)')
    parser.add_argument('--timeout', type=float, default=3)
    args = parser.parse_args()
    if not args.capture_existing:
        parser.error('No live action by default; review README before --capture-existing')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
        parser.error('--timeout must be in (0,30] seconds')
    deadline = time.monotonic()+args.timeout
    with socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0800)) as capture:
        capture.bind(('lo', 0))
        while time.monotonic() < deadline:
            capture.settimeout(max(.001, deadline-time.monotonic()))
            try:
                frame = capture.recv(65535)
            except socket.timeout:
                break
            payload = feedback_payload(frame)
            if payload is None:
                continue
            try:
                result = decode_feedback(payload)
            except ValueError:
                continue
            result['received_at_unix_s'] = time.time()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
    raise SystemExit('No supported feedback observed; no controller/client was started.')


if __name__ == '__main__':
    main()
