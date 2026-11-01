#!/usr/bin/env python3
"""Control/probe the storage-box servo board over a local Linux USB serial port."""

import argparse
import fcntl
import os
import select
import sys
import termios
import time

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0"
QUERY_ADDRESS = bytes.fromhex("68 07 00 FF 06 05 16")
ACTIONS = {
    "0": ("closed", bytes.fromhex("68 08 00 FF 14 00 13 16")),
    "close": ("closed", bytes.fromhex("68 08 00 FF 14 00 13 16")),
    "90": ("open", bytes.fromhex("68 08 00 FF 14 02 15 16")),
    "open": ("open", bytes.fromhex("68 08 00 FF 14 02 15 16")),
}


class ServoBoxError(RuntimeError):
    pass


def resolve_port(path):
    if os.path.exists(path):
        return path
    if path == DEFAULT_PORT and os.path.exists("/dev/ttyUSB0"):
        return "/dev/ttyUSB0"
    return path


def open_serial(path):
    """Open the controller as 115200 8N1 and take a cooperative exclusive lock."""
    resolved = resolve_port(path)
    fd = os.open(resolved, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = termios.B115200
        attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 5
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)
    except Exception:
        os.close(fd)
        raise
    return fd, resolved


def reply_is_framed(reply):
    """Perform the validation possible without an official response schema."""
    return len(reply) >= 4 and reply[0] == 0x68 and reply[-1] == 0x16


def transact(fd, frame, retries=2, response_timeout=0.8):
    """Send one frame and return the first framed controller response."""
    last_reply = b""
    for attempt in range(retries + 1):
        termios.tcflush(fd, termios.TCIFLUSH)
        os.write(fd, frame)
        termios.tcdrain(fd)
        deadline = time.monotonic() + response_timeout
        reply = bytearray()
        last_rx = None
        while time.monotonic() < deadline:
            wait = min(0.10, max(0.0, deadline - time.monotonic()))
            ready, _, _ = select.select([fd], [], [], wait)
            if ready:
                chunk = os.read(fd, 1024)
                if chunk:
                    reply.extend(chunk)
                    last_rx = time.monotonic()
            elif reply and last_rx is not None and time.monotonic() - last_rx >= 0.10:
                break
        last_reply = bytes(reply)
        if reply_is_framed(last_reply):
            return last_reply
        if attempt < retries:
            time.sleep(0.15)
    if last_reply:
        raise ServoBoxError("malformed controller reply: %s" % last_reply.hex(" "))
    raise ServoBoxError("controller did not reply")


def probe(port=DEFAULT_PORT, retries=2):
    """Query the controller address. This frame does not command servo motion."""
    fd, resolved = open_serial(port)
    try:
        reply = transact(fd, QUERY_ADDRESS, retries=retries)
        return resolved, reply
    finally:
        os.close(fd)


def command(position, port=DEFAULT_PORT, retries=2):
    """Probe, then command one of the two verified positions (0 or 90 degrees)."""
    if position not in ACTIONS:
        raise ServoBoxError("unsupported position: %s" % position)
    state, action_frame = ACTIONS[position]
    fd, resolved = open_serial(port)
    try:
        query_reply = transact(fd, QUERY_ADDRESS, retries=retries)
        action_reply = transact(fd, action_frame, retries=retries)
        return state, resolved, query_reply, action_reply
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(
        description="Storage box: probe=no motion, 0/close=close, 90/open=open"
    )
    parser.add_argument("position", choices=("probe",) + tuple(ACTIONS))
    parser.add_argument("--port", default=DEFAULT_PORT, help="serial device")
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.position == "probe":
            resolved, reply = probe(args.port, max(0, args.retries))
            print("PORT:", resolved)
            print("QUERY TX:", QUERY_ADDRESS.hex(" "))
            print("QUERY RX:", reply.hex(" "))
            print("PASS: controller online; no servo motion was requested")
            return 0
        state, resolved, query_reply, action_reply = command(
            args.position, args.port, max(0, args.retries)
        )
        print("PORT:", resolved)
        print("QUERY RX:", query_reply.hex(" "))
        print("ACTION TX:", ACTIONS[args.position][1].hex(" "))
        print("ACTION RX:", action_reply.hex(" "))
        print("COMMAND ACCEPTED: %s; physical position is not sensed" % state)
        return 0
    except FileNotFoundError as exc:
        print("serial device not found: %s" % exc, file=sys.stderr)
        return 2
    except PermissionError as exc:
        print("serial permission denied: %s" % exc, file=sys.stderr)
        return 3
    except BlockingIOError:
        print("serial device is already locked by another controller", file=sys.stderr)
        return 4
    except (OSError, ServoBoxError) as exc:
        print("servo controller error: %s" % exc, file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
