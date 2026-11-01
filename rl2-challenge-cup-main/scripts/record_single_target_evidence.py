#!/usr/bin/env python3
"""Record one autonomous target run as machine-readable evidence.

This node is observation-only: it does not call services, actions, or publish
motion/actuator commands.  Start it before the mission and stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


REQUIRED_STAGES = (
    "noncontact_start",
    "leave_base",
    "visual_confirmation",
    "navigate_dock",
    "autonomous_grasp",
    "independent_bin_confirmation",
    "carry_return",
    "designated_unload",
    "stable_stop",
)


class EvidenceRecorder(Node):
    def __init__(self, output_dir: Path, run_id: str) -> None:
        super().__init__("single_target_evidence_recorder")
        self.output_dir = output_dir
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.last_pose: Optional[Dict[str, Any]] = None
        self.last_pose_write = 0.0
        self.mission_state = "UNKNOWN"
        self.target_id = ""
        self.returned_base = False
        self.errors = []
        self.stages: Dict[str, Dict[str, Any]] = {
            name: {"status": "NOT_OBSERVED"} for name in REQUIRED_STAGES
        }

        self.events_file = (self.output_dir / "events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.pose_file = (self.output_dir / "poses.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self.pose_writer = csv.DictWriter(
            self.pose_file,
            fieldnames=(
                "elapsed_s", "wall_time_utc", "x", "y", "z", "qx", "qy",
                "qz", "qw", "frame_id", "mission_state", "target_id",
            ),
        )
        self.pose_writer.writeheader()

        latched = QoSProfile(depth=10)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        regular = QoSProfile(depth=50)
        regular.reliability = ReliabilityPolicy.RELIABLE

        for topic in (
            "/mission_supervisor_node/status",
            "/mission_supervisor_node/target_report",
            "/arm/status",
            "/storage_box/status",
            "/cmd_vel_safety_node/status",
            "/go2w_sport_cmd_bridge/status",
            "/rl2_nav/slam_pose_jump_guard/status",
        ):
            self.create_subscription(
                String, topic,
                lambda msg, t=topic: self._on_string(t, msg), latched,
            )
        for topic in (
            "/task_event",
            "/target_detection_status",
            "/robot_arm/target_status",
        ):
            self.create_subscription(
                String, topic,
                lambda msg, t=topic: self._on_string(t, msg), regular,
            )
        self.create_subscription(
            PoseStamped, "/slam/current_pose", self._on_pose, 20
        )
        # Future independent sensor contract.  Until a real sensor publishes
        # this topic, the report deliberately remains NOT_OBSERVED.
        self.create_subscription(
            Bool, "/storage_box/sample_present", self._on_sample_present, 10
        )
        self.create_timer(1.0, self._write_summary)
        self._record("recorder", "start", {
            "run_id": run_id,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "independent_bin_sensor_topic": "/storage_box/sample_present",
        })
        self.get_logger().info("Evidence output: %s" % self.output_dir)

    def _now(self) -> Dict[str, Any]:
        return {
            "elapsed_s": round(time.monotonic() - self.started_mono, 6),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _record(self, topic: str, kind: str, payload: Any) -> None:
        row = self._now()
        row.update({
            "topic": topic,
            "kind": kind,
            "payload": payload,
            "pose": self.last_pose,
            "mission_state": self.mission_state,
            "target_id": self.target_id,
        })
        self.events_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _decode(data: str) -> Any:
        try:
            return json.loads(data)
        except (TypeError, ValueError):
            return data

    def _stage(self, name: str, transition: str, evidence: str,
               result: str = "") -> None:
        stage = self.stages[name]
        now = self._now()
        key = "start" if transition == "start" else "end"
        if key in stage:
            return
        stage[key] = now
        stage[key + "_pose"] = self.last_pose
        stage[key + "_target_id"] = self.target_id
        stage.setdefault("evidence", []).append(evidence)
        if transition == "start":
            stage["status"] = "IN_PROGRESS"
        else:
            stage["status"] = result or "OBSERVED"
        self._record("recorder", "stage_%s" % transition, {
            "stage": name, "evidence": evidence, "result": result,
        })

    def _on_string(self, topic: str, msg: String) -> None:
        payload = self._decode(msg.data)
        self._record(topic, "message", payload)
        if topic == "/task_event":
            self._on_task_event(str(msg.data))
        elif topic == "/mission_supervisor_node/status" and isinstance(payload, dict):
            self.mission_state = str(payload.get("state", self.mission_state))
            if self.returned_base and self.mission_state in ("DONE", "IDLE"):
                self._stage(
                    "stable_stop", "end",
                    "supervisor state=%s after return" % self.mission_state,
                    "OBSERVED",
                )
        elif topic == "/mission_supervisor_node/target_report" and isinstance(payload, dict):
            seq = payload.get("seq")
            if seq is not None:
                self.target_id = "target_%s" % seq
        elif topic == "/arm/status" and isinstance(payload, dict):
            reported = str(payload.get("target_id", "") or "")
            if reported:
                self.target_id = reported
        elif topic == "/target_detection_status" and isinstance(payload, dict):
            if bool(payload.get("target_detected", False)):
                self._stage(
                    "visual_confirmation", "start",
                    "target_detection_status target_detected=true",
                )
        elif topic == "/storage_box/status" and isinstance(payload, dict):
            self._on_storage_status(payload)

    def _on_task_event(self, event: str) -> None:
        if event.startswith(("mission_started", "manual_grasp_mission_started")):
            self._stage("noncontact_start", "start", event)
            self._stage("noncontact_start", "end", event, "OBSERVED")
            self._stage("leave_base", "start", event)
        if event.startswith(("target_seen", "mission_pending_target_recorded")):
            self._stage("visual_confirmation", "start", event)
        if event.startswith("mission_visual_approach_started"):
            self._stage("leave_base", "end", event, "OBSERVED")
            self._stage("visual_confirmation", "end", event, "OBSERVED")
            self._stage("navigate_dock", "start", event)
        if event.startswith((
            "mission_target_geometric_arrival_confirmed",
            "mission_arm_handoff_started",
            "manual_grasp_goal_arrived",
        )):
            self._stage("navigate_dock", "end", event, "OBSERVED")
        if event.startswith((
            "mission_arm_waiting_for_target",
            "mission_arm_grasp_requested",
            "mission_arm_grasp_started",
        )):
            self._stage("autonomous_grasp", "start", event)
        if event.startswith("mission_arm_grasp_succeeded"):
            self._stage("autonomous_grasp", "end", event, "OBSERVED_SUCCESS")
        if event.startswith("mission_arm_grasp_failed"):
            self._stage("autonomous_grasp", "end", event, "FAILED")
            self.errors.append(event)
        if event.startswith("mission_return_started"):
            self._stage("carry_return", "start", event)
        if event.startswith("mission_arrived_base"):
            self.returned_base = True
            self._stage("carry_return", "end", event, "OBSERVED")
            self._stage("stable_stop", "start", event)
        if any(token in event for token in ("failed", "abort", "slam_lost")):
            self.errors.append(event)

    def _on_storage_status(self, payload: Dict[str, Any]) -> None:
        if payload.get("state") != "command_accepted":
            return
        commanded = str(payload.get("commanded_state", ""))
        if commanded == "closed" and self.stages["autonomous_grasp"].get(
                "status") == "OBSERVED_SUCCESS":
            self._record(
                "/storage_box/status", "payload_secured_command",
                "close command accepted; physical state is not sensed",
            )
        if commanded == "open" and self.returned_base:
            self._stage(
                "designated_unload", "start",
                "storage box open command accepted after return",
            )
            self._stage(
                "designated_unload", "end",
                "controller accepted open command; physical unload unverified",
                "COMMAND_ACCEPTED_UNVERIFIED",
            )

    def _on_sample_present(self, msg: Bool) -> None:
        self._record(
            "/storage_box/sample_present", "message", {"data": bool(msg.data)}
        )
        if msg.data:
            evidence = "independent /storage_box/sample_present=true"
            self._stage("independent_bin_confirmation", "start", evidence)
            self._stage(
                "independent_bin_confirmation", "end", evidence,
                "OBSERVED_SUCCESS",
            )

    def _on_pose(self, msg: PoseStamped) -> None:
        pose = msg.pose
        self.last_pose = {
            "frame_id": msg.header.frame_id,
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "qx": float(pose.orientation.x),
            "qy": float(pose.orientation.y),
            "qz": float(pose.orientation.z),
            "qw": float(pose.orientation.w),
        }
        now = time.monotonic()
        if now - self.last_pose_write < 0.2:
            return
        self.last_pose_write = now
        row = self._now()
        row.update(self.last_pose)
        row.update({
            "mission_state": self.mission_state,
            "target_id": self.target_id,
        })
        self.pose_writer.writerow(row)

    def _write_summary(self) -> None:
        missing = [
            name for name, stage in self.stages.items()
            if stage.get("status") in ("NOT_OBSERVED", "IN_PROGRESS")
        ]
        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_wall_time_utc": datetime.fromtimestamp(
                self.started_wall, timezone.utc
            ).isoformat(),
            "elapsed_s": round(time.monotonic() - self.started_mono, 3),
            "mission_state": self.mission_state,
            "target_id": self.target_id,
            "last_pose": self.last_pose,
            "stages": self.stages,
            "errors": self.errors,
            "missing_or_incomplete_stages": missing,
            "complete_50_point_chain": not missing and not self.errors,
            "limitations": [
                "Storage controller acknowledgement is not physical position feedback.",
                "Independent in-bin confirmation requires a real publisher on /storage_box/sample_present.",
            ],
        }
        temporary = self.output_dir / "summary.json.tmp"
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output_dir / "summary.json")

    def close(self) -> None:
        self._record("recorder", "stop", {})
        self._write_summary()
        self.events_file.close()
        self.pose_file.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root", default="evidence/single_target_runs",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_root).expanduser().resolve() / args.run_id
    if output_dir.exists():
        print("REFUSED: output directory already exists: %s" % output_dir)
        return 2
    rclpy.init()
    node = EvidenceRecorder(output_dir, args.run_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("Evidence saved to %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
