#!/usr/bin/env python3
"""Test the exact online PT inference path on an image; no ROS node or robot I/O."""
import argparse
import json
from pathlib import Path
import cv2
from sensor_msgs.msg import Image
from ultralytics import YOLO
from b2w_perception.yolo_target_perception_node import YoloTargetPerceptionNode

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image', type=Path)
parser.add_argument('--output', type=Path, default=ROOT/'output/offline_detection.json')
args = parser.parse_args()
image = cv2.imread(str(args.image))
if image is None:
    raise SystemExit(f'Cannot read image: {args.image}')
# Bypass ROS Node construction to guarantee that this check performs no DDS I/O.
node = object.__new__(YoloTargetPerceptionNode)
node.pt_model = YOLO(str(ROOT.parent/'sponge_best.pt'))
node.class_names = node.pt_model.names
node.class_map = {}
node.target_class, node.device = 'sponge', 'cpu'
node.imgsz, node.candidate_confidence_threshold = 640, 0.30
node.rgb_msg = Image(height=image.shape[0], width=image.shape[1], encoding='bgr8', step=image.strides[0], data=image.tobytes())
detection = node._detect_yolo_target()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'model': str(ROOT.parent/'sponge_best.pt'), 'names': node.class_names, 'image': str(args.image.resolve()), 'target': 'sponge', 'uv_confidence_wh': detection}, indent=2)+'\n')
print(args.output.read_text())
