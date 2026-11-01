#!/usr/bin/env python3
"""Detect sponges using B2W front/back SDK images. No depth or motion is fabricated."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='eno1')
    parser.add_argument('--camera', choices=('front','back','both'), default='front')
    parser.add_argument('--confidence', type=float, default=0.65)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'output/builtin')
    args = parser.parse_args()
    binary = ROOT/'build/native/capture_builtin'
    if not binary.is_file():
        raise SystemExit('Build first: ./scripts/build_native.sh')
    model = YOLO(str(ROOT.parent/'sponge_best.pt'))
    classes = [i for i,n in model.names.items() if n=='sponge']
    if not classes:
        raise SystemExit(f'Model does not contain sponge: {model.names}')
    args.output.mkdir(parents=True, exist_ok=True)
    cameras = ('front','back') if args.camera=='both' else (args.camera,)
    try:
        while True:
            begun = time.time()
            capture = subprocess.run([str(binary), args.interface, str(args.output), args.camera], timeout=12)
            if capture.returncode:
                raise RuntimeError('Camera capture failed; stale images will not be inferred')
            for camera in cameras:
                image = cv2.imread(str(args.output/f'{camera}.jpg'))
                if image is None:
                    raise RuntimeError('Camera payload is not a decodable image')
                result = model.predict(image, classes=classes, conf=args.confidence, device='cpu', imgsz=640, verbose=False)[0]
                boxes = [{'class':'sponge', 'confidence':float(b.conf.item()), 'xyxy':b.xyxy[0].cpu().tolist()} for b in result.boxes]
                status = {'camera':camera, 'capture_started_wall_s':begun, 'capture_and_inference_s':time.time()-begun,
                          'detections':boxes, 'depth_available':False, 'navigation_target_published':False}
                (args.output/f'{camera}_detections.json').write_text(json.dumps(status, indent=2)+'\n')
                annotated = result.plot()
                cv2.imwrite(str(args.output/f'{camera}_detections.jpg'), annotated)
                print(json.dumps(status), flush=True)
                if args.view:
                    cv2.imshow('B2W '+camera, annotated)
            if args.once or (args.view and cv2.waitKey(1)&0xff==ord('q')):
                break
            time.sleep(0.10)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
