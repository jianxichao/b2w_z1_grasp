"""Remote RealSense worker. Explicit serial; aligned RGB-D; no arm commands."""
import contextlib,json,os,sys
import cv2
import numpy as np
import pyrealsense2 as rs
if not hasattr(rs,'pipeline'):
    from pyrealsense2 import pyrealsense2 as rs


def main():
    serial=sys.argv[1]
    output=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    pipe,cfg=rs.pipeline(),rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,15)
    cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,15)
    with contextlib.redirect_stdout(sys.stderr):
        profile=pipe.start(cfg)
        try:
            scale=profile.get_device().first_depth_sensor().get_depth_scale()
            align=rs.align(rs.stream.color)
            for _ in range(20):pipe.wait_for_frames(10000)
            for line in sys.stdin:
                if line.strip()=='QUIT':break
                if line.strip()!='READ':raise ValueError('Unknown camera command')
                # Bound queue draining even if the device produces frames continuously.
                for _ in range(30):
                    if not pipe.poll_for_frames():break
                frames=align.process(pipe.wait_for_frames(10000))
                depth,color=frames.get_depth_frame(),frames.get_color_frame()
                if not depth or not color:raise RuntimeError('Missing aligned RGB-D')
                intr=color.profile.as_video_stream_profile().intrinsics
                payload=[]
                for ext,frame in [('.jpg',color),('.png',depth)]:
                    ok,data=cv2.imencode(ext,np.asanyarray(frame.get_data()))
                    if not ok:raise RuntimeError('Image encoding failed')
                    payload.append(data.tobytes())
                meta=dict(serial=serial,color_bytes=len(payload[0]),depth_bytes=len(payload[1]),
                          depth_scale_m=scale,fx=intr.fx,fy=intr.fy,ppx=intr.ppx,ppy=intr.ppy,
                          width=intr.width,height=intr.height,frame_number=color.get_frame_number(),
                          depth_timestamp_ms=depth.get_timestamp(),color_timestamp_ms=color.get_timestamp(),
                          timestamp_domain=str(depth.get_frame_timestamp_domain()),
                          color_timestamp_domain=str(color.get_frame_timestamp_domain()))
                output.write(json.dumps(meta).encode()+b'\n')
                for data in payload:output.write(data)
        finally:pipe.stop()


if __name__=='__main__':main()
