from pyrealsense2 import pyrealsense2 as rs
import numpy as np
import cv2, json, time
from pathlib import Path
out=Path('/tmp/realsense_check')
p=rs.pipeline(); c=rs.config()
c.enable_device('244222076021')
c.enable_stream(rs.stream.depth,640,480,rs.format.z16,15)
c.enable_stream(rs.stream.color,640,480,rs.format.bgr8,15)
started=False
try:
 profile=p.start(c); started=True
 scale=profile.get_device().first_depth_sensor().get_depth_scale()
 start=time.monotonic(); timestamps=[]
 for i in range(45):
  frames=p.wait_for_frames(10000)
  if frames.get_depth_frame() and frames.get_color_frame():
   timestamps.append(frames.get_depth_frame().get_timestamp())
 d=frames.get_depth_frame(); color=frames.get_color_frame()
 raw=np.asanyarray(d.get_data()).copy(); rgb=np.asanyarray(color.get_data()).copy()
 meters=raw*scale; valid=meters>0
 colored=cv2.applyColorMap(cv2.convertScaleAbs(np.clip(meters,0,3),alpha=255/3),cv2.COLORMAP_TURBO)
 colored[~valid]=0
 preview=np.hstack([rgb,colored])
 cv2.putText(preview,'RGB | DEPTH (0-3m, black=invalid)',(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
 for name,img in [('color.png',rgb),('depth_raw.png',raw),('depth_color.png',colored),('preview.jpg',preview)]:
  assert cv2.imwrite(str(out/name),img)
 intr=d.profile.as_video_stream_profile().intrinsics
 info={'device':'Intel RealSense D435i','serial':'244222076021','depth_scale_m':scale,'resolution':[640,480],'requested_fps':15,'frames_received':len(timestamps),'elapsed_s':round(time.monotonic()-start,3),'depth_fps_from_timestamps':round((len(timestamps)-1)*1000/(timestamps[-1]-timestamps[0]),2),'valid_depth_percent':round(float(valid.mean()*100),2),'center_depth_m':float(meters[240,320]),'depth_percentiles_m':np.percentile(meters[valid],[5,50,95]).tolist() if valid.any() else [],'intrinsics':{'fx':intr.fx,'fy':intr.fy,'ppx':intr.ppx,'ppy':intr.ppy,'model':str(intr.model),'coeffs':list(intr.coeffs)},'note':'Depth is native unaligned Z16; meters = raw * depth_scale_m. Color and depth pixels are not registered.'}
 (out/'info.json').write_text(json.dumps(info,indent=2))
 print(json.dumps(info,indent=2))
finally:
 if started:p.stop()
