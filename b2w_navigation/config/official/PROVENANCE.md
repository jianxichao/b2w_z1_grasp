# Official B2W reference geometry

Repository: https://github.com/unitreerobotics/unitree_ros
Revision: 15ebe53f4cd77e7d745ae83b9d7ff90e685e8998
URDF: https://github.com/unitreerobotics/unitree_ros/blob/15ebe53f4cd77e7d745ae83b9d7ff90e685e8998/robots/b2w_description/urdf/b2w_description.urdf
Product dimensions: https://www.unitree.com/b2-w/
Retrieved: 2026-10-27

- Nominal standing dimensions: 1.098 x 0.550 x 0.758 m.
- Navigation envelope: 1.30 x 0.80 m, padded from the nominal footprint.
- base_link -> lidar_link (used as raw rslidar frame): xyz=(0.34218,0,0.17851), rpy=(0,0,0).
- base_link -> f_dc_link: xyz=(0.42161,0.025,0.061851), rpy=(-2.3562,0,-1.5708).
- base_link -> r_dc_link: xyz=(-0.42244,-0.025,0.038026), rpy=(-2.3562,0,1.5708).

These are model mount frames, not measured optical calibration. No image intrinsics,
distortion coefficients, depth stream endpoint, or optical-frame equivalence is
inferred. Z1 must remain inside the padded navigation envelope.
base_footprint -> base_link uses live body height/roll/pitch feedback for flat-floor
testing, not the product's total standing height. Firmware body_height=0 falls back
to position.z; floor-frame interpretation still requires physical validation.
