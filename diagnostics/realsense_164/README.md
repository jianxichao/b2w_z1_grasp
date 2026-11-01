# RealSense 检查结果

相机连接在 192.168.123.164（unitree-b2-pc4）的 USB 上，通过 SSH 读取。
型号 Intel RealSense D435i，序列号 244222076021，USB 2.1。

- color.png：彩色 BGR 图像，保存为标准 PNG。
- depth_raw.png：原始 16 位深度 PNG；像素值乘以 info.json 中的 depth_scale_m 得到米，0 表示无效。
- depth_color.png：0–3 米伪彩色示意图；超过 3 米按 3 米着色，黑色为无效像素。
- preview.jpg：左彩色、右深度；两幅图未进行像素配准。
- info.json：相机内参、深度单位与本次采集统计。
- capture.py：机载电脑上的采集脚本副本，依赖机载电脑已有的 pyrealsense2、numpy、cv2。

再次采集：

```bash
ssh unitree@192.168.123.164 'python3 /tmp/realsense_check/capture.py'
scp unitree@192.168.123.164:/tmp/realsense_check/\*.png /home/robot/unitree_ws/diagnostics/realsense_164/
scp unitree@192.168.123.164:/tmp/realsense_check/info.json /home/robot/unitree_ws/diagnostics/realsense_164/
scp unitree@192.168.123.164:/tmp/realsense_check/preview.jpg /home/robot/unitree_ws/diagnostics/realsense_164/
```

远端脚本和输出位于 /tmp/realsense_check，重启后可能清除。采集结束已释放相机，未启动常驻服务或 ROS 发布节点。
