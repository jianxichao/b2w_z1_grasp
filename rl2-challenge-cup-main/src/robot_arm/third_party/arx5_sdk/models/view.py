import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# 加载模型
model = mujoco.MjModel.from_xml_path("X5.urdf")
data = mujoco.MjData(model)

# 获取 EEF 的 ID (确保名字与 URDF 一致)
eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link1")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # 1. 获取位置 (x, y, z) - 单位：米
        pos = data.xpos[eef_id]
        
        # 2. 获取姿态 (四元数 w, x, y, z)
        quat = data.xquat[eef_id]
        
        # 3. 转换为 RPY 弧度 (可选，如果习惯看欧拉角)
        # 注意 MuJoCo 四元数顺序是 (w, x, y, z)，Scipy 需要 (x, y, z, w)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rpy = r.as_euler('xyz', degrees=False)
        
        # 打印 Pose
        print(f"EEF ID: {eef_id}")
        print(f"EEF Pose (world Frame): Pos={pos.round(3)}, RPY={rpy.round(3)}")

        # 更加严谨的相对位置计算
        relative_pos = data.xpos[eef_id] - data.xpos[base_id]
        print(f"Base ID: {base_id}")
        print(f"EEF Pose (Base Frame): Pos={relative_pos.round(3)}")
        
        viewer.sync()
