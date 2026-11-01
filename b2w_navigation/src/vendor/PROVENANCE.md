Unitree ROS 2 message definitions (unitree_api, unitree_go) copied from:
https://github.com/unitreerobotics/unitree_ros2
Commit: 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
Retrieved: 2026-10-27
See UNITREE_LICENSE and each package.xml for upstream license information.
These packages only define messages; no robot controller is started by building.

Local CMake change: legacy rosidl_generator_dds_idl/Connext IDL generation is
optional. Standard ROS 2 C/C++/Python typesupport and message definitions are
unchanged. Navigation uses rmw_cyclonedds_cpp.

SportModeState.msg appends PathPoint[10] path_point to match this workspace's
unitree_sdk2/include/unitree/idl/go2/SportModeState_.hpp.
