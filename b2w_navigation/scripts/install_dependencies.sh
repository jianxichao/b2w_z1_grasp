#!/usr/bin/env bash
set -e
sudo apt-get update
sudo apt-get install ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-slam-toolbox ros-humble-pointcloud-to-laserscan \
    ros-humble-rmw-cyclonedds-cpp python3-pytest python3-yaml
