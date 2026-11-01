#!/usr/bin/env bash
# Activate the project Python environment, then source Unitree Humble and this
# workspace without allowing nounset to break generated setup files.

_rl2_restore_nounset=0
case $- in
  *u*) _rl2_restore_nounset=1 ;;
esac

set +u
_rl2_workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rl2_venv="${_rl2_workspace_dir}/.venv/bin/activate"
if [[ ! -f "${_rl2_venv}" ]]; then
  echo "RL2 project environment is missing: ${_rl2_venv}" >&2
  echo "Create the workspace .venv before sourcing this script." >&2
  unset _rl2_workspace_dir _rl2_venv _rl2_restore_nounset
  return 1
fi

source "${_rl2_venv}"
source /home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh

if [[ ! -f "${_rl2_workspace_dir}/install/setup.bash" ]]; then
  echo "RL2 workspace is not built: ${_rl2_workspace_dir}/install/setup.bash is missing." >&2
  echo "Run colcon build --symlink-install in ${_rl2_workspace_dir} first." >&2
  unset _rl2_workspace_dir _rl2_venv _rl2_restore_nounset
  return 1
fi

source "${_rl2_workspace_dir}/install/setup.bash"
export ROS_LOG_DIR="${_rl2_workspace_dir}/log/ros"
export TMPDIR="${_rl2_workspace_dir}/.tmp"
export PIP_CACHE_DIR="${_rl2_workspace_dir}/.cache/pip"
export TORCH_HOME="${_rl2_workspace_dir}/.cache/torch"
export YOLO_CONFIG_DIR="${_rl2_workspace_dir}/.cache/ultralytics"
export MPLCONFIGDIR="${_rl2_workspace_dir}/.cache/matplotlib"
mkdir -p \
  "${ROS_LOG_DIR}" "${TMPDIR}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" \
  "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}"

echo "RL2 workspace environment: ${_rl2_workspace_dir} (Python: ${VIRTUAL_ENV})"

unset _rl2_workspace_dir _rl2_venv
if [[ ${_rl2_restore_nounset} -eq 1 ]]; then
  unset _rl2_restore_nounset
  set -u
else
  unset _rl2_restore_nounset
fi
