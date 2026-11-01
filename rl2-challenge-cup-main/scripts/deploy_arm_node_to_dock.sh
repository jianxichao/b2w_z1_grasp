#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCK_HOST="${DOCK_HOST:-unitree@192.168.123.18}"
DOCK_ROOT="${DOCK_ROOT:-/home/unitree/rl2_arm_runtime}"
CACHE="${RL2_DOCK_WHEEL_CACHE:-/tmp/rl2-dock-wheels}"
STAGE="$(mktemp -d /tmp/rl2-arm-dock.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$CACHE" "$STAGE/src" "$STAGE/wheels" "$STAGE/scripts"

ARX_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'arx5_interface-0.1.4-cp38-*aarch64*.whl' -print -quit)"
if [[ -z "$ARX_WHEEL" ]]; then
  python3 -m pip download \
    --only-binary=:all: --no-deps \
    --platform manylinux_2_24_aarch64 \
    --platform manylinux_2_28_aarch64 \
    --python-version 38 --implementation cp --abi cp38 \
    --dest "$CACHE" arx5-interface==0.1.4
  ARX_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'arx5_interface-0.1.4-cp38-*aarch64*.whl' -print -quit)"
fi
[[ -n "$ARX_WHEEL" ]] || { echo "ARX5 ARM64 wheel download failed" >&2; exit 2; }
cp "$ARX_WHEEL" "$STAGE/wheels/"

ORBBEC_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'pyorbbecsdk2-2.1.2-cp38-*aarch64*.whl' -print -quit)"
if [[ -z "$ORBBEC_WHEEL" ]]; then
  python3 -m pip download \
    --only-binary=:all: --no-deps \
    --platform manylinux_2_27_aarch64 \
    --python-version 38 --implementation cp --abi cp38 \
    --dest "$CACHE" pyorbbecsdk2==2.1.2
  ORBBEC_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'pyorbbecsdk2-2.1.2-cp38-*aarch64*.whl' -print -quit)"
fi
ORT_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'onnxruntime-1.16.3-cp38-*aarch64*.whl' -print -quit)"
if [[ -z "$ORT_WHEEL" ]]; then
  python3 -m pip download \
    --only-binary=:all: --no-deps \
    --platform manylinux2014_aarch64 \
    --python-version 38 --implementation cp --abi cp38 \
    --dest "$CACHE" onnxruntime==1.16.3
  ORT_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'onnxruntime-1.16.3-cp38-*aarch64*.whl' -print -quit)"
fi
NUMPY_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'numpy-1.24.4-cp38-*aarch64*.whl' -print -quit)"
if [[ -z "$NUMPY_WHEEL" ]]; then
  python3 -m pip download \
    --only-binary=:all: --no-deps \
    --platform manylinux2014_aarch64 \
    --python-version 38 --implementation cp --abi cp38 \
    --dest "$CACHE" numpy==1.24.4
  NUMPY_WHEEL="$(find "$CACHE" -maxdepth 1 -type f -name 'numpy-1.24.4-cp38-*aarch64*.whl' -print -quit)"
fi
[[ -n "$ORBBEC_WHEEL" && -n "$ORT_WHEEL" && -n "$NUMPY_WHEEL" ]] || {
  echo "Vision ARM64 wheel download failed" >&2; exit 2;
}
cp "$ORBBEC_WHEEL" "$ORT_WHEEL" "$NUMPY_WHEEL" "$STAGE/wheels/"

# Only the arm-side ROS packages and the compact ONNX model are deployed.
# Navigation, SLAM, PyTorch weights, tests and operator documents remain local.
cp -a "$ROOT/src/rl2_interfaces" "$STAGE/src/"
cp -a "$ROOT/src/rl2_manipulation" "$STAGE/src/"
cp -a "$ROOT/src/robot_arm" "$STAGE/src/"
rm -rf \
  "$STAGE/src/rl2_interfaces/test" \
  "$STAGE/src/rl2_manipulation/test" \
  "$STAGE/src/rl2_manipulation/docs" \
  "$STAGE/src/robot_arm/tests" \
  "$STAGE/src/robot_arm/docs" \
  "$STAGE/src/robot_arm/meshes" \
  "$STAGE/src/robot_arm/mjcf" \
  "$STAGE/src/robot_arm/tools" \
  "$STAGE/src/robot_arm/third_party/arx5_sdk/build" \
  "$STAGE/src/robot_arm/third_party/arx5_sdk/lib/x86_64" \
  "$STAGE/src/robot_arm/third_party/arx5_sdk/python/arx5_interface"*.so
rm -f "$STAGE/src/robot_arm/models/yolo/best.pt"
find "$STAGE/src" -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +

cp "$ROOT/scripts/dock_arm_setup_can.sh" "$STAGE/scripts/"
cp "$ROOT/scripts/dock_arm_start.sh" "$STAGE/scripts/"
cp "$ROOT/scripts/dock_arm_preflight.sh" "$STAGE/scripts/"
cp "$ROOT/scripts/dock_arm_initialize.sh" "$STAGE/scripts/"
cp "$ROOT/scripts/dock_arm_calibrate_gripper.sh" "$STAGE/scripts/"
cp "$ROOT/scripts/cyclonedds_foxy_eth10.xml" "$STAGE/scripts/"
cp "$ROOT/scripts/99-orbbec.rules" "$STAGE/scripts/"
chmod +x "$STAGE/scripts/"*.sh

tar -C "$STAGE" -czf "$STAGE/rl2_arm_runtime.tar.gz" src wheels scripts

echo "Uploading arm-only runtime to $DOCK_HOST:$DOCK_ROOT"
ssh "$DOCK_HOST" "mkdir -p '$DOCK_ROOT'"
scp "$STAGE/rl2_arm_runtime.tar.gz" "$DOCK_HOST:$DOCK_ROOT/"

ssh -t "$DOCK_HOST" "
  set -e
  cd '$DOCK_ROOT'
  tar --warning=no-timestamp -xzf rl2_arm_runtime.tar.gz
  if ! python3 -m pip install --user --no-deps --force-reinstall wheels/arx5_interface-*.whl; then
    echo 'Dock pip is too old for the wheel platform tag; extracting the standard wheel directly.'
    USER_SITE=\$(python3 -m site --user-site)
    mkdir -p \"\$USER_SITE\"
    python3 -m zipfile -e wheels/arx5_interface-*.whl \"\$USER_SITE\"
  fi
  USER_SITE=\$(python3 -m site --user-site)
  mkdir -p "\$USER_SITE"
  rm -rf "\$USER_SITE"/numpy "\$USER_SITE"/numpy.libs "\$USER_SITE"/numpy-*.dist-info
  python3 -m zipfile -e wheels/numpy-*.whl "\$USER_SITE"
  for wheel in wheels/pyorbbecsdk2-*.whl wheels/onnxruntime-*.whl; do
    python3 -m zipfile -e "\$wheel" "\$USER_SITE"
  done
  set +u
  source /opt/ros/foxy/setup.bash
  set -u
  python3 -c 'import arx5_interface, pyorbbecsdk, onnxruntime; print(arx5_interface.__file__); print(pyorbbecsdk.__file__); print(onnxruntime.__version__)'
  stamp=\$(date +%s).\$\$
  for generated in build install log; do
    if [ -e \"\$generated\" ]; then
      mv \"\$generated\" \"/tmp/rl2_arm_\${generated}_previous_\$stamp\"
    fi
  done
  colcon build --symlink-install --packages-select rl2_interfaces robot_arm rl2_manipulation
"

echo
echo "Deployment complete. No CAN interface was created and no robot motion was started."
echo "Next on dock: $DOCK_ROOT/scripts/dock_arm_preflight.sh"
