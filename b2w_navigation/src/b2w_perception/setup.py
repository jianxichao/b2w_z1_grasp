from glob import glob
from setuptools import find_packages, setup

package_name = "b2w_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml") + glob("config/*.xml"),
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/scripts", glob("scripts/*.sh")),
        (
            f"share/{package_name}/models/yolo",
            glob("models/yolo/*.onnx")
            + glob("models/yolo/*.json"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RL2 Challenge Cup Team",
    maintainer_email="0305pkj@sjtu.edu.cn",
    description="Target detection and 3D pose estimation for science targets.",
    license="MIT",
    options={
        "develop": {"script_dir": f"$base/lib/{package_name}"},
        "install": {"install_scripts": f"$base/lib/{package_name}"},
    },
    entry_points={
        "console_scripts": [
            "yolo_target_perception_node = b2w_perception.yolo_target_perception_node:main",
        ],
    },
)
