import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cerberus_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Kevin Thiha",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="Launch files composing the full Cerberus ROS2 graph.",
    license="MIT",
    tests_require=["pytest"],
)
