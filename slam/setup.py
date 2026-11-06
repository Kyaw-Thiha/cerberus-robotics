from setuptools import find_packages, setup

package_name = "slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kevin",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="Pluggable multi-backend SLAM",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ground_truth_publisher = slam.slam.ground_truth.ground_truth_node:main",
            "fast_lio2_adapter = slam.slam.backends.fast_lio2.adapter_launch:main",
            "bievr_lio_adapter = slam.slam.backends.bievr_lio.adapter_launch:main",
            "fast_livo2_adapter = slam.slam.backends.fast_livo2.adapter_launch:main",
            "orb_slam3_adapter = slam.slam.backends.orb_slam3.adapter_launch:main",
            "multicam_slam_adapter = slam.slam.backends.multicam_slam.adapter_launch:main",
            "stereo_to_depth_node = slam.slam.backends.multicam_slam.stereo_to_depth_node:main",
            "gs_livo_adapter = slam.slam.backends.gs_livo.adapter_launch:main",
        ],
    },
)
