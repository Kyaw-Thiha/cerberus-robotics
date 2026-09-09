from setuptools import find_packages, setup

package_name = "locomotion_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    # onnxruntime is also a hard runtime import in inference_node.py but is
    # provided by the pixi environment (pixi.toml), not pip's install
    # process for this ament_python package -- not listed here since a
    # `pip install` of this package alone wouldn't be how it's deployed.
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Kevin Thiha",
    maintainer_email="k.thiha10.mail@gmail.com",
    description="ROS2 inference node for a trained Cerberus locomotion policy.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = locomotion_policy.inference_node:main",
        ],
    },
)
