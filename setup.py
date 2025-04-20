from setuptools import setup, find_packages

setup(
    name="video-logs-extract",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "ultralytics",
        "pyyaml"
    ],
) 