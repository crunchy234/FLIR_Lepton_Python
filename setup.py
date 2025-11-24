"""
Setup script for FLIR Lepton 3.5 Python Library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="flir-lepton",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Python library for FLIR Lepton 3.5 thermal camera on Raspberry Pi and Jetson",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FLIR_Lepton_Python",
    py_modules=["flir_lepton"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Hardware Drivers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "display": ["opencv-python>=4.5.0"],
        "imaging": ["Pillow>=8.0.0"],
        "control": ["smbus2>=0.4.0"],
    },
)
