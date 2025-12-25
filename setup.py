"""
Setup script for CANS/GQS Framework
The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cans-gqs",
    version="1.0.0",
    author="ContributorX Ltd.",
    description="A Formal n-Dimensional Framework for Computational Geometry and Dynamic Simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "numba>=0.54.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "full": [
            "clifford>=1.4.0",
            "scikit-learn>=1.0.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
        ],
    },
)
