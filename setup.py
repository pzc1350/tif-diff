from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tif-diff",
    version="0.1.0",
    author="pzc1350",
    author_email="your-email@example.com",
    description="基于SAM的先进遥感图像智能解译系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pzc1350/tif-diff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "rasterio>=1.3.0",
        "geopandas>=0.13.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "tif-diff=main:main",
            "tif-diff-quick=quick_start:quick_process",
        ],
    },
)