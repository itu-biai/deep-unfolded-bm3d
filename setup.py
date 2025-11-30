"""
Setup configuration for Deep Unfolded BM3D
"""
from setuptools import setup, find_packages

setup(
    name="du-bm3d",
    version="0.1.0",
    description="Deep Unfolded BM3D: Non-Local Collaborative Filtering as Trainable Neural Network",
    author="Kerem Basim, Mehmet Ozan Unal, Metin Ertas, Isa Yildirim",
    author_email="basim21@itu.edu.tr, unalmehmet@itu.edu.tr, ertas@istanbul.edu.tr, iyildirim@itu.edu.tr",
    url="https://github.com/itu-biai/deep_unfolded_bm3d",
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-image>=0.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "bm3d>=4.1.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3",
        "Operating System :: OS Independent",
    ],
)
