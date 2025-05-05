from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    description="Advanced Transformer architecture implementation in PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/advanced-transformer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
