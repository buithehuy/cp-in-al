from setuptools import setup, find_packages

setup(
    name="cp-in-al",
    version="0.1.0",
    description="Active Learning with Conformal Prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "hydra-core>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.8",
)
