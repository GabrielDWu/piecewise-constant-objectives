from setuptools import find_packages, setup

setup(
    name="piecewise-constant-objectives",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "torch>=2.4.1",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm"
    ],
    extras_require={
        "test": ["pytest"],
    },
)