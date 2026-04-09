"""
setup.py for the SGCRL isolation study package.

Install with:
    pip install -e .

This makes `from envs import ...`, `from agent import ...` etc. work from any
directory without manually patching sys.path.
"""

from setuptools import setup, find_packages

setup(
    name="sgcrl-isolation-study",
    version="1.0.0",
    description="Isolation study for Single-Goal Contrastive RL (SGCRL) — "
                "HPC-ready codebase for NYU Torch SLURM cluster.",
    author="NYU SGCRL Research",
    python_requires=">=3.9",
    packages=find_packages(
        where=".",
        include=["isolation_study*"],
        exclude=["results*", "slurm*", "__pycache__*"],
    ),
    py_modules=[
        "agent",
        "envs",
        "metrics",
        "run_experiment",
        "run_sweep",
        "generate_plots",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "sgcrl-run=run_experiment:main",
            "sgcrl-sweep=run_sweep:main",
            "sgcrl-plots=generate_plots:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
