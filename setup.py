from setuptools import setup, find_packages

setup(
    name="gpcr-coupling",
    version="1.0.0",
    description="Paired GPCR-G protein coupling prediction using protein language models",
    author="Guohao Lü, Yuxin Xia",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "gpcr-coupling=gpcr_coupling.cli:main",
        ],
    },
    python_requires=">=3.8",
)
