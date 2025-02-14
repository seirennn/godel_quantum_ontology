"""
Setup configuration for the Divine Algorithm package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="divine-algorithm",
    version="0.1.0",
    author="Gödel Quantum Research Team",
    author_email="contact@godelquantum.com",
    description="A quantum computational proof of God's existence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/godelquantum/divine-algorithm",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "pylint>=2.17.5",
        ],
        "symbolic": [
            "symengine>=0.9.2,<0.10",
            "sympy>=1.12"
        ]
    },
    entry_points={
        "console_scripts": [
            "divine-algorithm=divine_algorithm.main:main",
        ],
    },
)