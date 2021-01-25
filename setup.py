# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

test_packages = [
    "black==20.8b1",
    "flake8==3.8.3",
    "isort==5.5.3",
    "pytest==6.0.2",
    "pytest-cov==2.10.1",
]

dev_packages = [
    "jupyterlab==2.2.8",
    "mkdocs==1.1.2",
    "mkdocs-macros-plugin==0.5.0",
    "mkdocs-material==6.2.4",
    "mkdocs-redirects==1.0.1",
    "mkdocstrings==0.14.0",
    "pre-commit==2.7.1",
]

setup(
    name="tagifai",
    version="0.1",
    license="MIT",
    description="Tag suggestions for projects on Made With ML.",
    author="Goku Mohandas",
    author_email="hello@madewithml.com",
    url="https://madewithml.com/",
    keywords=[
        "machine-learning",
        "artificial-intelligence",
        "madewithml",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages,
    },
    entry_points={
        "console_scripts": [
            "tagifai = tagifai.main:app",
        ],
    },
)
