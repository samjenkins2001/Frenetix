# Standard imports
import subprocess

# Third party imports
import setuptools


def git(*args):
    return subprocess.check_output(["git"] + list(args))


# get latest tag
try:
    latest = git("describe", "--tags").decode().strip()
    latest = latest.split("-")[0]
except subprocess.CalledProcessError:
    latest = "2023.1"


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Remove extra index urls
requirements = [
    requirement
    for requirement in requirements
    if "--extra-index-url" not in requirement
]

setuptools.setup(
    name="commonroad-reactive-planner",
    version=latest,
    description="Reactive Planner: Sampling-based Frenet Planner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.lrz.de/av2.0/commonroad/commonroad-reactive-planner",
    author='Institute of Automotive Technology, Technical University of Munich',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
)

# EOF
