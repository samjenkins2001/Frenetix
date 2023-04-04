from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    required = f.read().splitlines()


setup(
    name="commonroad-reactive-planner",
    version="2023.1",
    description="Reactive Planner: Sampling-based Frenet Planner",
    long_description_content_type='text/markdown',
    long_description=readme,
    url="https://gitlab.lrz.de/av2.0/commonroad/commonroad-reactive-planner",
    author='Cyber-Physical Systems Group, Technical University of Munich',
    author_email='commonroad@lists.lrz.de',
    license='GNU General Public License v3.0',
    packages=find_packages(exclude=['doc', 'unit_tests']),
    zip_safe=False,
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'commonroad_vehicle_models>=3.0.2',
        'matplotlib>=3.7.1',
        'networkx>=2.6',
        'numpy>=1.24.2',
        'methodtools',
        'omegaconf>=2.1.1',
        'pytest>=6.2.5',
        'shapely>=2.0.1',
        'scipy>=1.7.0',
        'commonroad-io==2023.1',
        'commonroad-drivability-checker==2022.2.1',
        ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ]
)
