# Mapping of Radio Beacons by Particle Filtering

[![ROS](https://img.shields.io/badge/ROS-Melodic-brightgreen)](https://wiki.ros.org/melodic)
[![Python](https://img.shields.io/badge/Python-3.6-blue)](https://www.python.org/)
[![Gazebo](https://img.shields.io/badge/Gazebo-9-red)](https://gazebosim.org/)

## Description

This project implements a Particle Filter for mapping radio beacons using an aerial robot. Built with ROS and Python, it demonstrates probabilistic SLAM techniques based on radio signal observations.

## Installation

```bash
# Clone the repository
git clone https://github.com/josgarvil/Mapping-of-radio-beacons-by-Particle-Filter.git

# Add to your ROS workspace
cd ~/catkin_ws/src
ln -s /path/to/Mapping-of-radio-beacons-by-Particle-Filter .
cd ~/catkin_ws
catkin_make
source devel/setup.sh
```

## Usage

```bash
catkin_make && roscore
source devel/setup.sh && roslaunch mapping_adr mapping.launch
# or
source devel/setup.sh && roslaunch mapping_adr simulation.launch
```

* Launch sensor subscribers
* Run the particle filter algorithm
* Visualize the map using RViz
* Launch Gazebo simulation

## Technologies

* ROS
* Python
* Particle Filters

## Author
José García Villalón