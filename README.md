# Mapping of Radio Beacons by Particle Filtering

## 🧠 Description

This project implements a Particle Filter for mapping radio beacons using an aerial robot. Built with ROS and Python, it demonstrates probabilistic SLAM techniques based on radio signal observations.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/josgarvil/Mapping-of-radio-beacons-by-Particle-Filter.git

# Add to your ROS workspace
cd ~/catkin_ws/src
ln -s /path/to/Mapping-of-radio-beacons-by-Particle-Filter .
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 🚀 Usage

```bash
roslaunch mapping_pf mapping.launch
```

* Launch sensor subscribers
* Run the particle filter algorithm
* Visualize the map using RViz

## 📁 Project Structure

```arduino
mapping_pf/
├── launch/
│   └── mapping.launch
├── scripts/
│   └── particle_filter.py
└── msg/
```
## 🛠️ Technologies

* ROS
* Python
* Particle Filters

## 👨‍💻 Author
José García Villalón – GitHub