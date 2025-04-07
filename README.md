# Intelligent Robotic Manipulation: Advanced topics in Robot Perception, Planning and Control: Final Project

The overall goal of this project is to grasp an object with a robot arm and place it in a goal basket while avoiding obstacles. To facilitate understanding all aspects that need to be considered to achieve this goal, the project can be divided into five sub-tasks. Perception (task 1) to detect the graspable object, Controller (task 2) to move the robot arm, sample and execute a grasp (task 3), localize and track obstacles (task 4) and plan the trajectory to place the object in the goal, while avoiding the obstacles (task 5). The project is implemented using a pybullet simulation, but the methods don't use privilege information from the sim in the process (only for tests).

## Setup Instructions

### 1. Clone the repository
### 2. Create a virtual environment with Python 3.8 and activate it
### 3. Install requirements
```shell
pip install -r requirements.txt
```
### 4. Install pybullet object models
```shell
git clone https://github.com/eleramp/pybullet-object-models.git # inside the final_project folder
pip install -e pybullet-object-models/
```

## Codebase Structure

```shell

irobman-wise-2425-final-project # code base
├── configs
│   └── test_config.yaml # config file
├── main.py # test full setup
├── README.md
├── test_control.py # test moving the arm to a specific position
├── test_grasping.py # test grasping objects
├── test_perception.py # test detecting objects and estimating their pose
├── test_pnp.py # test pick and place operation without obstacles
├── test_tracking.py # test obstacle tracking
└── src
    ├── control.py # moving the robot arm
    ├── grasping.py # find grasps and execute pick and place
    ├── objects.py # contains all objects and obstacle definitions
    ├── perception.py # detect objects and estimate pose
    ├── planning.py # plan movements to avoid obstacles
    ├── robot.py # robot class
    ├── simulation.py # simulation class
    ├── tracking.py # track obstacles
    └── utils.py # helpful utils
README.md
requirements.txt

```

## Results

The project includes tests to evaluate the performance of each sub-task. Results are logged and visualized to provide insights into the system's capabilities.

To execute the tests, run them from the final_project folder.

### Run all tests
```shell
python irobman-wise-2425-final-project/launch.py
```
### Test Perception
```shell
python irobman-wise-2425-final-project/test_perception.py
```
### Test Control
```shell
python irobman-wise-2425-final-project/test_control.py
```
### Test Grasping
```shell
python irobman-wise-2425-final-project/test_grasping.py
```
### Test Pick-and-place
```shell
python irobman-wise-2425-final-project/test_pnp.py
```
### Test Tracking
```shell
python irobman-wise-2425-final-project/test_tracking.py
```
### Test Full Setup
```shell
python irobman-wise-2425-final-project/main.py
```

## Authors

Elena Bock, Dominik Schiller

