# Fast Differentiable Particle Filters for Visual-Lidar Localization

Hello!
This is our final project for [ROB 498-004 (Deeprob)](https://deeprob.org/w25/) in Winter 2025. This is our implementation of Differentiable particle filters based on work by [Jonschkowski et.al](https://github.com/tu-rbo/differentiable-particle-filters). We modified the model to utilize run on a [Mbot](https://mbot.robotics.umich.edu), an educational robotics platform created at the University of Michigan and used extensively across courses in the [Robotics Department](https://robotics.umich.edu).

To do this we made the following changes
1. Translate the original code from Tensorflow 1 to Pytorch
2. Write a script to collect a dataset from the mbot including slam pose, odom pose, camera images, and lidar scans.
3. Modify the model to create a Lidar Embedding in addition to a Camera Embedding and concatinate the two for use in subsequent stages

## Report
Our report is available here: [FinalReport.pdf](/FinalReport.pdf)

## Demo
We drove the mbot in the Maze pictured below:
![Original Maze](/images/Maze.jpeg "Original Maze")

We then visualized our paths where:
- Blue = DPF prediction
- Green = SLAM pose (treated as ground truth)
- Red = Odometry pose

![Path 1](/images/path1.png "Path 1")
![Path 2](/images/path2.png "Path 2")
![Path 3](/images/path3.png "Path 3")

This shows that our DPF has relatively good performance and compares well against traditional SLAM.

## Code Structure
- The **mbot** folder contains scripts to run on the mbot
    1. data_collection.py is used to build the dataset
    2. live_run.py is used to run the DPF on the mbot
    3. generate_map.py generates a map image based on the map file and run log
- The **mbot_web_app** folder contains a modified version of the Mbot web app that allows us to teleop an Mbot Classic. The standard web app only works with Mbot Omni.
-  The **models_trained** folder contains our trained model
- The **original experiments** folder contains the raw converted version of the original DPF code
- The **particle_filter** folder contains our actual DPF code

## Dependancies
Our project requires the following python libraries:
- pytorch
- numpy
- pandas
- pillow
- [picamera2](https://github.com/raspberrypi/picamera2)
- [Mbot Bridge API](https://mbot.robotics.umich.edu/docs/tutorials/bridge/)


## People

This project was created by:
- Raphael Alluson (allusson@umich.edu)
- Saipranav Janyavula (sjanyavu@umich.edu)
- Siddharth Kotapati (sidkot@umich.edu)

