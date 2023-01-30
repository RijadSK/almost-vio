# Almost-VIO

Computer vision project about visual odometry

# Pipeline

```mermaid
graph LR;
    Images-->Conv2d_Odometry;
    Images-->Conv2d_Intertial;
    Conv2d_Odometry-->Dense_Odometry;
    Conv2d_Intertial-->Conv1d_Inertial;
    Inertial_Sequence-->Conv1d_Inertial;
    Conv1d_Inertial-->Dense_Odometry;
    Dense_Odometry-->Odometry;
```

# Dataset

The data are 'corridor 1' of 'Euroc / DSO 512x512 dataset' from: https://vision.in.tum.de/data/datasets/visual-inertial-dataset (francesco)

source: https://github.com/AaltoVision/ADVIO (francesco)

# Reference

- http://mrsl.grasp.upenn.edu/loiannog/tutorial_ICRA2016/VO_Tutorial.pdf (Blandinie)
