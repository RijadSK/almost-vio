# Almost-VIO

Computer vision project about visual odometry

# Pipeline

```mermaid
graph LR;
    Images-->Conv_Odometry;
    Images-->Conv_Intertial;
    Conv_Odometry-->Dense_Odometry;
    Conv_Intertial-->Sequence_Inertial;
    Sequence_Inertial-->Dense_Odometry;
    Dense_Odometry-->Odometry;
```

# Dataset source

- https://vision.in.tum.de/data/datasets/visual-inertial-dataset (francesco)

# Reference

- http://mrsl.grasp.upenn.edu/loiannog/tutorial_ICRA2016/VO_Tutorial.pdf (Blandinie)
