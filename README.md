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

source: https://github.com/AaltoVision/ADVIO (francesco)

## Utils

- **video_to_frame.py**: it extracts the frames from the video in order to prepare the data
- **sanitize_labels.py**: it checks if the frames extrated by _video_to_frame.py_ are present in the labels(_frames.csv_) in order to have coherent frames and labels

# Reference

- http://mrsl.grasp.upenn.edu/loiannog/tutorial_ICRA2016/VO_Tutorial.pdf (Blandinie)
