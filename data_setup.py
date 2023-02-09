"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset.
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def resampling(sample: np.ndarray, frq_new: int) -> np.ndarray:
    time_unit = 1 / frq_new
    resampled = []
    old_time = 0
    current_time = old_time + time_unit

    for idx, s in enumerate(sample):
        if current_time < s and idx != 0:
            # memorizing the closest frame between the previous or the next
            if abs(sample[idx - 1] - current_time) < abs(sample[idx] - current_time):
                resampled.append(sample[idx - 1])
            else:
                resampled.append(sample[idx])

            current_time += time_unit

    return np.array(resampled)


def extract_inertial_data(sample, df):
    times_inertials = df.iloc[:, 0].to_numpy()
    inertials = df.iloc[:, 1:].to_numpy()

    if not os.path.exists(path + "inertials/"):
        os.mkdir(path + "inertials/")

    for s in tqdm(sample):
        for idx, t in enumerate(times_inertials):
            if t < s and idx != 0:
                # store 2 inertial datapoints for each sample
                couple_inertials = np.stack([inertials[idx - 1], inertials[idx]])
                np.save(f"{path}inertials/{s}", couple_inertials)


path = "./data/advio-01/iphone/"
in_file_frames = "./data/advio-01/iphone/frames.csv"
in_file_accellerometer = "./data/advio-01/iphone/accelerometer.csv"
output_file = "./data/advio-01/iphone/frames_cleaned.csv"
path_frames = "./data/advio-01/iphone/frames"

print("Set up...")

df_frames = pd.read_csv(in_file_frames, header=None)
df_inertial = pd.read_csv(in_file_accellerometer, header=None)
list_dir = os.listdir(path_frames)
list_dir = np.array(list_dir)

# timestamps at 60Hz
ts = df_frames.iloc[:, 0].to_numpy()

# resampling to 50Hz
print(f"\nResampling")
tsr = resampling(ts, 50)
print(f"Done! {tsr.shape}")

# selecting the frames that matches timestamp
frames = df_frames.iloc[:, 1].to_numpy()
frames = frames[np.where(np.isin(ts, tsr))]
frame_names = np.array([f"{f}.jpg" for f in frames])

# extracting accellerometer data
print(f"\nExtracting inertial data")
inertial_data = extract_inertial_data(tsr, df_inertial)
print(f"Done! {inertial_data.shape}")

# saving df
data = np.stack([tsr, frame_names])
df = pd.DataFrame(
    data.T,
)
df.to_csv(output_file, index=False, header=False)

print("Done!")
