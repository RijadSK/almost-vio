"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset (download link present in README.md).
"""
import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from matplotlib.image import imread


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

    inertial_data = np.zeros((sample[100:].shape[0], 2, 3))

    for sidx, s in enumerate(tqdm(sample[100:])):
        for idx, t in enumerate(times_inertials):
            if t > s and idx != 0:
                # store 2 inertial datapoints for each sample
                couple_inertials = np.stack(
                    [
                        inertials[idx - 1] / np.array([1, 9.81, 9.81]),
                        inertials[idx] / np.array([1, 9.81, 9.81]),
                    ]
                )
                inertial_data[sidx, :, :] = couple_inertials
                break

    return inertial_data


def sync_poses(sample, df):

    if not os.path.exists(path + "labels/"):
        os.mkdir(path + "labels/")

    labels = np.zeros((sample.shape[0], 2, 3))

    for sidx, s in enumerate(tqdm(sample)):
        for pose_idx, pose in enumerate(df.iloc[:, 0]):
            if pose > s and pose_idx != 0:
                # store 2 pose datapoints for each sample
                couple_truth = np.stack(
                    [
                        df.iloc[pose_idx - 1, 1:4].to_numpy(),
                        df.iloc[pose_idx, 1:4].to_numpy(),
                    ]
                )
                labels[sidx, :, :] = couple_truth
                break

    return labels


path = "./data/advio-01/iphone/"
in_frames = "./data/advio-01/iphone/frames.csv"
in_accellerometer = "./data/advio-01/iphone/accelerometer.csv"
in_labels = "./data/advio-01/ground-truth/pose.csv"

output_file = "./data/advio-01/iphone/frames_synced.csv"
path_frames = "./data/advio-01/iphone/frames"


df_frames = pd.read_csv(in_frames, header=None)
df_inertial = pd.read_csv(in_accellerometer, header=None)
df_label = pd.read_csv(in_labels, header=None)

list_dir = os.listdir(path_frames)
list_dir = np.array(list_dir)


print("Set up...")

# timestamps at 60Hz
ts = df_frames.iloc[:, 0].to_numpy()

# resampling to 50Hz
print(f"\nResampling")
tsr = resampling(ts, 50)
print(f"Done - {tsr.shape}")

# selecting the frames that matches timestamp
frames = df_frames.iloc[:, 1].to_numpy()
frames = frames[np.where(np.isin(ts, tsr))]
frame_names = np.array([f"{f}.jpg" for f in frames])

# extracting accellerometer data
print(f"\nExtracting inertial data")
inertials = extract_inertial_data(tsr, df_inertial)
np.save(path + "inertials.npy", inertials)
del inertials
print(f"Done")

# building labels
print(f"\nBuilding labels")
labels = sync_poses(tsr, df_label)
np.save(path + "labels.npy", labels)
del labels
print(f"Done")

data = np.stack([tsr, frame_names])
df = pd.DataFrame(
    data.T,
)

""" for idx, ts in enumerate(tsr):
    ts = df.iloc[idx,0]
    frame_name = df.iloc[idx,1]

    frames_path = path + "frames/"
    current_path = frames_path + f"{ts}/"
    print(current_path)
    os.mkdir(current_path)
    shutil.move(frames_path + frame_name, current_path + frame_name) """

# saving df
df.to_csv(output_file, index=False, header=False)

# PACKING DATA AS TENSORS


def normalize(data):
    data = data - data.min()
    data = data / data.max()
    return data


# image packing
path = "./data/advio-01/iphone/frames/"
df = pd.read_csv("./data/advio-01/iphone/frames_synced.csv", header=None)
files = df.iloc[100:, 1].to_numpy()
full_frames = np.zeros((13002 - 100, 224, 224, 3)).astype(np.float32)

for idx, file in enumerate(files):
    frame = imread(path + file)
    frame = normalize(frame)
    frame = frame.astype(np.float32)

    full_frames[idx, :, :, :] = frame
    del frame

np.save("./data/advio-01/iphone/frames.npy", full_frames)
print(full_frames.shape)
print("Done")

# packing buffer
print("\nPacking inertials")
path = "./data/advio-01/iphone/inertials/"
df = pd.read_csv("./data/advio-01/iphone/frames_synced.csv", header=None, dtype=str)
files = df.iloc[:, 0].to_numpy()
full_inertials = np.zeros((13002 - 100, 100, 2, 3)).astype(np.float32)

for idx in tqdm(range(100, files.shape[0])):

    inertial_buffer = np.zeros((100, 2, 3)).astype(np.float32)
    for delta in range(100, 0, -1):
        inertial = np.load(path + files[idx - delta] + ".npy")
        inertial /= np.array(
            [1, 9.81, 9.81]
        )  # fast patch to not normalized inertials on y and z
        inertial_buffer[100 - delta, :, :] = inertial

    full_inertials[idx - 100, :, :, :] = inertial_buffer
    del inertial_buffer

np.save("./data/advio-01/iphone/inertial_buffer.npy", full_inertials)
print(full_inertials.shape)
print("Done")

print("\nCompleted!")
