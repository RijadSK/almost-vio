"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset (download link present in README.md).
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib.image import imread


def resampling(ts_samples: np.ndarray, frq_new: int) -> np.ndarray:
    time_unit = 1 / frq_new
    resampled = []
    old_time = 0
    current_time = old_time + time_unit

    for idx, ts in enumerate(ts_samples):
        if current_time < ts and idx != 0:
            # memorizing the closest frame between the previous or the next
            if abs(ts_samples[idx - 1] - current_time) < abs(ts_samples[idx] - current_time):
                resampled.append(ts_samples[idx - 1])
            else:
                resampled.append(ts_samples[idx])

            current_time += time_unit

    return np.array(resampled)


def extract_inertial_data(ts_samples, df):
    ts_inertials = df.iloc[:, 0].to_numpy()
    inertials = df.iloc[:, 1:].to_numpy()
    amount_samples = ts_samples.shape[0]
    inertial_data = np.zeros((amount_samples, 2, 3))

    for sidx, ts in enumerate(tqdm(ts_samples)):
        for idx, t in enumerate(ts_inertials):
            if idx != 0 and ts_inertials[idx-1] <= ts and t > ts:
                # store 2 inertial datapoints for each sample
                couple_inertials = np.stack(
                    [
                        inertials[idx - 1] / np.array([1, 9.81, 9.81]),
                        inertials[idx] / np.array([1, 9.81, 9.81]),
                    ]
                )
                inertial_data[sidx, :, :] = couple_inertials
                break
            elif idx != 0 and ts_inertials[idx-1] > ts:
                raise Exception("timestamp not syncend correctly synced with INERTIALS")

    return inertial_data


def sync_poses(ts_samples, df):
    ts_poses = df.iloc[:, 0].to_numpy()
    poses = df.iloc[:, 1:4].to_numpy()
    amount_samples = ts_samples.shape[0]
    labels = np.zeros((amount_samples, 2, 3))

    for sidx, ts in enumerate(tqdm(ts_samples)):
        for pidx, ts_pose in enumerate(ts_poses):
            if pidx != 0 and ts_poses[pidx-1] <= ts and ts_pose > ts:
                # store 2 pose datapoints for each sample
                couple_truth = np.stack(
                    [
                        poses[pidx - 1],
                        poses[pidx],
                    ]
                )
                labels[sidx, :, :] = couple_truth
                break
            elif pidx != 0 and ts_poses[pidx-1] > ts:
                raise Exception("timestamp not syncend correctly synced with POSE LABELS")

    return labels

def pack_buffer(inertials, buffer_size):
    amount_samples = inertials.shape[0]
    full_inertials = np.zeros((amount_samples - buffer_size, buffer_size, 2, 3)).astype(np.float32)

    for idx in tqdm(range(buffer_size, amount_samples)):
        full_inertials[idx - buffer_size] = inertials[idx - buffer_size : idx]

    return full_inertials

def normalize(data):
    data = data - data.min()
    data = data / data.max()
    return data

def diff_frames(path, scene_name, frames):
    path = path + "frames/"

    amount_samples = frames.shape[0]

    for i in range(amount_samples-1):
        frame = imread(path + f"{frames[i]}.jpg")
        next_frame = imread(path + f"{frames[i+1]}.jpg")

        diff = next_frame - frame
        diff = np.moveaxis(diff, -1, 0) # (channels, heigth, width)
        np.save(path + f"{scene_name}_{i+1}", diff)


buffer_size=100
scene_name = "advio-01"
path = f"./data/{scene_name}/iphone/"
in_frames = path + "frames.csv"
in_inertials = path + "accelerometer.csv"
in_labels = "./data/advio-01/ground-truth/pose.csv"

out_synced = path + "frames_synced.csv"

df_frames = pd.read_csv(in_frames, header=None)
df_inertial = pd.read_csv(in_inertials, header=None)
df_label = pd.read_csv(in_labels, header=None)

# timestamps at 60Hz
ts = df_frames.iloc[:, 0].to_numpy()

# resampling to 50Hz
print(f"\nResampling")
tsr = resampling(ts, 50)
print(f"Done - {tsr.shape}")

# selecting the frames that matches timestamp
frames = df_frames.iloc[:, 1].to_numpy()
frames = frames[np.where(np.isin(ts, tsr))]

# diff frames
print(f"\nBuilding labels")
diff_frames(path, scene_name, frames)
print(f"Done")

# since diff consider a subset of frames, i have to crop the others to match shapes
frames = frames[1:]
tsr = tsr[1:]

# building labels
print(f"\nBuilding labels")
labels = sync_poses(tsr, df_label)
print(f"Done")

# extracting accellerometer data
print(f"\nExtracting inertial data")
inertials = extract_inertial_data(tsr, df_inertial)
print(f"Done")

# packing buffer
print("\nBuilding buffer inertials")
full_inertials = pack_buffer(inertials, buffer_size)
np.save(path + "inertial_buffer.npy", full_inertials)
print(f"Done - {full_inertials.shape}")

# since buffer consider a subset of data, i have to crop the others to match shapes
labels = labels[buffer_size:]
inertials = inertials[buffer_size:]

np.save(path + "labels.npy", labels)
np.save(path + "inertials.npy", inertials)

del full_inertials
del inertials
del labels

# saving
frame_names = np.array([f"{scene_name}_{f}.jpg" for f in frames])
data = np.stack([tsr, frame_names])
df = pd.DataFrame(data.T)
df.to_csv(out_synced, index=False, header=False)

print("\nCompleted!")
