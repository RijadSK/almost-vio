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
    
    if not os.path.exists(out_inertials):
        os.mkdir(out_inertials)

    times_inertials = df.iloc[:, 0].to_numpy()
    inertials = df.iloc[:, 1:].to_numpy()
    amount_samples = ts_samples.shape[0]
    inertial_data = np.zeros((amount_samples, 2, 3))

    for sidx, ts in enumerate(tqdm(ts_samples)):
        for idx, t in enumerate(times_inertials):
            if t > ts and idx != 0 and inertials[idx-1] < ts:
                # store 2 inertial datapoints for each sample
                couple_inertials = np.stack(
                    [
                        inertials[idx - 1] / np.array([1, 9.81, 9.81]),
                        inertials[idx] / np.array([1, 9.81, 9.81]),
                    ]
                )
                inertial_data[sidx, :, :] = couple_inertials
                break
            else:
                raise Exception("timestamp not syncend correctly synced with INERTIALS")

    return inertial_data


def sync_poses(ts_samples, df):

    if not os.path.exists(out_labels):
        os.mkdir(out_labels)

    ts_poses = df.iloc[:, 0]
    amount_samples = ts_samples.shape[0]
    labels = np.zeros((amount_samples, 2, 3))

    for sidx, ts in enumerate(tqdm(ts_samples)):
        for pidx, ts_pose in enumerate(ts_poses):
            if ts_pose > ts and pidx != 0 and ts_poses[pidx-1] < ts:
                # store 2 pose datapoints for each sample
                couple_truth = np.stack(
                    [
                        df.iloc[pidx - 1, 1:4].to_numpy(),
                        df.iloc[pidx, 1:4].to_numpy(),
                    ]
                )
                labels[sidx, :, :] = couple_truth
                break
            else:
                raise Exception("timestamp not syncend correctly synced with POSE LABELS")

    return labels

def normalize(data):
    data = data - data.min()
    data = data / data.max()
    return data

buffer_size=100
path = "./data/advio-01/iphone/"
in_frames = path + "frames.csv"
in_inertials = path + "accelerometer.csv"
in_labels = "./data/advio-01/ground-truth/pose.csv"

out_synced = path + "frames_synced.csv"
out_frames = path + "frames"
out_inertials = path + "inertials"
out_labels = path + "labels"

df_frames = pd.read_csv(in_frames, header=None)
df_inertial = pd.read_csv(in_inertials, header=None)
df_label = pd.read_csv(in_labels, header=None)

# timestamps at 60Hz
ts = df_frames.iloc[:, 0].to_numpy()

# resampling to 50Hz
print(f"\nResampling")
tsr = resampling(ts, 50)
print(f"Done - {tsr.shape}")

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

# PACKING DATA AS TENSORS


""" # image packing
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
print("Done") """

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

# selecting the frames that matches timestamp and saving
frames = df_frames.iloc[:, 1].to_numpy()
frames = frames[np.where(np.isin(ts, tsr))]
frame_names = np.array([f"{f}.jpg" for f in frames])
 
data = np.stack([tsr, frame_names])
df = pd.DataFrame(data.T, index=False, header=False)
print("pandas header/index correctness check")
print(data[:5])
print(df[:5])
df.to_csv(out_synced, index=False, header=False)

print("\nCompleted!")
