"""
Script that packs different scenes as a single the dataset 
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

scenes = [8, 9, 10, 13, 15, 16, 17, 19]

dataset_frames = None
dataset_buffer = None
dataset_labels = None
dataset_inertials = None

print("\nPacking the dataset:")
for i in tqdm(scenes):
  scene_name = f"advio-{i:02}"
  path = f"./data/{scene_name}/"
  path_iphone = path + "iphone/"

  in_synced = path_iphone + "frames_synced.csv"
  df_synced = pd.read_csv(in_synced, header=None)
  frames = df_synced.iloc[:, 1].to_numpy()

  labels = np.load(path_iphone + "labels.npy")
  inertials = np.load(path_iphone + "inertials.npy")
  inertial_buffer = np.load(path_iphone + "inertial_buffer.npy")

  dataset_frames = frames if dataset_frames is None else np.concatenate([dataset_frames, frames])
  dataset_buffer = inertial_buffer if dataset_buffer is None else np.concatenate([dataset_buffer, inertial_buffer])
  dataset_labels = labels if dataset_labels is None else np.concatenate([dataset_labels, labels])
  dataset_inertials = inertials if dataset_inertials is None else np.concatenate([dataset_inertials, inertials])


np.save("./dataset_frames", dataset_frames)
np.save("./dataset_buffer", dataset_buffer)
np.save("./dataset_labels", dataset_labels)
np.save("./dataset_inertials", dataset_inertials)

assert dataset_frames.shape[0] == dataset_buffer.shape[0] == dataset_labels.shape[0] == dataset_inertials.shape[0]

print(f"Completed -> {dataset_frames.shape}")