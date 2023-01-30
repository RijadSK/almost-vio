"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset.
"""
import pandas as pd
import numpy as np
import os

input_file = "./data/advio-01/iphone/frames.csv"
output_file = "./data/advio-01/iphone/frames_synced.csv"
path_frames = "./data/advio-01/iphone/frames"

print("Syncing...")

df = pd.read_csv(input_file)
list_dir = os.listdir(path_frames)
list_dir = np.sort(np.array(list_dir))

length = df.shape[0] if df.shape[0] < list_dir.shape[0] else list_dir.shape[0]
list_dir = list_dir[:length]
list_dir = list_dir.reshape(-1, 1)
df = df.to_numpy()
print(df[:2])
print(list_dir[:2])
df = np.concatenate([df, list_dir], axis=1)
df = pd.DataFrame(df)

df.to_csv(output_file, index=False)

print("Synced")
