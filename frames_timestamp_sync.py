"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset.
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

input_file = "./data/advio-01/iphone/frames.csv"
output_file = "./data/advio-01/iphone/frames_sanitize.csv"
path_frames = "./data/advio-01/iphone/frames"

print("Sanitizing...")

df = pd.read_csv(input_file, header=None)
list_dir = os.listdir(path_frames)
list_dir = np.array(list_dir)

frames = df.iloc[:, 1].to_numpy()

for i in tqdm(frames):
  if not np.any(list_dir == f"{i}.jpg"):

    # drop frames not present
    index_to_drop = df[df.iloc[:,1] == i].index[0]
    df = df.drop([index_to_drop], axis=0)

# saving sanitized df
df.to_csv(output_file, index=False)

print("Done!")
