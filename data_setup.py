"""
Script for the syncing of the extracted frames (given by video_to_frames.py) 
and the timestap of the ADVIO dataset.
"""
import pandas as pd
import numpy as np
import os

def resampling(sample: np.ndarray, frq_new:int) -> np.ndarray:
    print(f"\tResampling at {frq_new}Hz")
    
    time_unit = 1/frq_new
    resampled = []

    old_time = 0
    current_time = old_time + time_unit

    for idx, s in enumerate(sample):
        if current_time < s and idx != 0:
            # memorizing the closest frame between the previous or the next
            if abs(sample[idx-1] - current_time) < abs(sample[idx] - current_time):
              resampled.append(sample[idx-1] )
            else:
              resampled.append(sample[idx])
                
            current_time += time_unit

    resampled = np.array(resampled)
    
    print(f"\tDone! {resampled.shape}")
    return resampled

input_file = "./data/advio-01/iphone/frames.csv"
output_file = "./data/advio-01/iphone/frames_cleaned.csv"
path_frames = "./data/advio-01/iphone/frames"

print("Set up...")

df = pd.read_csv(input_file, header=None)
list_dir = os.listdir(path_frames)
list_dir = np.array(list_dir)

# timestamps at 60Hz
ts = df.iloc[:, 0].to_numpy()

# resampling to 50Hz
tsr = resampling(ts, 50)

# selecting the frames that matches timestamp
frames = df.iloc[:, 1].to_numpy()
frames = frames[np.where(np.isin(ts,tsr))]
frame_names = np.array([f"{f}.jpg" for f in frames])

data = np.stack([tsr, frame_names])
df = pd.DataFrame(data.T, )

# saving df
df.to_csv(output_file, index=False, header=False)

print("Done!")
