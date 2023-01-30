import cv2
import os
import time
from tqdm import tqdm

start_time = time.time()

path = "./data/advio-02/iphone/"
file_extension = ".mov"
filename = path + "frames"+ file_extension

vid = cv2.VideoCapture(filename)
fps = vid.get(cv2.CAP_PROP_FPS)
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nExtracting {filename} ...\n")
while vid.isOpened():
    for i in tqdm(range(frame_count+1)):
      ret,frame = vid.read()

      if not os.path.exists(path+"frames/"):
          os.mkdir(path+"frames/")
      
      try:
        cv2.imwrite(path+"frames/"+f"frame_{i:05}.jpg", frame)
      except Exception as e:
        break

      if cv2.waitKey(10) & 0xFF == ord('q'):
          break


end_time = time.time()
elapsed_time = end_time - start_time
print(f'\nIt took {elapsed_time}s to extract {frame_count/fps}s of video')
print(f"Done, check the images at {path+'frames'}")
vid.release()