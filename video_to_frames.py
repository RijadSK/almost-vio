import cv2
import os
import time
from tqdm import tqdm

start_time = time.time()

path = "./data/advio-03/iphone/"
file_extension = ".mov"
filename = path + "frames"+ file_extension

if not os.path.exists(path+"frames/"):
    os.mkdir(path+"frames/")

vid = cv2.VideoCapture(filename)
fps = vid.get(cv2.CAP_PROP_FPS)
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nExtracting {filename} ...\n")

if vid.isOpened():
    for i in range(frame_count):
                    
        try:
          if cv2.waitKey(10) & 0xFF == ord('q'):
            break

          ret,frame = vid.read()
          
          cv2.imwrite(path+"frames/"+f"frame_{i:05}.jpg", frame)
        except Exception as e:
          break

        print(f"Completed {i/(frame_count+1)*100:1.2f} %", end="\r")

end_time = time.time()
elapsed_time = end_time - start_time

print(f'\nIt took {elapsed_time/60:0.0m}min the video')
print(f"Check the images at {path+'frames'}")

vid.release()