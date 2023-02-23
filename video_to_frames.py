"""
Script for the extraction of the frames from the video of the ADVIO dataset
"""

import cv2
import os
import time

output_size = (224, 224)
file_extension = ".mov"

for i in range(1, 24):
    start_time = time.time()

    path = f"./data/advio-{i:02}/iphone/"
    filename = path + "frames" + file_extension

    if not os.path.exists(path + "frames/"):
        os.mkdir(path + "frames/")

    vid = cv2.VideoCapture(filename)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"""\nExtracting: 
  \t{filename}
  \t{fps} fps
  \t{frame_count} frames\n"""
    )

    if vid.isOpened():
        for i in range(frame_count):

            try:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    print("Quit signal cv2")
                    break

                ret, frame = vid.read()
                resized_frame = cv2.resize(
                    frame, output_size, interpolation=cv2.INTER_AREA
                )

                cv2.imwrite(path + "frames/" + f"{i}.jpg", resized_frame)
            except Exception as e:
                print(f"Exception occurred: {e}")
                break

            print(f"Completed {i/(frame_count+1)*100:1.2f} %", end="\r")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nIt took {elapsed_time/60:0.0f}min")
    print(f"Check the images at {path+'frames'}")

    vid.release()
