#!/usr/bin/ext python3.13
from pathlib import Path

import numpy as np
import cv2

from matplotlib import pyplot as plt

HOME_DIR = Path(__file__).parent

ORG_DIR = HOME_DIR / "original_videos" / "CiCfl-8.mp4"
vidcap = cv2.VideoCapture(ORG_DIR)

PX, PY = 80, 0

HGT = 800  # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WDT = 550  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

FPS = vidcap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG') #'XVID')  # Codec for MP4

output = cv2.VideoWriter(
    HOME_DIR / "processed_videos" / "CiCfl-8_diff_abs-Full.avi",
    fourcc,
    FPS,
    (WDT, HGT),
    0
)


def read_frame():
    flag, image = vidcap.read()
    if not flag:
        return flag, image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[PY:PY+HGT, PX:PX+WDT]
    image = np.array(255*(image / 255) ** 0.6, dtype="uint8")
    return flag, image


success, im1 = read_frame()
im2 = None
frame_counter = 0
MAX_FRAME = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Write FPS: {FPS}")

while success:
    # cv2.imwrite(f"frame{frame_counter}.png", image)  # save single frame
    im2 = im1
    success, im1 = read_frame()

    if not success:
        break

    print(f"Frame {frame_counter} / {MAX_FRAME}")

    diff = np.interp(np.abs(im1-0.95*im2), (0, 128), (0, 255)).astype("uint8")
    #cv2.imshow("diff", diff)
    #cv2.waitKey(1)

    output.write(diff)
    frame_counter += 1

    # if frame_counter > FPS:
    #     break

vidcap.release()
output.release()

cv2.destroyAllWindows()
