#!/usr/bin/ext python3.13
from pathlib import Path

import numpy as np
import cv2

from matplotlib import pyplot as plt

HOME_DIR = Path(__file__).parent

ORG_DIR = HOME_DIR / "original_videos" / "CiCfl-8.mp4"
vidcap = cv2.VideoCapture(ORG_DIR)

PX, PY = 80, 0

HGT = 800
WDT = 550


def read_frame():
    flag, image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[PY:PY+HGT, PX:PX+WDT]
    image = np.array(255*(image / 255) ** 0.6, dtype="uint8")
    return flag, image


success, im1 = read_frame()
im2 = None
frame_counter = 0
MAX_FRAME = 20

while success:
    # cv2.imwrite(f"frame{frame_counter}.png", image)  # save single frame
    im2 = im1
    success, im1 = read_frame()

    if not success:
        break

    print(f"Frame {frame_counter} / {MAX_FRAME}")

    # plt.imshow(np.clip(im1-0.95*im2, a_min=0, a_max=255))
    diff = np.interp(im1-0.95*im2, (-128, 128), (0, 255))
    diff[0, 0] = 255
    plt.imshow(diff)
    # plt.imshow(im1)
    plt.show()

    frame_counter += 1
    if frame_counter > MAX_FRAME:
        break
