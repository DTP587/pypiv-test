#!/usr/bin/ext python3.13
from pathlib import Path

import numpy as np
import cv2

from matplotlib import pyplot as plt
from openpiv import pyprocess, preprocess, filters, validation

HOME_DIR = Path(__file__).parent
OUT_DIR = HOME_DIR / "outputs"

ORG_DIR = HOME_DIR / "original_videos" / "CiCfl-8.mp4"
vidcap = cv2.VideoCapture(ORG_DIR)


PX, PY = 80, 0

HGT = 800
WDT = 550


FPS = vidcap.get(cv2.CAP_PROP_FPS)
MAX_FRAME = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# Write video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for MP4

output = cv2.VideoWriter(
    OUT_DIR / "CiCfl-8.speed.avi",
    fourcc,
    FPS,
    (1100, 1600),  # (784, 1142),
    True  # Set to True if using color frames, False for grayscale
)


def read_frame():
    flag, image = vidcap.read()
    if not flag:
        return flag, image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[PY:PY+HGT, PX:PX+WDT]
    image = np.array(255*(image / 255) ** 1.2, dtype="uint8")  # 0.6
    return flag, image


success, im1 = read_frame()

im2 = None
frame_counter = 0

winsize = 40 # 30 # pixels, interrogation window size in frame A
searchsize = 55 # 40  # pixels, search in image B
overlap = 35 # 15 # pixels, 50% overlap
dt = 1. / 240. # sec, time interval between pulses

xs, ys = pyprocess.get_coordinates(
    image_size=im1.shape,
    search_area_size=searchsize,
    overlap=overlap
) # if searchsize error, check image is grayscale

while success:
    print(f"Reading frame: {frame_counter} / {MAX_FRAME - 1}")

    im1 = preprocess.standardize_array(im1)

    im2 = im1
    success, im1 = read_frame()

    if not success:
        break

    u0, v0, s2n = pyprocess.extended_search_area_piv(
        im1,
        im2,
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak'
    )

    u0, v0 = filters.replace_outliers(
        u0,
        v0,
        validation.sig2noise_val(s2n, threshold=1.0)
    )

    u0, v0 = filters.gaussian(u0, v0, 1.02)

    fig, ax = plt.subplots()
    ax.quiver(
        xs,
        ys[::-1],
        -u0*dt,
        -v0*dt,
        s2n,
        cmap="inferno", #"Blues",  #"plasma",
        angles="xy",
        scale_units="xy",
        scale=0.25,
    )

    ax.imshow(im1)
    ax.set_aspect('equal')
    fig.patch.set_visible(False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    fig.set_size_inches(WDT/100, HGT/100)  # (WDT/140, HGT/140)
    plt.axis('off')
    # plt.show()
    # plt.close()

    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

    output.write(img_plot)
    plt.close()

    frame_counter += 1

    if frame_counter > FPS:
        break


vidcap.release()
output.release()

cv2.destroyAllWindows()
