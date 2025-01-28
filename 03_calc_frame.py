#!/usr/bin/ext python3.13
from pathlib import Path

import numpy as np
import cv2

from scipy.signal import correlate
from matplotlib import pyplot as plt

HOME_DIR = Path(__file__).parent

PRS_DIR = HOME_DIR / "processed_videos" / "CiCfl-8_diff.avi"
vidcap = cv2.VideoCapture(PRS_DIR)

PX, PY = 80, 0

HGT = 800  # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WDT = 550  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

FPS = vidcap.get(cv2.CAP_PROP_FPS)


def read_frame():
    flag, image = vidcap.read()
    if not flag:
        return flag, image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return flag, image


# def read_frame():
#     flag, image = vidcap.read()
#     if not flag:
#         return flag, image
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     image = image[PY:PY+HGT, PX:PX+WDT]
#     image = np.array(255*(image / 255) ** 0.6, dtype="uint8")
#     return flag, image


def ZNCC(image_1, image_2):
    """Returns the Zero-Normalised Cross-Correlation of two images."""

    im1 = image_1.copy()
    im2 = image_2.copy()

    npx = im1.size

    sig1 = im1.mean()
    sig2 = im2.mean()

    std1 = im1.std()
    std2 = im2.std()

    xcorr = correlate(
        (im1 - sig1) / std1,
        (im2 - sig2) / std2,
        mode = 'full',
        method = 'auto'
    ) / npx

    return xcorr


def vel_field(im1, im2, size):
    ys = np.arange(0, im1.shape[0], size)
    xs = np.arange(0, im1.shape[1], size)

    dys = np.zeros((ys.size, xs.size))
    dxs = np.zeros((ys.size, xs.size))
    noise = np.zeros((ys.size, xs.size))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            wn1 = im1[y: y + size, x: x + size]
            wn2 = im2[y: y + size, x: x + size]

            xcorr = ZNCC(wn2, wn1)

            noise[iy, ix] = np.abs( xcorr.mean() / xcorr.std() )

            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(xcorr), xcorr.shape)
                - np.array([size, size])
                + 1
            )

    # draw velocity vectors from the center of each window
    ys = ys + size / 2
    xs = xs + size / 2
    return xs, ys, dxs, dys, noise


success, im1 = read_frame()
im2 = None
frame_counter = 0


while success:
    print(f"Reading frame: {frame_counter}")
    # cv2.imwrite(f"frame{frame_counter}.png", image)  # save single frame
    im2 = im1
    success, im1 = read_frame()

    if not success:
        break

    # imw = ZNCC(im1, im2)
    # plt.imshow(imw)
    # plt.show()

    xs, ys, dxs, dys, noise = vel_field(im1, im2, 25)

    norm_drs = np.sqrt(dxs ** 2 + dys ** 2)

    plt.quiver(
        xs,
        ys[::-1],
        dxs,
        -dys,
        noise,  # norm_drs,
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=0.25,
    )
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    # cv2.imshow("diff", diff)
    # cv2.waitKey(0)

    frame_counter += 1

    if frame_counter > FPS:
        break

vidcap.release()

cv2.destroyAllWindows()
