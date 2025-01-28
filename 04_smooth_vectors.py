#!/usr/bin/ext python3.13
from pathlib import Path

import numpy as np
import cv2

from scipy.signal import correlate
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

HOME_DIR = Path(__file__).parent
OUT_DIR = HOME_DIR / "outputs"

VID_DIR = HOME_DIR / "processed_videos" / "CiCfl-8_diff.avi"
vidcap = cv2.VideoCapture(VID_DIR)

ORG_DIR = HOME_DIR / "original_videos" / "CiCfl-8.mp4"
orgcap = cv2.VideoCapture(ORG_DIR)


PX, PY = 80, 0

HGT = 800
WDT = 550


FPS = vidcap.get(cv2.CAP_PROP_FPS)
MAX_FRAME = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for MP4

output = cv2.VideoWriter(
    OUT_DIR / "CiCfl-8-diff.speed.avi",
    fourcc,
    FPS,
    (1100, 1600),  # (784, 1142),
    True  # Set to True if using color frames, False for grayscale
)


def read_org_frame():
    flag, image = orgcap.read()
    if not flag:
        return flag, image
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image[PY:PY+HGT, PX:PX+WDT]
    #image = np.array(255*(image / 255) ** 1.2, dtype="uint8")  # 0.6
    return flag, image


def read_vid_frame():
    flag, image = vidcap.read()
    if not flag:
        return flag, image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return flag, image


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


# Inner xcorr Kernel
kernel = np.ones((5, 5)) / 25.

# Gaussian Kernel
kernem = np.array([[1., 2., 1.],
                   [2., 4., 2.],
                   [1., 2., 1.]]) / 16.

# Outer velocity Kernel smooth
# kernem = np.array([[0.5,  1., 0.5],
#                    [1.0, 10., 1.0],
#                    [0.5,  1., 0.5]]) / 16.

def vel_field(im1, im2, size):
    """Calculate velocity field."""
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

            xcorr = convolve(xcorr, kernel)

            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(xcorr), xcorr.shape)
                - np.array([size, size])
                + 1
            )

        #     plt.imshow(xcorr)
        #     plt.show()
        # assert False

    # Smoothing kernel
    dys = convolve(dys, kernem)
    dxs = convolve(dxs, kernem)

    # draw velocity vectors from the center of each window
    ys = ys + size / 2
    xs = xs + size / 2
    return xs, ys, dxs, dys


_, imO = read_org_frame()
success, im1 = read_vid_frame()

im2 = None
frame_counter = 0

window_size = 25
idy = int(HGT/window_size)
idx = int(WDT/window_size)

res_dxs = np.ndarray([MAX_FRAME-1, idy, idx])
res_dys = np.ndarray([MAX_FRAME-1, idy, idx])
res_norm = np.ndarray([MAX_FRAME-1, idy, idx])


while success:
    print(f"Reading frame: {frame_counter} / {MAX_FRAME - 1}")
    # cv2.imwrite(f"frame{frame_counter}.png", image)  # save single frame
    im2 = im1
    success, imO = read_org_frame()
    success, im1 = read_vid_frame()

    if not success:
        break

    xs, ys, dxs, dys = vel_field(im1, im2, 25)

    norm_drs = np.sqrt(dxs ** 2 + dys ** 2)

    # output for saving numpy array
    res_dxs[frame_counter] = dxs
    res_dys[frame_counter] = dys
    res_norm[frame_counter] = norm_drs

    fig, ax = plt.subplots()
    ax.quiver(
        xs[3:-2],
        ys[::-1][2:-4],
        -dxs[2:-4, 3:-2],
        -dys[2:-4, 3:-2],
        0.5*norm_drs[2:-4, 3:-2],
        cmap="inferno", #"Blues",  #"plasma",
        angles="xy",
        scale_units="xy",
        scale=0.25,
    )
    ax.imshow(imO)
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

    # print(img_plot.shape)
    output.write(img_plot)
    plt.close()

    frame_counter += 1

    if frame_counter > FPS:
        break

np.savez(OUT_DIR / 'results.npz', dxs=res_dxs, dys=res_dys, norm=res_norm)

vidcap.release()
orgcap.release()
output.release()

cv2.destroyAllWindows()
