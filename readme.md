# PIV with `opencv-python`, `open-piv` and `scipy`

This is a little test of PIV capabilities using `opencv`, `ffmpeg`, `scipy` and `open-piv`.

01 - testing import of video file a few frames at a time
02 - move to comparing frames of the video file and extracting the output as a video
03 - tracking the particles and extracting general direction trends without any overheads
04 - try some hacky methods of smoothing the vectors using the difference between frames to highlight change

## Precedents

 - [Loading files](https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames)
 - [Zero-Normalised Cross-Correlation of two images](https://en.wikipedia.org/wiki/Cross-correlation)
 - [PIV basics](https://openpiv.readthedocs.io/en/latest/src/piv_basics.html)
 - [Example from scratch](https://github.com/forughi/PIV/blob/master/Python_Code.py)
