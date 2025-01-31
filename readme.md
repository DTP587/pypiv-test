# PIV with `opencv-python`, `openpiv-python` and `scipy`

Test of PIV capabilities using `opencv`, `openpiv-python` and `scipy`.


## Requirements

Check `requirements.txt` for a full list. `opencv-python` requires `ffmpeg` accessible in your environment. Recommend making a virtual environment:

```sh
python -m venv pypiv-test
source pypiv-test/bin/activate  # This line will differ on Windows 
```

Then let pip install the requirements.

```sh
pip install -r requirements.txt
```

Used python 3.13, but any version above 3 should work fine.


## File Description

01 - Testing import of video file a few frames at a time
02 - Move to comparing frames of the video file and extracting the output as a video
03 - Tracking the particles and extracting general direction trends with python
04 - Improve results of 03, with smoothing and using frame difference to infer speed
05 - Used openpiv builtins to test capabilities


## Precedents

 - [Loading files](https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames)
 - [Zero-Normalised Cross-Correlation of two images](https://en.wikipedia.org/wiki/Cross-correlation)
 - [PIV basics](https://openpiv.readthedocs.io/en/latest/src/piv_basics.html)
 - [Example from scratch](https://github.com/forughi/PIV/blob/master/Python_Code.py)
