# Models

This assignment uses **no trained model artifacts**.

The detector is OpenCV's `cv2.HOGDescriptor` combined with the pretrained
people-detection SVM returned by
`cv2.HOGDescriptor_getDefaultPeopleDetector()` — a 3,780-dimensional linear
SVM over HOG features of a 64x128 detection window, shipped inside the
`opencv-python` package itself and trained by OpenCV on the INRIA Person
*training* split.

No weights are fine-tuned, re-trained, or downloaded here, so there is
nothing to commit under this folder. The "model" is fully described by:

- the installed OpenCV version (see `requirements.txt`), and
- the detector configuration in `src/config.py`
  (`WIN_STRIDE`, `PADDING`, `SCALE`, `HIT_THRESHOLD`,
  `CONFIDENCE_THRESHOLD`, `NMS_THRESHOLD`).
