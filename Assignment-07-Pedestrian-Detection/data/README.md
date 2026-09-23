# Data

This folder holds the **INRIA Person Dataset** (Dalal & Triggs, CVPR 2005),
used to evaluate the classical OpenCV HOG + SVM pedestrian detector.

## Source / provenance

- Original publication: N. Dalal and B. Triggs, "Histograms of Oriented
  Gradients for Human Detection", CVPR 2005.
- Official site `http://pascal.inrialpes.fr/data/human/` is offline, so the
  complete original archive (`INRIAPerson.tar`, 1,016,094,720 bytes) is
  downloaded from the University of Central Florida course mirror:
  <http://cs.ucf.edu/courses/cap6412/fall2009/misc/INRIAPerson.tar>
- `python -m src.main --evaluate` (and the test-suite helper
  `src.data_loader.ensure_dataset()`) download and extract the archive
  automatically when it is missing.

## Layout after extraction

```
data/
├── downloads/INRIAPerson.tar      # original archive (~969 MB), gitignored
└── INRIAPerson/Test/              # extracted test split (~276 MB), gitignored
    ├── pos/                       # 288 test images with pedestrians (.png)
    ├── annotations/               # 288 PASCAL-style .txt files
    │   └── crop_000001.txt        # "Bounding box for object N "PASperson"
    │                              #  (Xmin, Ymin) - (Xmax, Ymax) : (x1, y1) - (x2, y2)"
    └── neg/                       # 453 pedestrian-free images (.jpg)
```

- 288 annotated test images, **589 ground-truth `PASperson` boxes** in total
  (measured from the extracted files; up to 16 persons in one image).
- Coordinates are inclusive pixel corners; the parser converts them to
  OpenCV's (x, y, width, height) convention.
- The `Train/` split of the archive is NOT extracted: OpenCV's
  `HOGDescriptor_getDefaultPeopleDetector()` was already trained on it, and
  this assignment uses that pretrained SVM (no training is performed).
