# Data

This folder contains **no raw data in the repository**. The GTSRB dataset is
downloaded and extracted automatically to `data/downloads/` the first time
you run `python -m src.train`.

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)** — Stefan Houben,
Johannes Stallkamp, Jan Salmen, Marc Schlipsing, Sven Iversen; IJCNN 2011.

- Official download host used: <https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/>
  (the benchmark organisers' official archive, linked from the GTSRB homepage
  <https://benchmark.ini.rub.de/gtsrb_dataset.html>)
- `GTSRB_Final_Training_Images.zip` — 39,209 labelled training images
- `GTSRB_Final_Test_Images.zip` — 12,630 test images
- `GTSRB_Final_Test_GT.zip` — official test labels (`GT-final_test.csv`)

## Layout after download/extraction

```
data/downloads/
└── GTSRB/
    ├── Final_Training/Images/000xx/*.ppm + GT-000xx.csv   (43 class folders)
    └── Final_Test/Images/*.ppm
        GT-final_test.csv                                   (Filename;...;ClassId)
```

Extracted size ~570 MB; it is gitignored and reproducible from the official
source. No fabricated or resized copies of the data are committed.
