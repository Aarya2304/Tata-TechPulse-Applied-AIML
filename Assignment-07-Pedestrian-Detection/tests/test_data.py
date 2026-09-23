"""Tests for dataset utilities (synthetic annotations + real data when present)."""

from __future__ import annotations

import pytest

from src import config
from src.data_loader import GroundTruthImage, list_negative_images, parse_annotation_file

SAMPLE_ANNOTATION = '''# PASCAL Annotation Version 1.00

Image filename : "Test/pos/crop_000001.png"
Image size (X x Y x C) : 491 x 720 x 3
Database : "The INRIA Rho^ne-Alpes Annotated Person Database"
Objects with ground truth : 2 { "PASperson" "PASperson" }

# Details for object 1 ("PASperson")
Original label for object 1 "PASperson" : "UprightPerson"
Center point on object 1 "PASperson" (X, Y) : (267, 111)
Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (142, 66) - (340, 646)

# Details for object 2 ("PASperson")
Original label for object 2 "PASperson" : "UprightPerson"
Center point on object 2 "PASperson" (X, Y) : (60, 100)
Bounding box for object 2 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (10, 50) - (110, 250)
'''


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A tiny fake dataset tree: one image + one annotation."""
    pos_dir = tmp_path / "Test" / "pos"
    ann_dir = tmp_path / "Test" / "annotations"
    pos_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    # minimal valid PNG (1x1) saved via cv2
    import numpy as np
    import cv2

    img = np.zeros((720, 491, 3), dtype=np.uint8)
    cv2.imwrite(str(pos_dir / "crop_000001.png"), img)
    ann = ann_dir / "crop_000001.txt"
    ann.write_text(SAMPLE_ANNOTATION, encoding="latin-1")
    return tmp_path / "Test"


def test_parse_annotation_converts_corners_to_xywh(synthetic_dataset):
    gt = parse_annotation_file(synthetic_dataset / "annotations" / "crop_000001.txt",
                               synthetic_dataset / "pos")
    assert gt.image_path.name == "crop_000001.png"
    assert gt.boxes == [(142, 66, 198, 580), (10, 50, 100, 200)]  # w=x2-x1, h=y2-y1
    assert gt.labels == ["PASperson", "PASperson"]
    assert gt.n_persons == 2


def test_parse_annotation_skips_degenerate_boxes(tmp_path):
    """A zero-width GT box is dropped; the valid one is kept."""
    ann = tmp_path / "crop_bad.txt"
    text = SAMPLE_ANNOTATION.replace("(10, 50) - (110, 250)", "(10, 50) - (10, 250)")
    ann.write_text(text, encoding="latin-1")
    # create a matching image so parsing reaches the box stage
    import cv2
    import numpy as np

    cv2.imwrite(str(tmp_path / "crop_bad.png"), np.zeros((720, 491, 3), dtype=np.uint8))
    gt = parse_annotation_file(ann, tmp_path)
    assert gt.boxes == [(142, 66, 198, 580)]  # degenerate box removed
    assert gt.n_persons == 1


def test_ground_truth_image_defaults():
    gt = GroundTruthImage(image_path=__import__("pathlib").Path("x.png"))
    assert gt.boxes == [] and gt.labels == [] and gt.n_persons == 0


def test_real_dataset_loads_when_present():
    """If the dataset is extracted, validate its real headline statistics."""
    if not (config.POS_DIR.is_dir() and config.ANNOTATIONS_DIR.is_dir()):
        pytest.skip("INRIA test split not extracted yet (run: python -m src.main --evaluate)")
    from src.data_loader import load_ground_truth

    gt_images = load_ground_truth()
    assert len(gt_images) == 288
    total_persons = sum(g.n_persons for g in gt_images)
    assert total_persons == 589
    assert all(g.boxes for g in gt_images)
    # boxes must be inside the image
    import cv2

    for g in gt_images[:5]:
        img = cv2.imread(str(g.image_path))
        h, w = img.shape[:2]
        for x, y, bw, bh in g.boxes:
            assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h


def test_negative_images_listing():
    """Pedestrian-free image listing is valid when the dataset is present."""
    if not config.NEG_DIR.is_dir():
        pytest.skip("INRIA negative split not extracted yet")
    negs = list_negative_images()
    assert len(negs) >= 400
    assert all(p.suffix.lower() in {".jpg", ".png"} for p in negs)
