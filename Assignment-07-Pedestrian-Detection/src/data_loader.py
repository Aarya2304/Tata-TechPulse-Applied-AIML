"""Dataset utilities: download, verification, extraction and ground-truth parsing.

Dataset: INRIA Person Dataset (Dalal & Triggs, CVPR 2005). The official
pascal.inrialpes.fr server is no longer reachable, so the complete original
archive is fetched from the University of Central Florida course mirror.
Only the annotated test split (Test/pos + Test/annotations + Test/neg) is
extracted and used for evaluation; the training crops are not needed because
OpenCV ships a pretrained people-detection SVM.
"""

from __future__ import annotations

import re
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from src import config

# ---------------------------------------------------------------------------
# Download / extraction
# ---------------------------------------------------------------------------


def download_inria_tar(dest: Path | None = None, url: str | None = None) -> Path:
    """Download the INRIA Person archive if it is not already present.

    Uses curl when available (progress + resume support) and falls back to
    urllib otherwise. The expected size is verified afterwards.
    """
    dest = Path(dest) if dest is not None else config.TAR_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url or config.TAR_URL
    if dest.exists() and dest.stat().st_size == config.TAR_SIZE_BYTES:
        return dest
    print(f"Downloading {config.DATASET_NAME} archive ({config.TAR_SIZE_BYTES/1e6:.0f} MB)...")
    import subprocess

    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-S", "-L", "-o", str(dest), "-C", "-", "--retry", "3", url],
            check=True,
        )
    else:  # pragma: no cover - fallback path
        urllib.request.urlretrieve(url, dest)
    size = dest.stat().st_size
    if size != config.TAR_SIZE_BYTES:
        raise IOError(f"Downloaded archive has {size} bytes, expected {config.TAR_SIZE_BYTES}")
    return dest


def _is_within(member: tarfile.TarInfo, base: str) -> bool:
    """Return True if *member* is a regular file under the archive path *base*."""
    parts = member.name.split("/")
    return not member.isdir() and len(parts) > 2 and f"{parts[0]}/{parts[1]}" == base


def extract_test_split(tar_path: Path | None = None, dest: Path | None = None) -> Path:
    """Extract only Test/pos, Test/annotations and Test/neg from the archive.

    Members that already exist on disk with the same size are skipped, so the
    call is idempotent and cheap to repeat.
    """
    tar_path = Path(tar_path) if tar_path is not None else config.TAR_PATH
    dest = Path(dest) if dest is not None else config.DATA_DIR
    if not tar_path.exists():
        download_inria_tar(tar_path)
    wanted = ("INRIAPerson/Test/pos", "INRIAPerson/Test/annotations", "INRIAPerson/Test/neg")
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf.getmembers():
            if not any(_is_within(member, w) for w in wanted):
                continue
            target = dest / member.name
            if target.exists() and target.stat().st_size == member.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return dest / "INRIAPerson" / "Test"


def ensure_dataset() -> Path:
    """Make sure the extracted test split exists, downloading if necessary."""
    if config.POS_DIR.is_dir() and config.ANNOTATIONS_DIR.is_dir():
        return config.INRIA_TEST_DIR
    extract_test_split()
    return config.INRIA_TEST_DIR


# ---------------------------------------------------------------------------
# Ground-truth parsing (PASCAL-style annotation files)
# ---------------------------------------------------------------------------
# Example line:
#   Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (142, 66) - (340, 646)
_BOX_RE = re.compile(
    r'Bounding box for object \d+ "(?P<label>[A-Za-z]+)" '
    r"\(Xmin, Ymin\) - \(Xmax, Ymax\) : "
    r"\((?P<xmin>\d+), (?P<ymin>\d+)\) - \((?P<xmax>\d+), (?P<ymax>\d+)\)"
)
_FILENAME_RE = re.compile(r'Image filename\s*:\s*"(?P<name>[^"]+)"')
PERSON_LABEL = "PASperson"


@dataclass
class GroundTruthImage:
    """Ground-truth annotation for a single evaluation image."""

    image_path: Path
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # x, y, w, h
    labels: list[str] = field(default_factory=list)

    @property
    def n_persons(self) -> int:
        return sum(1 for label in self.labels if label == PERSON_LABEL)


def parse_annotation_file(ann_path: Path, images_dir: Path | None = None) -> GroundTruthImage:
    """Parse one PASCAL-style INRIA annotation file into boxes.

    Coordinates are converted from (Xmin, Ymin)-(Xmax, Ymax) corners to
    (x, y, width, height) as used by OpenCV.
    """
    ann_path = Path(ann_path)
    text = ann_path.read_text(encoding="latin-1")

    # Prefer the image path recorded inside the annotation; fall back to the
    # same stem as the annotation file inside ``images_dir``.
    image_path: Path | None = None
    match = _FILENAME_RE.search(text)
    if match:
        candidate = config.PROJECT_ROOT / "data" / match.group("name")
        if candidate.exists():
            image_path = candidate
    if image_path is None and images_dir is not None:
        for suffix in (".png", ".jpg"):
            candidate = Path(images_dir) / (ann_path.stem + suffix)
            if candidate.exists():
                image_path = candidate
                break
    if image_path is None:
        raise FileNotFoundError(f"Image for annotation {ann_path.name} not found")

    boxes: list[tuple[int, int, int, int]] = []
    labels: list[str] = []
    for m in _BOX_RE.finditer(text):
        xmin, ymin = int(m.group("xmin")), int(m.group("ymin"))
        xmax, ymax = int(m.group("xmax")), int(m.group("ymax"))
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:  # skip degenerate annotations defensively
            continue
        boxes.append((xmin, ymin, w, h))
        labels.append(m.group("label"))
    return GroundTruthImage(image_path=image_path, boxes=boxes, labels=labels)


def load_ground_truth(images_dir: Path | None = None,
                      annotations_dir: Path | None = None) -> list[GroundTruthImage]:
    """Load and validate ground truth for every annotated image (sorted by name)."""
    images_dir = Path(images_dir) if images_dir is not None else config.POS_DIR
    annotations_dir = (
        Path(annotations_dir) if annotations_dir is not None else config.ANNOTATIONS_DIR
    )
    results: list[GroundTruthImage] = []
    for ann_path in sorted(annotations_dir.glob("*.txt")):
        try:
            results.append(parse_annotation_file(ann_path, images_dir))
        except FileNotFoundError:
            continue  # annotation without a matching image on disk
    return results


def list_negative_images() -> list[Path]:
    """Return pedestrian-free test images (used for false-positive checks)."""
    return sorted(config.NEG_DIR.glob("*.jpg")) + sorted(config.NEG_DIR.glob("*.png"))
