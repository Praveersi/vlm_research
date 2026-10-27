"""Central configuration: paths, models, prompt, seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Paths ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEN1FLOODS_DIR = RAW_DIR / "sen1floods11_v1.1"
S2_DIR = SEN1FLOODS_DIR / "data" / "S2L1CHand"   # 13-band L1C, files named <id>_S2Hand.tif
LABEL_DIR = SEN1FLOODS_DIR / "data" / "LabelHand"  # files named <id>_LabelHand.tif
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_128_DIR = DATA_DIR / "processed_128"
PROCESSED_128_COVERED_DIR = DATA_DIR / "processed_128_covered"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- Image / Tile Configuration ---
IMAGE_SIZE_ORIGINAL = 512
IMAGE_SIZE_TARGET = 128


# --- Dataset ---
HF_REPO_ID = "blumenstiel/Sen1Floods11"
HF_FILENAME = "sen1floods11_v1_1.tar.gz"

# Sentinel-2 L1C band order in the GeoTIFF (from its GDAL_METADATA):
# B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 -> RGB = B4,B3,B2 = indices 3,2,1
RGB_BAND_IDX = (3, 2, 1)
RGB_PERCENTILE = 98

# --- Reproducibility ---
SEED = 42
