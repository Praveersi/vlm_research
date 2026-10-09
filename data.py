"""Load Sen1Floods11 hand-labeled S2 image/mask pairs and build 8-bit RGB."""
from pathlib import Path

import numpy as np
import tifffile

import config


def list_ids():
    """Sorted chip ids that have both an S2 image and a hand label."""
    s2 = {p.name.removesuffix("_S2Hand.tif") for p in config.S2_DIR.glob("*_S2Hand.tif")}
    lab = {p.name.removesuffix("_LabelHand.tif") for p in config.LABEL_DIR.glob("*_LabelHand.tif")}
    return sorted(s2 & lab)


def load_pair(chip_id):
    """Return (s2 [13,H,W] int16, mask [H,W] int16 with -1/0/1)."""
    s2 = tifffile.imread(config.S2_DIR / f"{chip_id}_S2Hand.tif")
    mask = tifffile.imread(config.LABEL_DIR / f"{chip_id}_LabelHand.tif")
    assert s2.shape[0] == 13 and s2.shape[1:] == mask.shape, (chip_id, s2.shape, mask.shape)
    return s2, mask


def to_rgb8(s2, percentile=config.RGB_PERCENTILE):
    """Bands 4/3/2 -> [H,W,3] uint8.

    One scale factor (the percentile over all three bands' valid pixels) keeps the
    true-colour balance. Pixels that are 0 in every band are sensor no-data and are
    excluded from the percentile so black borders don't skew the stretch.
    """
    rgb = s2[list(config.RGB_BAND_IDX)].astype(np.float32).transpose(1, 2, 0)
    valid = (rgb > 0).any(axis=-1)
    scale = np.percentile(rgb[valid], percentile) if valid.any() else 1.0
    rgb = np.clip(rgb / max(scale, 1e-6), 0, 1)
    return (rgb * 255).round().astype(np.uint8)


def water_fraction(mask):
    """Water pixels / valid (non -1) pixels."""
    valid = mask != -1
    return float((mask[valid] == 1).sum() / valid.sum()) if valid.any() else 0.0


def save_pair_figure(ids, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#7f7f7f", "#d9c27a", "#1f5fbf"])  # -1 no data, 0 dry, 1 water
    fig, axes = plt.subplots(len(ids), 3, figsize=(10, 3.4 * len(ids)))
    for row, cid in zip(axes, ids):
        s2, mask = load_pair(cid)
        rgb = to_rgb8(s2)
        row[0].imshow(rgb)
        row[0].set_title(f"{cid} RGB (B4/B3/B2)", fontsize=9)
        row[1].imshow(mask, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        row[1].set_title(f"mask  water={water_fraction(mask):.1%} of valid", fontsize=9)
        row[2].imshow(rgb)
        row[2].contour(mask == 1, levels=[0.5], colors="red", linewidths=0.6)
        row[2].set_title("overlay (red = water edge)", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.text(0.5, 0.005, "mask: grey = no data/cloud (-1), tan = dry (0), blue = water (1)", ha="center")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ids = list_ids()
    print(f"matched image/mask pairs: {len(ids)}")
    rng = np.random.default_rng(config.SEED)
    sample = sorted(rng.choice(ids, size=5, replace=False).tolist())
    out = config.FIGURES_DIR / "phase0_pairs.png"
    save_pair_figure(sample, out)
    print(f"sample: {sample}\nsaved {out}")
