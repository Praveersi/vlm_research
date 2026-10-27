"""C2: river-vs-flood test set (permanent water vs. flood water).

Uses the Sen1Floods11 JRCWaterHand layer (binary: 1 = JRC permanent water, 0 = not)
to split hand-labelled water into
  permanent  = JRC == 1
  flood_only = (LabelHand == 1) & ~permanent

Tiles chips into 128x128 exactly like oversample_data.py (same tile ids), and samples
  permanent_only: permanent_water_fraction >= 0.02 (JRC permanent AND LabelHand water,
                  i.e. the permanent water is visible in this image), water_fraction >= 0.02
                  and flood_only_fraction < 0.02
  mixed         : permanent_fraction >= 0.02 and flood_only_fraction >= 0.05
(seed 42 per group, at most 3 tiles per parent chip per group, up to 150 tiles per group).
Parent chips in EXCLUDED_PARENTS are dropped from both groups before sampling.
For mixed tiles, only flood_only pixels are replaced with sampled dry pixels
(generate_covered_images.camouflage_image); permanent water is left untouched.

Outputs: data/c2/{orig,flood_only_removed}/<tile_id>.png, data/c2/metadata.json,
results/figures/c2_examples.png
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tifffile
from PIL import Image

import config
from data import list_ids, load_pair, to_rgb8
from generate_covered_images import camouflage_image

C2_DIR = config.DATA_DIR / "c2"
JRC_DIR = config.SEN1FLOODS_DIR / "data" / "JRCWaterHand"
TILE = config.IMAGE_SIZE_TARGET
MAX_NODATA = 0.5
MAX_PER_PARENT = 3
MAX_PER_GROUP = 150
MIN_PERMANENT = 0.02
MAX_FLOOD_ONLY_PERMANENT_GROUP = 0.02
MIN_WATER_PERMANENT_GROUP = 0.02
MIN_FLOOD_ONLY_MIXED = 0.05
# Removed after visual check: India_591317 tiles are featureless grey water; Ghana_866994
# is hazy and JRC marks whole tiles permanent where LabelHand shows dry land.
EXCLUDED_PARENTS = {"India_591317", "Ghana_866994"}

# Same wording as the pilot (processed_128/metadata.json) so the existing notebook runs
# unchanged, plus flood- vs. water-specific Yes/No questions: a river IS surface water,
# so "Yes" to the pilot POPE prompt is correct on permanent_only tiles.
PILOT_META = config.PROCESSED_128_DIR / "metadata.json"
EXTRA_PROMPTS = {
    "yn_flood": "Is there flooding in this satellite image? Answer only Yes or No.",
    "yn_water": "Is there any water (river, lake, or flood) in this satellite image? Answer only Yes or No.",
}


def load_jrc(chip_id):
    return tifffile.imread(JRC_DIR / f"{chip_id}_JRCWaterHand.tif")


def tile_stats(chip_ids):
    """Tiles every chip like oversample_data.py and computes JRC-based fractions over valid pixels."""
    tiles = []
    for i, chip_id in enumerate(chip_ids):
        _, mask = load_pair(chip_id)
        permanent = load_jrc(chip_id) == 1
        flood_only = (mask == 1) & ~permanent
        h, w = mask.shape
        for r_idx, r in enumerate(range(0, h, TILE)):
            for c_idx, c in enumerate(range(0, w, TILE)):
                m = mask[r:r + TILE, c:c + TILE]
                valid = m != -1
                nodata_fraction = 1.0 - valid.mean()
                if nodata_fraction > MAX_NODATA:
                    continue
                n_valid = valid.sum()
                tiles.append({
                    "tile_id": f"{chip_id}_r{r_idx}_c{c_idx}",
                    "parent_chip_id": chip_id,
                    "country": chip_id.split("_")[0],
                    "row_idx": r_idx,
                    "col_idx": c_idx,
                    "tile_bbox_in_parent": [r, c, r + TILE, c + TILE],
                    "water_fraction": round(float((m == 1).sum() / n_valid), 4),
                    "permanent_fraction": round(float((permanent[r:r + TILE, c:c + TILE] & valid).sum() / n_valid), 4),
                    "permanent_water_fraction": round(
                        float((permanent[r:r + TILE, c:c + TILE] & (m == 1)).sum() / n_valid), 4),
                    "flood_only_fraction": round(float((flood_only[r:r + TILE, c:c + TILE] & valid).sum() / n_valid), 4),
                    "nodata_fraction": round(float(nodata_fraction), 4),
                })
        if (i + 1) % 100 == 0 or i + 1 == len(chip_ids):
            print(f"  tiled {i + 1}/{len(chip_ids)} chips -> {len(tiles)} tiles with nodata <= {MAX_NODATA}")
    return tiles


def sample_group(candidates, rng):
    """Shuffles candidates and keeps at most MAX_PER_PARENT per parent, up to MAX_PER_GROUP."""
    order = rng.permutation(len(candidates))
    per_parent = Counter()
    picked = []
    for k in order:
        t = candidates[k]
        if per_parent[t["parent_chip_id"]] >= MAX_PER_PARENT:
            continue
        per_parent[t["parent_chip_id"]] += 1
        picked.append(t)
        if len(picked) == MAX_PER_GROUP:
            break
    return sorted(picked, key=lambda t: t["tile_id"])


def save_examples_figure(selected, out_path, n_per_group=4):
    """RGB | LabelHand | JRC permanent | flood_only_removed, n examples per group."""
    label_cmap = ListedColormap(["#7f7f7f", "#d9c27a", "#1f5fbf"])  # -1 no data, 0 dry, 1 water
    jrc_cmap = ListedColormap(["#f2f2f2", "#08306b"])
    rows = [(g, t) for g in ("permanent_only", "mixed") for t in selected[g][:n_per_group]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.8 * len(rows)))
    for ax_row, (group, t) in zip(axes, rows):
        r0, c0, r1, c1 = t["tile_bbox_in_parent"]
        _, mask = load_pair(t["parent_chip_id"])
        jrc = load_jrc(t["parent_chip_id"])
        ax_row[0].imshow(Image.open(C2_DIR / t["image_path"]))
        ax_row[0].set_title(f"{group}\n{t['tile_id']}", fontsize=8)
        ax_row[1].imshow(mask[r0:r1, c0:c1], cmap=label_cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax_row[1].set_title(f"LabelHand  water={t['water_fraction']:.1%}", fontsize=8)
        ax_row[2].imshow(jrc[r0:r1, c0:c1], cmap=jrc_cmap, vmin=0, vmax=1, interpolation="nearest")
        ax_row[2].set_title(f"JRC permanent={t['permanent_fraction']:.1%}\nflood_only={t['flood_only_fraction']:.1%}",
                            fontsize=8)
        if "flood_only_removed_path" in t:
            ax_row[3].imshow(Image.open(C2_DIR / t["flood_only_removed_path"]))
            ax_row[3].set_title("flood_only_removed", fontsize=8)
        else:
            ax_row[3].text(0.5, 0.5, "n/a (no flood_only)", ha="center", va="center", fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.text(0.5, 0.003, "LabelHand: grey = no data/cloud, tan = dry, blue = water.   JRC: dark blue = permanent water",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_c2(max_chips=None):
    chip_ids = list_ids()[:max_chips] if max_chips else list_ids()
    print(f"=== C2 river-vs-flood set: {len(chip_ids)} chips ===")
    tiles = tile_stats(chip_ids)
    n_before = len(tiles)
    tiles = [t for t in tiles if t["parent_chip_id"] not in EXCLUDED_PARENTS]
    print(f"Excluded {n_before - len(tiles)} tiles from parents {sorted(EXCLUDED_PARENTS)}")

    groups = {
        # JRC permanent pixels must also be hand-labelled water: drops tiles where JRC
        # marks a channel that is dry (or mislocated) in this image.
        "permanent_only": [t for t in tiles if t["permanent_water_fraction"] >= MIN_PERMANENT
                           and t["water_fraction"] >= MIN_WATER_PERMANENT_GROUP
                           and t["flood_only_fraction"] < MAX_FLOOD_ONLY_PERMANENT_GROUP],
        "mixed": [t for t in tiles if t["permanent_fraction"] >= MIN_PERMANENT
                  and t["flood_only_fraction"] >= MIN_FLOOD_ONLY_MIXED],
    }
    print("\nQualifying tiles before sampling:")
    for g, cands in groups.items():
        print(f"  {g:15s}: {len(cands):4d} tiles from {len({t['parent_chip_id'] for t in cands})} parent chips")

    # Own RNG per group, so changing one group's rule does not reshuffle the other.
    selected = {g: sample_group(cands, np.random.default_rng(config.SEED)) for g, cands in groups.items()}

    with open(PILOT_META, "r") as f:
        pilot_prompts = next(iter(json.load(f).values()))["prompts"]
    prompts = {**pilot_prompts, **EXTRA_PROMPTS}

    for sub in ("orig", "flood_only_removed"):  # drop images from earlier builds
        if (C2_DIR / sub).exists():
            shutil.rmtree(C2_DIR / sub)
    (C2_DIR / "orig").mkdir(parents=True, exist_ok=True)
    (C2_DIR / "flood_only_removed").mkdir(parents=True, exist_ok=True)
    camo_rng = np.random.default_rng(config.SEED)

    by_parent = defaultdict(list)
    for g, ts in selected.items():
        for t in ts:
            t["group"] = g
            by_parent[t["parent_chip_id"]].append(t)

    for chip_id in sorted(by_parent):
        s2, mask = load_pair(chip_id)
        rgb = to_rgb8(s2)
        permanent = load_jrc(chip_id) == 1
        for t in by_parent[chip_id]:
            r0, c0, r1, c1 = t["tile_bbox_in_parent"]
            tile_rgb = rgb[r0:r1, c0:c1]
            t["image_path"] = f"orig/{t['tile_id']}.png"
            Image.fromarray(tile_rgb).save(C2_DIR / t["image_path"])
            if t["group"] == "mixed":
                # Hide permanent pixels from camouflage_image: they are neither water to
                # replace (1) nor dry pixels to sample from (0).
                m = mask[r0:r1, c0:c1].copy()
                m[permanent[r0:r1, c0:c1] & (m != -1)] = 2
                edited, _ = camouflage_image(tile_rgb, m, camo_rng)
                t["flood_only_removed_path"] = f"flood_only_removed/{t['tile_id']}.png"
                Image.fromarray(edited).save(C2_DIR / t["flood_only_removed_path"])

    metadata = {
        "meta": {
            "seed": config.SEED,
            "tile_size": TILE,
            "max_nodata_fraction": MAX_NODATA,
            "max_tiles_per_parent_per_group": MAX_PER_PARENT,
            "excluded_parents": sorted(EXCLUDED_PARENTS),
            "fractions_over": "valid pixels (LabelHand != -1)",
            "permanent": "JRCWaterHand == 1",
            "permanent_water": "permanent & (LabelHand == 1)",
            "flood_only": "(LabelHand == 1) & ~permanent",
            "groups": {
                "permanent_only": f"permanent_water_fraction >= {MIN_PERMANENT} and water_fraction >= "
                                  f"{MIN_WATER_PERMANENT_GROUP} and flood_only_fraction < {MAX_FLOOD_ONLY_PERMANENT_GROUP}",
                "mixed": f"permanent_fraction >= {MIN_PERMANENT} and flood_only_fraction >= {MIN_FLOOD_ONLY_MIXED}",
            },
            "qualifying_before_sampling": {g: len(c) for g, c in groups.items()},
            "selected": {g: len(s) for g, s in selected.items()},
        },
        "tiles": {t["tile_id"]: {**t, "prompts": prompts} for g in selected for t in selected[g]},
    }
    with open(C2_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSelected (seed 42, <= 3 per parent):")
    for g, ts in selected.items():
        pf = np.array([t["permanent_fraction"] for t in ts])
        ff = np.array([t["flood_only_fraction"] for t in ts])
        print(f"  {g:15s}: {len(ts):3d} tiles, {len({t['parent_chip_id'] for t in ts}):3d} parents, "
              f"countries={dict(Counter(t['country'] for t in ts))}")
        print(f"  {'':15s}  permanent median={np.median(pf):.1%}  flood_only median={np.median(ff):.1%}")
    print(f"Saved metadata: {C2_DIR / 'metadata.json'}")

    fig_path = config.FIGURES_DIR / "c2_examples.png"
    save_examples_figure(selected, fig_path)
    print(f"Saved figure:   {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the C2 river-vs-flood test set")
    parser.add_argument("--max-chips", type=int, default=None, help="Limit parent chips for rapid testing")
    args = parser.parse_args()
    build_c2(args.max_chips)
