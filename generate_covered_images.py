"""Phase 2: Generate counterfactual camouflaged/covered images.

Replaces flood/water pixels with sampled dry-land pixels from the same scene
to test whether Vision-Language Models hallucinate based on contextual priors
(e.g., riverbanks, vegetation, valleys) rather than visual evidence.
Supports both the baseline 512x512 dataset and the Phase 1 128x128 tiled dataset.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

import config
from data import list_ids, load_pair, to_rgb8


def camouflage_image(rgb, mask, rng):
    """Replaces water pixels (mask == 1) with sampled dry pixels (mask == 0)."""
    water_mask = (mask == 1)
    dry_mask = (mask == 0)
    
    rgb_covered = rgb.copy()
    if water_mask.any():
        if dry_mask.any():
            dry_pixels = rgb[dry_mask]
            num_water_pixels = int(water_mask.sum())
            sampled_dry_pixels = rng.choice(dry_pixels, size=num_water_pixels, replace=True)
            rgb_covered[water_mask] = sampled_dry_pixels
        else:
            # Fallback if tile has only water and clouds: realistic dry land terrain color
            rgb_covered[water_mask] = [100, 120, 80]
        return rgb_covered, True
    return rgb_covered, False


def generate_covered_512():
    """Generates camouflaged images for the baseline 512x512 chips."""
    ids = list_ids()
    out_dir = config.DATA_DIR / "processed_covered"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 512x512 covered images for {len(ids)} chips...")
    count = 0
    rng = np.random.default_rng(config.SEED)
    
    for chip_id in ids:
        s2, mask = load_pair(chip_id)
        rgb = to_rgb8(s2)
        rgb_covered, covered = camouflage_image(rgb, mask, rng)
        if covered:
            count += 1
            
        out_png = out_dir / f"{chip_id}.png"
        Image.fromarray(rgb_covered).save(out_png)
        
    print(f"Saved {len(ids)} images to {out_dir}")
    print(f"Water camouflaged in {count} chips.\n")


def generate_covered_128():
    """Generates camouflaged images for the Phase 1 128x128 tiled dataset."""
    meta_path = config.PROCESSED_128_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run oversample_data.py first.")
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    out_dir = config.PROCESSED_128_COVERED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 128x128 covered images for {len(metadata)} tiles...")
    rng = np.random.default_rng(config.SEED)
    count = 0
    
    # Cache parent masks so we only load each 512x512 GeoTIFF once
    cached_parent_id = None
    cached_mask = None
    
    for tile_id, item in metadata.items():
        parent_id = item["parent_chip_id"]
        if parent_id != cached_parent_id:
            _, cached_mask = load_pair(parent_id)
            cached_parent_id = parent_id
            
        r0, c0, r1, c1 = item["tile_bbox_in_parent"]
        tile_mask = cached_mask[r0:r1, c0:c1]
        
        # Load the normal 128x128 RGB
        tile_rgb = np.array(Image.open(config.ROOT / item["image_path"]))
        
        rgb_covered, covered = camouflage_image(tile_rgb, tile_mask, rng)
        if covered:
            count += 1
            
        out_png = out_dir / f"{tile_id}.png"
        Image.fromarray(rgb_covered).save(out_png)
        
    print(f"Saved {len(metadata)} covered 128x128 images to {out_dir}")
    print(f"Water camouflaged in {count} tiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Generate Camouflaged/Covered Images")
    parser.add_argument("--dataset", choices=["128", "512", "all"], default="128",
                        help="Target dataset: '128' (Phase 1 128x128), '512' (baseline), or 'all'")
    args = parser.parse_args()
    
    if args.dataset in ("512", "all"):
        generate_covered_512()
    if args.dataset in ("128", "all"):
        generate_covered_128()
