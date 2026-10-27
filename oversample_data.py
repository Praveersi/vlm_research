"""Phase 1: Spatial Tiling & Oversampling to 128x128 resolution.

This script tiles 512x512 Sentinel-2 chips and masks into non-overlapping 128x128
sub-chips (4x4 grid = 16 sub-chips per chip, oversampling the dataset to 7,136 images).
Preserves native 10m Sentinel-2 ground sampling distance without interpolation blur,
isolates localized flood pockets, and generates standardized metadata with evaluation prompts.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

import config
from data import list_ids, load_pair, to_rgb8


def get_dominant_quadrant_128(tile_mask):
    """Finds the 64x64 quadrant in a 128x128 tile with the most water pixels."""
    h, w = tile_mask.shape
    h2, w2 = h // 2, w // 2
    quadrants = {
        "top-left": tile_mask[:h2, :w2],
        "top-right": tile_mask[:h2, w2:],
        "bottom-left": tile_mask[h2:, :w2],
        "bottom-right": tile_mask[h2:, w2:]
    }
    
    max_water = -1
    dom_quad = "none"
    for name, quad in quadrants.items():
        water_count = int((quad == 1).sum())
        if water_count > max_water:
            max_water = water_count
            dom_quad = name
            
    if max_water <= 0:
        return "none"
    return dom_quad


def categorize_flood_tier(water_fraction, nodata_fraction):
    """Categorizes tile into stratified evaluation tiers."""
    if nodata_fraction >= 0.50:
        return "cloud_obscured"
    if water_fraction == 0.0:
        return "dry_control"
    if water_fraction < 0.15:
        return "moderate_flood"
    return "severe_flood"


def save_tiling_sample_figure(parent_id, rgb_full, mask_full, tiles_meta, out_path):
    """Saves a verification figure showing the 512x512 full chip and its 16 128x128 tiles."""
    cmap = ListedColormap(["#7f7f7f", "#d9c27a", "#1f5fbf"])  # -1 no data, 0 dry, 1 water
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 6)
    
    # Left side: 512x512 RGB and Mask (2x2 grid each)
    ax_rgb = fig.add_subplot(gs[0:2, 0:2])
    ax_rgb.imshow(rgb_full)
    ax_rgb.set_title(f"Original 512x512 RGB: {parent_id}", fontsize=10)
    ax_rgb.axis("off")
    
    ax_mask = fig.add_subplot(gs[2:4, 0:2])
    ax_mask.imshow(mask_full, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax_mask.set_title("Original 512x512 Mask (Grey=Cloud/Nodata, Tan=Dry, Blue=Water)", fontsize=9)
    ax_mask.axis("off")
    
    # Right side: 4x4 grid of 128x128 tiles
    for tile in tiles_meta:
        r_idx = tile["row_idx"]
        c_idx = tile["col_idx"]
        sub_ax = fig.add_subplot(gs[r_idx, 2 + c_idx])
        tile_img = Image.open(config.ROOT / tile["image_path"])
        sub_ax.imshow(tile_img)
        wf = tile["stats"]["water_fraction"] * 100
        sub_ax.set_title(f"r{r_idx}c{c_idx}: {wf:.1f}% w", fontsize=7.5)
        sub_ax.axis("off")
        
    fig.suptitle(f"Phase 1: 128x128 Spatial Tiling & Oversampling (16 tiles from {parent_id})", fontsize=12, y=0.98)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def process_and_oversample(tile_size=128, filter_pure_nodata=True, max_chips=None):
    """Tiles all 512x512 chips into 128x128 sub-chips and compiles metadata."""
    ids = list_ids()
    if max_chips:
        ids = ids[:max_chips]
        
    out_dir = config.PROCESSED_128_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Starting Phase 1: 128x128 Spatial Tiling & Oversampling ===")
    print(f"Total parent 512x512 chips to process: {len(ids)}")
    print(f"Tile target resolution: {tile_size} x {tile_size}")
    print(f"Output directory: {out_dir}")
    print(f"Filter 100% sensor nodata tiles: {filter_pure_nodata}\n")
    
    metadata = {}
    tier_counts = {"dry_control": 0, "moderate_flood": 0, "severe_flood": 0, "cloud_obscured": 0}
    skipped_pure_nodata = 0
    sample_tiles_for_viz = []
    sample_parent_id = None
    sample_rgb = None
    sample_mask = None
    
    for chip_idx, chip_id in enumerate(ids):
        s2, mask = load_pair(chip_id)
        rgb = to_rgb8(s2)
        h, w = mask.shape
        
        chip_sample_tiles = []
        
        for r_idx, r in enumerate(range(0, h, tile_size)):
            for c_idx, c in enumerate(range(0, w, tile_size)):
                tile_rgb = rgb[r:r + tile_size, c:c + tile_size]
                tile_mask = mask[r:r + tile_size, c:c + tile_size]
                
                total_pixels = int(tile_mask.size)
                no_data_pixels = int((tile_mask == -1).sum())
                nodata_fraction = float(no_data_pixels / total_pixels)
                
                # Check for 100% blank border tiles
                if filter_pure_nodata and nodata_fraction >= 0.999:
                    skipped_pure_nodata += 1
                    continue
                    
                valid_pixels = total_pixels - no_data_pixels
                if valid_pixels > 0:
                    water_pixels = int((tile_mask == 1).sum())
                    water_fraction = float(water_pixels / valid_pixels)
                else:
                    water_fraction = 0.0
                    
                dom_quadrant = get_dominant_quadrant_128(tile_mask)
                is_mostly_cloud = nodata_fraction > 0.50
                tier = categorize_flood_tier(water_fraction, nodata_fraction)
                tier_counts[tier] += 1
                
                tile_id = f"{chip_id}_r{r_idx}_c{c_idx}"
                out_png = out_dir / f"{tile_id}.png"
                Image.fromarray(tile_rgb).save(out_png)
                
                # Prompt definitions
                prompt_free_text = (
                    "Please provide a detailed free-text report describing the presence and extent of surface water/inundation in this 128x128 satellite image."
                )
                prompt_json = (
                    "Analyze the 128x128 image for surface water/inundation. Return your analysis as a structured JSON object with the following fields: "
                    "'flood_present' (boolean), 'water_percentage' (float, 0-100), and 'location' (string, e.g., 'top-left', 'bottom-right', 'center', or 'none')."
                )
                prompt_pope = "Is there surface water/inundation present in this image? Answer only Yes or No."
                
                tile_entry = {
                    "tile_id": tile_id,
                    "parent_chip_id": chip_id,
                    "row_idx": r_idx,
                    "col_idx": c_idx,
                    "image_path": str(out_png.relative_to(config.ROOT)),
                    "image_size": [tile_size, tile_size],
                    "tile_bbox_in_parent": [r, c, r + tile_size, c + tile_size],
                    "target_label": "surface water/inundation",
                    "stats": {
                        "water_fraction": round(water_fraction, 4),
                        "nodata_fraction": round(nodata_fraction, 4),
                        "dominant_quadrant": dom_quadrant,
                        "is_mostly_cloud": is_mostly_cloud,
                        "flood_tier": tier,
                        "flood_present": water_fraction > 0.005
                    },
                    "prompts": {
                        "free_text": prompt_free_text,
                        "structured_json": prompt_json,
                        "pope": prompt_pope
                    }
                }
                
                metadata[tile_id] = tile_entry
                chip_sample_tiles.append(tile_entry)
                
        # Save first chip with water for visual validation figure
        if sample_parent_id is None and any(t["stats"]["water_fraction"] > 0.10 for t in chip_sample_tiles):
            sample_parent_id = chip_id
            sample_rgb = rgb
            sample_mask = mask
            sample_tiles_for_viz = chip_sample_tiles
            
        if (chip_idx + 1) % 50 == 0 or (chip_idx + 1) == len(ids):
            print(f"Processed {chip_idx + 1}/{len(ids)} parent chips -> {len(metadata)} valid 128x128 tiles generated.")

    # Save central metadata file
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n=== Phase 1 Preprocessing Complete ===")
    print(f"Total 128x128 tiles saved: {len(metadata)}")
    if filter_pure_nodata:
        print(f"Excluded 100% blank sensor border tiles: {skipped_pure_nodata}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Class Breakdown across 128x128 tiles:")
    for tier, count in tier_counts.items():
        pct = (count / len(metadata) * 100) if metadata else 0
        print(f"  - {tier:16s}: {count:5d} ({pct:5.1f}%)")
        
    # Save visual verification figure
    if sample_parent_id and sample_tiles_for_viz:
        viz_out = config.FIGURES_DIR / "phase1_128_tiling_sample.png"
        save_tiling_sample_figure(sample_parent_id, sample_rgb, sample_mask, sample_tiles_for_viz, viz_out)
        print(f"Verification figure saved to: {viz_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: 128x128 Spatial Tiling & Oversampling")
    parser.add_argument("--tile-size", type=int, default=config.IMAGE_SIZE_TARGET, help="Tile resolution (default: 128)")
    parser.add_argument("--keep-pure-nodata", action="store_true", help="Keep pure blank (100 percent nodata) border tiles")
    parser.add_argument("--max-chips", type=int, default=None, help="Limit parent chips for rapid testing")
    args = parser.parse_args()
    
    process_and_oversample(
        tile_size=args.tile_size,
        filter_pure_nodata=not args.keep_pure_nodata,
        max_chips=args.max_chips
    )
