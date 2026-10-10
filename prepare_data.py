import json
import numpy as np
from PIL import Image

import config
from data import list_ids, load_pair, to_rgb8

def get_dominant_quadrant(mask):
    """Finds the quadrant with the most water pixels."""
    h, w = mask.shape
    h2, w2 = h // 2, w // 2
    quadrants = {
        "top-left": mask[:h2, :w2],
        "top-right": mask[:h2, w2:],
        "bottom-left": mask[h2:, :w2],
        "bottom-right": mask[h2:, w2:]
    }
    
    max_water = -1
    dom_quad = "none"
    for name, quad in quadrants.items():
        water_count = (quad == 1).sum()
        if water_count > max_water:
            max_water = water_count
            dom_quad = name
            
    if max_water == 0:
        return "none"
    return dom_quad

def prepare_data():
    ids = list_ids()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    
    print(f"Processing {len(ids)} chips...")
    for chip_id in ids:
        s2, mask = load_pair(chip_id)
        
        # 1. Render RGB using the shared 98th-percentile stretch method from data.py
        rgb = to_rgb8(s2)
        out_png = config.PROCESSED_DIR / f"{chip_id}.png"
        Image.fromarray(rgb).save(out_png)
        
        # 2. Compute fractions
        total_pixels = mask.size
        no_data_pixels = (mask == -1).sum()
        nodata_fraction = float(no_data_pixels / total_pixels)
        
        valid_pixels = total_pixels - no_data_pixels
        if valid_pixels > 0:
            water_pixels = (mask == 1).sum()
            water_fraction = float(water_pixels / valid_pixels)
        else:
            water_fraction = 0.0
            
        # 3. Dominant quadrant
        dom_quadrant = get_dominant_quadrant(mask)
        
        # 4. Decide rule for mostly-cloud chips (Flag them if > 50% nodata)
        is_mostly_cloud = nodata_fraction > 0.50
        
        # Define Ground Truth Prompts for the models
        prompt_free_text = "Please provide a detailed free-text report describing the presence and extent of surface water/inundation in this satellite image."
        
        prompt_json = (
            "Analyze the image for surface water/inundation. Return your analysis as a structured JSON object with the following fields: "
            "'flood_present' (boolean), 'water_percentage' (float, 0-100), and 'location' (string, e.g., 'top-left', 'bottom-right', 'center', or 'none')."
        )
        
        prompt_pope = "Is there surface water/inundation present in this image? Answer only Yes or No."

        metadata[chip_id] = {
            "image_path": str(out_png.relative_to(config.ROOT)),
            "target_label": "surface water/inundation",
            "stats": {
                "water_fraction": round(water_fraction, 4),
                "nodata_fraction": round(nodata_fraction, 4),
                "dominant_quadrant": dom_quadrant,
                "is_mostly_cloud": is_mostly_cloud
            },
            "prompts": {
                "free_text": prompt_free_text,
                "structured_json": prompt_json,
                "pope": prompt_pope
            }
        }
    
    # Save metadata
    meta_path = config.PROCESSED_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Data preparation complete.")
    print(f"Processed images saved to: {config.PROCESSED_DIR}")
    print(f"Metadata and prompts saved to: {meta_path}")

if __name__ == "__main__":
    prepare_data()
