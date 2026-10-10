import config
from data import list_ids, load_pair, to_rgb8
import numpy as np
from PIL import Image

def generate_covered():
    ids = list_ids()
    out_dir = config.DATA_DIR / "processed_covered"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating covered images for {len(ids)} chips...")
    count = 0
    rng = np.random.default_rng(config.SEED)
    
    for chip_id in ids:
        s2, mask = load_pair(chip_id)
        
        # Render standard RGB
        rgb = to_rgb8(s2)
        
        water_mask = mask == 1
        dry_mask = mask == 0
        
        # Check if there is any water to cover
        if water_mask.any():
            if dry_mask.any():
                # Extract all dry land pixels from this image
                dry_pixels = rgb[dry_mask]
                
                # Randomly sample from the dry land pixels to fill the water area
                num_water_pixels = water_mask.sum()
                sampled_dry_pixels = rng.choice(dry_pixels, size=num_water_pixels, replace=True)
                
                rgb[water_mask] = sampled_dry_pixels
            else:
                # If the entire image is water/clouds, use a generic dry land color
                rgb[water_mask] = [100, 120, 80]
                
            count += 1
            
        out_png = out_dir / f"{chip_id}.png"
        Image.fromarray(rgb).save(out_png)
        
    print(f"Saved {len(ids)} images to {out_dir}")
    print(f"Actually covered water in {count} images.")

if __name__ == "__main__":
    generate_covered()
