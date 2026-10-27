"""Phase 3 Helper: Package 128x128 datasets into a single zip archive for Google Drive.

Compacts data/processed_128/ (images + metadata) and data/processed_128_covered/
into a single zip file (~150-200 MB) for rapid 1-click upload to Google Drive.
"""

import os
from pathlib import Path
import zipfile
import config


def package_datasets():
    out_zip = config.DATA_DIR / "sen1floods_128.zip"
    
    print("=== Packaging 128x128 Datasets for Google Drive ===")
    dirs_to_pack = [
        config.PROCESSED_128_DIR,
        config.PROCESSED_128_COVERED_DIR
    ]
    
    for d in dirs_to_pack:
        if not d.exists():
            raise FileNotFoundError(f"Missing directory: {d}. Ensure Phase 1 and Phase 2 are complete.")
            
    print(f"Target archive: {out_zip}")
    print("Compressing images and metadata...")
    
    total_files = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zip_ref:
        for folder in dirs_to_pack:
            folder_name = folder.name
            for file_path in folder.glob("*.*"):
                archive_name = f"{folder_name}/{file_path.name}"
                zip_ref.write(file_path, arcname=archive_name)
                total_files += 1
                if total_files % 1000 == 0:
                    print(f"  Archived {total_files} files...")

    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"\nPackaging complete!")
    print(f"Total files archived: {total_files}")
    print(f"Archive size: {zip_size_mb:.1f} MB")
    print(f"Saved to: {out_zip.resolve()}")
    print("\nNext step: Upload this 'sen1floods_128.zip' file to your Google Drive under:")
    print("  MyDrive/vlm_flood_research/data/sen1floods_128.zip\n")


if __name__ == "__main__":
    package_datasets()
