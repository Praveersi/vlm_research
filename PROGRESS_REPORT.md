# Project Progress Report: VLM Flood Detection Research

**Date:** October 10, 2026
**Project Phase:** Data Preparation & Experimental Setup

This document outlines the chronological steps and milestones achieved in the project so far. 

---

### 1. Environment & Dependency Setup
*   Initialized a Python virtual environment (`venv`) to ensure a clean workspace.
*   Installed all required dependencies for data processing and machine learning: `numpy`, `tifffile`, `matplotlib`, `huggingface_hub`, `torch`, `torchvision`, `transformers`, `accelerate`, and `qwen-vl-utils`.

### 2. Dataset Acquisition
*   **Script Created:** `download_data.py`
*   **Action:** Successfully downloaded the **Sen1Floods11** benchmark dataset (`sen1floods11_v1_1.tar.gz`) directly from Hugging Face.
*   **Result:** Extracted the 13-band Sentinel-2 L1C optical images and the expert hand-labeled ground-truth masks into the `data/raw/` directory.

### 3. Baseline Data Preprocessing (Step 1)
*   **Script Created:** `prepare_data.py`
*   **Action:** Iterated through all 446 image chips in the dataset to standardize the inputs for the Vision-Language Model (VLM).
*   **Transformations:** 
    *   Converted the 13-band `.tif` files into standard 8-bit true-color RGB `.png` images using a 98th-percentile contrast stretch.
    *   Saved all processed images to `data/processed/`.
*   **Calculations & Rules:** 
    *   Computed the exact `water_fraction` and `nodata_fraction` for every image.
    *   Calculated the `dominant_quadrant` (top-left, top-right, bottom-left, bottom-right) where the majority of the water is located.
    *   Implemented a rule to flag "mostly-cloud" chips (any chip with >50% no-data).

### 4. Prompt Engineering & Ground Truth
*   **Action:** Designed three distinct styles of prompts to evaluate how the VLM responds to different instruction formats:
    1.  **Free Text Report:** Evaluates open-ended scene understanding (e.g., distinguishing normal water bodies from active floods).
    2.  **Structured JSON:** Forces the model to return strict data (`flood_present`, `water_percentage`, `location`) for automated pipelines.
    3.  **POPE Style:** A strict binary "Yes/No" question to calculate standard hallucination metrics.
*   **Result:** Saved all the computed statistics, file paths, and the three prompts for every image into a unified `data/processed/metadata.json` file.

### 5. Experimental Design: Hallucination Testing
*   **Script Created:** `generate_covered_images.py`
*   **Action:** Designed a robust occlusion experiment to test if the VLM hallucinates floods based on context rather than visual evidence.
*   **Mechanism:** The script identifies the water pixels in the ground-truth mask and "camouflages" them by randomly sampling and pasting **dry land pixels** from the surrounding area over the water. This ensures the model sees natural-looking dry land instead of a suspicious black box. 
*   **Result:** Successfully generated 394 camouflaged images and saved them to `data/processed_covered/`.

### 6. Model Inference Preparation
*   **Script Created:** `qwen_inference.py` (Local Inference) and `qwen_colab.ipynb` (Cloud Inference).
*   **Action:** Set up the codebase to run the **Qwen2.5-VL-7B-Instruct** model. 
*   **Result:** Provided a Colab notebook with 4-bit quantization (`bitsandbytes`) configured so the 7B parameter model can run efficiently on free cloud GPUs (like the T4) without crashing due to memory limits.

---

### 7. Phase 1 Implementation: 128x128 Spatial Tiling & Dataset Oversampling
*   **Script Created:** `oversample_data.py`
*   **Action:** Decomposed all 446 $512 \times 512$ Sentinel-2 chips and ground truth masks into a $4 \times 4$ spatial grid of non-overlapping $128 \times 128$ sub-chips.
*   **Benefits:**
    *   Preserved native 10m Sentinel-2 optical resolution without downsampling/blur.
    *   Expanded dataset by $16\times$, yielding **6,816 clean $128 \times 128$ images** after removing 320 pure sensor nodata border tiles.
    *   Stratified class balance:
        *   `dry_control`: 2,263 chips (33.2%) — essential negative controls for hallucination measurement.
        *   `moderate_flood`: 2,940 chips (43.1%) — tests detection sensitivity.
        *   `severe_flood`: 1,124 chips (16.5%) — tests quantitative water fraction regression.
        *   `cloud_obscured`: 489 chips (7.2%) — tests robustness under cloud noise.
*   **Outputs:**
    *   Saved all 6,816 images to `data/processed_128/`.
    *   Generated complete index and ground-truth prompt metadata at `data/processed_128/metadata.json`.
    *   Saved verification figure to `results/figures/phase1_128_tiling_sample.png`.

---

### 8. Phase 2 Implementation: Counterfactual Camouflaged / Masked Dataset ($128 \times 128$)
*   **Script Updated:** `generate_covered_images.py`
*   **Action:** Inpainted all floodwater pixels across the $128 \times 128$ dataset by sampling authentic dry-land pixels from the surrounding local scene.
*   **Results:**
    *   Camouflaged water in **4,267 tiles** that contained inundation, while maintaining identical terrain, riverbank geometry, vegetation, and scene context.
    *   Saved all **6,816 paired counterfactual images** to `data/processed_128_covered/`.
    *   Generated visual verification figure at `results/figures/phase2_camouflage_sample.png`.

---

### 9. Phase 3 Implementation: Google Drive & Dual-VLM Cloud Setup
*   **Script Created:** `package_for_drive.py`
    *   Compressed all 13,633 images (`processed_128/` and `processed_128_covered/`) and metadata into a single compact archive: `data/sen1floods_128.zip` (354.0 MB).
*   **Notebook Created:** `vlm_benchmark_colab.ipynb`
    *   **Persistent Hugging Face Caching:** Automatically mounts Google Drive and routes `HF_HOME` to `MyDrive/vlm_flood_research/hf_cache`, downloading 15 GB weights once and loading in seconds on future runs.
    *   **4-bit NormalFloat (NF4) Quantization:** Allows both **Qwen2.5-VL-7B-Instruct** and **LLaVA-1.5-7B-hf** to load into ~5 GB VRAM on free Google Colab T4 GPUs without memory crashes.
    *   **Resumable Streaming Batch Inference:** Checkpoints predictions line-by-line into JSONL files directly on Google Drive (`MyDrive/vlm_flood_research/results/`), ensuring zero loss of progress if a Colab session disconnects.

---

### 10. Phase 4 Implementation: Two-Stage Model Inference Execution
*   **Action:** Executed dual-VLM inference across 1,500 normal and 1,500 counterfactual masked image tiles using the Google Colab pipeline.
*   **Result:** Successfully generated and collected 6,000 total predictions:
    *   `results/model_result/qwen2_5_vl_7b_stage1_results.jsonl` (1,500 entries)
    *   `results/model_result/qwen2_5_vl_7b_stage2_results.jsonl` (1,500 entries)
    *   `results/model_result/llava_1_5_7b_stage1_results.jsonl` (1,500 entries)
    *   `results/model_result/llava_1_5_7b_stage2_results.jsonl` (1,500 entries)

---

### 11. Phase 5 Implementation: Evaluation Engine & Metrics Computation
*   **Script Upgraded:** `eval_metrics.py`
*   **Action:** Evaluated all 6,000 model predictions across Detection Capability, Hallucination Susceptibility, Spatial Faithfulness, Quantification Error, and Sliced Flood Tiers.
*   **Key Findings:**
    *   **Recall / Sensitivity:** **Qwen 2.5 VL** achieved **59.24%** vs. **LLaVA 1.5's 27.45%** (+31.8% advantage for Qwen). On severe flood scenes, Qwen achieved **70.1% accuracy**.
    *   **Hallucination Susceptibility:** When water was camouflaged with dry terrain, Qwen exhibited high contextual hallucination (**50.6% POPE HR, 77.6% JSON HR**), demonstrating that it predicts flood based on surrounding valley/terrain visual priors. LLaVA showed low false alarms (**5.0% dry land, 10.0% masked**) due to an extreme conservative bias (predicting "No" 84.3% of the time).
    *   **Spatial Faithfulness:** Qwen achieved **10.87%** quadrant citation accuracy on True Positives, whereas LLaVA defaulted to "none"/"center" (0.0%).
*   **Outputs Generated:**
    *   `results/metrics_summary.json`
    *   `results/metrics_summary.csv`
    *   `results/figures/accuracy_and_hallucination_comparison.png`
    *   `results/figures/confusion_matrices_comparison.png`
    *   `results/figures/flood_tier_slice_performance.png`
    *   `results/figures/normal_vs_masked_hallucination.png`
    *   `results/figures/water_percentage_regression.png`

---

### 12. Phase 6 Implementation: Publication Visualizations & Final Academic Report
*   **Statistical Validation:** Computed 95% bootstrap confidence intervals ($N=1,000$ iterations) demonstrating statistically significant advantages for Qwen 2.5 VL in Recall ($p < 0.001$) and F1-Score ($p < 0.001$).
*   **Publication-Quality Visualizations Generated:**
    *   `results/figures/model_comparison_radar.png`: Multi-axis radar chart illustrating sensitivity, precision, faithfulness, and specificity trade-offs.
    *   `results/figures/qualitative_samples.png`: 4-case high-resolution plate displaying Normal RGB, Camouflage Inpainting, Ground Truth statistics, and model textual reasoning.
    *   `results/figures/accuracy_and_hallucination_comparison.png`: Detection capability vs. hallucination susceptibility.
    *   `results/figures/confusion_matrices_comparison.png`: Side-by-side confusion matrices.
    *   `results/figures/flood_tier_slice_performance.png`: Detection accuracy across flood severity tiers.
    *   `results/figures/normal_vs_masked_hallucination.png`: Counterfactual inpainting persistence test.
    *   `results/figures/water_percentage_regression.png`: Scatterplot of predicted vs. ground truth water percentage.
*   **Academic Report Compiled:** Authored `results/FINAL_REPORT.md`, providing a complete scientific synthesis rigorously answering all three central research questions.

---

## 🏆 Project Milestone: 100% COMPLETE

All phases from Phase 0 to Phase 6 in `IMPLEMENTATION.md` are fully completed, verified, and preserved in the repository.
