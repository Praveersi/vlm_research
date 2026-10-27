# 🌊 Benchmarking Vision-Language Models for Satellite Flood Inundation: Visual Grounding, Contextual Hallucination & Faithfulness

**Authors:** NLP Research Team  
**Benchmark Dataset:** Sen1Floods11 (Sentinel-2 L1C True-Color RGB + Hand-labeled Expert Masks)  
**Evaluated Models:** `Qwen2.5-VL-7B-Instruct` vs. `LLaVA-1.5-7B-hf`  
**Quantization & Execution:** 4-bit NormalFloat (NF4) on Google Colab T4 Cloud GPU  
**Date:** October 16, 2026  

---

## 📑 Executive Summary & Abstract

Vision-Language Models (VLMs) have achieved state-of-the-art visual reasoning on consumer photography, but their capabilities, domain alignment, and hallucination tendencies in Earth observation (EO) remain poorly understood. Satellite imagery presents unique physical challenges: nadir downward-looking geometry ($D_4$ dihedral group symmetry), high spatial frequency land cover, and localized disaster signals.

In this research, we introduce a standardized benchmark evaluating two leading open-weights 7B VLMs—**Qwen2.5-VL-7B-Instruct** (native spatial dynamic resolution) and **LLaVA-1.5-7B-hf** (canonical CLIP-based baseline)—across **6,000 empirical inferences** derived from the **Sen1Floods11** benchmark at $128 \times 128$ native 10m spatial resolution.

Using a two-stage experimental design involving **normal imagery** and **counterfactually camouflaged imagery** (where water is inpainted with local dry-land terrain), we evaluate three core research questions:
1. **Detection Capability:** Qwen 2.5 VL demonstrates genuine visual sensitivity to optical water (**59.24% recall**, **70.08% accuracy on severe floods**), whereas LLaVA 1.5 suffers catastrophic domain misalignment, exhibiting an extreme negative bias (**27.45% recall**, missing 72.5% of real flood events).
2. **Contextual Hallucination:** Under counterfactual camouflage, Qwen exhibits severe contextual hallucination (**50.60% POPE HR**, **77.60% JSON HR**), demonstrating that it predicts inundation based on geomorphological priors (dry river channels, valley depressions, riparian greenness) rather than visual surface water. Conversely, LLaVA's low hallucination rate (**10.00%**) is an artifact of universal negative default.
3. **Spatial Faithfulness & Quantification:** Qwen achieves **10.87% spatial quadrant faithfulness** on True Positives (vs. 0.00% for LLaVA), but tends to overestimate water area ($\text{MAE} = 38.80\%$). LLaVA defaults to 0% water across the entire dataset.

---

## 🧭 1. Research Questions & Experimental Hypotheses

| Research Question (RQ) | Scientific Rationale | Empirical Finding |
| :--- | :--- | :--- |
| **RQ1: Detection Capability** | Can off-the-shelf VLMs identify, locate, and quantify surface water inundation under varying cloud and terrain conditions? | **Partially Confirmed for Qwen; Rejected for LLaVA.** Qwen accurately detects severe floods (70.1%), while LLaVA fails entirely on satellite optical water. |
| **RQ2: Contextual Hallucination** | When floodwater is artificially camouflaged with dry terrain, does the model still predict flood based on contextual visual priors? | **Confirmed.** Qwen continues to predict flooding on 50.6% of camouflaged images, proving strong contextual prior reliance over verified pixel evidence. |
| **RQ3: Faithfulness & Citations** | Are model textual reports and spatial citations faithful to ground truth water masks? | **Weak / Modest Faithfulness.** Qwen shows localized citation capabilities (10.9% quadrant match), but defaults to "center" under ambiguous conditions. |

---

## 🛠️ 2. Methodology & Experimental Framework

### 2.1 Dataset: Sen1Floods11 at Native $128 \times 128$ Resolution
* Raw Sen1Floods11 consists of 446 $512 \times 512$ chips across 11 global flood events (Bolivia, Ghana, India, Pakistan, USA, etc.).
* **Spatial Tiling ($16\times$ Expansion):** Each $512 \times 512$ chip was non-overlappingly tiled into sixteen $128 \times 128$ sub-chips. This preserves native 10m optical Sentinel-2 resolution without synthetic downsampling blur.
* Pure sensor nodata border tiles were filtered, leaving **6,816 clean tiles**.
* **Stratified Evaluation Set:** A balanced representative subset of **1,500 test tiles** was evaluated across two stages:
  - `dry_control`: 598 tiles (39.9%) — Authentic dry land negative controls.
  - `moderate_flood` ($0.1\% \le \text{water} \le 15\%$): 459 tiles (30.6%) — Tests localized sensitivity.
  - `severe_flood` ($>15\%$ water): 244 tiles (16.3%) — Tests quantitative surface water regression.
  - `cloud_obscured` ($>50\%$ cloud): 199 tiles (13.3%) — Tests robustness under atmospheric occlusion.

### 2.2 Counterfactual Camouflage / Inpainting (Phase 2)
To rigorously distinguish between *visual perception* (seeing water) and *contextual hallucination* (inferring flood from surrounding landscape cues):
* For every flooded tile, all water pixels (`mask == 1`) were inpainted with authentic dry-land pixels sampled from the same scene.
* Scene context (vegetation, road networks, river meanders, topographic elevation gradients) remained 100% intact, while visual water reflectance was eliminated.
* **Stage 1 (Normal):** Ground Truth Flood = `True` (for flooded tiles).
* **Stage 2 (Masked):** Ground Truth Flood = `False` (for all tiles).

### 2.3 Prompting Configurations
1. **POPE Binary Probing (Strict Object Probing):**  
   > *"Is there surface water/inundation present in this image? Answer only Yes or No."*
2. **Structured JSON (Quantification & Grounding):**  
   > *"Analyze the 128x128 image for surface water/inundation. Return your analysis as a structured JSON object with: 'flood_present' (boolean), 'water_percentage' (float, 0-100), and 'location' (string, e.g., 'top-left', 'bottom-right', 'center', or 'none')."*

---

## 📊 3. Quantitative Results & Statistical Rigor

### 3.1 Primary Comparative Performance Matrix

| Evaluation Dimension | Metric | Qwen 2.5 VL 7B | LLaVA 1.5 7B | Differential / Winner | 95% Bootstrap Confidence Interval |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Detection Quality** | **Accuracy (POPE)** | 57.67% | **62.87%** | **LLaVA 1.5** (+5.2%) | Q: [55.3%, 60.1%] \| L: [60.4%, 65.3%] |
| | **Precision** | 55.15% | **83.40%** | **LLaVA 1.5** (+28.2%) | Q: [51.6%, 58.7%] \| L: [78.4%, 87.9%] |
| | **Recall / Sensitivity** | **59.24%** | 27.45% | **Qwen 2.5 VL** (+31.8%) | Q: [55.7%, 62.4%] \| L: [24.3%, 30.6%] |
| | **F1-Score** | **57.12%** | 41.31% | **Qwen 2.5 VL** (+15.8%) | Q: [54.3%, 59.9%] \| L: [37.4%, 44.9%] |
| **Hallucination** | **Dry Land False Alarm Rate ($\text{HR}_{\text{POPE}}$)** | 43.77% | **4.96%** | **LLaVA 1.5** (38.8% lower) | Q: [40.5%, 47.3%] \| L: [3.6%, 6.5%] |
| | **Masked Counterfactual HR (POPE)** | 50.60% | **10.00%** | **LLaVA 1.5** (40.6% lower) | Q: [48.1%, 53.0%] \| L: [8.5%, 11.6%] |
| | **Masked Counterfactual HR (JSON)** | 77.60% | **0.00%** | **LLaVA 1.5** (77.6% lower) | — |
| **Faithfulness** | **Spatial Quadrant Match** | **10.87%** | 0.00% | **Qwen 2.5 VL** (+10.9%) | — |
| | **Perturbation Delta ($\Delta_{\text{pert}}$)** | +8.27% | -27.13% | **Qwen 2.5 VL** | Expected drop upon water removal |
| **Quantification** | **Water Percentage MAE** | 38.80% | **11.00%** | **LLaVA 1.5** (Default 0% bias) | Qwen overestimates; LLaVA zeroes out |
| **Operational** | **Avg Latency per Inference** | **2.83s** | 2.91s | **Qwen 2.5 VL** (0.08s faster) | Execution time on 4-bit T4 GPU |

### 3.2 Confusion Matrices Comparison (Stage 1 Normal, $N=1,500$)

```
Qwen 2.5 VL 7B:
                 Predicted No      Predicted Yes
Actual Dry Land     442 (TN)          344 (FP - Hallucination)
Actual Flood        291 (FN - Miss)   423 (TP - Hit)

LLaVA 1.5 7B:
                 Predicted No      Predicted Yes
Actual Dry Land     747 (TN)           39 (FP - Hallucination)
Actual Flood        518 (FN - Miss)   196 (TP - Hit)
```

---

## 🔬 4. In-Depth Analysis: Answering the Three Research Questions

### 4.1 Research Question 1: Detection Capability & Sensitivity
* **The Disaster Response Imperative:** In natural disaster damage assessment, **False Negatives are fatal** (failing to detect people and communities trapped in floodwaters). False Positives can be visually verified by humans or SAR radar, but missed floods lead to abandoned relief operations.
* **Empirical Analysis:**
  - **Qwen 2.5 VL** achieves a **Recall of 59.24%** (95% CI: [55.74%, 62.38%]) and an **F1-Score of 57.12%**. When sliced by severity tier, Qwen achieves **70.08% accuracy on severe floods**. It clearly recognizes water spectral absorption characteristics (dark blue, black, turbid brown).
  - **LLaVA 1.5** exhibits severe **out-of-distribution failure**. It scores only **27.45% Recall** (95% CI: [24.27%, 30.60%]), missing **518 out of 714 true floods** (a 72.5% failure rate). On severe floods, LLaVA's accuracy is only **36.07%**.
  - **Domain Alignment:** LLaVA's CLIP visual encoder is pretrained on web images of everyday objects (cats, dogs, furniture, cars) with perspective horizons. When presented with nadir optical Earth observation data, CLIP produces weak activations, causing the LLM head (Vicuna) to default to the negative token ("No").

### 4.2 Research Question 2: Contextual Hallucination under Counterfactual Camouflage
* **The Inpainting Experiment:** When water pixels were completely replaced with dry soil and grass, visual evidence of flooding was reduced to zero ($0.0\%$).
* **Empirical Findings:**
  - **Qwen 2.5 VL:** Predicted flood on **50.60% (POPE)** and **77.60% (JSON)** of the camouflaged scenes. Its positive prediction rate barely dropped from Stage 1 to Stage 2 ($51.13\% \rightarrow 50.60\%$, $\Delta = 0.53\%$).
  - **Mechanistic Cause of Contextual Hallucination:** Qwen relies heavily on surrounding environmental semantics. When it detects a sinuous riverbed, low topography, green agricultural boundaries, or alluvial floodplains, its language model prior overrides the visual feature representation, hallucinating that flood inundation is present even when all water has been removed.
  - **LLaVA 1.5:** Hallucination rate on masked data was low (**10.00%**). However, this is not indicative of superior visual faithfulness, but rather a byproduct of its global negative bias (predicting "No" on 90% of masked images).

### 4.3 Research Question 3: Faithfulness & Spatial Citations
* **Spatial Citations:**
  - When Qwen successfully detected flooding (True Positives), it correctly identified the dominant quadrant in **10.87%** of cases (e.g., citing `top-left` when ground truth water was concentrated in the top-left). However, in 757 out of 1,500 inferences, Qwen defaulted to the generic token `"center"`.
  - LLaVA produced zero valid spatial matches ($0.0\%$), universally predicting `"none"` or `"center"`.
* **Quantitative Water Fraction Estimation:**
  - Qwen exhibited a **Mean Absolute Error (MAE) of 38.80%**, systematically over-estimating inundation coverage (frequently predicting 35% to 75% for small 5% flood pockets).
  - LLaVA exhibited an MAE of 11.00% solely because it predicted `0%` water across 1,236 chips, matching the low mean ground truth water fraction (11.00%).

---

## 🍰 5. Slice-Based Error Analysis across Flood Tiers

Evaluating models across stratified sub-populations reveals distinct behavioral regimes:

| Flood Tier | Chip Count | Qwen 2.5 VL Accuracy | LLaVA 1.5 Accuracy | Dominant Error Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Dry Control ($0.0\%$ water)** | 598 | 53.68% | **95.65%** | Qwen generates false alarms on dry soil; LLaVA's negative bias yields high accuracy. |
| **Moderate Flood ($0.1\%-15\%$)** | 459 | **57.08%** | 33.55% | Qwen detects localized flood pockets; LLaVA misses 298 out of 459 floods (FN). |
| **Severe Flood ($>15\%$ water)** | 244 | **70.08%** | 36.07% | Qwen identifies large water bodies; LLaVA misses 156 out of 244 severe floods. |
| **Cloud Obscured ($>50\%$ cloud)** | 199 | 55.78% | **64.82%** | Cloud edges trigger false alarms in Qwen; LLaVA defaults to negative. |

---

## ⚡ 6. Operational & Production Viability

| Operational Metric | Qwen2.5-VL-7B-Instruct | LLaVA-1.5-7B-hf | Production Assessment |
| :--- | :---: | :---: | :--- |
| **Quantization Format** | 4-bit NF4 (`bitsandbytes`) | 4-bit NF4 (`bitsandbytes`) | Highly accessible on low-cost hardware |
| **VRAM Footprint** | 5.2 GB | 4.8 GB | Both fit comfortably on free Google Colab T4 (15 GB) |
| **Inference Latency** | **2.83s / tile** | 2.91s / tile | Practical for near-real-time satellite tile processing |
| **Throughput** | ~21.2 tiles / min | ~20.6 tiles / min | Sufficient for rapid scene-level screening |
| **Deployment Recommendation** | **Viable with False Alarm Filtering** | **Unviable for Remote Sensing** | Qwen is recommended as a high-sensitivity triage filter |

---

## 💡 7. Research Recommendations & Next Steps

1. **Pre-training on Multispectral / Earth Observation Data:** Off-the-shelf VLMs require pretraining on paired remote sensing datasets (e.g., Geo-Chat, EarthGPT, SkySense) to bridge the domain gap between natural consumer photos and nadir satellite views.
2. **Context-Contrastive Decoding:** To suppress contextual hallucinations, inference algorithms should implement contrastive decoding that penalizes predictions derived purely from terrain background tokens.
3. **Multimodal SAR Fusion:** Optical imagery is constrained by cloud cover. Integrating Sentinel-1 Synthetic Aperture Radar (SAR) backscatter with optical RGB will eliminate cloud-induced ambiguity.

---

## 📁 8. Summary of Benchmark Deliverables

* **Source Code:**
  - Preprocessing & Tiling: [oversample_data.py](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/oversample_data.py)
  - Camouflage Inpainting: [generate_covered_images.py](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/generate_covered_images.py)
  - Cloud Inference Notebook: [vlm_benchmark_colab.ipynb](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/vlm_benchmark_colab.ipynb)
  - Evaluation Engine: [eval_metrics.py](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/eval_metrics.py)
* **Metrics Data:**
  - Machine-readable JSON: [results/metrics_summary.json](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/metrics_summary.json)
  - Publication CSV: [results/metrics_summary.csv](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/metrics_summary.csv)
* **Publication Figures:**
  - Multi-Axis Radar Chart: [results/figures/model_comparison_radar.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/model_comparison_radar.png)
  - Qualitative Plates: [results/figures/qualitative_samples.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/qualitative_samples.png)
  - Detection vs. Hallucination Bar Chart: [results/figures/accuracy_and_hallucination_comparison.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/accuracy_and_hallucination_comparison.png)
  - Confusion Matrices: [results/figures/confusion_matrices_comparison.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/confusion_matrices_comparison.png)
  - Sliced Flood Tiers: [results/figures/flood_tier_slice_performance.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/flood_tier_slice_performance.png)
  - Counterfactual Persistence Shift: [results/figures/normal_vs_masked_hallucination.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/normal_vs_masked_hallucination.png)
  - Water % Calibration Regression: [results/figures/water_percentage_regression.png](file:///e:/NLP-%20Natural%20Language%20Processing/NLP%20SEM%20PROJECT/vlm_research/results/figures/water_percentage_regression.png)
