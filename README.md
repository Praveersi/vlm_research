# VLM Flood Hallucination Research (Sen1Floods11)

Do vision-language models actually *see* floods in satellite images, or do they guess from context?
We test **Qwen2.5-VL-7B-Instruct** and **LLaVA-1.5-7B** on 128×128 Sentinel-2 RGB tiles from
**Sen1Floods11**, using the expert hand-labelled water masks (`LabelHand`) and the JRC permanent-water
layer (`JRCWaterHand`) as ground truth.

Two experiments are complete:

1. **Pilot (v1):** Yes/No + JSON questions on 1,500 tiles, then the same tiles with the water
   painted over with dry-land pixels. Re-scored with a corrected evaluation.
2. **C2, river vs. flood:** does a model call *permanent* water (rivers, lakes) a flood, and does it
   react when only the *flood* water is removed?

---

## What has been done

| Step | What | Output |
| --- | --- | --- |
| Data download | Sen1Floods11 v1.1 from Hugging Face (`blumenstiel/Sen1Floods11`), 446 hand-labelled chips | `data/raw/` |
| RGB conversion | Sentinel-2 B4/B3/B2 → 8-bit RGB, 98th-percentile stretch per chip | `data.py`, `prepare_data.py` |
| Tiling | Each 512×512 chip cut into a 4×4 grid of 128×128 tiles; all-nodata border tiles dropped → 6,816 tiles | `oversample_data.py`, `data/processed_128/` |
| Camouflage images | Water pixels replaced with randomly sampled dry pixels from the same tile | `generate_covered_images.py`, `data/processed_128_covered/` |
| Pilot inference (Colab) | Qwen + LLaVA, 4-bit NF4, on 1,500 tiles × {normal, camouflaged} × {Yes/No, JSON} = 6,000 answers | `vlm_benchmark_colab.ipynb`, `results/model_result/` |
| **Corrected pilot evaluation** | Rewritten scorer: see "Pilot results" below | `eval_metrics.py`, `results/pilot_v1/metrics_corrected.json` |
| **C2 test set** | Permanent-water vs. flood-water tiles from `JRCWaterHand`, plus "flood removed" images | `build_c2.py`, `data/c2/`, `results/figures/c2_examples.png` |
| **C2 inference (Colab)** | Qwen + LLaVA, Yes/No only, 738 jobs per model, resumable | `notebooks/run_c2.ipynb`, `results/c2/*.jsonl` |
| **C2 scoring** | % Yes with bootstrap CIs, paired persistence, exact McNemar test | `score_c2.py`, `results/c2/c2_table.md`, `results/figures/c2_bars.png` |

Outputs of the *original* pilot evaluation (old metrics, figures and the old scoring script) are kept
unchanged in `results/pilot_v1/` for reference.

---

## Pilot results (corrected)

What changed from the original evaluation:

- **Persistence** is measured only on wet tiles (water > 0.5%) that the model called flooded on the
  normal image: what % still get "Yes" after the water is painted over. The original "masked
  hallucination rate" was averaged over all tiles, including dry tiles whose "masked" image is
  identical to the normal one (verified: 0 of 702 differ).
- **Dry-tile false alarms** (water = 0) are reported separately.
- **95% CIs** come from a bootstrap that resamples **parent chips**, not tiles (tiles from one chip are
  correlated): 2,000 resamples, seed 42.
- **Degenerate answers are flagged, not scored:** if one value makes up > 90% of a model's answers
  for a field, every metric built on it is marked `degenerate`. LLaVA's JSON is always
  `flood_present: false, water_percentage: 0`.
- All numbers are computed; there is no hard-coded "winner" text.

1,500 tiles from 103 parent chips: 702 dry, 712 wet, 86 trace (0–0.5% water, excluded).
Format: value [95% CI] (n tiles).

| Metric | Qwen2.5-VL-7B | LLaVA-1.5-7B |
| :--- | :---: | :---: |
| Yes/No detection rate (wet) | 59.4% [53.9, 65.2] (712) | 27.5% [21.9, 33.5] (712) |
| Yes/No false-alarm rate (dry) | 44.0% [38.4, 49.6] (702) | 4.4% [2.8, 6.2] (702) |
| Yes/No accuracy | 57.7% [53.9, 61.4] | 61.3% [55.6, 67.3] |
| Yes/No F1 | 58.6% [52.7, 63.7] | 41.8% [34.7, 48.8] |
| **Persistence** (Yes before → still Yes after water removed) | **76.4% [71.5, 81.3]** (423) | **39.3% [30.6, 48.2]** (196) |
| Same answer on identical dry images (determinism check) | 98.3% | 100.0% |
| JSON detection rate (wet) / false-alarm rate (dry) | 76.5% / 76.3% | degenerate |
| JSON water % mean absolute error (percentage points) | 39.1 [35.4, 42.7] | degenerate |
| JSON quadrant match | 11.4% [8.2, 14.5] | degenerate |

Takeaways:
- **Qwen's JSON `flood_present` carries no signal:** it says True as often on dry tiles as on wet ones.
- **Qwen's Yes/No answer separates wet from dry only weakly** (59% vs. 44% Yes).
- **Qwen usually keeps saying "flood" after the water is removed.**
- **LLaVA is conservative:** few false alarms, but it misses most floods.
- **Persistence has no baseline yet.** We can't yet tell how much of the 76.4% comes from Qwen
  ignoring the camouflage rather than from its weak detection to begin with.

---

## C2: river vs. flood

**Tile definitions** (per pixel, fractions over valid pixels, i.e. `LabelHand != -1`):
- `permanent = JRCWaterHand == 1`. `JRCWaterHand` is binary (1 = JRC permanent water).
  96% of its valid pixels are also hand-labelled water.
- `flood_only = (LabelHand == 1) & ~permanent`

**Groups** (tiles with > 50% nodata skipped; ≤ 3 tiles per parent chip; seed 42; ≤ 150 per group):

| Group | Rule | Tiles | Parent chips |
| --- | --- | --- | --- |
| `permanent_only` | permanent ∩ hand-labelled water ≥ 2%, hand-labelled water ≥ 2%, flood_only < 2% | 69 | 44 |
| `mixed` | permanent ≥ 2%, flood_only ≥ 5% | 150 | 82 |

- **`permanent_only` uses "visible" permanent water:** the JRC pixels must also be hand-labelled
  water. This drops tiles where JRC marks a river that is dry in the image.
- **Two chips are excluded after a visual check:** `India_591317` (featureless grey water) and
  `Ghana_866994` (haze; JRC disagrees with the hand label).
- **`flood_only_removed`:** for `mixed` tiles, only the flood_only pixels are replaced with sampled
  dry pixels; permanent water is kept.

**Prompts** (greedy decoding, `max_new_tokens=5`):
- `yn_flood`: "Is there flooding in this satellite image? Answer only Yes or No."
- `yn_water`: "Is there any water (river, lake, or flood) in this satellite image? Answer only Yes or No."

**Results:** % answering Yes, 95% CI from resampling parent chips within each group (2,000 resamples, seed 42).

| Metric | Qwen2.5-VL-7B | LLaVA-1.5-7B |
| :--- | :---: | :---: |
| permanent_only, yn_water | 58.0% [43.1, 72.6] | 75.4% [60.6, 88.6] |
| permanent_only, yn_flood (river called a flood) | 15.9% [6.9, 26.8] | 17.4% [8.6, 26.9] |
| permanent_only with flood_only < 0.5% (n=20), yn_flood | 10.0% [0.0, 28.6] | 10.0% [0.0, 25.0] |
| mixed, yn_flood: original → flood removed | 23.3% → 26.7% | 40.7% → 30.7% |
| mixed, yn_water: original → flood removed | 71.3% → 68.0% | 82.0% → 78.7% |
| Persistence (Yes on original → still Yes after removal) | 57.1% [40.7, 73.1] (n=35) | 65.6% [52.4, 78.0] (n=61) |
| Switch (No on original → Yes after removal) | 17.4% [10.7, 25.2] (n=115) | 6.7% [2.2, 12.4] (n=89) |
| Exact McNemar test (yn_flood, mixed, original vs. removed) | b=15, c=20, p=0.50 | b=21, c=6, p=0.006 |

Takeaways:
- **Neither model often calls permanent water a flood** (about 16–17% say Yes to "flooding").
- **Both models miss most real floods** on mixed tiles.
- **Qwen does not react to the flood being removed.** Its answers flip in both directions at
  similar rates.
- **LLaVA does react to the removal,** mostly switching Yes → No.

See `results/c2/c2_table.md` and `results/figures/c2_bars.png`.

### Caveats
- **The McNemar p-values are optimistic:** the test treats tiles as independent, but up to 3 tiles
  share a parent chip.
- **The small subsets have wide CIs.** The strict permanent_only subset is only 20 tiles, and
  Sri Lanka contributes 23 of the 69 permanent_only tiles.
- **Dry-pixel camouflage leaves visible speckle** where large areas are replaced (see
  `results/figures/c2_examples.png`). A model may react to the artefact rather than to the
  missing water.
- **Most permanent_only tiles contain a little flood water:** 49 of 69 have 0.5–2%. Treat the group
  as "mostly permanent water", not "no flood at all".
- **`results/FINAL_REPORT.md` is out of date.** It was written for the original pilot evaluation, and
  its numbers and statistical claims predate the corrected metrics above. Do not cite it without
  re-checking.

---

## Repository layout

```
config.py                    paths, seed (42), RGB band order
data.py                      load S2/LabelHand pairs, RGB conversion
download_data.py             fetch Sen1Floods11 from Hugging Face
prepare_data.py              512x512 baseline PNGs + metadata
oversample_data.py           128x128 tiling -> data/processed_128/
generate_covered_images.py   dry-pixel camouflage -> data/processed_128_covered/
package_for_drive.py         zip the 128x128 data for Colab
vlm_benchmark_colab.ipynb    pilot inference (Qwen + LLaVA)
eval_metrics.py              corrected pilot evaluation
build_c2.py                  C2 tile sets, flood_only_removed images, example figure
scripts/make_run_c2_notebook.py   generates notebooks/run_c2.ipynb (edit this, not the .ipynb)
notebooks/run_c2.ipynb       C2 inference on Colab
score_c2.py                  C2 scoring, tables and figure
results/
  model_result/              pilot raw answers (4 x 1,500 JSONL)
  pilot_v1/                  original pilot outputs + metrics_corrected.json
  c2/                        C2 raw answers, c2_table.md, c2_metrics.json
  figures/                   data-pipeline figures, c2_examples.png, c2_bars.png
```

`data/` is not in git (raw data, images and zips are too large). Rebuild it with the steps below.

---

## How to reproduce

**Local (CPU is enough):** Python 3.11 with `numpy`, `scipy`, `matplotlib`, `pillow`, `tifffile`,
`huggingface_hub`.

```bash
python download_data.py            # then extract the .tar.gz inside data/raw/
python oversample_data.py          # 128x128 tiles
python generate_covered_images.py --dataset 128
python eval_metrics.py             # corrected pilot metrics (needs results/model_result/)
python build_c2.py                 # C2 tiles + data/c2/metadata.json + example figure
python score_c2.py                 # C2 tables + figure (needs results/c2/*.jsonl)
```

To package C2 for Colab, zip the *contents* of `data/c2/` (so `metadata.json` is at the zip root)
as `c2.zip`.

**Colab (GPU, T4 is enough), C2 inference:**
1. Upload `c2.zip` to `MyDrive/vlm_flood_research/data/c2.zip`.
2. Open `notebooks/run_c2.ipynb`, set `MODEL` to `qwen` or `llava`, and run all cells.
3. Answers are appended to `MyDrive/vlm_flood_research/results/c2/<model>.jsonl`. Re-running skips
   finished jobs.
4. Model weights are cached on the Colab disk (`/content/hf_cache`), not on Drive, so they
   re-download each session.
5. To switch models, use Runtime → Restart session first. Qwen needs a recent `transformers`; LLaVA
   uses the pinned `4.46.3` from the pilot.
6. Copy the two `.jsonl` files into `results/c2/` and run `python score_c2.py`.
