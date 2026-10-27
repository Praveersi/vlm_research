"""Corrected evaluation of the pilot (v1) Qwen 2.5 VL vs. LLaVA 1.5 results.

Tile groups (by ground-truth water_fraction):
  dry   : water_fraction == 0
  wet   : water_fraction > 0.005
  trace : 0 < water_fraction <= 0.005 (excluded from every metric, count reported)

Metrics (all computed from the JSONL logs, nothing hard-coded):
1. Stage 1 POPE: detection rate on wet tiles, false-alarm rate on dry tiles,
   accuracy / precision / F1 over dry + wet tiles.
2. Persistence: among wet tiles where the model said Yes on stage 1, the % that
   still say Yes on stage 2 (water camouflaged). Dry tiles are NOT used for any
   stage 2 metric, because their stage 2 image is identical to stage 1; they only
   give a stage1 == stage2 answer-agreement sanity check.
3. The same detection / false-alarm / persistence metrics for the JSON
   flood_present field, plus JSON water % MAE and quadrant match rate.
4. 95% CIs from a bootstrap that resamples PARENT chips (tiles of one parent
   are correlated), 2000 resamples, seed 42.
5. Degeneracy: if one value accounts for >90% of a model's stage 1 answers on a
   field, every metric built on that field is marked "degenerate" and its value
   is not reported.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import config

MODELS = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B-Instruct",
    "llava_1_5_7b": "LLaVA-1.5-7B-hf",
}
WET_THRESHOLD = 0.005
DEGENERATE_SHARE = 0.90
N_BOOTSTRAP = 2000
QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")

# metric name -> (answer fields it depends on, human-readable label, unit)
METRICS = {
    "pope_detection_rate":        (("pope",), "POPE detection rate (wet, stage 1)", "%"),
    "pope_dry_false_alarm_rate":  (("pope",), "POPE false-alarm rate (dry, stage 1)", "%"),
    "pope_accuracy":              (("pope",), "POPE accuracy (dry + wet, stage 1)", "%"),
    "pope_precision":             (("pope",), "POPE precision (dry + wet, stage 1)", "%"),
    "pope_f1":                    (("pope",), "POPE F1 (dry + wet, stage 1)", "%"),
    "pope_persistence":           (("pope",), "POPE persistence (wet, Yes on s1 -> Yes on s2)", "%"),
    "pope_dry_s1_s2_agreement":   (("pope",), "POPE dry-tile s1/s2 agreement (sanity check)", "%"),
    "json_parse_rate":            ((), "JSON parse rate (stage 1)", "%"),
    "json_detection_rate":        (("json_flood_present",), "JSON detection rate (wet, stage 1)", "%"),
    "json_dry_false_alarm_rate":  (("json_flood_present",), "JSON false-alarm rate (dry, stage 1)", "%"),
    "json_persistence":           (("json_flood_present",), "JSON persistence (wet, True on s1 -> True on s2)", "%"),
    "json_water_mae":             (("json_water_percentage",), "JSON water % MAE (dry + wet, stage 1)", "pp"),
    "json_quadrant_match_rate":   (("json_flood_present", "json_location"), "JSON quadrant match (wet, flood_present True)", "%"),
}


def parse_pope(text):
    """Returns True/False for a leading Yes/No, None if the answer is not parseable."""
    if not isinstance(text, str):
        return None
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    if words[0] == "yes":
        return True
    if words[0] == "no":
        return False
    return None


def parse_json(text):
    """Extracts flood_present / water_percentage / location; missing or invalid fields are None."""
    res = {"parsed": False, "flood_present": None, "water_percentage": None, "location": None}
    if not isinstance(text, str):
        return res
    s = text.replace(r"\_", "_")  # LLaVA escapes underscores
    match = re.search(r"\{.*?\}", s, re.DOTALL)
    if not match:
        return res
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return res
    res["parsed"] = True

    fp = data.get("flood_present")
    if isinstance(fp, bool):
        res["flood_present"] = fp
    elif isinstance(fp, str) and fp.strip().lower() in ("true", "false"):
        res["flood_present"] = fp.strip().lower() == "true"

    wp = data.get("water_percentage")
    try:
        if isinstance(wp, str):
            wp = wp.replace("%", "").strip()
        if wp is not None and not isinstance(wp, bool):
            res["water_percentage"] = float(wp)
    except (TypeError, ValueError):
        pass

    loc = data.get("location")
    if isinstance(loc, str) and loc.strip():
        res["location"] = loc.strip().lower().replace(" ", "-")
    return res


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tiles(s1_records, s2_records):
    """Joins stage 1 and stage 2 records of one model into one row per tile."""
    s2_by_id = {r["tile_id"]: r for r in s2_records}
    tiles = []
    for r in s1_records:
        r2 = s2_by_id[r["tile_id"]]
        gt = r["ground_truth"]
        wf = gt["water_fraction"]
        j1 = parse_json(r["predictions"].get("json_raw"))
        j2 = parse_json(r2["predictions"].get("json_raw"))
        tiles.append({
            "tile_id": r["tile_id"],
            "parent": r["parent_chip_id"],
            "water_fraction": wf,
            "group": "dry" if wf == 0 else ("wet" if wf > WET_THRESHOLD else "trace"),
            "quadrant": gt.get("dominant_quadrant", "").lower(),
            "s1_pope": parse_pope(r["predictions"].get("pope_raw")),
            "s2_pope": parse_pope(r2["predictions"].get("pope_raw")),
            "s1_json": j1,
            "s2_json": j2,
        })
    return tiles


def tile_counts(t):
    """Per-tile (numerator, denominator, tile counted) contribution to each metric, in METRICS order."""
    dry, wet = t["group"] == "dry", t["group"] == "wet"
    used = dry or wet
    y1, y2 = t["s1_pope"] is True, t["s2_pope"] is True
    j1, j2 = t["s1_json"], t["s2_json"]
    f1, f2 = j1["flood_present"] is True, j2["flood_present"] is True
    tp, fp, fn = wet and y1, dry and y1, wet and not y1

    wp = j1["water_percentage"]
    has_wp = used and wp is not None
    err = abs(wp - t["water_fraction"] * 100) if has_wp else 0.0
    has_loc = wet and f1 and j1["location"] is not None and t["quadrant"] in QUADRANTS

    c = {
        "pope_detection_rate":       (wet and y1, wet),
        "pope_dry_false_alarm_rate": (dry and y1, dry),
        "pope_accuracy":             (tp or (dry and not y1), used),
        "pope_precision":            (tp, tp or fp),
        "pope_f1":                   (2 * tp, 2 * tp + fp + fn),
        "pope_persistence":          (wet and y1 and y2, wet and y1),
        "pope_dry_s1_s2_agreement":  (dry and t["s1_pope"] == t["s2_pope"], dry),
        "json_parse_rate":           (used and j1["parsed"], used),
        "json_detection_rate":       (wet and f1, wet),
        "json_dry_false_alarm_rate": (dry and f1, dry),
        "json_persistence":          (wet and f1 and f2, wet and f1),
        "json_water_mae":            (err, has_wp),
        "json_quadrant_match_rate":  (has_loc and j1["location"] == t["quadrant"], has_loc),
    }
    # n = tiles in the metric's population (F1's denominator 2TP+FP+FN is not a tile count)
    n = {m: bool(c[m][1]) for m in METRICS}
    n["pope_f1"] = used
    return [(float(c[m][0]), float(c[m][1]), float(n[m])) for m in METRICS]


def degeneracy(tiles):
    """Most common stage 1 value and its share, per answer field (dry + wet tiles, parsed answers only)."""
    used = [t for t in tiles if t["group"] != "trace"]
    fields = {
        "pope": [t["s1_pope"] for t in used],
        "json_flood_present": [t["s1_json"]["flood_present"] for t in used],
        "json_water_percentage": [t["s1_json"]["water_percentage"] for t in used],
        "json_location": [t["s1_json"]["location"] for t in used],
    }
    out = {}
    for name, values in fields.items():
        values = [v for v in values if v is not None]
        if not values:
            out[name] = {"mode": None, "share": None, "n": 0, "degenerate": True}
            continue
        mode, count = Counter(values).most_common(1)[0]
        share = count / len(values)
        out[name] = {"mode": mode, "share": round(share, 4), "n": len(values),
                     "degenerate": share > DEGENERATE_SHARE}
    return out


def evaluate_model(tiles, parents, boot_idx):
    """Point estimates + parent-level bootstrap CIs for one model."""
    p_index = {p: i for i, p in enumerate(parents)}
    counts = np.zeros((len(parents), len(METRICS), 3))
    for t in tiles:
        counts[p_index[t["parent"]]] += np.array(tile_counts(t))

    total = counts.sum(axis=0)
    boot = counts[boot_idx].sum(axis=1)  # (n_boot, n_metrics, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_vals = np.where(boot[..., 1] > 0, boot[..., 0] / boot[..., 1], np.nan)

    deg = degeneracy(tiles)
    results = {}
    for k, (name, (fields, label, unit)) in enumerate(METRICS.items()):
        num, den, n = total[k]
        scale = 100.0 if unit == "%" else 1.0
        entry = {"label": label, "unit": unit, "numerator": round(num, 2), "denominator": round(den, 2),
                 "n_tiles": int(n)}
        bad = [f for f in fields if deg[f]["degenerate"]]
        if bad:
            entry.update(status="degenerate", value=None, ci95=None,
                         reason="; ".join(f"{f} = {deg[f]['mode']!r} for {deg[f]['share']:.1%} of answers"
                                          for f in bad))
        elif den == 0:
            entry.update(status="undefined", value=None, ci95=None, reason="denominator is 0")
        else:
            lo, hi = np.nanpercentile(boot_vals[:, k], [2.5, 97.5])
            entry.update(status="ok", value=round(num / den * scale, 2),
                         ci95=[round(lo * scale, 2), round(hi * scale, 2)])
        results[name] = entry

    groups = Counter(t["group"] for t in tiles)
    unparsed = {
        "pope_stage1": sum(t["s1_pope"] is None for t in tiles),
        "pope_stage2": sum(t["s2_pope"] is None for t in tiles),
        "json_stage1": sum(not t["s1_json"]["parsed"] for t in tiles),
        "json_stage2": sum(not t["s2_json"]["parsed"] for t in tiles),
    }
    return {"tile_groups": dict(groups), "unparsed": unparsed, "field_degeneracy": deg, "metrics": results}


def format_cell(m):
    if m["status"] != "ok":
        return m["status"]
    unit = "" if m["unit"] == "pp" else "%"
    return f"{m['value']:.1f}{unit} [{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}] (n={m['n_tiles']})"


def markdown_table(per_model):
    keys = list(per_model)
    lines = ["| Metric | " + " | ".join(MODELS[k] for k in keys) + " |",
             "| :--- | " + " | ".join(":---:" for _ in keys) + " |"]
    for name, (_, label, _) in METRICS.items():
        lines.append(f"| {label} | " + " | ".join(format_cell(per_model[k]["metrics"][name]) for k in keys) + " |")
    return "\n".join(lines)


def evaluate_all(results_dir, out_path):
    rdir = Path(results_dir)
    raw = {m: (load_jsonl(rdir / f"{m}_stage1_results.jsonl"), load_jsonl(rdir / f"{m}_stage2_results.jsonl"))
           for m in MODELS}
    tiles = {m: build_tiles(*raw[m]) for m in MODELS}

    # One shared set of parent resamples -> CIs are paired across models and metrics.
    parents = sorted({t["parent"] for m in MODELS for t in tiles[m]})
    rng = np.random.default_rng(config.SEED)
    boot_idx = rng.integers(0, len(parents), size=(N_BOOTSTRAP, len(parents)))

    per_model = {m: evaluate_model(tiles[m], parents, boot_idx) for m in MODELS}
    table = markdown_table(per_model)

    summary = {
        "meta": {
            "source": str(rdir),
            "n_tiles": len(tiles[next(iter(MODELS))]),
            "n_parent_chips": len(parents),
            "wet_threshold_water_fraction": WET_THRESHOLD,
            "degenerate_share_threshold": DEGENERATE_SHARE,
            "bootstrap": {"unit": "parent_chip", "n_resamples": N_BOOTSTRAP, "seed": config.SEED,
                          "ci": "percentile 2.5-97.5"},
            "notes": [
                "Trace tiles (0 < water_fraction <= 0.005) are excluded from all metrics.",
                "No stage 2 metric uses dry tiles: their stage 2 image is identical to stage 1.",
                "Unparseable Yes/No answers count as 'not Yes'; counts are under 'unparsed'.",
                "Degeneracy is judged on stage 1 answers of dry + wet tiles.",
            ],
        },
        "models": {m: {"model_name": MODELS[m], **per_model[m]} for m in MODELS},
        "table_markdown": table,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Tiles: {summary['meta']['n_tiles']}  parents: {len(parents)}")
    for m in MODELS:
        print(f"{MODELS[m]}: groups={per_model[m]['tile_groups']} unparsed={per_model[m]['unparsed']}")
        for field, d in per_model[m]["field_degeneracy"].items():
            flag = "  <-- DEGENERATE" if d["degenerate"] else ""
            print(f"    {field:22s} mode={d['mode']!r:12} share={d['share']}{flag}")
    print("\n" + table + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrected pilot evaluation with parent-level bootstrap CIs")
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR / "model_result"))
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "pilot_v1" / "metrics_corrected.json"))
    args = parser.parse_args()
    evaluate_all(args.results_dir, args.out)
