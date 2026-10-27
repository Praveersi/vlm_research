"""Scores the C2 river-vs-flood Yes/No runs (results/c2/<model>.jsonl).

Per model, with 95% CIs from a bootstrap over parent chips (2000 resamples, seed 42;
parents are resampled within each group, the same resamples for both models):
1. permanent_only: % Yes for yn_water / yn_flood, on all tiles and on the strict subset
   with flood_only_fraction < 0.005.
2. mixed: % Yes for yn_flood / yn_water on orig vs. flood_only_removed.
3. Paired, mixed, yn_flood: persistence = among tiles Yes on orig, % still Yes after
   flood_only_removed; switch-to-Yes = among tiles No on orig, % Yes after.
4. Exact McNemar test (orig vs. flood_only_removed, mixed, yn_flood).

Answers are parsed from the first word: yes / no / other. "other" counts as not-Yes.
Outputs: results/c2/c2_table.md, results/c2/c2_metrics.json, results/figures/c2_bars.png
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

import config

MODELS = {"qwen": "Qwen2.5-VL-7B", "llava": "LLaVA-1.5-7B"}
COLORS = {"qwen": "#2a78d6", "llava": "#eb6834"}  # validated categorical slots 1-2
N_BOOTSTRAP = 2000
STRICT_FLOOD_ONLY = 0.005
C2_RESULTS = config.RESULTS_DIR / "c2"
C2_META = config.DATA_DIR / "c2" / "metadata.json"


def parse_answer(text):
    words = re.findall(r"[a-z]+", str(text).lower())
    if words and words[0] in ("yes", "no"):
        return words[0]
    return "other"


def parent_of(tile_id):
    return tile_id.split("_r")[0]


def load_answers(path):
    """{tile_id: {(condition, prompt_id): yes/no/other}} plus answer counts; first record wins on duplicates."""
    answers = defaultdict(dict)
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["condition"], r["prompt_id"])
            if key in answers[r["tile_id"]]:
                continue
            a = parse_answer(r["answer"])
            answers[r["tile_id"]][key] = a
            counts[a] += 1
    return answers, counts


def is_yes(ans, cond, prompt):
    return ans.get((cond, prompt)) == "yes"


def answered(ans, cond, prompt):
    return (cond, prompt) in ans


def rate(cond, prompt, subset=lambda t: True):
    """% Yes among tiles (in subset) that have an answer for (cond, prompt)."""
    def f(t, ans):
        den = answered(ans, cond, prompt) and subset(t)
        return (den and is_yes(ans, cond, prompt), den)
    return f


def persistence(t, ans):
    den = is_yes(ans, "orig", "yn_flood") and answered(ans, "flood_only_removed", "yn_flood")
    return (den and is_yes(ans, "flood_only_removed", "yn_flood"), den)


def switch_to_yes(t, ans):
    den = ans.get(("orig", "yn_flood")) == "no" and answered(ans, "flood_only_removed", "yn_flood")
    return (den and is_yes(ans, "flood_only_removed", "yn_flood"), den)


strict = lambda t: t["flood_only_fraction"] < STRICT_FLOOD_ONLY

# (key, group, label, per-tile function returning (numerator, denominator))
METRICS = [
    ("po_yn_water",          "permanent_only", "permanent_only: % Yes, yn_water",                         rate("orig", "yn_water")),
    ("po_yn_flood",          "permanent_only", "permanent_only: % Yes, yn_flood",                         rate("orig", "yn_flood")),
    ("po_strict_yn_water",   "permanent_only", "permanent_only strict (flood_only < 0.5%): % Yes, yn_water", rate("orig", "yn_water", strict)),
    ("po_strict_yn_flood",   "permanent_only", "permanent_only strict (flood_only < 0.5%): % Yes, yn_flood", rate("orig", "yn_flood", strict)),
    ("mx_orig_yn_flood",     "mixed", "mixed orig: % Yes, yn_flood",                                      rate("orig", "yn_flood")),
    ("mx_removed_yn_flood",  "mixed", "mixed flood_only_removed: % Yes, yn_flood",                        rate("flood_only_removed", "yn_flood")),
    ("mx_orig_yn_water",     "mixed", "mixed orig: % Yes, yn_water",                                      rate("orig", "yn_water")),
    ("mx_removed_yn_water",  "mixed", "mixed flood_only_removed: % Yes, yn_water",                        rate("flood_only_removed", "yn_water")),
    ("mx_persistence",       "mixed", "mixed yn_flood persistence: Yes on orig -> still Yes after removal", persistence),
    ("mx_switch_to_yes",     "mixed", "mixed yn_flood switch: No on orig -> Yes after removal",           switch_to_yes),
]


def mcnemar(tiles, answers):
    """Exact McNemar on mixed yn_flood. b = Yes->not-Yes, c = not-Yes->Yes (paired tiles only)."""
    b = c = n = 0
    for tid, t in tiles.items():
        ans = answers.get(tid, {})
        if t["group"] != "mixed" or not (answered(ans, "orig", "yn_flood") and answered(ans, "flood_only_removed", "yn_flood")):
            continue
        n += 1
        o, r = is_yes(ans, "orig", "yn_flood"), is_yes(ans, "flood_only_removed", "yn_flood")
        b += o and not r
        c += r and not o
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"n_pairs": n, "b_yes_to_no": int(b), "c_no_to_yes": int(c), "p_value": float(p)}


def score_model(tiles, answers, group_parents, boot_idx):
    results = {}
    for key, group, label, fn in METRICS:
        parents = group_parents[group]
        p_index = {p: i for i, p in enumerate(parents)}
        counts = np.zeros((len(parents), 2))
        for tid, t in tiles.items():
            if t["group"] == group:
                counts[p_index[parent_of(tid)]] += np.array(fn(t, answers.get(tid, {})), dtype=float)
        num, den = counts.sum(axis=0)
        boot = counts[boot_idx[group]].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(boot[:, 1] > 0, boot[:, 0] / boot[:, 1], np.nan)
        entry = {"label": label, "group": group, "numerator": int(num), "n": int(den)}
        if den == 0:
            entry.update(value=None, ci95=None)
        else:
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            entry.update(value=round(100 * num / den, 1), ci95=[round(100 * lo, 1), round(100 * hi, 1)],
                         n_parents=int((counts[:, 1] > 0).sum()))
        results[key] = entry
    return results


def fmt(m):
    if m["value"] is None:
        return "n/a (n=0)"
    return f"{m['value']:.1f}% [{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}] (n={m['n']})"


def markdown(per_model, mcn, parse_counts, meta_counts):
    keys = list(per_model)
    lines = [
        "# C2 river vs. flood: Yes/No results",
        "",
        f"Tiles: {meta_counts}. 95% CIs: bootstrap over parent chips within each group "
        f"({N_BOOTSTRAP} resamples, seed {config.SEED}). n = tiles in the denominator.",
        "",
        "| Metric | " + " | ".join(MODELS[k] for k in keys) + " |",
        "| :--- | " + " | ".join(":---:" for _ in keys) + " |",
    ]
    for key, _, label, _ in METRICS:
        lines.append(f"| {label} | " + " | ".join(fmt(per_model[k][key]) for k in keys) + " |")
    lines += [
        "",
        "## Exact McNemar test: mixed, yn_flood, orig vs. flood_only_removed",
        "",
        "b = Yes on orig, not Yes after removal; c = not Yes on orig, Yes after removal.",
        "Treats tiles as independent (ignores parent-chip clustering), so the p-value is optimistic.",
        "",
        "| Model | pairs | b | c | p-value |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for k in keys:
        m = mcn[k]
        lines.append(f"| {MODELS[k]} | {m['n_pairs']} | {m['b_yes_to_no']} | {m['c_no_to_yes']} | {m['p_value']:.3g} |")
    lines += ["", "Parsed answers: " + "; ".join(f"{MODELS[k]} {dict(parse_counts[k])}" for k in keys), ""]
    return "\n".join(lines)


def plot_bars(per_model, out_path):
    panels = [
        ("permanent_only (orig)", [("po_yn_water", "yn_water"), ("po_yn_flood", "yn_flood"),
                                   ("po_strict_yn_water", "yn_water\nstrict"), ("po_strict_yn_flood", "yn_flood\nstrict")]),
        ("mixed: orig vs. flood_only_removed", [("mx_orig_yn_flood", "yn_flood\norig"), ("mx_removed_yn_flood", "yn_flood\nremoved"),
                                                ("mx_orig_yn_water", "yn_water\norig"), ("mx_removed_yn_water", "yn_water\nremoved")]),
        ("mixed, yn_flood, paired", [("mx_persistence", "Yes stays Yes\n(persistence)"), ("mx_switch_to_yes", "No turns Yes")]),
    ]
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [4, 4, 2.4]}, facecolor=surface)
    width = 0.38
    for ax, (title, items) in zip(axes, panels):
        ax.set_facecolor(surface)
        x = np.arange(len(items))
        for i, model in enumerate(per_model):
            vals = [per_model[model][k]["value"] or 0 for k, _ in items]
            cis = [per_model[model][k]["ci95"] or [0, 0] for k, _ in items]
            err = np.array([[v - lo for v, (lo, _) in zip(vals, cis)], [hi - v for v, (_, hi) in zip(vals, cis)]])
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width, color=COLORS[model], edgecolor=surface, linewidth=2,
                   label=MODELS[model], zorder=2)
            ax.errorbar(x + offset, vals, yerr=err, fmt="none", ecolor=ink, elinewidth=1.2, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in items], fontsize=9, color=ink)
        ax.set_title(title, fontsize=10.5, color=ink, loc="left")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    axes[0].set_ylabel("% answering Yes", color=ink)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2)
    fig.text(0.01, 0.01, "Error bars: 95% CI, bootstrap over parent chips. strict = flood_only < 0.5%. "
             "removed = flood_only pixels replaced, permanent water kept.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=surface)
    plt.close(fig)


def main(results_dir):
    with open(C2_META, "r") as f:
        tiles = json.load(f)["tiles"]
    group_parents = {g: sorted({parent_of(tid) for tid, t in tiles.items() if t["group"] == g})
                     for g in ("permanent_only", "mixed")}
    rng = np.random.default_rng(config.SEED)
    boot_idx = {g: rng.integers(0, len(p), size=(N_BOOTSTRAP, len(p))) for g, p in group_parents.items()}

    per_model, mcn, parse_counts = {}, {}, {}
    for model in MODELS:
        answers, parse_counts[model] = load_answers(Path(results_dir) / f"{model}.jsonl")
        per_model[model] = score_model(tiles, answers, group_parents, boot_idx)
        mcn[model] = mcnemar(tiles, answers)

    meta_counts = {g: sum(t["group"] == g for t in tiles.values()) for g in group_parents}
    meta_counts["strict permanent_only"] = sum(t["group"] == "permanent_only" and strict(t) for t in tiles.values())
    table = markdown(per_model, mcn, parse_counts, meta_counts)

    out_dir = Path(results_dir)
    (out_dir / "c2_table.md").write_text(table, encoding="utf-8")
    with open(out_dir / "c2_metrics.json", "w") as f:
        json.dump({"bootstrap": {"unit": "parent_chip, within group", "n_resamples": N_BOOTSTRAP, "seed": config.SEED},
                   "models": per_model, "mcnemar_yn_flood_mixed": mcn}, f, indent=2)
    fig_path = config.FIGURES_DIR / "c2_bars.png"
    plot_bars(per_model, fig_path)

    print(table)
    print(f"Saved: {out_dir / 'c2_table.md'}\nSaved: {out_dir / 'c2_metrics.json'}\nSaved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score C2 river-vs-flood Yes/No results")
    parser.add_argument("--results-dir", default=str(C2_RESULTS))
    args = parser.parse_args()
    main(args.results_dir)
