# C2 river vs. flood: Yes/No results

Tiles: {'permanent_only': 69, 'mixed': 150, 'strict permanent_only': 20}. 95% CIs: bootstrap over parent chips within each group (2000 resamples, seed 42). n = tiles in the denominator.

| Metric | Qwen2.5-VL-7B | LLaVA-1.5-7B |
| :--- | :---: | :---: |
| permanent_only: % Yes, yn_water | 58.0% [43.1, 72.6] (n=69) | 75.4% [60.6, 88.6] (n=69) |
| permanent_only: % Yes, yn_flood | 15.9% [6.9, 26.8] (n=69) | 17.4% [8.6, 26.9] (n=69) |
| permanent_only strict (flood_only < 0.5%): % Yes, yn_water | 50.0% [20.0, 83.3] (n=20) | 65.0% [36.8, 93.3] (n=20) |
| permanent_only strict (flood_only < 0.5%): % Yes, yn_flood | 10.0% [0.0, 28.6] (n=20) | 10.0% [0.0, 25.0] (n=20) |
| mixed orig: % Yes, yn_flood | 23.3% [16.9, 30.4] (n=150) | 40.7% [31.8, 49.4] (n=150) |
| mixed flood_only_removed: % Yes, yn_flood | 26.7% [18.8, 35.1] (n=150) | 30.7% [22.1, 40.0] (n=150) |
| mixed orig: % Yes, yn_water | 71.3% [63.0, 79.7] (n=150) | 82.0% [74.7, 89.0] (n=150) |
| mixed flood_only_removed: % Yes, yn_water | 68.0% [59.2, 76.7] (n=150) | 78.7% [71.5, 85.7] (n=150) |
| mixed yn_flood persistence: Yes on orig -> still Yes after removal | 57.1% [40.7, 73.1] (n=35) | 65.6% [52.4, 78.0] (n=61) |
| mixed yn_flood switch: No on orig -> Yes after removal | 17.4% [10.7, 25.2] (n=115) | 6.7% [2.2, 12.4] (n=89) |

## Exact McNemar test: mixed, yn_flood, orig vs. flood_only_removed

b = Yes on orig, not Yes after removal; c = not Yes on orig, Yes after removal.
Treats tiles as independent (ignores parent-chip clustering), so the p-value is optimistic.

| Model | pairs | b | c | p-value |
| :--- | :---: | :---: | :---: | :---: |
| Qwen2.5-VL-7B | 150 | 15 | 20 | 0.5 |
| LLaVA-1.5-7B | 150 | 21 | 6 | 0.00592 |

Parsed answers: Qwen2.5-VL-7B {'no': 403, 'yes': 335}; LLaVA-1.5-7B {'no': 326, 'yes': 412}
