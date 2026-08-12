#!/usr/bin/env python3
"""Complexity table with unreliable latency cells replaced by their closest clean data.

A cell is suspect when either
  (a) its std exceeds --rel_tol of its mean -- a transient stall during timing, or
  (b) it exceeds --ratio_tol x the median of its clean peers -- a sustained stall,
      which (a) misses because every iteration was slow.

Peers = rows sharing the same encoder and the same model, i.e. the prompt-set
variants, whose latency differs only by prompt count and is otherwise identical.
A suspect cell is replaced by the clean peer median. Cells with no clean peer are
left measured and reported, never silently kept as if verified.
"""
import argparse
import numpy as np, pandas as pd

LAT = ["streaming_latency_ms", "encoder_latency_ms", "model_latency_ms", "end_to_end_latency_ms"]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="profile_reports/full_complexity_run18415449_mean.csv")
ap.add_argument("--out", default="profile_reports/COMPLEXITY_TABLE_estimated_x1.7.csv")
ap.add_argument("--scale", type=float, default=1.7)
ap.add_argument("--rel_tol", type=float, default=0.25)
ap.add_argument("--ratio_tol", type=float, default=2.0)
a = ap.parse_args()

d = pd.read_csv(a.csv).reset_index(drop=True)
d["_peer"] = d["encoder_name"].astype(str) + "|" + d["model_name"].astype(str)

report = []
for col in LAT:
    std = d[col.replace("_ms", "_std_ms")]
    noisy = (std / d[col].replace(0, np.nan)) > a.rel_tol

    # Clean peer median, computed from the low-variance cells only.
    med = d[~noisy].groupby("_peer")[col].median()
    peer_med = d["_peer"].map(med)
    # Sustained stalls: low variance, but far above the clean peers.
    outlier = peer_med.notna() & (d[col] > a.ratio_tol * peer_med)

    suspect = noisy | outlier
    fixable = suspect & peer_med.notna() & (~d.index.isin(d[~suspect].index) | True)
    # Only impute where a clean peer median actually exists and differs from this row.
    fixable = suspect & peer_med.notna()

    d[col + "_measured"] = d[col]
    d.loc[fixable, col] = peer_med[fixable]
    d[col + "_estimated"] = fixable

    for i in d.index[suspect]:
        report.append({
            "model": d.at[i, "model_name"], "prompt": d.at[i, "prompt_type"],
            "encoder": d.at[i, "encoder_name"], "column": col,
            "measured": round(d.at[i, col + "_measured"] * a.scale, 3),
            "estimate": (round(d.at[i, col] * a.scale, 3) if fixable[i] else "KEPT (no clean peer)"),
            "why": "high variance" if noisy[i] else "outlier vs peers",
        })

for c in LAT:
    d[c] = d[c] * a.scale
    d[c + "_measured"] = d[c + "_measured"] * a.scale
    d[c.replace("_ms", "_std_ms")] = d[c.replace("_ms", "_std_ms")] * a.scale
d["latency_scale"] = a.scale
d = d.drop(columns=["_peer"])
d.to_csv(a.out, index=False)

r = pd.DataFrame(report)
pd.set_option("display.width", 220)
print(f"Suspect cells: {len(r)} of {len(d)*len(LAT)}\n")
print(r.to_string(index=False) if len(r) else "(none)")
n_est = int(sum(d[c + '_estimated'].sum() for c in LAT))
print(f"\nEstimated {n_est} cells from clean peers; {len(r)-n_est} kept as measured.")
print(f"Saved: {a.out}")
