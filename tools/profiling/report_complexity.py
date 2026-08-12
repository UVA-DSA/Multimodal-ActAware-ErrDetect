#!/usr/bin/env python3
"""Render the window-model complexity table with latencies scaled to the reference hardware.

Params and FLOPs are hardware-independent and are reported as measured. The four
latency columns are measured on this GPU and then multiplied by --latency_scale to
put them on the same footing as the supplementary table's hardware. Raw measured
values stay in the source CSV and in the *_raw columns of the emitted CSV.
"""
import argparse, os
import pandas as pd

LAT = ["streaming_latency_ms", "encoder_latency_ms", "model_latency_ms", "end_to_end_latency_ms"]
STD = [c.replace("_ms", "_std_ms") for c in LAT]

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="profile_reports/full_complexity.csv")
ap.add_argument("--latency_scale", type=float, default=1.7)
ap.add_argument("--out", default="profile_reports/full_complexity_scaled.csv")
a = ap.parse_args()

df = pd.read_csv(a.csv)
for c in LAT + STD:
    if c in df.columns:
        df[c + "_raw"] = df[c]
        df[c] = df[c] * a.latency_scale

df["latency_scale"] = a.latency_scale
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)

def short(row):
    n, p = str(row["model_name"]), str(row.get("prompt_type", ""))
    return n if p in ("", "n/a", "nan") else f"{n} [{p}]"

hdr = f"{'family':<9} {'model':<44} {'encoder':<20} {'enc_params':>11} {'mdl_params':>11} {'GFLOPs':>9} {'stream':>8} {'enc':>8} {'mdl':>8} {'e2e':>8}"
print(f"\nLatencies x{a.latency_scale} (hardware-scaled). Params/FLOPs as measured. Window = 10 samples, batch 1.\n")
print(hdr); print("-" * len(hdr))
for _, r in df.iterrows():
    print(f"{r['model_family']:<9} {short(r)[:44]:<44} {str(r['encoder_name'])[:20]:<20} "
          f"{int(r['encoder_params']):>11,} {int(r['model_params']):>11,} "
          f"{r['flops_per_window']/1e9:>9.2f} "
          f"{r['streaming_latency_ms']:>8.3f} {r['encoder_latency_ms']:>8.3f} "
          f"{r['model_latency_ms']:>8.3f} {r['end_to_end_latency_ms']:>8.3f}")
print(f"\nSaved: {a.out}  (raw measured values preserved in *_raw columns)")
