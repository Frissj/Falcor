# Score the roulette sweep on the GATE's metric, plus the 1-spp firefly census.
#
# WHY THIS EXISTS: VNA_Matrix_Compare.py reports signedRel, a signed mean over
# ALL pixels. The project's gate is a different statistic - the cloudy-region
# mean, masked to the top-40% luminance of 01_ref (see VNA_BUILD_HANDOFF_4,
# "Reference baselines"). For heavy-tailed estimators like Russian roulette the
# two disagree badly: sweep 1 read +2.6% on signedRel and +2.8% on cloudy-mean
# for q20, but for q10 it was +0.159% (FAIL) vs +0.021% (PASS). Judging roulette
# on signedRel produces the wrong verdict. This script computes the gate metric.
#
# It also reports the 1-spp firefly census, which is what actually decides
# adoption: the gate measures 256-frame convergence, but the shipping target is
# 1 spp, where a single 100x pixel is visible.
#
#   py scripts/VNA_RouletteScore.py

import glob
import os

import numpy as np
import OpenEXR

DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"
GATE = 0.15  # percent, cloudy-region mean
LUMA = np.array([0.2126, 0.7152, 0.0722])


def load(path):
    f = OpenEXR.File(path)
    ch = f.channels()
    for k in ("RGBA", "RGB"):
        if k in ch:
            return np.asarray(ch[k].pixels, dtype=np.float64)[..., :3]
    return np.stack([np.asarray(ch[c].pixels, dtype=np.float64) for c in ("R", "G", "B")], axis=-1)


def newest(pattern):
    g = glob.glob(os.path.join(DIR, pattern))
    return max(g, key=os.path.getmtime) if g else None


ref_path = newest("01_ref_vp1_converged.*.exr")
if ref_path is None:
    raise SystemExit("01_ref_vp1_converged not found in " + DIR)
ref = load(ref_path)
ref_lum = ref @ LUMA
# Mask defined by the REFERENCE, not by each image - otherwise every config gets
# a different mask and the means are not comparable.
mask = ref_lum >= np.percentile(ref_lum, 60.0)
ref_mean = ref_lum[mask].mean()

ref1_path = newest("01_ref_vp1_1spp.*.exr")
ref1 = load(ref1_path) if ref1_path else None

# rq* = roulette sweeps, rc* = rad-cache/tracker-RR sweeps (VNA_RadCacheSweep.py).
names = sorted(
    {os.path.basename(p).split("_vp1_")[0] for p in glob.glob(os.path.join(DIR, "rq*_vp1_*.exr"))} |
    {os.path.basename(p).split("_vp1_")[0] for p in glob.glob(os.path.join(DIR, "rc*_vp1_*.exr"))}
)

print(f"reference: {os.path.basename(ref_path)}   cloudyMean {ref_mean:.4f}   gate +-{GATE}%")
print()
print(f"{'config':<16}{'cloudyMean':>11}{'rel%':>10}{'gate':>7}   "
      f"{'1spp MSE':>11}{'>10':>7}{'>100':>7}{'>1000':>7}")
print("-" * 80)

for n in names:
    cp = newest(f"{n}_vp1_converged.*.exr")
    row = f"{n:<16}"
    if cp:
        a = load(cp)
        m = (a @ LUMA)[mask].mean()
        rel = (m - ref_mean) / ref_mean * 100.0
        row += f"{m:>11.4f}{rel:>+10.4f}{('PASS' if abs(rel) <= GATE else 'FAIL'):>7}"
    else:
        row += f"{'-':>11}{'-':>10}{'-':>7}"

    op = newest(f"{n}_vp1_1spp.*.exr")
    if op is not None and ref1 is not None and load(op).shape == ref1.shape:
        b = load(op)
        d = b - ref1
        mse = float((d * d).mean())
        ad = np.abs(d).max(axis=-1)
        row += f"{mse:>11.3e}{int((ad>10).sum()):>7}{int((ad>100).sum()):>7}{int((ad>1000).sum()):>7}"
    else:
        row += f"{'-':>11}{'-':>7}{'-':>7}{'-':>7}"
    print(row)

print()
print("1-spp MSE is a firefly lottery - read the >10 / >100 / >1000 counts, not")
print("the MSE alone. A single 800-brightness pixel is MSE 0.33 at 2M px.")
print("Adoption test: gate PASS *and* firefly counts no worse than the baseline.")
