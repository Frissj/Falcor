# Null-distribution check for the trRR image gate (run 132 follow-up).
#
# The walk-level probe is now clean (wave-flush miscompile removed): the
# escape walk under RR is UNBIASED at the estimator level. The only remaining
# evidence against trRR is the image-level -0.45..-0.86% - measured between
# runs whose RNG realizations DIFFER (RR coins/kills shift the downstream
# stream). If pure seed changes with NO RR move the 256-frame cloudy-mean by
# a similar magnitude, the gate readings are realization noise, not bias.
#
# Pairs:
#   NULL (no RR anywhere, only seed differs): ntb vs ntb_s1 / ntb_s2, s1 vs s2
#   GATE (RR vs no-RR, matched seeds):        ntrE vs ntb, ntrE_s1 vs ntb_s1,
#                                             ntrE_s2 vs ntb_s2
#
#   py scripts/VNA_TrRRNullScore.py

import glob
import os

import numpy as np
import OpenEXR

DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"
LUMA = np.array([0.2126, 0.7152, 0.0722])


def load(p):
    f = OpenEXR.File(p)
    ch = f.channels()
    for k in ("RGBA", "RGB"):
        if k in ch:
            return np.asarray(ch[k].pixels, dtype=np.float64)[..., :3]
    return np.stack([np.asarray(ch[c].pixels, dtype=np.float64) for c in ("R", "G", "B")], axis=-1)


def newest(pattern):
    g = glob.glob(os.path.join(DIR, pattern))
    return max(g, key=os.path.getmtime) if g else None


def rel(a_name, b_name, w=256):
    ap = newest(f"{a_name}_vp1_w{w}.*.exr")
    bp = newest(f"{b_name}_vp1_w{w}.*.exr")
    if ap is None or bp is None:
        return None
    a, b = load(ap), load(bp)
    if a.shape != b.shape:
        return None
    bl = b @ LUMA
    mask = bl >= np.percentile(bl, 60.0)
    return ((a @ LUMA)[mask].mean() - bl[mask].mean()) / bl[mask].mean() * 100.0


print("NULL pairs (seed change only, no RR) - the gate's noise floor:")
for a, b in [("ntb_s1", "ntb"), ("ntb_s2", "ntb"), ("ntb_s2", "ntb_s1")]:
    r = rel(a, b)
    print(f"  {a:8s} vs {b:8s}: {r:+.3f}%" if r is not None else f"  {a} vs {b}: MISSING")

print("GATE pairs (RR on vs off, matched seeds):")
for a, b in [("ntrE", "ntb"), ("ntrE_s1", "ntb_s1"), ("ntrE_s2", "ntb_s2")]:
    r = rel(a, b)
    print(f"  {a:8s} vs {b:8s}: {r:+.3f}%" if r is not None else f"  {a} vs {b}: MISSING")
