# Scorer for the de-confounding matrix (VNA_RadCacheSweep.py, sweep 2).
#
# REWRITTEN 2026-07-20 after the trRR hunt closed: the original scorer built
# its cloudy mask from the BASE REALIZATION'S OWN 60th luma percentile. That
# mask preferentially selects pixels where the base's noise ran HIGH, which
# inflates the base mean inside the mask and reads as a NEGATIVE offset for
# any comparison image - the same selection-bias disease as probe v2 binning
# by Tref. Measured damage on real captures (shared mask vs per-pair mask):
# ntrE_s1 vs ntb_s1 went from -0.423% "FAIL" to +0.013%, ntrE_s2 vs ntb_s2
# from -0.497% "FAIL" to -0.014%. Tracker RR lost a day to this artifact.
#
# Fixes:
#   1. The mask is built ONCE per window from the MEAN of ALL captures at
#      that window/shape. Any single realization's noise is diluted by the
#      row count (~13x), so the mask cannot co-select with either side of a
#      comparison. Never mask by either image being compared.
#   2. The no-RR seed trio (ntb / ntb_s1 / ntb_s2) is scored first as the
#      REALIZATION NOISE FLOOR: pure seed changes with identical estimators.
#      A pair delta is only evidence of bias if it clears BOTH the gate and
#      that floor. (Sweep 5's "systematic -0.6%" died the moment this floor
#      was computed: the null spread was -0.68..-0.79%.)
#
#   py scripts/VNA_RadCacheScore.py
#
# Discriminators (unchanged):
#   - Temporal-ON rows vs t_base at 256 AND 1024 frames:
#       bias            -> rel% roughly window-INVARIANT and beyond gate+floor
#       chain artifact  -> |rel%| SHRINKS markedly from w256 to w1024
#   - Temporal-OFF pairs: independent frames, textbook noise floor.

import glob
import os

import numpy as np
import OpenEXR

DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"
GATE = 0.15
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


_luma_cache = {}


def luma_of(name, w):
    key = (name, w)
    if key not in _luma_cache:
        p = newest(f"{name}_vp1_w{w}.*.exr")
        _luma_cache[key] = load(p) @ LUMA if p else None
    return _luma_cache[key]


_mask_cache = {}


def shared_mask(w, shape):
    """Cloudy mask from the MEAN over every capture at this window+shape.
    Independent (to ~1/rows) of any single realization - see header."""
    key = (w, shape)
    if key not in _mask_cache:
        acc = None
        n = 0
        for p in glob.glob(os.path.join(DIR, f"*_vp1_w{w}.*.exr")):
            img = load(p)
            if img.shape[:2] != shape:
                continue
            l = img @ LUMA
            acc = l if acc is None else acc + l
            n += 1
        if acc is None:
            _mask_cache[key] = None
        else:
            m = acc / n
            _mask_cache[key] = m >= np.percentile(m, 60.0)
    return _mask_cache[key]


def cloudy_rel(name, base_name, w):
    bl = luma_of(base_name, w)
    al = luma_of(name, w)
    if bl is None or al is None or al.shape != bl.shape:
        return None
    mask = shared_mask(w, bl.shape)
    bm = bl[mask].mean()
    return (al[mask].mean() - bm) / bm * 100.0


# ---- Realization noise floor: identical estimator, seed-only changes. ----
print("realization noise floor (no RR anywhere, seed change only, w256):")
floor = 0.0
for a, b in [("ntb_s1", "ntb"), ("ntb_s2", "ntb"), ("ntb_s2", "ntb_s1")]:
    r = cloudy_rel(a, b, 256)
    if r is None:
        print(f"  {a:8s} vs {b:8s}: MISSING")
        continue
    floor = max(floor, abs(r))
    print(f"  {a:8s} vs {b:8s}: {r:+.4f}%")
print(f"  floor = {floor:.4f}%  (a pair is evidence of bias only beyond gate AND floor)")
print()


def verdict(r):
    if r is None:
        return ""
    if abs(r) <= GATE:
        return "PASS"
    if abs(r) <= floor:
        return f"beyond gate but INSIDE noise floor ({floor:.2f}%) - not evidence"
    return "FAIL (beyond gate and floor)"


print(f"gate +-{GATE}% (relative to same-window internal baseline, shared mask)")
print()
print(f"{'config':<9}{'rel% @256':>11}{'rel% @1024':>12}   verdict heuristic")
print("-" * 70)
for n, base in [("t_rr7", "t_base"), ("t_p100", "t_base"), ("t_p25", "t_base"),
                # Lever-1 warp-RR: judged against its own baseline (t_p25),
                # so the delta is the roulette alone, not the whole CV stack.
                ("t_wrr8", "t_p25")]:
    r256 = cloudy_rel(n, base, 256)
    r1024 = cloudy_rel(n, base, 1024)
    row = f"{n:<9}"
    row += f"{r256:>+11.4f}" if r256 is not None else f"{'-':>11}"
    row += f"{r1024:>+12.4f}" if r1024 is not None else f"{'-':>12}"
    v = ""
    if r256 is not None and r1024 is not None:
        if abs(r1024) <= GATE:
            v = "PASS at 1024" + (" (256 offset = chain artifact)" if abs(r256) > GATE else "")
        elif abs(r1024) <= floor:
            v = f"beyond gate but inside noise floor ({floor:.2f}%) - needs seeds/window, not a FAIL"
        elif abs(r1024) < 0.5 * abs(r256):
            v = "shrinking with window -> chain-dominated, still settling"
        else:
            v = "window-invariant beyond gate AND floor -> REAL BIAS"
    print(row + "   " + v)

print()
for n, base, what in [
    ("ntr", "ntb", "combined (sweep-2 row)"),
    ("ntrN", "ntb", "NEE walk only (bit0)"),
    ("ntrE", "ntb", "escape walk only (bit3)"),
    ("ntrE_fm", "ntb_fm", "escape RR, adaptive M OFF (vs its own control)"),
    ("ntrE_nf", "ntb_nf", "escape RR, footprint mips OFF (vs its own control)"),
    ("ntrE_s1", "ntb_s1", "escape RR, RNG epoch 7777 (vs its own control)"),
    ("ntrE_s2", "ntb_s2", "escape RR, RNG epoch 31337 (vs its own control)"),
]:
    r = cloudy_rel(n, base, 256)
    if r is None:
        continue
    print(f"{n:<9}{r:>+11.4f}{'':>12}   temporal-off, {what}: {verdict(r)}")
