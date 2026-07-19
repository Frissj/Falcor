# VNA validation matrix - comparison step. Plain python (no Mogwai):
#
#   py scripts/VNA_Matrix_Compare.py
#
# Reads the converged EXRs captured by VNA_Matrix.py and prints, for every
# config vs 01_ref at the same viewpoint: mean / p99 / max absolute difference
# and MSE, in raw linear-HDR units. No pass/fail threshold is baked in;
# instead the pairwise spread among the Stage-A variants (03..06) - which are
# all the SAME estimator with different noise - is printed as an empirical
# noise floor. A config whose diff sits far above that floor is a bug.
#
# The 1-spp EXRs are also compared against the converged reference to report
# per-config 1-spp variance (RIS should BEAT non-RIS here; that is its job).

import glob
import itertools
import os
import sys

import numpy as np
import OpenEXR  # official bindings; opencv-python ships without EXR support

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

def load(name, vp, kind):
    pattern = os.path.join(OUT_DIR, "{}_vp{}_{}.*.exr".format(name, vp, kind))
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None
    try:
        channels = OpenEXR.File(files[-1]).channels()
    except Exception as e:
        print("  !! failed to read {}: {}".format(files[-1], e))
        return None
    if "RGBA" in channels or "RGB" in channels:  # interleaved layout
        key = "RGBA" if "RGBA" in channels else "RGB"
        return channels[key].pixels[..., :3].astype(np.float64)
    return np.stack([channels[c].pixels for c in ("R", "G", "B")], axis=-1).astype(np.float64)

def diff_stats(a, b):
    d = np.abs(a - b).mean(axis=2)  # per-pixel mean over RGB
    # Signed relative mean: the bias detector. Noise averages toward 0 over
    # ~2M pixels; a systematic estimator error shows as a shift that does NOT
    # shrink with more pixels or frames.
    signed_rel = (a - b).mean() / max(b.mean(), 1e-12)
    return d.mean(), np.percentile(d, 99), d.max(), ((a - b) ** 2).mean(), signed_rel

def main():
    exrs = glob.glob(os.path.join(OUT_DIR, "*_converged.*.exr"))
    if not exrs:
        sys.exit("No converged EXRs in {} - run VNA_Matrix.py in Mogwai first.".format(OUT_DIR))

    # Discover (config, viewpoint) pairs from filenames.
    runs = sorted({tuple(os.path.basename(f).split("_vp")[0:1] +
                         [os.path.basename(f).split("_vp")[1].split("_")[0]])
                   for f in exrs})

    refs = {}
    for vp in sorted({vp for _, vp in runs}):
        refs[vp] = load("01_ref", vp, "converged")
        if refs[vp] is None:
            sys.exit("Missing 01_ref vp{} - the matrix needs its reference.".format(vp))

    print("\n== CONVERGED vs 01_ref (linear HDR) ==")
    print("{:<22} {:>4} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "config", "vp", "mean|d|", "p99|d|", "max|d|", "MSE", "signedRel"))
    print("(signedRel is the bias detector: noise -> ~0, systematic error -> a stable shift)")
    for name, vp in runs:
        if name == "01_ref":
            continue
        img = load(name, vp, "converged")
        if img is None:
            continue
        mean, p99, mx, mse, srel = diff_stats(img, refs[vp])
        print("{:<22} {:>4} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>+12.4%}".format(
            name, vp, mean, p99, mx, mse, srel))

    # Empirical noise floor: Stage-A variants are identical estimators, so
    # their pairwise converged diffs are pure residual noise at this frame
    # count. Everything else should sit in the same range.
    stage_a = ["03_ris", "04_ris_sweep", "05_ris_adaptm", "06_ris_both"]
    imgs = {n: load(n, "1", "converged") for n in stage_a}
    pairs = [(a, b) for a, b in itertools.combinations(stage_a, 2)
             if imgs[a] is not None and imgs[b] is not None]
    if pairs:
        print("\n== NOISE FLOOR: pairwise diffs among Stage-A variants (vp1) ==")
        means, srels = [], []
        for a, b in pairs:
            mean, p99, mx, mse, srel = diff_stats(imgs[a], imgs[b])
            means.append(mean)
            srels.append(abs(srel))
            print("{:<22} vs {:<14} mean {:>10.3e}  max {:>10.3e}  signedRel {:>+10.4%}".format(
                a, b, mean, mx, srel))
        print("floor: mean|d| {:.3e}, |signedRel| {:.4%} - configs far above are SUSPECT".format(
            max(means), max(srels)))

    print("\n== 1-SPP variance vs converged 01_ref (lower = better sampler) ==")
    print("{:<22} {:>4} {:>14}".format("config", "vp", "1-spp MSE"))
    for name, vp in runs:
        img = load(name, vp, "1spp")
        if img is None or refs.get(vp) is None:
            continue
        mse = ((img - refs[vp]) ** 2).mean()
        print("{:<22} {:>4} {:>14.3e}".format(name, vp, mse))

if __name__ == "__main__":
    main()
