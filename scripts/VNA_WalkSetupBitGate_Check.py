# Checker for VNA_WalkSetupBitGate.py - BIT identity, not a noise floor.
#
#   py scripts/VNA_WalkSetupBitGate_Check.py
#
# PASS means every pixel of every viewpoint is bit-for-bit equal between
# useSharedWalkSetup OFF (original per-walk preamble) and ON (per-instance
# hoist). There is no tolerance and there must not be one: both arms run the
# same estimator over the same RNG stream, so the only correct answer is zero.
#
# A nonzero max-abs-diff is NOT "small enough". The likely causes, in order:
#   - the setup is being reused across a grid change (walkGridIdx not updated,
#     or the loop hands back interleaved instances and the guard is wrong),
#   - the brick cache is being held across a Grid it is not valid for, which is
#     the exact failure the single-grid rule at GridVolumeSampler.slang:553
#     exists to prevent,
#   - the two arms did not actually run the same config (check the log).
#
# This script deliberately does NOT print a percentage error. Reporting 0.0001%
# invites treating a correctness failure as a quality tradeoff.

import glob
import os
import sys

import numpy as np
import OpenEXR  # official bindings; opencv-python ships without EXR support

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\walksetup"
SPP = 320
VIEWPOINTS = (0, 1, 2)


def load(name, vp):
    pattern = os.path.join(OUT_DIR, "{}_vp{}_s{}.*.exr".format(name, vp, SPP))
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None, None
    path = files[-1]
    try:
        channels = OpenEXR.File(path).channels()
    except Exception as e:
        print("  !! failed to read {}: {}".format(path, e))
        return None, path
    if "RGBA" in channels or "RGB" in channels:  # interleaved layout
        key = "RGBA" if "RGBA" in channels else "RGB"
        return channels[key].pixels[..., :3], path
    return np.stack([channels[c].pixels for c in ("R", "G", "B")], axis=-1), path


def compare(name_a, name_b, vp):
    """Returns (verdict, detail). verdict in {'same','differ','missing'}."""
    a, pa = load(name_a, vp)
    b, pb = load(name_b, vp)
    if a is None or b is None:
        return 'missing', "{} / {}".format(
            "ok" if a is not None else "no " + name_a,
            "ok" if b is not None else "no " + name_b)
    if a.shape != b.shape:
        return 'differ', "shape {} vs {}".format(a.shape, b.shape)
    # Compare the raw bits, not the values: this also catches a -0.0/+0.0 or
    # NaN-payload difference that == would call equal.
    if a.tobytes() == b.tobytes():
        return 'same', "{} pixels".format(a.shape[0] * a.shape[1])
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    nz = d > 0
    return 'differ', ("{} of {} channel values differ, max abs {:.9g}, "
                      "median of differing {:.9g}"
                      .format(int(nz.sum()), d.size, float(d.max()),
                              float(np.median(d[nz]))))


def main():
    print("WALKSETUP BIT GATE")
    print("Reading {}".format(OUT_DIR))
    print("")

    # ---- CONTROL FIRST -----------------------------------------------------
    # offA vs offB is the SAME config twice. If it does not come back identical
    # the renderer is not reproducible and NO off-vs-on verdict from this
    # harness means anything - including a PASS, which would then only prove
    # that two nondeterministic runs happened to land together.
    print("CONTROL - offA vs offB (same config twice):")
    control_ok = True
    for vp in VIEWPOINTS:
        verdict, detail = compare("offA", "offB", vp)
        print("  vp{}: {} - {}".format(vp, verdict.upper(), detail))
        if verdict != 'same':
            control_ok = False
    print("")

    if not control_ok:
        print("INCONCLUSIVE - the harness is not reproducible, so it cannot")
        print("test bit-identity. Do NOT read the off-vs-on comparison below")
        print("as evidence about the hoist either way.")
        print("Find the nondeterminism first: anything accumulated with atomics")
        print("(radiance cache splats, stat buffers feeding the image) or any")
        print("state surviving updatePass will do it.")
        print("")

    # ---- THE ACTUAL QUESTION ----------------------------------------------
    print("TEST - offA vs on (original vs per-instance hoist):")
    failures = []
    missing = []
    for vp in VIEWPOINTS:
        a, pa = load("offA", vp)
        b, pb = load("on", vp)
        if a is None or b is None:
            missing.append(vp)
            print("  vp{}: MISSING ({} / {})".format(
                vp, "ok" if a is not None else "no off capture",
                "ok" if b is not None else "no on capture"))
            continue
        if a.shape != b.shape:
            failures.append(vp)
            print("  vp{}: FAIL - shape {} vs {}".format(vp, a.shape, b.shape))
            continue

        # Compare the raw bits, not the values: this also catches a -0.0/+0.0
        # or NaN-payload difference that == would call equal.
        same_bits = a.tobytes() == b.tobytes()
        if same_bits:
            print("  vp{}: PASS - bit-identical ({} pixels)".format(vp, a.shape[0] * a.shape[1]))
            continue

        failures.append(vp)
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        nd = int((d > 0).sum())
        idx = np.unravel_index(int(np.argmax(d)), d.shape)
        print("  vp{}: FAIL - {} of {} channel values differ, max abs {:.9g} at {}"
              .format(vp, nd, d.size, float(d.max()), idx))
        print("        off={} on={}".format(pa, pb))

    print("")
    if not control_ok:
        print("VERDICT: INCONCLUSIVE - control failed, see above. The hoist is")
        print("neither confirmed nor refuted by this run.")
        return 2
    if missing and not failures:
        print("INCONCLUSIVE - captures missing for vp{}. Re-run the gate."
              .format(", vp".join(str(v) for v in missing)))
        return 2
    if failures:
        print("FAIL - the hoist is NOT bit-identical. It is wrong; do not time it.")
        print("Check walkGridIdx is reset per RayQuery and updated on every")
        print("rebuild, in all four loops in VolumeInstanceSampler.slang.")
        return 1

    print("PASS - bit-identical on every viewpoint.")
    print("")
    print("NOW check the MECHANISM, which this script cannot see:")
    print("  grep '\\[COST\\]' in the Mogwai log; brickCache hit% must be HIGHER")
    print("  on the 'on' arm than the 'off' arm. Identical images with a flat")
    print("  hit% means the hoist never fired - harmless, worthless, drop it.")
    print("  Only if hit% rose is a timing A/B worth running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
