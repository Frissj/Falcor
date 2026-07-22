# Wave-uniform majorant mip gate - verdict step. Plain python (no Mogwai):
#
#   py scripts/VNA_WaveMipGate_Compare.py
#
# Reads the newest Mogwai log written by VNA_WaveMipGate.py plus the converged
# EXRs it captured, and prints an explicit PASS / FAIL for each question the
# gate asks. Nothing here re-renders; run the driver first.
#
# THE THREE CHECKS, IN THE ORDER THEY MUST BE READ
#
#   1. MECHANISM. For every lift > 0 config, [STEPOCC]'s lift fraction must be
#      nonzero. If it is ~0 the warps' lanes already agreed on the mip, the
#      lever never ran, and checks 2 and 3 say nothing about the idea. This is
#      reported FIRST and gates the rest, because "no effect" and "no
#      opportunity" produce identical occupancy numbers.
#
#   2. OCCUPANCY. [STEPOCC] shade occ against the 'off' config. This is the
#      quantity the lever exists to move, and it is more trustworthy than gpuMs
#      here: the lever deliberately trades ALU loop trips for tex taps, so time
#      can wash while the intended mechanism works perfectly. gpuMs is reported
#      alongside, never instead.
#
#   3. IMAGE. Unbiased at every lift, but NOT byte-identical - cell count drives
#      the sample stream - so this is a statistical check against a MEASURED
#      noise floor, not a bit compare.
#
# WHY THE FLOOR IS MEASURED AND NOT ASSUMED. The distWeight gate ended stuck at
# "differs by ~10x the expected noise floor" with no way to separate a bug from
# a heavy tail, because the floor was expected. Here the driver renders the
# reference config twice with different seeds. That pair is the same estimator,
# so it differs by NOISE ALONE, at this spp, on this scene, in this run - it IS
# the floor. Every lift is judged as a multiple of it.
#
# DRIFT GUARD. The driver interleaves two rounds of every config. If a config's
# two rounds disagree by more than the spread BETWEEN configs, thermal or clock
# drift is larger than the effect and the run is noise. That is checked here and
# reported as INCONCLUSIVE rather than folded into an average.

import glob
import os
import re
import sys

import numpy as np
import OpenEXR  # official bindings; opencv-python ships without EXR support

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\wavemip"
LOG_GLOB = r"C:\Users\Friss\Documents\Clouds\Falcor\build\windows-vs2022\bin\Release\Mogwai.exe.*.log"

# Must match VNA_WaveMipGate.py CONFIGS / IMAGE_CONFIGS, in order. The log has
# no config labels in it (script print() does not reach the Mogwai log), so the
# mapping is positional on [WORK] frame-0 blocks. Reorder the driver and you
# must reorder this.
CONFIGS = ['off', 'lift0', 'lift1', 'lift2', 'lift3']
ROUNDS = 2
BASELINE = 'off'
# Must match the driver. Counters are logged every risStatsInterval frames, so
# the warm-up emits them too - those frames are cache cold-start and must not
# reach the average. Anything at or below this frame index is discarded.
WARM = 192
IMAGE_CONFIGS = ['refA', 'refB', 'img1', 'img2', 'img3']
REF_A, REF_B = 'refA', 'refB'
SPPS = [256, 4096]

# A lift is clean if its difference from the reference sits within this multiple
# of the seed-to-seed floor. 4x rather than 1x because the floor is itself an
# estimate from a single pair, and MSE of a noisy image is a high-variance
# statistic; 4x still catches anything structural, which shows up as orders of
# magnitude, not factors of two.
FLOOR_MULT = 4.0
# Bias band. signed_rel is the detector that does NOT shrink with more pixels,
# so it carries the bias verdict. Floor-relative, with an absolute back-stop at
# the project's established +-0.15% matrix band so a freakishly clean floor
# cannot make the test impossible to pass.
BIAS_FLOOR_MULT = 3.0
BIAS_ABS_MIN = 0.0015
# Below this the lever provably did not run on a meaningful share of cells.
MIN_LIFT_FRACTION = 0.5  # percent


def newest_log():
    files = glob.glob(LOG_GLOB)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_log(path):
    """Split the log into per-config blocks and average each block's counters.

    Segmentation is positional: the VolumePathTracer frame counter resets to 0
    on every updatePass, so '[WORK] frame 0' opens a new block.

    Two ordering facts this depends on, both verified against a real log:

      1. Within one frame the pass emits [LOOPOCC], [STEPOCC], [SHADEOCC],
         [WORK], [COST] - so [STEPOCC] arrives BEFORE the [WORK] line that would
         open a new block. Dropping frame 0 outright is therefore load-bearing,
         not cosmetic: without it the frame-0 [STEPOCC] of config N+1 would be
         charged to config N.

      2. Counters are logged throughout the WARM window too, and those frames
         are cache cold-start. Keeping only frames > WARM removes them, and
         removes any spurious leading block as a side effect (it would contain
         nothing past the threshold).
    """
    blocks = []
    cur = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            mw = re.search(r'\[WORK\] frame (\d+) res (\d+x\d+) gpuMs ([\d.]+)', line)
            if mw:
                frame = int(mw.group(1))
                if frame == 0:
                    cur = {'gpuMs': [], 'occ': [], 'liftPct': [], 'liftAvg': [],
                           'res': set()}
                    blocks.append(cur)
                elif cur is not None and frame > WARM:
                    cur['gpuMs'].append(float(mw.group(3)))
                    # The gate runs at full WINDOW res on purpose (960x540 changes
                    # the warp divergence this lever targets), so resolution is
                    # not pinned and a mid-run window resize would silently
                    # invalidate every comparison. Recorded here, checked below.
                    cur['res'].add(mw.group(2))
                continue
            ms = re.search(
                r'\[STEPOCC\] frame (\d+) \| lift \d+ \| main occ [\d.]+ .*?'
                r'\| shade occ ([\d.]+) .*?\| lifted ([\d.]+)% of \d+ cells, avg \+([\d.]+) mip',
                line)
            if ms and cur is not None and int(ms.group(1)) > WARM:
                cur['occ'].append(float(ms.group(2)))
                cur['liftPct'].append(float(ms.group(3)))
                cur['liftAvg'].append(float(ms.group(4)))
    # A block with no measured frames is not a config - drop it rather than let
    # it shift the positional mapping by one.
    return [b for b in blocks if b['gpuMs']]


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def load(name, spp):
    pattern = os.path.join(OUT_DIR, "{}_vp1_s{}.*.exr".format(name, spp))
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
    # ~500k pixels; a systematic estimator error shows as a shift that does NOT
    # shrink with more pixels or frames.
    signed_rel = (a - b).mean() / max(b.mean(), 1e-12)
    return {
        'mean': d.mean(),
        'p99': float(np.percentile(d, 99)),
        'max': d.max(),
        'mse': ((a - b) ** 2).mean(),
        'signed_rel': signed_rel,
    }


def perf_phase():
    print("=" * 78)
    print("PHASE 1 - PERF")
    print("=" * 78)

    log = newest_log()
    if not log:
        print("  FAIL: no Mogwai log found at {}".format(LOG_GLOB))
        return False
    print("  log: {}".format(os.path.basename(log)))

    blocks = parse_log(log)
    need = len(CONFIGS) * ROUNDS
    if len(blocks) < need:
        print("  FAIL: found {} config blocks, expected at least {}. Did the "
              "driver finish?".format(len(blocks), need))
        return False
    # The driver runs the perf phase FIRST, so the blocks it produced are the
    # first `need` of them; the image phase appends more and is ignored here.
    blocks = blocks[:need]

    # RESOLUTION GUARD. The gate deliberately does not pin resolution - a
    # lane-occupancy lever has to be measured at the res its target phenomenon
    # lives at, and the repo's own rule (WdasSkyVNA.py _RENDER_RES) is that
    # 960x540 changes the warp divergence. The price is that a window resize
    # mid-run would compare two different workloads, and Mogwai log 183 shows
    # that happening for real. So: every block must agree, or the run is void.
    seen = set()
    for b in blocks:
        seen |= b['res']
    if len(seen) != 1:
        print("  FAIL: resolution changed mid-run ({}). The window was resized, "
              "so the configs did not render the same workload. Re-run without "
              "touching the window.".format(", ".join(sorted(seen)) or "none"))
        return False
    print("  res: {} (constant across all {} blocks)".format(seen.pop(), len(blocks)))

    rows = {}
    for r in range(ROUNDS):
        for i, name in enumerate(CONFIGS):
            b = blocks[r * len(CONFIGS) + i]
            rows.setdefault(name, []).append({
                'gpuMs': mean(b['gpuMs']),
                'occ': mean(b['occ']),
                'liftPct': mean(b['liftPct']),
                'liftAvg': mean(b['liftAvg']),
            })

    print()
    print("  {:<7} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "config", "gpuMs", "occ", "lift%", "avg+mip", "r1-r2"))
    for name in CONFIGS:
        rs = rows[name]
        g = mean([x['gpuMs'] for x in rs])
        o = mean([x['occ'] for x in rs])
        lp = mean([x['liftPct'] for x in rs])
        la = mean([x['liftAvg'] for x in rs])
        drift = abs(rs[0]['gpuMs'] - rs[1]['gpuMs']) if len(rs) > 1 else 0.0
        print("  {:<7} {:>9.3f} {:>9.3f} {:>9.2f} {:>9.2f} {:>9.3f}".format(
            name, g, o, lp, la, drift))

    base = rows[BASELINE]
    base_ms = mean([x['gpuMs'] for x in base])
    base_occ = mean([x['occ'] for x in base])

    # DRIFT GUARD: if a config's two rounds disagree by more than the spread
    # between configs, the run measured the machine, not the lever.
    worst_drift = max(abs(rows[n][0]['gpuMs'] - rows[n][1]['gpuMs'])
                      for n in CONFIGS) if ROUNDS > 1 else 0.0
    config_spread = (max(mean([x['gpuMs'] for x in rows[n]]) for n in CONFIGS)
                     - min(mean([x['gpuMs'] for x in rows[n]]) for n in CONFIGS))
    print()
    print("  drift check: worst round-to-round {:.3f} ms vs between-config "
          "spread {:.3f} ms".format(worst_drift, config_spread))
    if worst_drift >= config_spread:
        print("  INCONCLUSIVE: drift >= effect. Discard this run, do not average "
              "it. Re-run on a cold machine.")
        return False

    ok = True
    print()
    for name in CONFIGS:
        if name == BASELINE:
            continue
        rs = rows[name]
        lp = mean([x['liftPct'] for x in rs])
        o = mean([x['occ'] for x in rs])
        g = mean([x['gpuMs'] for x in rs])
        d_occ = o - base_occ
        d_ms = 100.0 * (g - base_ms) / base_ms if base_ms else 0.0
        if name == 'lift0':
            # Estimator-identical to 'off'; this line prices the wave reduction
            # instruction alone. Not a pass/fail, but the lever has to win this
            # back before it wins anything.
            print("  lift0 (instruction cost only): {:+.2f}% gpuMs, occ {:+.3f}"
                  .format(d_ms, d_occ))
            continue
        if lp < MIN_LIFT_FRACTION:
            print("  {}: FAIL (mechanism) - lifted only {:.2f}% of cells. The "
                  "warps' lanes already agreed on the mip; the lever never ran, "
                  "so its occupancy and timing say nothing.".format(name, lp))
            ok = False
            continue
        verdict = "PASS" if d_occ > 0 else "FAIL"
        if d_occ <= 0:
            ok = False
        print("  {}: {} (occupancy) - lifted {:.2f}% of cells, occ {:.3f} -> "
              "{:.3f} ({:+.3f}), gpuMs {:+.2f}%"
              .format(name, verdict, lp, base_occ, o, d_occ, d_ms))

    print()
    print("  NOTE: occupancy is the verdict, gpuMs is context. The lever trades")
    print("  ALU loop trips for tex taps, and tex is not in the top-5 SOL list,")
    print("  so time can wash while the mechanism works. A gpuMs regression WITH")
    print("  occupancy up means the trip counts converged but the extra null")
    print("  collisions cost more than the saved iterations - tune the lift, do")
    print("  not conclude the idea failed.")
    return ok


def image_phase():
    print()
    print("=" * 78)
    print("PHASE 2 - IMAGE")
    print("=" * 78)

    # Only the REFERENCE PAIR is mandatory - it is what defines the noise floor,
    # and without it nothing can be judged. Individual lift configs are reported
    # as they are found and skipped when they are not: a run that died partway
    # (or was killed for taking too long) still holds real verdicts for every
    # config that did finish, and throwing those away would mean re-rendering
    # them for nothing.
    if load(REF_A, SPPS[-1]) is None or load(REF_B, SPPS[-1]) is None:
        print("  SKIPPED: reference pair ({} / {}) not in {}"
              .format(REF_A, REF_B, OUT_DIR))
        print("  (driver was run with PERF_ONLY = True, or died before refB)")
        return None
    absent = [n for n in IMAGE_CONFIGS
              if n not in (REF_A, REF_B) and load(n, SPPS[-1]) is None]
    if absent:
        print("  PARTIAL: no captures for {} - judging the rest. Re-run to cover "
              "them.".format(", ".join(absent)))

    ok = True
    for spp in SPPS:
        a = load(REF_A, spp)
        b = load(REF_B, spp)
        if a is None or b is None:
            print("  s{}: SKIPPED (reference pair missing)".format(spp))
            continue
        floor = diff_stats(a, b)
        print()
        print("  s{} noise floor (refA vs refB - same estimator, different seed):"
              .format(spp))
        print("    mean {:.6f}  p99 {:.6f}  max {:.6f}  mse {:.6g}  signed_rel {:+.4f}%"
              .format(floor['mean'], floor['p99'], floor['max'], floor['mse'],
                      100.0 * floor['signed_rel']))
        mse_tol = FLOOR_MULT * floor['mse']
        bias_tol = max(BIAS_FLOOR_MULT * abs(floor['signed_rel']), BIAS_ABS_MIN)
        print("    tolerances: mse <= {:.6g} ({}x floor), |signed_rel| <= {:.4f}%"
              .format(mse_tol, FLOOR_MULT, 100.0 * bias_tol))
        for name in IMAGE_CONFIGS:
            if name in (REF_A, REF_B):
                continue
            img = load(name, spp)
            if img is None:
                print("    {}: SKIPPED (missing)".format(name))
                continue
            st = diff_stats(img, a)
            mse_ok = st['mse'] <= mse_tol
            bias_ok = abs(st['signed_rel']) <= bias_tol
            # s4096 carries the bias verdict; s256 carries the variance one. A
            # clean s4096 with a loud s256 is the transmittance walk, which is
            # not majorant-invariant - noted, not failed.
            hard = (spp == SPPS[-1])
            verdict = "PASS" if (mse_ok and bias_ok) else ("FAIL" if hard else "NOISY")
            if hard and not (mse_ok and bias_ok):
                ok = False
            print("    {}: {}  mse {:.6g} ({:.2f}x floor)  signed_rel {:+.4f}%  "
                  "mean {:.6f}  p99 {:.6f}"
                  .format(name, verdict, st['mse'],
                          st['mse'] / floor['mse'] if floor['mse'] else float('inf'),
                          100.0 * st['signed_rel'], st['mean'], st['p99']))

    print()
    print("  NOTE: analog delta tracking returns the medium's TRUE free-flight")
    print("  distribution for any bounding majorant, so the DISTANCE sampler")
    print("  should be distributionally identical, not merely unbiased - s256")
    print("  noise should match too. The delta-tracking TRANSMITTANCE walk is")
    print("  the one legitimate exception; a NOISY s256 with a PASS s4096 is")
    print("  that walk, not a bias.")
    return ok


def main():
    perf_ok = perf_phase()
    img_ok = image_phase()

    print()
    print("=" * 78)
    if not perf_ok:
        print("VERDICT: NO-GO - the lever did not raise lane occupancy (or the run")
        print("was too noisy to tell). Do not promote it.")
        return 1
    if img_ok is None:
        print("VERDICT: PERF PASS, image UNVERIFIED. Re-run with PERF_ONLY = False")
        print("before promoting - this lever is not byte-identical, so occupancy")
        print("alone is not enough to ship it.")
        return 2
    if not img_ok:
        print("VERDICT: NO-GO - occupancy moved but the converged image did not")
        print("match the reference inside its own measured noise floor. A coarser")
        print("majorant is still a bound, so an unbiasedness failure here is an")
        print("IMPLEMENTATION bug in gvsWaveLiftMip, not a property of the idea.")
        return 1
    print("VERDICT: GO - occupancy up, converged image inside the measured noise")
    print("floor. Pick the lift with the best occupancy/gpuMs trade, re-confirm at")
    print("full res (the 4.12x came from a full-res capture), then promote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
