# Compare for VNA_WalkSetupTimeGate.py.
#
#   py scripts/VNA_WalkSetupTimeGate_Compare.py [path-to-log]
#
# Reads the newest Mogwai log, segments it on the [WORK] frame counter (which
# resets to 0 on every updatePass), assigns blocks to configs in CONFIGS order
# repeated ROUNDS times, and reports [FRAME] wall over the measured window.
#
# THE NOISE FLOOR IS MEASURED, NOT ASSUMED. Each config runs ROUNDS times,
# interleaved, so the spread between rounds of the SAME config is this run's
# own noise. If the config gap does not clear that spread, the answer is
# INCONCLUSIVE - not "a small win". Log 194 drifted 22% at frozen counters on
# this machine; a gate that ignores that will report whatever the thermals did.
#
# GROUND TRUTH IS wall / FPS. gpuMs is printed for attribution only: in log
# 194's toggle the pass timer and the wall clock disagreed by more than 1 ms.

import glob
import os
import re
import sys

LOG_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\build\windows-vs2022\bin\Release"
CONFIGS = ['off', 'on']
ROUNDS = 3
WARM = 768   # Must match the driver: frames below this are on the retrain ramp.

FRAME_RE = re.compile(
    r"\[FRAME\] frame (\d+) \| wall ([\d.]+) ms \(([\d.]+) FPS.*?"
    r"pass ([\d.]+) ms.*?main ([\d.]+) shadeMain ([\d.]+)")
COST_RE = re.compile(r"\[COST\] frame (\d+).*?brickCache hit ([\d.]+)%")


def newest_log():
    files = glob.glob(os.path.join(LOG_DIR, "Mogwai.exe.*.log"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def parse(path):
    """Returns list of blocks; each block is dict(frames=[...], hits=[...])."""
    blocks = []
    cur = None
    prev = None
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            m = FRAME_RE.search(line)
            if m:
                fr = int(m.group(1))
                if prev is None or fr <= prev:
                    cur = {'frames': [], 'hits': []}
                    blocks.append(cur)
                prev = fr
                cur['frames'].append({
                    'f': fr, 'wall': float(m.group(2)), 'fps': float(m.group(3)),
                    'pass': float(m.group(4)), 'main': float(m.group(5)),
                    'shade': float(m.group(6))})
                continue
            m = COST_RE.search(line)
            if m and cur is not None:
                cur['hits'].append((int(m.group(1)), float(m.group(2))))
    return blocks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_log()
    if not path or not os.path.exists(path):
        print("No log found in {}".format(LOG_DIR))
        return 2
    print("WALKSETUP TIMING GATE")
    print("Log: {}".format(path))

    blocks = parse(path)
    want = len(CONFIGS) * ROUNDS
    print("Blocks found: {} (expected {})".format(len(blocks), want))
    if len(blocks) < want:
        print("")
        print("INCONCLUSIVE - fewer blocks than the driver should have produced.")
        print("The run did not finish, or an earlier run's blocks are mixed in.")
        return 2
    blocks = blocks[-want:]   # tolerate a warm-up block from a previous graph

    print("")
    print("Per block (measured window = logged frames >= {}):".format(WARM))
    print("  {:<5} {:<5} {:>8} {:>7} {:>8} {:>8} {:>8} {:>9}".format(
        "round", "cfg", "wall", "fps", "pass", "main", "shade", "brickHit"))

    results = {c: [] for c in CONFIGS}
    for i, blk in enumerate(blocks):
        rnd, cfg = divmod(i, len(CONFIGS))
        name = CONFIGS[cfg]
        win = [f for f in blk['frames'] if f['f'] >= WARM]
        if not win:
            print("  block {} has no frames past WARM - risStatsInterval too "
                  "coarse for MEASURE?".format(i))
            return 2
        rec = {k: mean([f[k] for f in win]) for k in ('wall', 'fps', 'pass', 'main', 'shade')}
        hits = [h for (fr, h) in blk['hits'] if fr >= WARM]
        rec['hit'] = mean(hits)
        rec['n'] = len(win)
        results[name].append(rec)
        print("  {:<5} {:<5} {:8.2f} {:7.1f} {:8.2f} {:8.2f} {:8.2f} {:8.1f}%".format(
            rnd, name, rec['wall'], rec['fps'], rec['pass'], rec['main'],
            rec['shade'], rec['hit']))

    # ---- MECHANISM CHECK ---------------------------------------------------
    print("")
    hoff = mean([r['hit'] for r in results['off']])
    hon = mean([r['hit'] for r in results['on']])
    print("MECHANISM - brickCache hit: off {:.1f}%  on {:.1f}%  ({:+.1f} pp)"
          .format(hoff, hon, hon - hoff))
    if hon - hoff < 1.0:
        print("  The hoist did NOT fire in this run. Nothing below is about the")
        print("  change; check GVS_SHARED_WALK_SETUP actually flipped.")
        return 2

    # ---- NOISE FLOOR FROM THE SAME CONFIG ----------------------------------
    print("")
    spread = {}
    for c in CONFIGS:
        w = [r['wall'] for r in results[c]]
        spread[c] = max(w) - min(w)
        print("  {} wall across rounds: {}  spread {:.2f} ms"
              .format(c, " ".join("{:.2f}".format(x) for x in w), spread[c]))
    floor = max(spread.values())

    woff, won = mean([r['wall'] for r in results['off']]), mean([r['wall'] for r in results['on']])
    delta = won - woff
    print("")
    print("RESULT  wall: off {:.2f} ms  on {:.2f} ms   delta {:+.2f} ms ({:+.1f}%)"
          .format(woff, won, delta, 100.0 * delta / woff))
    print("        fps : off {:.1f}     on {:.1f}"
          .format(mean([r['fps'] for r in results['off']]), mean([r['fps'] for r in results['on']])))
    print("        attribution (gpuMs, NOT ground truth): main {:+.2f}  shade {:+.2f}"
          .format(mean([r['main'] for r in results['on']]) - mean([r['main'] for r in results['off']]),
                  mean([r['shade'] for r in results['on']]) - mean([r['shade'] for r in results['off']])))
    print("        same-config noise floor: {:.2f} ms".format(floor))

    print("")
    if abs(delta) <= floor:
        print("INCONCLUSIVE - the config gap ({:.2f} ms) does not clear this "
              "run's own".format(abs(delta)))
        print("noise floor ({:.2f} ms). The mechanism fired but the frame did "
              "not move".format(floor))
        print("measurably. Do not report this as a win; re-run with more ROUNDS")
        print("if you want a tighter floor.")
        return 2
    if delta < 0:
        print("PASS - the hoist is faster by {:.2f} ms ({:.1f}%), clearing a "
              "{:.2f} ms floor.".format(-delta, -100.0 * delta / woff, floor))
        return 0
    print("FAIL - the hoist is SLOWER by {:.2f} ms, clearing a {:.2f} ms floor."
          .format(delta, floor))
    print("More cache hits but less speed means the reuse is not where the time")
    print("was. Revert it rather than keeping a change that costs frames.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
