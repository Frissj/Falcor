# VNA Build Handoff 9 — the "shoot the messenger, twice" session (2026-07-20 night)

Supersedes HANDOFF_8. Every number below was measured this session; every
verdict has the run number that produced it.

---

## HEADLINE

- **Tracker RR and the radiance-cache CV are EXONERATED and SHIPPED.**
  HANDOFF_8's central mystery (-0.45..-0.86% "bias") was never in the
  renderer: it was (1) a GPU wave-op miscompile corrupting the probe's stat
  flush (runs 127-131) and (2) the scorer masking by the base realization's
  own luma percentile (selection bias, same disease as probe v2 binning by
  Tref). Fresh sweep under the fixed scorer: t_rr7 +0.05%, t_p100 -0.05%,
  t_p25 -0.02% @1024 - ALL PASS. Ship config now: `trRRThreshold 0.05`,
  `trRRMode 31`, rad cache adopted at p 0.25.
- **The trRR prize landed**: brickCands 13.6 -> 6.4/px, rqIters 30.1M ->
  13.1M, escapeT bucket 0.0 cells/px (dead trackers abort the traversal).
- **[WORK] now prints the frame split**: `gpuMs T (main A + shade B)` from a
  nested GPU timer. Current ship @2560x1351: ~27-30ms = main ~11 + shade ~16.
  shadeMain still leads (~48% of total frame).
- **Perf-lever campaign: 3 built, all 3 measured OUT** (in-session A/Bs, the
  only kind this machine permits - cross-session gpuMs drifts +-25%, bitten
  AGAIN this session):
  - Scatter-queue sort: LOSS (bounces occ 0.223 sorted vs 0.228 unsorted).
    Screen order was already the smart order. Default off, §4.
  - Brick prefetch: LOSS (~-1ms; main 13.1 -> 11.2 with it OFF). Speculation
    waste at segment ends + registers beat the 6.2% promote win. Default off, §4.
  - Warp-aware residual roulette: **FAILED ITS GATE, -3.0% @1024, real bias**
    - and the cause is a code defect, not the idea (§5). Fix sketched; OFF.
- **The real yield of the session is measurement infrastructure** (§3): the
  det-key paired probe + coin telemetry + self-check canaries, the shared-mask
  scorer with a printed noise floor, the main/shade timer split, per-pixel
  bounce-count telemetry, and three hard-won discipline rules (§6).

## 1. THE trRR EXONERATION — how HANDOFF_8's verdict fell apart

Chronology (all 2026-07-20 evening, runs = Mogwai.exe.NNN.log):
1. Probe v4 (det-key bins, run 126): walk E-bias read -35%/-23% thick bins.
   Believed. WRONG - see (4).
2. Probe v5 ([TRRPROBE2] coin telemetry, runs 127-128): surv > coins per bin
   while per-lane invariant checks read 0 violations - set-theoretically
   impossible => the flush itself was corrupt. Cause: WaveActiveSum +
   WaveIsFirstLane + [unroll] + atomics miscompiles (reconvergence shape).
   Runs 127/128 were dropping ~40% of wave adds from EVERY slot.
3. Per-lane atomics for coin slots (run 129): counters sane; coin measured
   per-event FAIR (theta*survives == sum TrBefore +-1%) while the walk still
   read -42%. Coin acquitted. Decorrelated hash coin (bit4, run 131): bias
   unchanged to 0.002pp => stream-conditioning theory dead. Mass accounting:
   coins touch <=125 T-units of bin1's 9758 while 4076 were "missing" - the
   RR block COULD NOT be the thief (33x too small). Something else was lying.
4. Removed the LAST wave ops from the probe (run 132): walk E-bias ~0 in
   EVERY bin, all phases (mode 8 / bit4 / A/A null identical). The -35/-42%
   was the wave-flush miscompile all along, in every run since v4 shipped.
5. Scorer autopsy (same night): per-pair mask = base realization's own 60th
   luma percentile selects pixels where the base's noise ran HIGH => every
   comparison reads low. Shared mean-of-all-rows mask: ntrE_s1 -0.42% "FAIL"
   -> +0.01%; noise floor (ntb seed trio) was mostly mask artifact too.
   Sweep rescored: everything PASSes. Fresh sweep (this run): same.

Residual curiosities, non-blocking: ntrE vs ntb (seed 0 only) reads ~-0.28,
right at the floor - the ntb seed-0 realization runs high (also seen at
+0.79% under a worse mask); both matched-seed pairs PASS. [TRRPROBE2-CHK]
negTr = 0 armed AND ref: the [0,2]-truncation suspect never fires in this
scene. bin0 coin bias +0.25% +- 0.33%/frame: statistically zero.

## 2. CURRENT SHIP STATE (WdasSkyVNA.py)

rouletteMinQ 0.125 · tauCache ON (coarseCells 0) · radCache ON cut 3 p 0.25
train 8 EMA 0.10 · **trRRThreshold 0.05, trRRMode 31** (all sites + bit4
hash coin - bit4 is hygiene, proven identical, kept because it is free) ·
radWarpRRLanes 0 (FAILED gate, §5) · useCompaction ON · useScatterSort OFF
(§4) · useBrickPrefetch OFF (§4) · wavefront OFF (unchanged) · temporal ON.

@2560x1351: gpuMs ~27-30 (main ~11 + shade ~16), cells 81.5M/frame
(candGen 19.0/px + shadeNEE 4.6/px), brickCache hit 61.6%, occSkip 54.3%.

**Mode-word gotcha**: the props parser clamps trRRMode - it silently ate 24
until the clamp was widened to 31 (run 130 lost to this). Any new bit needs
that clamp bumped: VolumePathTracer.cpp property parse for kTrRRMode.

## 3. MEASUREMENT INFRASTRUCTURE BUILT THIS SESSION (all in the build)

- **[TRRPROBE] v4**: paired RR/ref escape walks binned by DETERMINISTIC key
  exp(-coarseOpticalDepth(view ray)). Slots 46..53, fx multiples of 20 so
  theta-atoms land on integers, PER-LANE atomic flush.
- **[TRRPROBE2] + [TRRPROBE2-CHK]**: per-coin telemetry (count/sumBefore/
  survives per det-key bin, slots 54..65) + canaries (slots 66..70:
  survGtCoins must=0, refCoins must=0, coinLanes = RR exposure, negTr
  armed/ref). Arm via any mode-bit-3 row; VNA_TrRRProbe.py runs a 3-phase
  self-contained probe (mode 8 / mode 24 / A/A null) in ~2 min.
- **trRR bit4**: coin from jenkins-hash of (frame seed, ray bits, coin
  ordinal); does NOT advance sg. Proven behavior-identical here; makes the
  estimator provably generator-independent.
- **[WORK] main/shade gpuMs split**: nested GpuTimer around phase A. THE tool
  for attributing any future change (it decided the prefetch verdict alone).
- **queueClasses histogram** on [SHADEOCC] (slots 73..76) + **per-pixel
  measured bounce count** in reservoir `_pad1` (written by shadeMain,
  bounce>=1, 0 = no-path sentinel). NOTE: `_pad0` (the T hint) is
  RR-QUANTIZED since trRR shipped - exactly 0 for killed walks, >= ~theta
  for survivors - it can never again rank work in thick pixels. Anything
  wanting a work predictor uses `_pad1`.
- **Brick-prefetch promote counter** (slot 71, [COST] "promote").
  kRisStatSlots = 77.
- **Scorer (VNA_RadCacheScore.py) rewritten**: mask = 60th percentile of the
  MEAN of all captures at the window (co-selection diluted ~1/rows); prints
  the measured realization noise floor (no-RR seed trio) FIRST; verdicts
  require beyond gate AND floor. VNA_TrRRNullScore.py = standalone null/gate
  pair check.

## 4. LEVERS MEASURED OUT (kept compiled as A/B records, defaults OFF)

- **Scatter-queue sort** (classify/offset/scatter between argsMain and
  shadeMain, image-identical by per-pixel seeding). Run 139 in-session A/B
  with a WORKING predictor (bounce-count classes 320k/75k/29k/7k): gpuMs
  wash, bounces occ 0.223 sorted vs 0.228 unsorted, march occ 0.301 vs
  0.309. Append order is screen order; neighbors already share path length
  AND bricks. Reopen ONLY as class-major + spatial-minor STABLE sort (the
  atomic-cursor scatter is unstable within class - that instability is what
  destroyed locality). Predictor history in scatterClassMain comments.
- **Brick prefetch** (2-entry cache + speculative next-brick loads in
  stepToNextCollision; value-carry texel pipeline in the residual walk).
  Run 143 in-session A/B via the [WORK] split: main 13.1 ON vs 11.2 OFF,
  shade delta inside noise, net ~ -1ms. Waste at segment ends + registers
  beat a 6.2% promote conversion. COMPILE-TIME define GVS_BRICK_PREFETCH
  (run 137: a runtime flag cannot A/B register pressure - checkbox rebuilds
  the program). Reopen only with continuation-likelihood gating.
  (Run 137's gpuMs 38 was also the first version adding ptr/occ loads to
  every no-tap residual node - hit rate 61.6 -> 46.4 - fixed before the
  final verdict, so the OFF verdict is against the CLEAN implementation.)

## 5. WARP-AWARE RESIDUAL ROULETTE — FAILED GATE, CAUSE KNOWN, FIX SKETCHED

t_wrr8 (vs t_p25): **-2.40% @256, -3.02% @1024, window-invariant, 10x the
0.28% noise floor. REAL.** Do not adopt; radWarpRRLanes stays 0.

Root cause (code inspection after the number came in):
`WaveActiveCountBits(true)` sits INSIDE the `radIsResidual` branch - a
DIVERGENT branch - so `alive` counts only residual lanes, not live lanes.
A full warp with 2 residual paths reads alive=2 => p=0.25 per bounce
instead of no-op. The roulette therefore ran far more aggressively than
designed and COMPOUNDED per bounce (a survivor of five p=0.25 coins
carries 1024x throughput). The estimator is still formally unbiased, but
its mass sits in compensations so rare that a 1024-frame mean reads -3%:
kill-side loss is realized every frame, the balancing fireflies are not.
Practically indistinguishable from bias at any usable window.

Redesign requirements (all three, then re-gate):
1. Count ALL live lanes: hoist `alive = WaveActiveCountBits(true)` to the
   loop head, OUTSIDE any divergent branch (and remember runs 127/128:
   wave ops in divergent flow around atomics have miscompiled here).
2. Floor p (e.g. p = max(alive/lanes, 0.5)) and/or cap CUMULATIVE
   compensation per path so the tail cannot compound past ~4x.
3. Use the bit4-style hash coin (exists, free).
Also worth knowing: measured avg path is 4.2 bounces (not HANDOFF_8's 8);
avg warp-max 19; paths >=20 bounces are 7.1k of 432k queued (1.6%) but sit
in 41% of warps.

## 6. MEASUREMENT DISCIPLINE — additions (each bit us tonight)

- **NEVER use wave intrinsics in probe/stat flushes.** WaveActiveSum +
  WaveIsFirstLane + [unroll] + InterlockedAdd miscompiled (dropped adds,
  per-bin impossibilities). Per-lane atomics everywhere; counter costs are
  negligible against being lied to. Three separate probe generations were
  corrupted before this was caught (v1-v3's "wild numbers" now suspect too).
- **Never mask/bin/select by any quantity derived from either side of a
  comparison.** Third bite: probe v2/v3 (bin by Tref), the scorer (mask by
  base percentile). Deterministic or pooled-independent keys only.
- **Cross-session gpuMs is NOISE on this machine (+-25%).** Only in-session
  A/Bs count. Runtime toggles cannot A/B register pressure - use a define
  and let the checkbox rebuild the program in-process.
- **Check the run's config fingerprint before reading its numbers.** Two
  runs (140, 142) were wasted on pass-default configs (no --script):
  fingerprint = samplerCalls 4.0 / coarseCells > 0 / escapeT > 0. Ship
  fingerprint = samplerCalls 1.7, coarseCells 0, escapeT 0.0.
- **Sentinels and quantized hints**: an "empty" histogram class is data
  (class 1 == 0 exposed the RR-quantization of _pad0). Print histograms for
  anything a classifier keys on, BEFORE trusting the classifier.
- **The clamp trap**: props parsers clamp; a new mode bit silently dies
  (run 130). grep the parse site when adding bits.

## 7. NEXT STEPS, RANKED

1. **Redesign the warp roulette per §5 and re-gate** (t_wrr8 row exists).
   It aims at the measured #1 cost: shade ~16ms at occ 0.22.
2. If (1) passes but occupancy barely moves: **stable class-major /
   spatial-minor sort** (§4 tombstone has the requirements; packed-bound
   arithmetic from run 139: warpMaxSum 256k -> ~75-83k, i.e. ~3x on the
   bounce-loop critical path, ceiling ~20-30% of frame - paper numbers with
   measured inputs, discount accordingly).
3. **Fresh Nsight capture** of the current ship build (last one predates
   trRR + all levers; per-dispatch ranges or GPU Trace so main/shadeMain
   separate). Re-rank LGSB vs pred-on before any new lever.
4. candGen is now the bigger phase-A term (19.0 cells/px): the parked
   ReSTIR-PG idea (RIS target from resampled history) is the matching
   lever. Note the escape-term froxel tau-control from HANDOFF_8 §8.4 is
   OBSOLETE - trRR already zeroed escapeT.
5. Parked, unchanged: ambient tau channel (Nubis), reservoir compression,
   duplication maps.

## 8. FILES TOUCHED THIS SESSION

GridVolumeSampler.slang (brick cache 2-entry + define-gated prefetch, coin
telemetry statics, bit4 coin, negTr counter) · VolumeInstanceSampler.slang
(TrRRCB seed + bit4 + coin site) · VolumePathTracer.cs.slang (probe v4/v5 +
CHK, sort kernels, warp-RR block, _pad1 feed, timer split, slots ->77) ·
VolumePathTracer.{cpp,h} (clamp 31, defines, timers, buffers, UI, [COST]/
[SHADEOCC]/[WORK] extensions) · VNA_RadCacheScore.py (shared mask + floor)
· VNA_RadCacheSweep.py (t_wrr8 row) · VNA_TrRRProbe.py (3-phase probe) ·
VNA_TrRRNullScore.py (new) · WdasSkyVNA.py (trRR re-enable, rad adopt,
radWarpRRLanes 0).

Git: levers are commits 68521b00 / cdf8d645 / 048c4924; fixes + verdicts in
3426689a, e0f8841b, 1aac209e, 014e750f. UNCOMMITTED in the tree right now:
mUseBrickPrefetch default false (the run-143 verdict). Commit policy per
user: NEVER commit without being told.
