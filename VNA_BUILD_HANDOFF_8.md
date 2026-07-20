# VNA Build Handoff 8 — the "cache the mean, roulette the residual" session (2026-07-20, cont.)

Supersedes HANDOFF_7's §7. Everything below was measured this session; every
"RESOLVED/DEAD" has the number that killed it next to it.

---

## HEADLINE

- **~10%+ real frame reduction shipped and image-verified**: sun-tau cache
  (−9.8% SM instructions, −8.6% CS warp latency on drift-proof counters),
  roulette knee q=0.125 (gate PASS, 2.3x margin), 8×8 CTA (smem launch stall
  11.2% → 0.4%).
- **The radiance-cache control variate is BUILT and running** (the ReSTIR-FG
  idea, unbiased form): deep-bounce tail's mean lives in a trained world-grid
  cache; paths past the cut estimate only the residual. shadeMain bounces
  −47%, warp critical path −37%. **NOT yet gate-adopted** — its sweeps were
  contaminated first by harness bugs, then by the trRR bias (below).
- **Tracker RR is PULLED (trRRThreshold=0 in the script): real bias, root
  cause NOT yet found.** Five probe/sweep iterations narrowed it: NOT the RR
  algebra (sim clean), NOT adaptive M (ablation identical), NOT footprint mips
  (identical), NOT the temporal chain (persists temporal-off), NOT generator
  choice (probe v3), and the walk-level E-probe is STILL inconclusive because
  every probe so far had a measurement flaw of its own (§5).
- **The prize if the RR hunt is won**: with RR + commit-shrink live, the fused
  walk's RayQuery traversal HALVES (brickCands 13.6→6.4/px, rqIters 30.1M→13.1M,
  cells 94M→81.5M). That is why the hunt is worth finishing.
- **Three gate-methodology defects found in one day** — all the same disease
  (statistic's support ≠ mechanism's support). Saved to auto-memory as
  feedback-statistic-support. See §6 before trusting ANY sweep.

---

## 1. SHIPPED AND VERIFIED (in canonical WdasSkyVNA.py)

| change | files | evidence |
|---|---|---|
| Sun-tau cache (Nubis3 slides 127-130 / UE `DIM_USE_TRANSMITTANCE_VOLUME` port) | `VolumePathTracer.{cs.slang,cpp,h}` — TauCB, `tauDirMain`, `tauBuildMain` | coarseCells 39.5M/frame → 0; SM inst −9.8%; local-load sectors 2.16→0.70%; user A/B'd image: identical. Target-only ⇒ unbiased for any cache content. |
| `rouletteMinQ: 0.125` | script | rq2 sweep: +0.064% cloudy (PASS), 1-spp >100-fireflies 2 vs baseline's 13. Knee is sharp: 0.15 FAILS (+0.33%). |
| 8×8 CTA for `main` | `.cs.slang` numthreads + comment | smem launch stall 11.21→0.43% (the 23552 B was ~92 B/thread of RayQuery traversal stack — driver-allocated, no groupshared exists; verified by DXIL disassembly). Occupancy unchanged (registers still bind, CTA shape cancels) but the register step function is now fine-grained: 112→18, 96→20 warps. |
| primaryList live-range cut + shift-path re-gather | `.cs.slang` | Register-neutral (128 cap driver-pinned, 6 configs bit-identical), kept for reduced spill traffic. instSlabs counter rises when shift path fires — expected, documented. |

## 2. BUILT, DEFAULT-OFF OR PENDING GATE

- **Radiance-cache CV** (`useRadCache`, ON in script pending gate): RadCB,
  `radResolveMain`, training/consumption split (1-in-8, hash), confidence gate
  0.5, EMA 0.10, cut bounce 3, residual survival p. `radCutBounce=0` is the
  LIVE off-switch. p=1.0 is algebraically net-zero — any gate shift at p100 is
  a CV bookkeeping bug, and the (contaminated) sweeps never cleanly tested it.
  Gate it AFTER trRR is resolved, with §6 discipline.
- **Wavefront phase B** (`useWavefront: False`): per-bounce requeue + tail
  finisher, byte-identical (RNG carried in PathState). **Measured monotonically
  slower in K** — the divergence is INTRA-bounce (single-bounce marching cost
  varies 10x+ between lanes), which requeueing cannot equalize. Kept compiled
  as the A/B record. Do not reopen without new data.
- **Tracker RR** (`trRRThreshold: 0.0` = pulled; mode bitmask bit0 NEE / bit1
  fused / bit2 shrink / bit3 non-fused escape): §5 is the hunt state.

## 3. INVESTIGATION RESULTS (probes stay in the build, kEnable-style)

- **[LOOPOCC]**: coarse DDA was 0.342 SIMD-occupancy, 115.5M warp-slots = 3.6x
  the whole RQ traversal → killed by the tau cache. RQ traversal itself 0.94 —
  never the problem.
- **[DIVERGE]/[SHADEOCC]**: main's 78% idle lanes = fully-idle warps (31.2/32
  busy in marching warps) — benign, un-packable. shadeMain fused loop 0.271
  bounce-occupancy (avg path 8, warp-max 29) → attacked by the CV, not by
  scheduling (wavefront measured dead).
- **Registers**: 128 is a driver TARGET, not demand — six configs bit-identical
  (incl. all-RayQuery-deleted and a 27-value live-range cut). Spill traffic
  small (LDL 2.2%→0.7% post-tau). Register work is CLOSED as a lever.
- **[TRRPROBE] slots 46..53**: paired escape walks RR/ref, binned. See §5 for
  why v1-v3 readings are each invalid and what v4 must do.

## 4. PAPERS/UE — settled verdicts

- **ReSTIR PT Enhanced**: compaction already ported; their §6.2.4 roulette
  answer is GATE-BLOCKED in this scene (deep MS carries energy; q>0.125 fails).
  Leftovers if ever needed: reservoir compression 88→64B, reciprocal pairing
  (only if spatial reuse is fixed), duplication maps (if boiling appears).
- **ReSTIR PG**: orthogonal (variance/guiding, not cost). One idea parked:
  fit the RIS target from resampled history instead of hand-built phase×tau.
- **Nubis3**: tau cache = their lighting grid, ported unbiased (target-only).
  Unmined nugget: second cache channel for sky-hemisphere ambient occlusion in
  the same 39k-walk bake.
- **ReSTIR-FG**: the mechanism (terminate into cached radiance) IS the rad
  cache CV, done unbiased. The repo itself stays surface-only/unusable.

## 5. THE trRR BIAS HUNT — state and the exact next step

Timeline of eliminations (each row = one sweep/probe):
1. All-configs FAIL −0.45..−0.86 @256 → confounded by temporal chain (M-cap 20
   mixes slowly; identical seeds replay identical realizations).
2. Temporal-off `ntr` FAIL −0.45% → real, not chain. Sim of the exact scheme:
   −0.08% ≈ clean → not the algebra.
3. Site split: NEE-only PASS −0.005%, escape-only FAIL −0.73%.
4. Ablations: adaptive-M-off −0.7285, footprint-mips-off −0.7284 — IDENTICAL
   (both ablations are behavioral no-ops in this scene; excluded anyway).
5. Seed test (7777/31337): −0.51/−0.56 → systematic core ~−0.6%, not stream
   realization.
6. Probe v1 (global paired sumT): ±0.02% "clean" — INVALID: sum dominated by
   sky T~1 where RR never fires; thick pixels invisible.
7. Probe v2 (binned by Tref): wild numbers — INVALID: TinyUniform ref
   generator; bin3 (RR can't fire) read −0.44% ⇒ generator-sensitive walk OR
   next item.
8. Probe v3 (same-family generator): bin3 STILL −0.38% ⇒ **the binning itself
   is selection-biased: conditioning a bin on the ref walk's own realization
   inflates E[ref|bin] for straddler pixels while the RR side is
   unconditioned. ALL v2/v3 bin readings are invalid by construction.**

**NEXT (do this first): probe v4 — bin by a DETERMINISTIC key.** Use
`coarseOpticalDepth(view ray)` (analytic, zero RNG — exp(−tau) of the mean
field) as the bin key. Both walks are then unconditioned within bins; per-bin
ratios become honest. If thick bins then show the bias, the mechanism is
inside the walk (suspects left: interaction with the internal per-node RR at
GridVolumeSampler.slang:658, FP32 at Tr~1e-6, or the [0,2]-factor recovery
truncation being mishandled somewhere real, not in the idealized sim). If v4
reads clean everywhere, the bias is NOT in the walk and the paired-probe
technique moves to total pixel radiance.

**Open flag from v2/v3**: bin3's residual −0.38% under matched generators is
either selection bias (v4 will say) or a REAL generator/walk sensitivity. If
any generator sensitivity survives v4, note that GridVolumeSampler's public
dispatch reseeds walks via TinyUniform(jenkinsHash) at :343/:378 — that would
put a generator question under the REFERENCE estimators too. Do not chase
this until v4 separates it.

## 6. MEASUREMENT DISCIPLINE — additions to HANDOFF_7 §5 (all bit us today)

- **Statistic support must match mechanism support** (memory:
  feedback-statistic-support). signedRel vs cloudy-mean; global sumT vs thick
  bins; single-seed gates vs stream-decorrelating changes. Bin/pair/seed-
  repeat by the firing condition or the measurement acquits the guilty.
- **256-frame windows are INSUFFICIENT for changes that perturb the RIS/
  candidate stream** with temporal reuse on: chain offsets halve 256→1024 but
  don't vanish. Options: 1024+ windows, temporal-off rows (only valid for
  code compiled in non-fused paths — USE_FUSED_SWEEP requires temporal!), or
  seed-averaged repeats (seedOffset property exists now).
- **Never bin a paired comparison by one side's own realization** (selection
  bias). Deterministic keys only.
- **updatePass with identical props can silently no-op** → accumulation resets
  must alternate DIFFERENT prop dicts (fixed in VNA_RadCacheSweep.py). Verify
  reset-worked via selfMSE arithmetic: true 1spp vs converged ≈ per-frame
  variance; ~N-frame accum ≈ var×(1/N−1/(N+M)).
- **resizeFrameBuffer is ignored while the window is maximized** (§5 trap,
  third bite). Sweeps must run windowed, or compare internal same-res rows.
- **Sweep scripts must exit()** (both sweeps do now). progress.txt last line
  is the completion receipt; "SWEEP COMPLETE" missing = partial captures.
- **[TRRPROBE] costs a second escape walk** — gpuMs in probe-armed rows is not
  representative. Image is untouched (decorrelated generator, value goes only
  to stat slots).

## 7. CURRENT CONFIG STATE

`WdasSkyVNA.py`: rouletteMinQ 0.125 · useTauCache True (res 64, interval 0) ·
useRadCache True (res 64, cut 3, p 0.25, train 8, EMA 0.10) · useWavefront
False · trRRThreshold 0.0 (PULLED) · trRRMode 15 · seedOffset 0 · compaction
True · everything else as HANDOFF_7 §8.

Stat slots: 31..35 [DIVERGE], 36..37 idle warps, 38..41 [LOOPOCC], 42..45
[SHADEOCC], 46..53 [TRRPROBE] (4 Tref bins × RR/ref, per-bin fixed-point —
REPLACE binning key per §5 before next use). kRisStatSlots = 54.

Sweeps: `VNA_RadCacheSweep.py` (shortcut "Falcor VNA RadCache Sweep") — rows:
t_base/t_rr7/t_p100/t_p25 @256+1024, ntb/ntrN/ntrE(+_fm/_nf/_s1/_s2).
Score: `VNA_RadCacheScore.py`. Legacy roulette rows: `VNA_RouletteScore.py`.

## 8. NEXT STEPS, RANKED

1. **Probe v4** (deterministic bin key) → close the trRR hunt one way or the
   other. The traversal-halving prize (§HEADLINE) rides on it.
2. **Gate the rad cache cleanly** (p100 first — must be exactly 0; then walk p
   down 0.25→0.10) under §6 discipline. Every gate-PASS step of p is free
   frame time.
3. If trRR wins its gate: re-enable, then re-measure [COST] — candGen's
   traversal halves and the frame re-ranks.
4. The escape term's froxel τ-control (UE FrustumVoxelGrid layout + Nubis
   8-frame amortization + our residual tracking) — designed, checked against
   sources, unbuilt. The last big unstructured cost after 1-3.
5. Parked: ambient tau channel (Nubis), PG-style learned target, reservoir
   compression, [SHADEOCC]-informed roulette *inside* the CV residual.
