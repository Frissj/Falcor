# VNA Build Handoff 12 — "the frame was never measured" (2026-07-22)

Supersedes HANDOFF_11. Every number below was measured this session, at
2560x1351 full window res unless stated. Where HANDOFF_11 was wrong it is
recorded as wrong, with the correction — those entries are the most useful part
of this document, and this time several of them are corrections to HANDOFF_11
itself.

FPS did not move this session. Read §4 before anything else.

---

## HEADLINE

**Nothing in this pass had ever measured the frame.** The GPU timer bracketed
this pass's own dispatches; AccumulatePass, tone mapping, present and all CPU
time sat outside it, and no line in any log reported wall-clock or FPS. So
"gpuMs 26.8" could have been the whole frame or a third of it, and every lever
in HANDOFF_11 was priced against a number whose relationship to frame rate was
unknown. That is now fixed (`[FRAME]`), and the answers changed the target.

**The pass owns 85–94% of the frame.** OUTSIDE PASS is 3–6 ms of a ~30 ms frame,
so optimising here *can* move FPS — the work was not doomed by being in the
wrong place. Ceiling from eliminating everything outside the pass: ~1.12x.

**`ds` (distance sampling) is 14.2 ms = 81% of shadeMain = 47% of the frame.**
It is the single largest item anywhere, larger than all of phase A. Everything
else is a rounding error by comparison.

**The 24% "Active Threads Per Warp" figure is not headroom. It is 1/H₃₂.**
Nsight's 4.12x "speedup ceiling" is the harmonic number H₃₂ = 4.0585. This one
line of algebra explains, and permanently closes, three separate failed levers.

| measured | value |
|---|---|
| wall | 30.2 ms → **33.1 FPS** |
| pass | 26.6 ms (85–94% of frame) |
| ├ main (phase A) | 10.4 ms |
| ├ **shadeMain** | **16.8 ms** — of which ds 14.2, NEE 3.4 |
| ├ radResolve | **0.01 ms** (was suspected; it is free) |
| └ other (argsMain) | 0.00 ms |
| OUTSIDE PASS | 3–6 ms |

---

## 1. THE 1/H₃₂ RESULT — READ THIS BEFORE PROPOSING ANY SCHEDULING LEVER

`tauToGo = -log(1 - rand())` (GridVolumeSampler.slang) is Exp(1). The DDA runs
until accumulated `dt * majorant` exceeds it, so a lane's trip count IS that
exponential divided by a constant. A warp costs `max(trips)` over 32 lanes, and
for 32 i.i.d. exponentials `E[max]/E[mean] = H₃₂ = 4.0585`.

    predicted occupancy = 1/H₃₂ = 0.2464
    measured [STEPOCC]  shade occ = 0.239
    measured Nsight     active threads/warp = 0.2427
    Nsight RangeSpeedupFactor = 4.1198  vs  H₃₂ = 4.0585

**Consequences, all of which are now settled:**

- Changing the majorant scales mean AND max equally — the constant cancels, so
  occupancy is invariant. This is why the wave-uniform mip lever produced 0.239
  at every lift setting (§2).
- The draws are i.i.d., so any grouping of 32 has the same expected max. No
  predictor can exist because the quantity is random by construction. That is
  why the work-class sort failed with a healthy class spread.
- Each bounce is its own exponential race, so splitting by bounce does not split
  the race. That is why wavefront requeueing failed.

**Do not propose a fourth lever that reorders, requeues, or aligns lanes.** The
only thing that changes 1/Hₙ is not letting each lane run its race to completion
in lockstep, i.e. bounding trips per iteration and deferring the remainder — a
CELL-granularity wavefront. The bounce-granularity version already died on state
round-trip cost, so rate it unlikely, not promising.

**The 4.12x was never collectable.** It is a statistical property of an unbiased
estimator, not waste.

---

## 2. WHAT WAS MEASURED OUT THIS SESSION — DO NOT RE-CHASE

| lever | result |
|---|---|
| wave-uniform majorant mip | **REFUTED.** Warps are already 95.3% mip-uniform |
| occupancy skip (as a perf feature) | **worth 0.02 ms** — ds is not atlas-read bound |
| radResolve 64³ dispatch | 0.01 ms. Free. Suspicion was wrong |
| argsMain dispatch | 0.00 ms |
| radSplat signed-suffix fix (H11 §2) | **no-op; its premise is structurally impossible** |

### 2.1 Wave-uniform majorant mip (built, default off, REFUTED)
`gvsWaveLiftMip` lifts each lane's DDA mip toward the coarsest any active lane
in its warp wants, to make step size — and so trip count — wave-uniform.
Unbiased at any lift (a coarser majorant still bounds σ), gated by
`useWaveUniformMip` (compile) + `waveMipLift` 0..3 (runtime).

**Measured (log 185): lift1, lift2 and lift3 all produced EXACTLY 4.7% of cells
lifted, avg +1.00 mip, and shade occ 0.239 → 0.240 → 0.239 → 0.239.**

Identical results across three lift values means `WaveActiveMax(ownMip) - ownMip`
is never greater than 1 — the clamp never binds. Only 4.7% of cells have any
lane disagreement at all, and when they do it is always exactly one level. There
was nothing to make uniform. Kept compiled as the A/B record.

### 2.2 Occupancy skip is not a frame-time feature (log 192)
`useOccupancySkip` is a runtime uniform that changes ONLY whether a tap is
answered from the register-resident occupancy bitmask or a real `atlasTex` read;
a clear bit means every voxel decodes to exactly 0, so the result is
bit-identical. Turning it OFF converts ~19.7M taps/frame into real atlas reads.

    wall occSkip ON   31.68 ms (sd 2.47)
    wall occSkip OFF  31.70 ms (sd 1.67)
    delta              0.02 ms = 0.1%

**`ds` is not bound on atlas read throughput.** Those reads are already fully
hidden. This kills the entire memory-locality direction — tap batching, cache
reshaping, brick relayout — before anything was built. It also means occSkip
itself buys nothing measurable in time despite skipping 54% of all taps.

*(The ms split from that run was INCONCLUSIVE — drift 1.70/1.95 ms against a
0.79 ms effect. See §4.4: the instrument, not the machine.)*

### 2.3 NEE is a minority cost (log 189, useNEE A/B, settled regions only)

    NEE ON   shadeMain 17.62   main 10.93   wall 32.40 (30.9 FPS)
    NEE OFF  shadeMain 14.21   main 11.08   wall 30.17 (33.1 FPS)
    delta    -3.42 ms (-19% of shadeMain)              +7.4% FPS

`main` did not move (10.93 → 11.08), which is the control working. Removing 31%
of shade taps bought 19% of shade time — sub-proportional, as always here.
**Deleting an entire lighting feature bought 2.2 ms.** That is the scale: no
single component except `ds` is big enough that removing it changes the picture.

**Do not read the frames immediately after the toggle.** shadeMain ramps 7.5 →
14 over ~600 frames as the radiance cache retrains after the pass rebuild. Only
frames ≥ ~1856 are comparable. The naive read of the first post-switch block
gives 9.4 ms for NEE — 2.7x too high.

### 2.4 radSplat: HANDOFF_11 §2 is wrong
H11 claims the deposited suffix "can legitimately be negative" and that clipping
it made the cache read high — *"that is the image coming out LIGHTER"* — and
says to check the negative rate is nonzero before crediting the fix.

**It is zero. 1.8M deposits over 45 frames, not one negative.** And it cannot be
otherwise: `radCutRad` is snapshotted inside the bounce loop while
`radiance += neeContrib` happens after it, so

    radiance_final - radCutRad = postCutScatter + neeContrib
    suffix = (postCutScatter + neeContrib) - neeContrib*(1 - fPost)
           =  postCutScatter + neeContrib*fPost      >= 0 always

with `fPost` in [0,1]. The one subtraction that could break this
(`radiance -= throughput * C.rgb`) is in the CONSUME branch, mutually exclusive
with the training branch that sets `radHasCut`. The signed accumulator is
harmless and was kept, but it fixed nothing, and **the leading explanation for
the image lightening is dead.** Slot 119 is now an invariant guard that must
read 0; the log prints `*** INVARIANT BROKEN ***` if it ever does not.

---

## 3. FINDINGS THAT STAND

- **`[FRAME]` is the top-level accounting that was missing.** Wall clock, FPS,
  pass share, OUTSIDE PASS, and a main / shadeMain / radResolve / other split.
  The split closes to ±0.01 ms against the pass total.
- **`[WORK]`'s "shade" was always essentially shadeMain** — argsMain and
  radResolve are 0.00/0.01 ms. That attribution was fine; it is now proven.
- **shadeMain is 1.87x slower per warp-op than main even after dividing out lane
  occupancy** (12.2 vs 6.5 M warp-ops/ms). Divergence is not the whole story
  there, and §2.2 has now ruled out atlas-read throughput as the remainder. What
  is left is instruction count and latency.
- **`[STEPOCC]` (slots 108–113) measures the loop `[LOOPOCC]` and `[SHADEOCC]`
  could not see** — stepToNextCollision's DDA, where the shipping distance
  sampler spends the 68% ds bucket. main occ 0.731 vs shade occ 0.239: the
  divergence lives entirely in phase B, among lanes that all hold real paths.
- **Shared per-brick walk setup** (`useSharedWalkSetup`, ON by default,
  bit-identical): `evalEscapeAndCandidatesRQ` ran the same preamble — getGrid,
  brick-cache reset, two 4x4 mat-vecs, a reciprocal — 1 + risCandidates times per
  brick on the same grid and ray. At 5.8 segments and 19.0 cells per pixel that
  is 17.4 preambles per pixel against 19.0 loop iterations. **STILL UNMEASURED**
  — the A/B toggle exists and has never been flipped. See §6.

---

## 4. MISTAKES MADE THIS SESSION — READ BEFORE TRUSTING A CONCLUSION

### 4.1 Treating Nsight's 4.12x as headroom
The whole wave-mip lever was built on "Active Threads Per Warp 24.27%, 4.12x
ceiling, 73% frame gain" being collectable. It is H₃₂. The arithmetic that
refutes it uses only numbers that were already quoted in the analysis. **Check
whether a profiler's "ceiling" is a property of your estimator before building
against it.**

### 4.2 Sizing a 27,000-frame gate from an unrepresentative gpuMs
The first VNA_WaveMipGate run took 20 minutes. It inherited WARM/MEASURE/
CONV_SPP from VNA_DistWeightGate.py — a **960x540** gate — and ran them at full
res, and the per-frame cost was taken from a log line that was not the same
config. Real mean was 34.4 ms, not the 13.5 assumed.

The spp scaling also runs the wrong way: `signed_rel` is a mean over every
pixel, so its standard error goes as 1/sqrt(pixels × spp). Full res has **6.7x**
the pixels of 960x540, so going full res BUYS bias sensitivity — 512 spp there
is comparable to 4096 spp at low res. The gate is now 9,280 frames.

### 4.3 Inferring time from counters, three times
- HANDOFF_11's five op-count levers: ops moved, time did not.
- NEE predicted at ~36% of shadeMain from its real-atlas-read share. **Actual
  19%** — overestimated 1.8x.
- occSkip predicted to matter because ds skips only 34% of taps vs main's 63%.
  **Actual 0.02 ms.**

Counters do not predict time in this kernel. Only an A/B does.

### 4.4 THE INSTRUMENT, NOT THE MACHINE — the [FRAME] defect
The first `[FRAME]` averaged the wall clock over all 64 frames of the interval
but sampled the GPU timers on the **logged frame only**. One frame of a 30 ms
workload swings ±4 ms (one block held 17.16 … 36.89). The occSkip gate exposed
it: `pass` claimed **+1.98 ms** while `wall` said **+0.02 ms**.

**Every single-frame GPU number this pass has ever printed carries that noise,
and gates were coming back INCONCLUSIVE because the instrument could not resolve
them, not because the effects were absent.**

FIXED: a ring of `kTimerRing = 8` timers per region. begin/end every frame into
slot `frame % 8`, resolve that slot, and read the slot 7 frames old —
`getElapsedTime()` maps a staging buffer and would block if read the same frame,
but by ring-wrap the copy has landed. Results accumulate and are averaged over
the interval, so the GPU split now has the same ~64 samples as the wall clock
(sd of the mean ~8x smaller). `[FRAME]` prints both sample counts as `n=` so the
two halves can never again be compared on unequal footing invisibly.

**Re-run anything that was gated on a single-frame GPU number.**

### 4.5 Optimising code that never runs
The first target chosen for ALU reduction was the instance loop in
`sampleDistanceTail`. `[COST]` reports `tailRays 0.00/entry` — the tail path is
never taken. Same for the overlap path (`overlap steps 0.00`). **Check the
counter for the path before optimising it.**

---

## 5. STATE OF THE TREE (nothing committed)

New in this session, all default-safe:

- **`[FRAME]`** (VolumePathTracer.cpp/.h) — wall/FPS/pass/OUTSIDE PASS + split,
  GPU timer ring, gated on `logWorkStats`.
- **`[STEPOCC]`** slots 108–113 — stepToNextCollision DDA trip divergence, main
  and shadeMain separately, plus wave-mip lift counters.
- **`useSharedWalkSetup`** (default **ON**, compile-time, bit-identical) —
  `WalkSetup`/`beginWalk` in GridVolumeSampler.slang; one preamble per brick
  shared by the transmittance walk and all nProc candidate walks. UNMEASURED.
- **`useWaveUniformMip`** + `waveMipLift` (default OFF, compile-time) — refuted,
  kept as the measured record.
- radSplat comment corrected; slot 119 re-purposed as an invariant guard.

Slots: 108–113 = `[STEPOCC]`, 114–117 free.

Scripts + desktop shortcuts: `VNA_WaveMipGate.py` (+ `_Compare.py`, prints
PASS/FAIL with a measured noise floor, a drift guard and a resolution guard),
`VNA_DsGate.py`. Both full window res, both `exit()` on completion.

**Config to restore:** `useNEE` must be ON — log 189 ran it off for measurement
and the image has no direct sun without it.

---

## 6. NEXT, RANKED

1. **Measure `useSharedWalkSetup`.** It is ON in the current build and has never
   been A/B'd. It is bit-identical, so the gate is a bit-compare plus a same-
   session timing flip — minutes, and it is the only unmeasured change in the
   tree. Now worth doing properly because §4.4 is fixed.
2. **Attack `ds` on instruction count.** 14.2 ms, 47% of the frame. Not atlas-
   bound (§2.2), not divergence-bound beyond 1/H₃₂ (§1). That leaves ALU and
   latency. ALU is the top SOL pipe at 43.1% with SM issue at 52.6%.
   `evalEscapeAndCandidatesRQ`'s per-brick work and `stepToNextCollision`'s
   per-cell work are the only two places it can be.
3. **Register pressure — 1.36x, 26.5% launch stall, still untouched.** With the
   4.12x gone this is the largest honest number left in analysis7. Warp
   occupancy is 15.3%. Evidence of spilling is in the yaml: local-load miss
   2.41%, local-store 2.05%, global-store 0.00% — indexable temps in local
   memory, and `main` has 69 allocas. `IntervalList` is kMaxInstances × 12 B
   live across the whole march. Pack it; do not recompute it (the
   coarseOpticalDepth note records recomputation costing 25%).
4. Explain the ~4 pp dark lift, ship vs nocache (`[LUMHIST]` bin 0 29.9% →
   25.9%, still present). **Both named suspects are now dead** — radSplat by
   construction (§2.4), `[CVBAL]` by sign (mean +0.014% over 45 frames, 19
   positive / 25 negative).
5. Parked: cell-granularity wavefront (§1), the only structural idea that could
   change 1/Hₙ, rated unlikely.

**Ground truth for any future claim: `[FRAME]` wall/FPS. Not gpuMs, not ops.**
