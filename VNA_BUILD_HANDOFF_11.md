# VNA Build Handoff 11 — "the bottleneck is occupancy, not op counts" (2026-07-22)

Supersedes HANDOFF_10. Every number below was measured this session. Where I got
something wrong mid-session it is recorded as wrong, with the correction — those
entries are the most useful part of this document.

---

## HEADLINE

**The frame is register-limited occupancy. It always was.** Nsight `analysis7`,
read PER RANGE rather than as sorted aggregates:

| range | % of frame | unused warp slots | VRAM | reg-alloc launch stall |
|---|---|---|---|---|
| main work | 99.7% | **59.8%** | — | **24.0%** |
| main work | 99.5% | 59.8% | — | 24.3% |
| main work | 96.3% | 61.4% | — | 26.5% |
| 8 small ranges | 0.07–0.7% each (~3% total) | 11–25% | 84–89% | — |

~60% of warp slots sit empty across the 99% of the frame that matters, and the
top launch-stall reason is running out of register-file space. This matches
`analysis.yaml` (58%) and `analysis5` (60%) — it has never moved.

**Five independent levers moved operation counts substantially and none moved
time.** That is the signature of a latency-bound kernel that cannot keep enough
warps resident to hide its memory latency. Op counts were never the constraint.

**The register high-water mark is in `main`/`shadeMain` and nobody has looked.**
The existing note in `coarseOpticalDepth` records the one prior attempt: removing
81 floats moved registers −2.8% and made the frame **25% slower** (32.5 → 40.7 ms)
because it traded ALU for registers on an SM-bound kernel. Its own conclusion was
"look in main/shadeMain before trying this again." That is still the open lever,
and it is the only one with ~60% headroom behind it rather than the 3–5% the
op-count levers were fighting over.

---

## 1. WHAT WAS MEASURED OUT — DO NOT RE-CHASE

| lever | result |
|---|---|
| warp-RR, residual tail-RR | flat (HANDOFF_10 §2) |
| homogenization | dead — 0.3% of cells collapsible |
| NEE re-marching the sun per bounce | **does not happen**, `sv 0.00 cells/bounce` |
| majorant granularity 4³ | majRatio 0.607, but p/c 1.43 vs break-even 4.0 → net loss |
| weighted delta tracking | −24% ds taps → −4.6% frame; at the noise floor |
| CV bookkeeping | `[CVBAL]` net −0.25%/−0.36%, survivors exactly 25% → correct |
| cache amortization | −11% frame, but **lightens the image**; consistent, not unbiased |

### 1.1 Majorant granularity (settled, code removed)
`[SUBHOMOG]` measured `majRatio sub/brick = 0.607` — descending the DDA to 4³
cells would cut ∫σ̄·ds by 39%. The `VNA_ResidualMipGate` then priced a cell
against a tap by solving `Δms = c·Δcells + p·Δtaps` across residualMip 0/1/2:
**c ≈ 0.130, p ≈ 0.187 ms per million, p/c ≈ 1.43**, against a break-even of 4.0
(the 4³ trade buys 0.59M taps for 2.37M cells). Predicted +0.20 ms on a 2.10 ms
shade. **No-go.** The `subRange` texture and `[SUBHOMOG]` binning were removed
2026-07-22 — see §5.

Read the histogram carefully if this is ever revisited: `[SUBHOMOG]`'s bottom bin
went *up* (67.2% → 78.3%), i.e. sub-cells are individually *less* uniform, while
the majorant still got tighter. Both are true. `mean/maj` answers the
homogenization question; only `majRatio` bears on cost.

### 1.2 Weighted delta tracking (built, default off, NOT cleared)
`distWeightK` softens the free-flight majorant to `lerp(cellMean, cellMajorant, K)`
and pays with a per-collision weight `max(1, σ/σ̄)`. Unbiased for any K
(Galtier et al. 2013). `P_c = min(σ/σ̄, 1)` makes the null-collision weight
exactly 1, so there is no running weight — the accept test is unchanged and at
K=1 the estimator is byte-identical.

Mechanism confirmed — `ds taps/call` falls monotonically 3.7 → 3.4 → 3.1 → 2.8.
Payoff: **−24% taps for −4.6% frame**, and K=0.75 came out *slower* than K=1.0,
putting the whole spread at the noise floor.

**Unresolved:** converged (4096spp) captures differ from K=1.0 by ~10× the
expected noise floor, monotone in K (MSE 0.00155 / 0.00221 / 0.00498). Either the
weight is wrong (bug) or the estimator is heavy-tailed and 4096spp is not
converged. Distinguishing them needs 32k+ spp at K=0.25. **Not worth the run
given the 4.6% payoff** — but do not describe this lever as verified-unbiased.

### 1.3 Cache amortization (built, default off, image-rejected)
`radAmortPeriod` = 1 correction frame in N; the rest consume the cache and die at
the cut. Frame-gated rather than pixel-gated **because a warp costs max(bounces),
not sum** — residual tail-RR removed 13.5% of bounces and bought 3%, whereas
uniform removal collapses warp max from 16+ to ~5. `[PATHLEN]` on a consume frame
confirms the mechanism: 100% raw-uncut, mean 2.7, **0% stragglers**.

Timing (gate log 167, 960×540, both rounds agreeing to 0.024 ms):

| config | shade ms | gpuMs |
|---|---|---|
| nocache | 2.767 | 4.786 |
| ship | 2.286 | 4.080 |
| amort8 | **1.725** | **3.626** (−24.5% shade, −11.1% frame) |
| floor | 1.625 | 3.444 |

**Rejected on image.** Consume frames render the coarse cache mean `C` raw, which
flattens contrast. `[LUMHIST]` fraction of pixels in the darkest bin: nocache
29.9% → ship 25.9% → amort8 23.4%. Total energy barely moves (which is why
`[ENERGY]` could not see it, and why the eye could).

---

## 2. FINDINGS THAT STAND

- **`[COST] shadeNEE` was mislabelled** — slots 20/21 are `shadeMain`'s whole-lane
  totals, so the bucket was `sampleDistance` + `scatterAtVertex` + the deferred
  shadow ray. Renamed `shadeAll`. HANDOFF_10 §7.2 was unanswerable from any log
  because of this.
- **`[BOUNCECOST]` split (the answer to §7.1):** ds 68.3% of shade ops
  (9.7 ops/bounce, 275k calls), nee 31.7% (19.1 ops/call, 65k calls), **sv 0.00**.
  One NEE shadow ray costs 2× a whole bounce.
- **NEE is not re-marching per bounce** (§7.2 answered: `sv 0.00`), but it *is*
  a live `evalTransmittance` once per path — and that walk already runs residual
  ratio tracking with the correct **mean** control
  (`sigmaRbar = max(maj-mean, mean-min)`). `[HOMOG]`'s 0.125 is why it only
  removes ~12% of collisions.
- **The radcache is worth 17% of shade** (nocache 2.767 vs ship 2.286).
- **`radSplat` had a one-sided clamp on an expectation-correct quantity.**
  `max(Ls, 0)` at the call site plus `clamp(L, 0, 4096)` inside, forced by an
  unsigned accumulator. The deposited suffix can legitimately be negative (the
  post-cut NEE share is a reservoir *count ratio*, not the exact split), so
  clipping the negative tail biases the cache high. **Fixed:** signed fixed point
  via `asuint(int(...))` / `asint()`, symmetric ±4096 clamp. Two's-complement
  addition is bit-identical for int and uint so the buffer type is unchanged.
  `[RADSPLAT]` reports the negative-deposit rate — **check it is nonzero before
  crediting this fix.**
- **Unexplained: ship already lifts dark pixels ~4pp vs nocache** (`[LUMHIST]`
  bin 0: 29.9% → 25.9%) with the residual correcting every frame and `[CVBAL]`
  clean. Amortization roughly doubles it. Below the visual threshold, but real.

---

## 3. LITERATURE — PRIMARY SOURCES READ, NOT SUMMARIES

- **VSPG** (Xu, Herholz, Manzi, Papas, Gross — ToG 43(6), SIGGRAPH Asia 2024).
  Real, unbiased, decouples scattering probability from the majorant via weighted
  reservoir sampling (Chao 1982). **Its own Discussion rules out this case:**
  *"the number of density queries increases compared to standard delta tracking…
  in scenes with extremely large and dense volumes (e.g., cloudscapes)."* Also
  55 MB + 102 MB of buffers at 1080p, a 5D parallax-aware VMM guiding structure
  (Ruppert 2020), equal-time comparisons at **2–5 minutes per frame**.
  *Portable part:* VNA already has RIS + reservoirs + temporal reuse, so the core
  idea could be prototyped without the guiding structure.
- **Gabor Fields** (Condor, Hermann, Yurtsever, Didyk — arXiv 2602.05081, ToG,
  rev. 16 July 2026). **Do not adopt.** Their own comparison: *"our method, while
  achieving similar quality and unbiased rendering at a ~4300x compression rate,
  is substantially slower"* and *"voxel grids are significantly faster."* Their
  "highly optimized voxel grid" baseline is SuperVoxels with local
  majorant/minorant + RRT for NEE shadow rays + a **weighted** free-flight
  distance sampler — i.e. VNA, feature for feature. The ~2× figures are relative
  to their own unaccelerated method, not to voxel grids. Wins are memory and LOD.
- **Progressive null-tracking** (Misso 2023) solves *unknown* majorants for
  black-box procedural volumes. VNA bakes exact majorants into `rangeMeanTex`.
  Solves a problem we do not have.
- **Multi-Density Woodcock Tracking** (EGPGV 2025) is multi-channel scientific
  visualisation. One density grid here. N/A.
- **Kettunen et al. 2021**, unbiased ray-marching transmittance estimator —
  unimplemented, targets the nee bucket (32% of shade). The one genuinely
  actionable item from the literature.

**Independent validation:** a separate group building a deliberately strong voxel
baseline converged on VNA's exact architecture. This project is at the strong
baseline, which is why every lever this session came back marginal.

---

## 4. MISTAKES MADE THIS SESSION — READ THIS BEFORE TRUSTING A CONCLUSION

- **"Full res is VRAM bandwidth-bound at 88%"** — WRONG. That number came from
  sorted aggregate values belonging to ranges that are <1% of the frame each.
  Per-range, the 99% work has no VRAM entry and 60% unused warp slots. Always
  segment the Nsight yaml by `RelativeFrameDuration` before quoting a metric.
- **"L2 SysMem misses regressed 14×"** — WRONG. As a *percentage* it was
  unchanged (11.15% → 11.38%); only the absolute cycle count scaled.
- **"Wire tauCache into evalNEE"** — WRONG. Residual ratio tracking with a mean
  control was already shipped; tauCache would recompute the same analytic term
  and do nothing about the residual.
- **"Frame-gated amortization beats Nubis"** — the timing held, the image did not.
- **The oscillating SM throughput was a clue, not noise.** It was the frame
  alternating between two regimes: big occupancy-limited dispatches (~40% full)
  and tiny bandwidth-saturated ones (~89% full, 84–89% VRAM). The dips total ~3%
  of frame time — informative, not a cost.

### Tooling gotchas
- **Script `print()` does NOT reach the Mogwai log.** Verified across every gate
  driver. Segment logs on the `[WORK]` frame counter, which resets to 0 on each
  `updatePass` — one block per config, in CONFIGS order.
- **`risStatsInterval` aliases with `radAmortPeriod`.** 64 % 8 == 0, so every
  logged frame was a CORRECTION frame and consume frames were invisible. The
  driver now switches to 63 automatically when `_AMORT_PERIOD != 0`, and
  `[PATHLEN]` prints `[CORRECTION]`/`[consume]`/`[amort-off]`.
- **`radTrainEvery` at 1e6 silently breaks the cache**: training off during warm
  means the cache never populates, the `C.a > 0.5` gate fails, the cut never
  fires, and every path runs full length — measuring `nocache` while believing
  you measured a consume-only floor. Use 64.
- **A settled diagnostic must not stay gated on `logWorkStats`.** `[SUBHOMOG]`
  did an uncached texture Load per mip-0 DDA cell (~80M/frame at 2560×1351) long
  after its question was answered.

---

## 5. STATE OF THE TREE (nothing committed)

`WdasSkyVNA.py`: `_RENDER_RES = None` (full res), `_DIST_WEIGHT_K = 1.0` (off),
`_AMORT_PERIOD = 0` (off), `radCutBounce = 3`. **Two `DIAGNOSTIC STATE` comments
remain in the driver and should be tidied.**

New knobs, all default-off / byte-identical:
- `distWeightK` (1.0 = exact analog tracking)
- `radAmortPeriod`, `radAmortTrainEvery` (0 = off)

New diagnostics: `[BOUNCECOST]`, `[ENERGY]`, `[CVBAL]`, `[CVEXIT]`, `[LUMHIST]`,
`[RADSPLAT]`, `[BRICKMEM]`. **`[ENERGY]` and `[LUMHIST]` use per-lane atomics with
no wave reduction** — unlike the older counters — so they are contention-heavy
and should be wave-reduced or removed before shipping.

Removed: `subRange` texture + `[SUBHOMOG]` (converter, `BrickedGrid`, `Grid.cpp`,
`Grid.slang`, sampler binning, flush, print). Stat slots 108–117 are now free.

Gates + desktop shortcuts: `VNA_ResidualMipGate`, `VNA_CacheAmortGate`,
`VNA_DistWeightGate`. All pin their own config, interleave rounds, and `exit()`.

---

## 6. NEXT, RANKED

1. **Attack register pressure in `main`/`shadeMain`.** ~60% unused warp slots,
   register allocation the top launch stall at 24–26%. Reduce *live state*, not
   recomputation — the prior attempt traded ALU for registers and lost 25%.
   Re-capture Nsight and read it **per range**.
2. **Explain the 4pp dark lift in ship vs nocache** (`[LUMHIST]`). `[CVBAL]` is
   clean, so the energy is entering through the residual's `S` collection, not
   the `C` terms.
3. **Kettunen 2021** on the nee bucket (32% of shade, 19.1 ops/call).
4. Wave-reduce or delete `[ENERGY]`/`[LUMHIST]`; tidy the driver's diagnostic
   comments.
5. Parked: VSPG core on the existing reservoirs; weighted-tracking convergence
   question (only if the lever is revisited).
