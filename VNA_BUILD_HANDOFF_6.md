# VNA Build Handoff 6 — the "profile it properly, then cut march cost" session (2026-07-20)

Supersedes HANDOFF_5's open items. Read HANDOFF_5 for history; **§1 below corrects
three of its load-bearing claims.**

---

## HEADLINE

- **HANDOFF_5's central strategic rule is wrong.** It said "memory is not the
  bottleneck… reduce instructions, do not relocate data." The SM sampling
  profiler says nearly every hot line is **LONG SCOREBOARD** — dependent-load
  *latency*. Idle bandwidth (L1TEX/VRAM ~10%) and latency-bound are not the same
  thing, and they have opposite fixes.
- **The shader profiler was never actually enabled** in HANDOFF_5's traces
  (`Total Samples: 0`). Every "where does the time go" claim before this session
  came from unit throughputs, not samples.
- **`coarseOpticalDepth` 13.56% → ~2%**, the single biggest change to this
  kernel. Divergence + icache, not memory.
- **Brick-coherent tap cache** shipped: hit rate 44.8% → **60.6%**, coverage
  100% of taps.
- **Frame ~27.4 ms → ~23.8 ms** at matched settled frames with bit-identical work
  counters. See §4 for why this one is believable when earlier claims were not.
- **The remaining cost is per-sample MARCH cost**, not sample count. §6 is the
  brainstorm for attacking it.

---

## 1. CORRECTIONS TO HANDOFF_5

1. **"There is no RT Core row at all."** False. RTCORE Throughput is 9.2–9.6%.
   It is absent from the metric set the old traces used, not from the GPU. The
   conclusion (not traversal-bound) survives; the stated evidence does not.
2. **"Memory is not the bottleneck / do not relocate data."** Bandwidth is idle,
   but LGSB dominates the stall columns. Data *placement and ordering* is exactly
   what pays now — every win in §3 is a data-movement win.
3. **"Shared memory idle at 36.9%, LDS has headroom."** That is a throughput
   percentage, not free capacity. Actual allocation is ~82–83 KB/SM against a
   ~100 KB ceiling. LDS staging (§5.2 of HANDOFF_5) would likely just move the
   limiter from registers to shared memory. **Do not start there.**

---

## 2. TOOLING — the thing that unblocked everything

`scripts/VNA_NsightShaderProf.cmd` + desktop **"Falcor VNA Shader Profiler"**.

Three flags had to be right, and each failed silently:
- `--real-time-shader-profiler` is **architecture-scoped**. Without
  `--architecture Ada` ngfx accepts it and configures nothing → `Total Samples: 0`.
- `--metric-set-id 2` (Ada "Top-Level Triage") — this is what exposes RTCORE.
- `--debug-shaders` (a Mogwai flag → `setGenerateDebugInfoEnabled`, **no rebuild
  needed**). Without it: "Failed to find any debug symbols", and Shader Source /
  Hot Spots / call trees are all silently disabled.
- `--max-duration-ms` is **hard-capped below 10000**; 16000 is rejected outright.

**What this tool cannot do:** register allocation per source line. That is a
whole-kernel property; no Nsight view breaks it down. Do not re-run it hoping to
find the register high-water mark — that costs an hour and returns nothing.

---

## 3. WHAT SHIPPED

| change | file | effect |
|---|---|---|
| set-bit iteration + single-instance specialization in `coarseOpticalDepth` | `VolumeInstanceSampler.slang` | **13.56% → ~2%** |
| brick-coherent tap cache | `GridVolumeSampler.slang` | hit 44.8%, 100% coverage |
| majorant shares brick cache + prefetches ptr | `GridVolumeSampler.slang` | `stepToNextCollision` **2.98% → 1.77%**; hit rate → **60.6%** |
| TLAS instance-info cache | `VolumeInstanceSampler.slang` | removed a 2-hop chase; **did not move `sampleDistanceRQ`** |
| software-pipelined mean-field fetch | `VolumeInstanceSampler.slang` | unattributed |
| `densities[]` array removal | `VolumeInstanceSampler.slang` | −1.1% registers on `shadeMain` |
| brick-cache counters in `[COST]` | `.cs.slang` + `.cpp` + `.h` | coverage now measurable |

**Correctness discipline used throughout:** every change was validated by
`cells`/`taps`/`coarseCells` totals being **bit-identical** to the prior run, not
by looking at the image. Two changes were pure specializations (identical math,
identical order); the caches are exact (keyed on brick / instance index).

**The one correctness hazard, documented in-code:** the brick cache holds an
atlas pointer valid for **one grid**. Two instances can share a brick coordinate
with different pointers → silently wrong densities, no crash, no counter. It is
reset at *every* walk entry (all six, including estimators the current config
does not use). Do not hoist those resets.

---

## 4. MEASUREMENT — three wrong calls this session, and the rule that came out

I claimed a win three times and was wrong twice:
1. Compared a trace taken **before the deploy** (shaders are copied to
   `build/.../bin/Release/shaders/` at build time; a source edit does not reach
   the running app). **Always `diff` source vs deployed before trusting a trace.**
2. Read frame 128 of a run that stopped at 192 against settled frames of longer
   runs — a 25% "win" that was warmup.
3. Read a "3% cache coverage" figure that was a **wave-reduction bug**: the stats
   block sits inside `if (WaveIsFirstLane())` and every neighbouring counter is
   `WaveActiveSum`-reduced *before* that branch. Reading a per-thread total inside
   it counts lane 0 only — undercounts by exactly the wave width (32×).

**Rule:** a frame-time claim needs (a) identical `res WxH` in `[WORK]`,
(b) frames past ~1000, (c) three-plus consecutive samples, (d) bit-identical work
counters. The ~27.4 → ~23.8 ms result meets all four; nothing earlier did.

---

## 5. ⚠ THE RESOLUTION TRAP — read before any further measurement

**Mogwai's framebuffer follows the window.** Resizing the window overrides
`m.resizeFrameBuffer(...)`. Observed: fullscreen ≈ 22 fps, small window ≈ 100 fps
— that is resolution, not the renderer.

`scripts/WdasSkyVNA.py` now ends with `m.resizeFrameBuffer(960, 540)` (§5.1 of
HANDOFF_5: the pass is ~93% of frame and every term is per-pixel, so this is the
4× lever, plus the LoD system wakes up because `footprintSpread` is inversely
proportional to `targetDim.y`).

**Verify it took**: `[WORK]` prints `res WxH`. If it still says `1920x1080`, the
window clobbered it. **No two runs are comparable unless that field matches.**

Quality cost must be measured, not assumed — and the existing references are
1080p, so diffing a 540p capture against `01_ref` measures *resampling*.
Regenerate references per resolution, then score with `VNA_RouletteScore.py`.

**Note:** the scene already loads `wdas_cloud_quarter.nvdb`. Screen resolution
(pixels) and VDB level (voxels) are orthogonal levers; you have one of each
available.

---

## 6. THE REAL PROBLEM: PER-SAMPLE MARCH COST — full brainstorm

Current board after §3 (8.1M samples):

| share | site | stall |
|---|---|---|
| 11.18% | `sampleDistanceRQ` — `q.Proceed()` | LGSB 46% |
| 8.70% | `gvsLookupStochasticCached` (×2) | LGSB 82–87% |
| 5.68% | `evalEscapeAndCandidatesRQ` | LGSB 43% |
| 3.22% | `instanceSigmaMean` | LGSB 81% |
| 2.69% | `evalTransmittanceRQ` | LGSB 53% |
| 1.97% | `stepDDA` | WAIT 32% |
| 1.77% | `stepToNextCollision` | WAIT 48% |

`samplerCalls` is only **2.1/px**. Underneath each: **21.7 + 8.5 cells** and
**9.3 + 5.5 taps** per pixel. **The cost is the march, not the sample count.**
This is why variance-reduction methods cannot give 4× here (see §7).

### 6.1 Reduce CANDIDATES fed to the RayQuery — biggest single lever left
`aabbTests` = 14.6/px ≈ 30M/frame, each a **shader callback** (procedural prims,
no RT-core traversal). `q.Proceed()` is 11.18% and is *not* cacheable — you
cannot cache inside fixed-function traversal. Only fewer candidates helps.
- **Coarser brick granularity** in the TLAS (currently 66,877 AABBs at 8³).
  Candidate count falls ~cubically against modestly more wasted marching.
- **BUT**: 540p already cuts candidates 4× (14.6/px × ¼ the pixels).
  **Re-measure at 540p before rebricking** — this may already be solved.
- Two-level BLAS (coarse AABBs containing fine) to prune earlier.

### 6.2 Reduce TAPS per cell — tighten the majorant
Every null collision is a wasted tap. Taps are 9.3+5.5/px against 21.7+8.5 cells.
- Tighter local majorants (per-cell rather than per-brick) → fewer null events.
  The mip-0 range texel is already in the brick cache — **free to try**.
- Ratio/residual tracking already exploits this; check whether `residualMip 0`
  is optimal *at 540p* (§4.1's wash was measured at 1080p).
- Analytic majorant along the ray segment instead of per-cell max.

### 6.3 Skip empty voxels INSIDE a brick — the UE lesson not yet taken
`NaniteRasterizer.usf:1244-1261` keeps an occupancy bitmask as **one 64-bit
register** and steps DDA with zero memory traffic per step.
- Store a per-brick occupancy bitmask (8³ = 512 bits, or 4³=64 bits at a coarser
  sub-level) alongside the range/ptr already cached.
- A tap that lands in a known-empty sub-cell is skipped **without touching
  atlasTex** — directly removes LGSB-bound taps.
- `Engine/Shaders/Private/Nanite/Voxel/` (`Voxel.ush`, `Brick.ush`) is the
  two-level DDA reference (16³ as 4³ of 4³ with saved root DDA state).
- **This is the highest-upside untried idea** and it fits the cache already built.

### 6.4 Hide the latency that remains
Everything ≤3% is LGSB-bound with **56% of warp slots unallocated**.
- More warps hides dependent loads. Registers matter only as a *means* to
  occupancy — and register cuts have returned ≤3% on 4 of 4 attempts, so this is
  a poor road.
- Software pipelining (issue next load before consuming current) — done for the
  mean field; applicable to the majorant→tap chain and the DDA cell loop.
- **Not applicable to delta tracking's tap chain**: the next tentative position
  needs a fresh RNG draw, and speculating it advances the stream → changes
  results. Do not "prefetch" there.

### 6.5 Improve memory locality
Bandwidth is idle, so this is purely about latency/hit rate.
- **Morton/Z-order the brick atlas** so neighbouring bricks are cache-adjacent —
  raises the 60.6% brick-cache hit rate and L1 residency for free at bake time.
- Pack the indirection pointer *into* `rangeMeanTex.w` (currently unused) to
  collapse the two-hop chase into one load. **Blocked**: 66,877 bricks needs >16
  bits; would need a 32-bit channel or a brick-index remap.
- Half-precision (`min16float`) for march arithmetic → smaller working set.

### 6.6 Shorten paths
`maxBounces 64`.
- **Weight window** for Russian roulette (HANDOFF_5 §5.3) — splits high-weight
  paths instead of letting survivor weight compound as `1/(1-q)^n`, allowing q
  well past 0.125 without the tail that killed q=0.175. **Preserves
  unbiasedness.** Still the best-understood unexploited lever.
- Path guiding (see §7).
- Transmittance early-out is already the unbiased `Tr < 0.1` roulette.

### 6.7 Reduce divergence
- Bin/sort pixels by expected work (prev-frame T) before phase A
  (HANDOFF_5 §5.5). Consistent with the occupancy story.
- The `coarseOpticalDepth` fix (§3) was exactly this class and returned ~11.5
  points — **the best return of the session**, so this class is worth more.
- Remaining scan-and-skip loops exist in overlap/tail/sweep paths (13 sites) but
  those do not execute in this config.

### 6.8 Structural / bigger swings
- **NRC or a radiance cache** for deep bounces (memory: NRC cold-start flash is a
  known separate artifact).
- Decoupled lighting (Nubis) — **blocked**: `evalNEE` samples the env map, so the
  shadow direction is stochastic; caching arbitrary directions is a 5D table.
  Verified in HANDOFF_5 §6.2.
- Nanite-style **streaming** of fine atlas bricks with an always-resident
  range/mean pyramid — degrades *unbiased* to the coarse field.

---

## 7. ON ReSTIR PG / AREA ReSTIR / ReSTIR FG — unresolved, do this first

The question asked: can one ray give a 4× difference?

**Skeptical prior, from this session's data:** ReSTIR variants reduce *variance
per sample*. `samplerCalls` is 2.1/px, so even a perfect reduction to 1.0 is ~2×,
and it would not touch the LGSB-bound marching underneath each sample. Precedent
against: HANDOFF_4 §7 found VNA's spatial reuse fails structurally on the σT
ratio, and HANDOFF_5 §6.3 concluded paired spatial reuse does not transfer.

**Where it could genuinely win, and this is the open question:** `maxBounces 64`.
If path guiding **shortens paths**, that reduces *marching*, not just sampling —
the one mechanism in this family that could plausibly reach 4×.

**Not yet read.** `C:\Users\Friss\Downloads\Documents\siga25_ReSTIR_PG.pdf`
(34 MB — extract with `pymupdf`, as with Nubis3). **Answer the path-length
question specifically**, not the variance-reduction claims.

---

## 8. IMMEDIATE NEXT STEPS

1. **Pin the resolution.** Confirm `res` in `[WORK]`. Nothing else is meaningful
   until this is stable.
2. Re-run the Shader Profiler at 540p. `sampleDistanceRQ` may no longer be #1.
3. If it is still #1 → §6.1 (coarser bricks). If not → §6.3 (in-brick occupancy
   bitmask), which is the highest-upside untried idea.
4. Read ReSTIR PG for the path-length question (§7).
5. `logWorkStats` / `logRisStats` are **`True`** — set both `False` before
   quoting any shipping perf number.
