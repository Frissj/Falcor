# VNA Build Handoff 10 — "the tail is a red herring" session (2026-07-21)

Supersedes HANDOFF_9. Every number below was measured this session with the
in-session A/B discipline (only same-run comparisons; cross-session gpuMs is
±25% noise on this machine). Verdicts carry the config that produced them.

---

## HEADLINE

- **Resolution is the shipped win: ~5×.** The whole cloud subgraph now renders
  at an INTERNAL 960×540 and Mogwai upscales the marked output to the native
  window at present. The old `m.resizeFrameBuffer(960,540)` was resizing the
  WINDOW (→ the window manager fought it back to 2560×1351 after ~130 frames —
  that is why it never stuck). Replaced with per-pass `outputSize=Fixed` on
  VolumePathTracer + Composite (both now carry the knob), driven from
  `WdasSkyVNA.py` `_LOWRES`. Window stays native, GUI stays sharp, no stretch.
  **VERIFIED in-game.**
- **Path-length divergence is a DEAD END for shade time. Proven, twice.**
  Two independent, correctly-built, unbiased levers shortened the straggler
  tail dramatically and bought ~0 time:
  - **warp-RR (HANDOFF_9 §5) REDESIGNED and GATED.** The −3% gate failure was
    the §5 bug exactly (`WaveActiveCountBits(true)` inside the divergent
    `radIsResidual` branch → counted residual lanes, not warp occupancy). All 3
    §5 fixes applied. Threshold 8: **gate PASS** (−0.06% @256, +0.02% @1024),
    occupancy 0.228→0.262, **shade time FLAT**. Threshold 24: marginal FAIL
    (+0.15% @1024, on the floor), shade still flat.
  - **residual tail-RR (Lever-2) BUILT and GATED.** Unconditional per-bounce RR
    on post-cut residuals (`radResidualTailQ`). At 0.5 it cut residual mean
    **9.0→5.5** and stragglers **12%→3%** — and shade median moved 14.44→13.92
    ms (~3%, inside the noise), full-frame gpuMs flat-to-worse.
  - Conclusion: **the shade time is bulk per-bounce work, not the tail.**
    Cutting the longest 12% of paths to 3% changes the warp's TOTAL work
    negligibly, so `warpMaxSum`/occupancy barely move. Stop chasing divergence.
- **Homogenization is DEAD** ([HOMOG] probe): 67% of traversed mip-0 cells have
  σ_mean/σ_maj < 0.125, only 0.3% ≥ 0.75. The cloud is fractal; there is no
  uniform interior to collapse into an analytic segment.
- **Nubis, decoded** (from the 2023 talk): it does NOT path-trace multiple
  scattering — `ms = dimensional_profile × Beer(precomputed sun optical depth)`,
  closed form, zero stragglers. Its 40% win came from **decoupling + precomputing
  the secondary sun-march** (the NEE equivalent) into a 256×256×32 grid amortized
  over 8 frames. Your radcache ≈ its MS approx but unbiased; your tauCache ≈ its
  sun grid. **Your stragglers ARE the residual = the unbiasedness tax Nubis
  refuses to pay.** That is the whole story of §2.

## 1. RESOLUTION FIX — mechanism + the one gotcha (SHIPPED)

`WdasSkyVNA.py` now sets `outputSize=Fixed, fixedOutputSize=_RENDER_RES` on
EVERY graph pass (`_LOWRES`, applied via `**_LOWRES`). The graph renders at
`_RENDER_RES` (960×540); Mogwai's present blit (`Mogwai.cpp:720`,
`blit(graphOutput → targetFbo)`) upscales full-to-full to the native window.
Camera aspect stays window-locked, so a smaller intermediate only softens,
never stretches (the buffer resolution cancels: NDC→WxH→W'xH' == NDC→W'xH').

- **VolumePathTracer** and **Composite** each got `outputSize`/`fixedOutputSize`
  (mirror AccumulatePass). Composite had NO size knob before (forced
  `defaultTexDims`) — it needed the addition or its full-res dispatch reads the
  smaller inputs OOB.
- **THE GOTCHA (cost a crash):** `RenderPassHelpers::calculateIOSize` returns
  **{0,0} for `IOSize::Default`** — an "inherit whatever is bound" sentinel,
  fine for a reflect() texture size (0 = inherit) but a **zero-size buffer
  crash** if used as a dispatch/allocation dim. VPT `execute` (targetDim →
  reservoir) and Composite `compile` (mFrameDim → dispatch) both now fall back
  to `defaultTexDims` when the result is 0. The RadCache Sweep runs Default mode
  (its SHIP dict predates the key) and crashed until this fallback.
- **Overridable per-run:** `_RENDER_RES = None` → `outputSize=Default` → full
  window res. Gate/test wrappers set `None` so they run full res (where the
  0.228 occupancy lives + references are); the daily shortcut stays 960×540.

## 2. THE TAIL IS A RED HERRING — the session's main result

`[PATHLEN]` (slots 85–93, 960×540, warp-RR off) split shade paths by radcache
role — count / mean bounces / stragglers ≥16:

| role       |   n    | mean | strag≥16 |
|------------|--------|------|----------|
| raw-uncut  | 46,200 | 2.6  |  0.0%    |
| cut/resid  | 10,600 | 9.0  | 12.0%    |
| training   |  8,100 | 7.1  |  8.6%    |

Raw paths are SHORT (the hard-cut-at-K idea from earlier is dead — nothing to
cut). The stragglers are residual + training, both post-cut. Root of the long
residuals: at the cut a survivor gets `throughput /= radResidualSurvival` (4×
at 0.25), inflating `maxThp` so the raw roulette floors at q=0.125 and barely
kills it → it runs to mean 9.

Then the decisive experiment: **cap those residuals and watch shade time.**
Residual tail-RR at 0.5 (t_wrr8 vs t_p25, same run, full res):

| | resid mean | strag≥16 | shade median | gpuMs mean |
|---|---|---|---|---|
| OFF | 9.0 | 12% | 14.44 ms | 23.90 |
| ON  | 5.5 |  3% | 13.92 ms | 24.41 |

Residuals gutted, shade unchanged (within noise, gpuMs worse). Combined with
warp-RR (same outcome), **two levers prove the path-length tail does not set
shade time.** It is the BULK per-bounce work — every path's density tap + NEE +
phase every bounce ([COST] shadeNEE/candGen) — which scales with all ~57k
post-medium paths, not the 12% tail. Low occupancy (0.23) is a symptom of
divergence, not the cost; removing divergence doesn't remove the work.

**Do not reopen warp-RR, residual-RR, or any straggler/divergence lever for
time.** They are unbiased and correct; they just don't help here. Both stay
OFF by default.

## 3. DIAGNOSTIC INFRASTRUCTURE (all in the build, gated on `logWorkStats`)

- **[PATHLEN]** slots 85–93. Per-shade-path bin at the shadeMain loop exit
  (past the out-of-range return; `bounce`/`radTraining`/`radIsResidual` final),
  per-lane atomics. 3 roles × {count, sumBounces, stragglers≥16}. THE tool for
  "where does path length live." (Learned from the [HOMOG] miss: place probes
  in the LIVE path and verify against a known count before trusting them.)
- **[HOMOG]** slots 77–84. σ_mean/σ_majorant histogram over mip-0 majorant-DDA
  cells, binned in `GridVolumeSampler::stepToNextCollision` (where the brick
  cache already holds both from one texel — zero added traffic). Verdict:
  fractal, homogenization dead.
- **[COST]/[WORK]** unchanged from HANDOFF_9. `kRisStatSlots = 94`.
- **VNA_WarpRRGate.py** — a 5-config gate driver (t_p25, t_wrr8, ntb/s1/s2) vs
  the sweep's 15+. ~3× fewer frames, drops straight into `VNA_RadCacheScore.py`
  unchanged (names match). Knobs at top: `RESIDUAL_TAIL_Q`, `WRR_LANES`. Ends
  with `exit()` (fire-and-forget). Renders at whatever the window is (full res;
  the gate compares t_wrr8 vs t_p25, both same size — no reference needed).
- Desktop shortcuts added: **WarpRR 8/16/24** (wrappers, full res + warp-RR),
  **WarpRR Gate**.

## 4. NUBIS — what to steal, what you already have (from the 2023 talk)

- MS is a probability field: `ms = dimensional_profile *
  exp(-summed_sun_density * remap(sun_dot,...))`. No bounces. (slide 142)
- Ambient `= pow(1-dimensional_profile, 0.5) * exp(-summed_ambient_density)`.
  Secondary (lightning) = spherical falloff × pseudo-attenuation. All closed
  form. (slides 147, 153)
- **The real perf lever (slides 127–130):** decouple the secondary sun-march
  from the view ray, precompute summed-density-toward-sun into a 256×256×32
  grid in a SEPARATE pass, sample it after the first 2 inline samples, amortize
  over 8 frames → **40% off the frame** and BETTER quality (long-distance
  shadows). This is the bulk-work lever, and it matches §2's finding.
- Mapping to VNA: radcache ≈ MS approx (but you keep the residual → unbiased →
  stragglers). tauCache ≈ the precomputed sun grid. **You already have the two
  Nubis tricks in unbiased form.** The open question §7 raises: does NEE still
  re-march the sun per bounce, or is tauCache fully covering it?

## 5. SHIP STATE (WdasSkyVNA.py, unchanged estimator from HANDOFF_9 §2)

trRRThreshold 0.05 · trRRMode 31 · tauCache ON coarseCells 0 · radCache ON cut
3 p 0.25 train 8 EMA 0.10 · rouletteMinQ 0.125 startBounce 3 · useCompaction ON
· wavefront OFF · temporal ON. **NEW: outputSize=Fixed 960×540 on the whole
chain.** `radWarpRRLanes 0`, `radResidualTailQ 0` (both built, both measured
out — leave off).

Ship fingerprint (verify before trusting a log): res 960×540, samplerCalls 1.7,
coarseCells 0, escapeT 0.0. `_RENDER_RES=None` runs full-res 2560×1351 for
gating.

## 6. LEVERS BUILT AND MEASURED OUT (kept, gate-able, default OFF)

- `radWarpRRLanes` (warp-RR): unbiased at 8 (gate PASS), flat time. Redesigned
  block at `VolumePathTracer.cs.slang` shade loop head.
- `radResidualTailQ` (residual tail-RR): unbiased (RR + 4× radRRComp cap + hash
  coin), flat time. Block right after warp-RR in the same loop.
- Both share the `radRRComp` 4× firefly cap. Both use the counter-based hash
  coin (generator-independent, distinct salts).

## 7. NEXT STEPS, RANKED (for continued optimization)

1. **Profile ONE bounce, not path length.** §2 says the cost is bulk
   per-bounce work. Before any new lever, measure where a bounce's time goes:
   density tap vs NEE (sun) vs phase vs reservoir. Use Nsight per-dispatch on
   shadeMain, or add a `[BOUNCECOST]` split to the stat slots. The next lever
   must aim at whichever dominates, for ALL paths — not the tail.
2. **Confirm NEE isn't re-marching the sun.** Nubis's 40% was precomputing this
   (§4). tauCache is supposed to cache sun optical depth; verify shadeNEE
   actually hits the cache and isn't paying a live shadow march per bounce.
   If it re-marches, that's the biggest bulk lever available and it's the
   Nubis move done unbiased.
3. **Wavefront / per-bounce compaction is now LOW priority** — it fixes
   divergence, which §2 proved isn't the cost. Only revisit if (1) shows the
   per-bounce work itself is already minimal and occupancy is the true wall.
4. **candGen (19 cells/px, main phase)** is the other bulk term; the fractal
   HOMOG result says the majorant is ~10× loose (→ ~90% null collisions).
   Tighter local majorants / more occupancy skip (already 54%) is the unbiased
   lever there. NOT homogenization (dead) and NOT coarser boxes (loosens the
   majorant → more nulls).
5. Parked, unchanged from HANDOFF_9: ambient tau channel, reservoir
   compression, duplication maps, ReSTIR-PG target.

## 8. FILES TOUCHED THIS SESSION

Render pass:
- `VolumePathTracer.h` — kRisStatSlots 94; outputSize/fixedOutputSize members +
  RenderPassHelpers include; mRadResidualTailQ.
- `VolumePathTracer.cpp` — outputSize plumbing (key/parse/props/reflect/execute/
  UI) + **calculateIOSize→defaultTexDims fallback**; [HOMOG] + [PATHLEN] prints;
  radResidualTailQ plumbing (key/parse/props/bind×2/UI).
- `VolumePathTracer.cs.slang` — warp-RR §5 redesign; residual tail-RR block
  (`gRadResidualTailQ` CB); [HOMOG] gGvsHomogBins flush accessor; [PATHLEN]
  per-lane binning at shadeMain loop exit.
- `VolumeInstanceSampler.slang` — [HOMOG] reverted from the dead candidate-sweep
  path (statHomogBins field removed).
- `GridVolumeSampler.slang` — [HOMOG] `gGvsHomogBins` + binning in
  `stepToNextCollision` (mip-0 majorant fetch).
- `Utils/Composite/Composite.{h,cpp}` — outputSize/fixedOutputSize +
  calculateIOSize fallback (shared util pass; additive, default = old behavior).

Scripts:
- `WdasSkyVNA.py` — `_RENDER_RES`/`_LOWRES` per-pass resolution (replaces
  resizeFrameBuffer); `_WARP_RR_LANES` override; `radWarpRRLanes` reads it.
- `WdasSkyVNA_wrr8/16/24.py` — full-res warp-RR test wrappers.
- `VNA_WarpRRGate.py` — 5-config gate driver (RESIDUAL_TAIL_Q / WRR_LANES).

Desktop shortcuts: Falcor VNA (warpRR 8/16/24), Falcor VNA WarpRR Gate.

Git: NOT committed (user commits manually — never commit without being told).
Everything above is uncommitted in the tree. Diagnostics are gated on
`logWorkStats`; ship builds pay nothing.
