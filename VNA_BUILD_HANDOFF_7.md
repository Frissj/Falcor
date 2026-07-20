# VNA Build Handoff 7 — the "it was never memory, it's the register cliff" session (2026-07-20)

Supersedes HANDOFF_6's open items. **§1 corrects three of its load-bearing claims,
including the one its whole strategy rested on.**

---

## HEADLINE

- **The frame is REGISTER-LIMITED, not memory-limited.** `main` compiles to 128
  registers → 65536/128 = 512 threads = **exactly 16 of 48 warps = 33.3%**
  occupancy, matching the measured 33.5% to the decimal. Nsight's Trace Analysis
  independently confirms it: `CS Warp Launch Stalled Register Allocation = 25.4%`
  is the **sole** entry under "Warp Launch Stalled by Reasons".
- **Occupancy is a STEP function and every prior register attempt measured the
  flat part.** At a 256-thread CTA: 2 CTAs = 16 warps (128 regs) → 3 CTAs = 24
  warps needs **≤80 regs** → 4 CTAs = 32 warps needs ≤64. Cutting 128→112→96 buys
  **nothing**. This retroactively explains "register cuts returned ≤3% on 4 of 4
  attempts" — they were never going to show anything.
- **128 is a CAP, not a measurement.** Four radically different builds — baseline,
  all instrumentation removed, dead paths removed, and the entire brick TLAS
  removed — all report **128 registers and 23552 B smem, bit-identical**. The
  kernel wants more than 128 in every configuration.
- **§6.1 is dead and §6.3 is closed**, both measured, both for reasons the old
  counters could not have shown.
- **Memory is idle**: VRAM 9.8%, L2 12.0%, RTCORE 10.1%, against SM 56.9%.

---

## 1. CORRECTIONS TO HANDOFF_6

1. **§6.1's premise was a misread counter.** It said `aabbTests` = 14.6/px ≈ 30M/
   frame, "each a shader callback (procedural prims)". It is not. `statAabbTests`
   is `+= getGridVolumeCount()` per `gatherIntervals` call
   (`VolumeInstanceSampler.slang`), i.e. **9 software slab tests against whole-
   instance bounds in the INTERVAL-LIST path**. It is unrelated to the brick TLAS
   and *cannot* respond to LoD mip. Renamed to `instSlabs` in `[WORK]` so this
   cannot recur. The real counter (`brickCands`, slot 29) had to be built first.
2. **§1's "memory placement and ordering is exactly what pays now" is wrong.**
   That was inferred from LGSB stall columns. Directly tested this session by
   eliminating **22.5M of 48.9M taps per frame** (46.1%) — the single largest
   possible data-movement win — and it moved the frame **0.8%, inside noise**.
   LGSB% says where warps *sit*, not what *costs*. With only 16 warps resident
   there is nothing else to issue, so removing loads does not help.
3. **"No Nsight view breaks down register allocation" is half wrong.** Per-*line*
   attribution is genuinely unavailable (needs SASS; GPU Trace correlates to DXIL
   only — the Languages dropdown offers HLSL/DXIL and nothing else, and the
   `Live Registers` column in a Shader Source CSV export is empty on all rows).
   But **whole-kernel `#Reg` is right there** in Shader Pipelines, and
   **Trace Analysis names register allocation as the launch-stall reason**. That
   is everything needed; the previous session stopped one panel short.

---

## 2. THE REGISTER CLIFF — the finding everything else now hangs off

Nsight → Shader Pipelines, per kernel:

| kernel | samples | #Warp | #Reg | Smem | CTA | live |
|---|---|---|---|---|---|---|
| `shadeMain` | 49% | 20 | 96 | 4864 | 64,1,1 | 92 |
| `main` | 46% | **16** | **128** | 23552 | 16,16,1 | 126 |

Ada: 65536 registers/SM, 48-warp ceiling. Shared memory is **not** co-limiting
(23552 B/CTA allows 4 CTAs; registers allow 2), and Trace Analysis lists no smem
reason — only register allocation.

**The invariance is the point.** `main` #Reg across four builds:

| build | #Reg | Smem | warps |
|---|---|---|---|
| baseline | 128 | 23552 | 16 |
| `logWorkStats`/`logRisStats` OFF (all stats compiled out) | 128 | 23552 | 16 |
| `useMergedTail` + `useSharedCandidateSweep` OFF | 128 | 23552 | 16 |
| `useBrickTlas` OFF — **all four RayQuery objects gone** | 128 | 23552 | 16 |

Confirmed the last one really removed the traversal: `RTCORE Throughput`
disappeared from the metric list entirely and the pass slowed to 72 ms.

**Consequence: bisection by feature-removal cannot find the consumer**, because
the reported number is insensitive to it by construction. Getting to ≤80 is not
"find 48 wasteful registers", it is "reduce peak live state below a ceiling you
are already exceeding" — a restructuring of the marching loop.

**Unknown and worth measuring: is it spilling?** If true demand exceeds 128 the
kernel spills to local memory and that traffic was invisible in everything
inspected today. Check local-memory / spill traffic in the Metrics tab.

### Ruled out as the register consumer (all measured, none of them it)
- **Instrumentation** — ~14 `stat*` fields + `gGvs*` statics: 128 → 128, live
  126 → 125. They are write-only under the defines and DXC eliminates them.
  **The diagnostics are register-free; there is no reason to run blind.**
- **RayQuery objects** — four sites (`evalTransmittanceRQ`, `sampleDistanceRQ`,
  `sampleCandidatesRQ`, `evalEscapeAndCandidatesRQ`), each scoped to its own
  function, **verified non-nested** so their lifetimes never overlap. Removing
  all four: 128 → 128.
- **`CandidateSet`** — it is `1 + kRisM + kRisM` and the script sets
  `risCandidates: 2`, so **5 registers**, not the 17 the `#define RIS_CANDIDATES 8`
  fallback would suggest. Read the configured value, not the default.
- **Dead code paths** — `useMergedTail` and `useSharedCandidateSweep` never
  execute in this config (`tailRays 0.00/entry`, `sweepCells 0.0`) and cost 0
  registers; they are `#if`'d out entirely.

---

## 3. WHAT SHIPPED

| change | files | effect |
|---|---|---|
| `brickCands` counter (gRisStats slot 29) | `VolumeInstanceSampler.slang`, `.cs.slang`, `.cpp`, `.h` | first real measure of RayQuery candidates; `aabbTests` renamed `instSlabs` |
| §6.3 per-brick occupancy mask | `GridConverter.h`, `BrickedGrid.h`, `Grid.cpp`, `Grid.slang`, `GridVolumeSampler.slang` | `occSkip 46.1%` of taps answered from a register-resident bitmask, **exact** |
| runtime A/B toggle for the skip | `.cs.slang` CB + `mUseOccupancySkip` + UI | same-session A/B, the only trustworthy kind on this machine |
| `--multi-pass-metrics` | `scripts/VNA_NsightShaderProf.cmd` | Trace Analysis was returning one populated row without it |

### §6.3 detail — the design decision that matters
The mask is aligned to **BC4 tiles (4x4x1)**, not the 2^3 cells §6.3 suggested,
giving 2*2*8 = 32 tiles per brick = **one uint32**. Reason: the atlas is
BC4-compressed (`Grid.cpp` uses `NanoVDBConverterBC4`) and BC4 fits endpoints
across all 16 texels of a tile, so a 2^3 cell straddling tiles can decode nonzero
even when its own source voxels were empty. Masking at that granularity would
return the minorant where the renderer should return more — a small systematic
**bias**, invisible in the image, fatal to an unbiased renderer. At tile
granularity the test is exact by construction (`tilemajorant == 0` → both BC4
endpoints 0 → every texel decodes to exactly 0 → `0*(maj-min)+min` is exactly the
minorant, which the brick cache already holds). **Do not "improve" this to finer
granularity without decoding the compressed block first.**

Empty bricks (`majorant == minorant`, ptr 0) get an all-zero mask, so their taps
skip too — those were previously reading garbage from atlas brick 0 and
multiplying by a zero range.

**Verdict: keep, but it is not a lever.** 46.1% of the frame's most-stalled load
removed for 0.8%. Perf-neutral, exact, and now measured.

---

## 4. WHAT IS DEAD, WITH THE MEASUREMENT THAT KILLED IT

- **§6.1 (reduce candidates)** — `brickCands 14.1/px → segments 13.6/px` is
  **96% acceptance**. Only ~0.5 candidates/px are wasted. Coarser bricks or a
  two-level BLAS would prune against a 4% rejection rate. The 66,877 AABBs at 8^3
  are already right for this scene.
- **§6.3 (in-brick occupancy)** — implemented, exact, 46.1% skip, 0.8%. Closed.
- **Per-instance LoD as a perf lever** — `[LOD]` shows every instance at mip 0 and
  **that is correct**: footprints are 0.3–3.5 wu against a 3.60 wu mip0 cell
  (~6 sub-pixel voxels/pixel), so UE's projected-error rule is *declining* to
  coarsen, exactly as intended. Forcing `mipPixelThreshold: 16.0` coarsens 6 of 9
  instances to mip 1–2 and moves `segments/px` 13.6 → 12.1 — at the cost of the
  blockiness the rule exists to prevent. **Leave it at 1.0.**
  Note the real gate is `floor(log2(footprint/cell0))`, NOT the
  `footprint <= cell0` early-out above it: the mip only steps at `>= 2*cell0`.
- **ReSTIR-FG as a drop-in** — `C:\Users\Friss\Documents\Clouds\ReSTIR-FG` is a
  Falcor fork but has **zero** volume support (no hits for
  participating/volumetric/gridvolume/medium). Every anchor is a surface concept:
  photons stored on diffuse surfaces by `bsdfProperties.roughness`, reservoirs
  initialized "at the first diffuse hit", `sd.computeNewRayOrigin()`,
  `guideNormal`. A cloud has no first diffuse hit. Photons are also stored as
  AABBs in a *second* acceleration structure — more traversal, which is the part
  already measured as efficient.
  **But the mechanism transfers**: final gathering *terminates* the path at
  cached radiance instead of continuing to bounce, and with `maxBounces 64` that
  cuts *marching*, which is the one thing in the ReSTIR family that could reach
  4x (HANDOFF_6 §7's actual question). The real analogue is **volumetric photon
  mapping / beam radiance estimate** (Jarosz et al.), same trick without the
  surface assumption. Caveat: a volumetric gather replaces marching with a
  spatial lookup, which is itself a dependent-load chain. Mine the repo for
  reservoir/photon-AS *infrastructure*, not the algorithm.

---

## 5. MEASUREMENT DISCIPLINE — read before quoting any number

HANDOFF_6 §4's four conditions (matched `res`, frames past ~1000, 3+ consecutive
samples, bit-identical work counters) are **necessary but NOT sufficient.**

- **Cross-session gpuMs on this machine drifts ~25% with bit-identical work
  counters.** Observed three times: 540p runs read 8.41–9.13, 8.12–9.81, and
  5.94–7.50 ms with every counter identical. All four conditions were met. Only
  **same-session A/B** (a runtime toggle, one process, no reload) is trustworthy.
  That is why the occupancy skip is a CB uniform and not a define.
- **The resolution trap is still live and still bites.** `resizeFrameBuffer` only
  resizes the WINDOW (`SampleApp.cpp:562`); `handleWindowSizeChange()` then sets
  the FBO from `getClientAreaSize()`. So the FBO is recomputed from the window on
  *every* window event, forever. A run started at 960x540 and silently became
  2582x1462 at frame 576. **Headless (`mpWindow == nullptr`) is immune** and is
  the right answer for profiling runs.
- **A suspicious invariance is a finding, not a nuisance.** `aabbTests` landing on
  exactly 14.6 through a six-instance mip change was the tell that the counter
  could not see the brick TLAS. It was noticed and talked past, and a wrong
  conclusion ("§6.1 is dead because candidates are insensitive to mip") was
  published off it. The same pattern later gave the *correct* answer: 128/128/
  128/128 across four builds is what proved the cap.
- **Read the configured value, not the `#define` default.** `RIS_CANDIDATES`
  defaults to 8 in the header; the script sets 2. A "found it, 17 registers"
  conclusion was drawn off the default and was wrong.
- **Shaders deploy at build time** to `build/.../bin/Release/shaders/`. A source
  edit does not reach a running app. But **pass properties are defines**, so
  `logWorkStats`, `useBrickTlas`, `mipPixelThreshold` etc. change with a script
  edit and a relaunch — **no C++ rebuild**. Falcor recompiles on the changed
  define (verify via the shader hash changing in Shader Pipelines).

---

## 6. TOOLING

`scripts/VNA_NsightShaderProf.cmd` — flags that each fail silently if wrong:
`--architecture Ada` (scopes the sampler; without it `Total Samples: 0`),
`--metric-set-id 2`, `--real-time-shader-profiler`, `--debug-shaders`,
`--max-duration-ms` hard-capped below 10000, and now **`--multi-pass-metrics`**
(without it Trace Analysis banners "analysis results will be limited" and
populates exactly one row).

- **ngfx has no headless export.** `--output-dir` writes traces; reading requires
  the GUI. There is no report/CSV/JSON dump option.
- Traces are `.ngfx-gputrace` (not `.gputrace`) — glob accordingly.
- **Where the answers live**: Shader Pipelines → `#Reg`, `#Warp`, `Smem`, `Live
  Registers` per kernel. Trace Analysis → "Warp Launch Stalled by Reasons".
  Summary → Top-Level Throughput + SM Warp Occupancy. Hotspots → per-line samples
  and stall reasons. Shader Source → per-line samples, **CSV-exportable**, but
  `Live Registers` is empty there (SASS-only, and GPU Trace has no SASS).

---

## 7. IMMEDIATE NEXT STEPS

1. **New capture with `--multi-pass-metrics`** (already wired). The categories
   that stayed empty — `SM Warp Issue Stalls`, `SM Warp Occupancy`,
   `Unit Throughputs` — should populate and may name the next constraint directly.
2. **Check for register spill / local-memory traffic** in the Metrics tab. If
   non-zero, true demand exceeds 128 and the restructuring job is bigger than
   "cut 48".
3. **The only live lever: get `main` to ≤80 registers.** Not by deleting features
   (proved insensitive) but by reducing peak live state across the marching loop.
   Nothing between 128 and 81 changes anything — do not measure intermediate cuts
   and conclude "no effect".
4. `shadeMain` is **49% of samples, more than `main`'s 46%**, at 96 regs / 20
   warps. It has been the less-examined kernel and deserves equal attention; it
   also needs ≤80 for its next step (12 CTAs / 24 warps).
5. Hotspots are **flat** — top line `sampleDistanceRQ` at 11.61%, then 5.71%,
   3.11%, 2.63%, and a long tail. There is no single hot line to fix, which is
   itself consistent with an occupancy limit rather than a code-level one.

---

## 8. CURRENT CONFIG STATE

`scripts/WdasSkyVNA.py` is restored to canonical: `useBrickTlas: True`,
`useMergedTail: True`, `useSharedCandidateSweep: True`, `mipPixelThreshold: 1.0`,
`logWorkStats: True`, `logRisStats: True`, `risStatsInterval: 64`. The occupancy
skip defaults ON and is toggleable in the UI ("Occupancy tap skip").

Diagnostic captures taken this session, in order:
`12_42_44` baseline · `12_55_40` stats off · `13_10_32` mergedTail+sweep off ·
`13_17_59` brickTlas off. All at 960x540 unless the window was touched.
