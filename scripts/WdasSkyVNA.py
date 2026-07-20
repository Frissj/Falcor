# THE WHOLE VNA STACK, ON. One launcher that runs everything built so far,
# together, on the nine-instance sky:
#
#   section 3   cloudlet instancing            (scene: 9 instances, 1 grid)
#   section 3   merged coarse tail             (far rays march ONE summed grid)
#   section 4   HW-BVH brick TLAS              (RayQuery over per-brick AABBs,
#                                               per-instance projected-error mip)
#   section 2   residual ratio tracking        (analytic T_c over the mean pyramid)
#   section 2/5 footprint-driven residual mip  (UE projected-error rule)
#   section 5   Stage-A RIS primary scatter    (M real collisions, shared sweep)
#   section 9   MegaLights NEE budget          (one shadow ray per path)
#   section 6   demodulated reconstruction     (accum(Lin) + accum(T) * bg)
#
#   Mogwai.exe --script scripts/WdasSkyVNA.py
#
# EVERY feature is set EXPLICITLY here - never rely on C++ defaults, so this
# launcher stays "everything on" even if a default changes.
#
# The reference shortcuts stay untouched on purpose: THIS graph is the fast
# path, THOSE are the ground truth it must match. If this image ever looks
# wrong, isolate the guilty stage with the validation scripts
# (VNA_ResidualSweep.py, VNA_RisValidate.py, WdasCloudDemod.py) instead of
# debugging everything at once.

from falcor import *

def render_graph_VNA():
    g = RenderGraph("VNAStack")

    VolumePathTracer = createPass("VolumePathTracer", {
        'maxBounces': 64,
        'useNEE': True,
        # UE MegaLights lesson: one reservoir-picked vertex does NEE per path
        # (one shadow ray/pixel instead of one per bounce). Unbiased; attacks
        # the measured-dominant shadeNEE bucket.
        'useSingleNeePerPath': True,
        # Section 2/P2: analytic control-variate transmittance.
        'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
        # MEASURED AND REJECTED (log 57): residualMip 1 cut cells 65.7M -> 59.5M
        # (-9.5%) but RAISED taps 32.9M -> 39.0M (+18.6%); per-pixel candGen went
        # 21.7->19.9 cells for 9.3->11.9 taps, i.e. taps rose more in absolute
        # terms than cells fell. Coarsening the control field grows sigmaRbar, so
        # the inner residual ratio-track loop takes more density taps per cell,
        # and a stochastic trilinear tap is not cheaper than a cell's two
        # majorant fetches. Net ~164M -> ~158M lookups: a wash. mip 2 is worse by
        # the same trend. The footprint rule clamping to 0 is CORRECT, not a bug:
        # control-field accuracy beats cell count here. Do not re-litigate.
        'residualMip': 0,
        # UE projected-error rule: per instance+segment, coarsest mean/range
        # mip whose cell stays under the pixel footprint. Unbiased at any mip.
        'footprintMip': True,
        'footprintScale': 1.0,
        # Section 5 Stage A: RIS on the primary vertex, candidates generated
        # as real collisions in ONE shared traversal.
        'useRIS': True,
        'risCandidates': 2,  # cut 4->2 under exact-MIS temporal reuse; converged-identical (p8_m2, -0.09%)
        'risMip': 2,
        'useSharedCandidateSweep': True,
        # Transmittance-adaptive candidate budget: nearly-transparent rays run
        # 1 process instead of M (82-88% of fixed-M processes were escaping).
        'useAdaptiveM': True,
        # Stage B: temporal reservoir reuse of the primary scatter vertex
        # (t-shift + confidence-capped merge). Effective M grows over frames.
        'useTemporalReuse': True,
        'temporalMCap': 20.0,
        # Stage C: spatial reuse - prev-frame reservoirs from Gaussian-offset
        # neighbors merge through the exact Stage B shift/guard/weight.
        # Sigma 16 px matches the classic 30 px disk (ReSTIR PT Enhanced).
        'useSpatialReuse': False,  # HELD: cross-ray volume reuse needs the sigmaT_c(y')/sigmaT_i(y_i) reweight (measured -2.2%/neighbor even under exact pairwise MIS)
        'spatialNeighbors': 2,
        'spatialRadiusPx': 16.0,
        # Defensive RIS target floor (x fully-lit isotropic). Bounds the
        # L/Lhat firefly mechanism the validation matrix isolated (single
        # 800-2200x pixels at 1 spp). Unbiased at any value.
        'risTargetFloor': 0.01,   # firefly bound; CLEAN under exact pairwise MIS (+0.08%, p8_m2_floor) - the old -1.1% was the constant-MIS feedback
        # Russian roulette survival floor. Was hardcoded 0.05; swept because
        # Nsight put shadeMain (the bounce loop) at ~56% of the frame and its
        # cost is path LENGTH. 0.10 is strictly better than 0.05 on EVERY axis
        # (rq2 sweep, gate metric = cloudy mean vs 01_ref):
        #   q=0.05  rel -0.0168%  1spp MSE 4.648e-01  >10: 360  >100: 13  sc 2.6
        #   q=0.10  rel +0.0044%  1spp MSE 3.521e-02  >10:  94  >100:  1  sc 2.1
        # 13x lower 1-spp MSE and 19% fewer sampler calls. Killing low-throughput
        # paths EARLY removes the L/Lhat firefly source (a path that limps on at
        # tiny throughput then hits a bright NEE sample) - the same mechanism
        # risTargetFloor bounds. The x1.11 reweight is far too mild to reintroduce it.
        # Knee is sharp: q=0.15 FAILS the gate (+0.331%). Do not raise past 0.125.
        # Starting roulette earlier (rouletteStartBounce 1/2) FAILED at every
        # floor tested - that lever is dead, leave it at 3.
        # 0.125 = the knee itself, adopted 2026-07-20: rq2 sweep scored it
        # +0.064% cloudy-mean (PASS, 2.3x inside the gate) with 1-spp firefly
        # counts (>100: 2) far below the q=0.05 baseline's 13. Buys ~10-15%
        # off shadeMain's warp critical path on top of q=0.10.
        'rouletteMinQ': 0.125,
        'rouletteStartBounce': 3,
        # Stream compaction (ReSTIR PT Enhanced 6.2.2 / UE dense-dispatch):
        # phase A queues the ~13% of pixels that scatter, an indirect phase B
        # shades one thread per real path. Attacks the measured 87%-idle
        # warp divergence in the shading/bounce buckets.
        'useCompaction': True,
        # Wavefront phase B: MEASURED OFF (2026-07-20). fps degrades
        # monotonically with requeue rounds - K=0 fastest - because the
        # divergence is INTRA-bounce (single-bounce marching cost varies 10x+
        # between lanes), which requeueing cannot equalize; the state
        # round-trip and per-round barriers only add. Byte-identical backend
        # kept compiled for A/B; do not re-enable without a new measurement
        # that contradicts this one.
        'useWavefront': False,
        # Section 4: HW-BVH brick TLAS (UE HeterogeneousVolumes port) with
        # per-instance projected-error mip selection.
        'useBrickTlas': True,
        # 1.0 is correct - do not "fix" the fact that [LOD] shows every instance
        # at mip 0. Footprints are 0.3-3.5 wu against a 3.60 wu mip0 cell, i.e.
        # ~6 sub-pixel voxels per pixel, so the projected-error rule is DECLINING
        # to coarsen, which is UE's Nanite Foliage rule working as intended
        # (brick size tracks ~1 pixel at every distance; coarser = blocky).
        # Measured 2026-07-20 at 540p: forcing 16.0 coarsens 6 of 9 instances to
        # mip 1-2 and moves segments/px 13.6 -> 12.1. It does NOT move
        # aabbTests/px off 14.6 - but that proves nothing, because statAabbTests
        # is `+= getGridVolumeCount()` per gatherIntervals call
        # (VolumeInstanceSampler.slang:246), i.e. 9 software slab tests against
        # whole-instance bounds in the INTERVAL-LIST path. It is unrelated to the
        # brick TLAS and cannot respond to mip. There is currently NO counter for
        # real brick-AABB intersection-shader invocations; add one before making
        # any claim about RayQuery candidate count.
        'mipPixelThreshold': 1.0,
        # Section 3: merged coarse tail (UE Nanite Assemblies lesson) - far
        # rays march one world-space summed grid.
        'useMergedTail': True,
        'tailRes': 64,
        'tailGateVoxels': 32.0,
        # Diagnostics OFF for the shipping path. Logging every frame costs a
        # full GPU sync (submit-and-wait) per frame: measured 24 ms of GPU
        # work stretching to a 77 ms wall frame (13 fps) from the stall
        # alone, and work-stats additionally recompile the sampler with
        # per-step counters. Re-enable in the UI (or here) to diagnose cost;
        # the interval keeps any re-enabled logging to one sync per 60 frames.
        # TEMPORARILY ON to diagnose the 30 ms steady state. Interval 64 is one
        # sync per 64 frames and matches the frame cadence of the logs 52/53
        # blocks, so the numbers compare directly. Set back to False/False/60
        # before quoting any shipping perf number.
        # OCCUPANCY EXPERIMENT (2026-07-20). Nsight says `main` uses 128
        # registers -> 65536/128 = 512 threads = exactly 16 warps = 33.3%
        # occupancy (measured CS occupancy 33.5%). Registers are the SOLE
        # limiter: smem 23552 B/CTA would allow 4 CTAs, registers allow 2.
        # Occupancy is a STEP function at a 256-thread CTA:
        #     2 CTAs = 16 warps (33%)  <- here, at 128 regs
        #     3 CTAs = 24 warps (50%)  <- needs <= 80 regs
        #     4 CTAs = 32 warps (67%)  <- needs <= 64 regs
        # 128->112->96 buys NOTHING. This is why register cuts measured as
        # null results 4 times: they were on the flat part of the step.
        #
        # These two flags drive USE_WORK_STATS / USE_RIS_STATS /
        # GRID_VOLUME_SAMPLER_STATS (VolumePathTracer.cpp ~822-826), which gate
        # the readback of ~14 uint stat* fields on VolumeInstanceSampler plus
        # the gGvs* statics. With them off those fields are write-only and
        # should be dead-code eliminated. Being a define, this needs NO C++
        # rebuild - Falcor recompiles the shader on the changed define.
        #
        # RESULT (measured, both captures 2026-07-20): turning these OFF moved
        # `main` from 128 registers to ... 128 registers. Live registers 126 ->
        # 125, warps 16 -> 16, smem 23552 -> 23552. shadeMain: 96 -> 96 regs
        # (smem 4864 -> 2304, live 92 -> 87), still 20 warps.
        #
        # So THE INSTRUMENTATION IS FREE in register terms and does not cap
        # occupancy - the ~14 stat* fields are write-only under the defines and
        # DXC eliminates them. Left ON deliberately: there is no register reason
        # to run blind. (The per-frame readback sync is a separate, real cost -
        # that is what risStatsInterval is for.)
        #
        # The 128 registers are STRUCTURAL. Whatever gets `main` to <= 80 has to
        # come out of the live state held across the marching loop - the
        # RayQuery objects, the reservoir, the sampler's non-stat fields - not
        # out of diagnostics.
        'logWorkStats': True,
        'logRisStats': True,
        'risStatsInterval': 64,
    })
    g.addPass(VolumePathTracer, "VolumePathTracer")

    # Section 6: the two stochastic channels accumulate separately; background
    # is deterministic and never accumulated. Recombined AFTER accumulation.
    AccumLin = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumLin, "AccumLin")
    AccumT = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumT, "AccumT")
    Mul = createPass("Composite", {'mode': 'Multiply', 'outputFormat': 'RGBA32Float'})
    g.addPass(Mul, "Mul")
    Add = createPass("Composite", {'mode': 'Add', 'outputFormat': 'RGBA32Float'})
    g.addPass(Add, "Add")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VolumePathTracer.Lin",           "AccumLin.input")
    g.addEdge("VolumePathTracer.transmittance", "AccumT.input")
    g.addEdge("AccumT.output",                  "Mul.A")
    g.addEdge("VolumePathTracer.background",    "Mul.B")
    g.addEdge("Mul.out",                        "Add.A")
    g.addEdge("AccumLin.output",                "Add.B")
    g.addEdge("Add.out",                        "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    # Un-demodulated color through its own accumulator: flip between this and
    # ToneMapper.dst in the output dropdown - converged, they must match.
    AccumColor = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumColor, "AccumColor")
    ToneMapperRef = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperRef, "ToneMapperRef")
    g.addEdge("VolumePathTracer.color", "AccumColor.input")
    g.addEdge("AccumColor.output",      "ToneMapperRef.src")
    g.markOutput("ToneMapperRef.dst")

    # Work counters, as always.
    g.markOutput("VolumePathTracer.work")
    return g

graph = render_graph_VNA()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))
m.scene.selectViewpoint(1)

# ---------------------------------------------------------------------------
# RESOLUTION - 960x540.
#
# The VolumePathTracer pass is ~93% of the frame and EVERY term in it is
# per-pixel, so quartering the pixels quarters the pass. This is the single
# largest lever in the project and it is one line.
#
# COMPOUNDING BONUS - the LoD system switches itself on. footprintSpread is
# 2 * tanHalfFovY / targetDim.y * footprintScale, i.e. INVERSELY proportional to
# vertical resolution. At 1080p footprints are 0.2-1.8 wu against a 3.60 wu mip0
# cell, so selectResidualMip clamps to mip 0 on every instance every frame and
# the whole LoD path is dead (this is why footprintScale 2 measured as a null
# result - it could not fire). At 540p footprints double to 0.4-3.6 wu and start
# crossing the threshold, so far instances begin selecting mip 1 on their own.
# Extra saving on top of the 4x, for free, with no code change.
#
# Reference point: Nubis3 ships at exactly 960x540; their 2.2-4ms headline is at
# this resolution, not 1080p.
#
# COSTS: image quality, and a temporal upscale pass if you want 1080p output.
# The temporal machinery (Stage B reservoirs, accumulation) already exists.
#
# MEASURE THE QUALITY COST, do not assume it: capture here and at 1280x720 and
# score both with scripts/VNA_RouletteScore.py against the SAME gate (cloudy-mean
# masked to top-40% luminance of 01_ref). Note the existing reference baselines
# are 1080p, so a like-for-like comparison needs references regenerated at each
# resolution - a raw diff against the 1080p 01_ref measures resampling, not the
# renderer.
m.resizeFrameBuffer(960, 540)
