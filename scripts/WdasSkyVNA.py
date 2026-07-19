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
        'residualMip': 0,
        # UE projected-error rule: per instance+segment, coarsest mean/range
        # mip whose cell stays under the pixel footprint. Unbiased at any mip.
        'footprintMip': True,
        'footprintScale': 1.0,
        # Section 5 Stage A: RIS on the primary vertex, candidates generated
        # as real collisions in ONE shared traversal.
        'useRIS': True,
        'risCandidates': 4,
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
        'useSpatialReuse': True,
        'spatialNeighbors': 2,
        'spatialRadiusPx': 16.0,
        # Defensive RIS target floor (x fully-lit isotropic). Bounds the
        # L/Lhat firefly mechanism the validation matrix isolated (single
        # 800-2200x pixels at 1 spp). Unbiased at any value.
        'risTargetFloor': 0.01,
        # Stream compaction (ReSTIR PT Enhanced 6.2.2 / UE dense-dispatch):
        # phase A queues the ~13% of pixels that scatter, an indirect phase B
        # shades one thread per real path. Attacks the measured 87%-idle
        # warp divergence in the shading/bounce buckets.
        'useCompaction': True,
        # Section 4: HW-BVH brick TLAS (UE HeterogeneousVolumes port) with
        # per-instance projected-error mip selection.
        'useBrickTlas': True,
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
        'logWorkStats': False,
        'logRisStats': False,
        'risStatsInterval': 60,
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
