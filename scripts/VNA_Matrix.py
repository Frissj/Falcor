# VNA validation matrix (VNA_BUILD_HANDOFF_3 "RUN THIS BEFORE ANYTHING ELSE").
# Every feature added on the optimization day is an ESTIMATOR SWITCH: converged
# images must be IDENTICAL to the reference. Any converged delta is a bug, not
# a tradeoff. This script runs the whole matrix unattended and captures linear
# HDR EXRs; VNA_Matrix_Compare.py (plain python, no Mogwai) turns them into a
# numbers table.
#
#   1) Mogwai.exe --script scripts/VNA_Matrix.py        (renders + captures)
#   2) py scripts/VNA_Matrix_Compare.py                 (prints the table)
#
# Matrix steps -> configs (handoff numbering):
#   1 reference sanity          01_ref (all new options OFF, non-RIS)
#   2 backend equivalence       02_bricktlas   vs 01_ref
#   3 Stage A stack             03..06 (RIS x {sweep} x {adaptiveM}) vs 01_ref,
#                               plus 1-spp captures for the variance comparison
#   4 temporal (static camera)  07_temporal    vs 01_ref
#   5 single NEE                08_singlenee   vs 01_ref
#   6 footprint mip             09_footprint   vs 01_ref
#   7 merged tail, forced       10_tail4 (gate 4 voxels so it ENGAGES) vs 01_ref
#   + full shipping stack       11_full (WdasSkyVNA options) vs 01_ref
#   + full stack + forced tail  12_full_tail4  vs 01_ref
# Tail/full configs also run at the distant viewpoint (VP3) where footprints
# are large enough for the tail to actually engage; their logWorkStats is ON
# (throttled) so the log's [COST] tailRays proves engagement - a tail test
# where tailRays stays 0.00 tested nothing.
#
# Step 8 of the handoff (VNA_RisValidate.py, WdasCloudDemod.py) stays separate:
# those scripts already exist and gate their own invariants.

import os

from falcor import *

# 256 frames is enough: bias is detected by the SIGNED mean diff over ~2M
# pixels (compare script), which averages per-pixel noise down by sqrt(2M) -
# a sub-percent systematic shift is visible way above the floor. Raise via
# env var VNA_MATRIX_FRAMES only if a config sits marginally above the floor
# and you need the noise to shrink to call it (bias stays put, noise drops).
FRAMES_CONVERGED = int(os.environ.get("VNA_MATRIX_FRAMES", "256"))
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

# All new options OFF, non-RIS. Every option explicit (house policy).
BASE = {
    'maxBounces': 64,
    'useNEE': True,
    'useSingleNeePerPath': False,
    'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
    'residualMip': 0,
    'footprintMip': False,
    'footprintScale': 1.0,
    'useRIS': False,
    'risCandidates': 4,
    'risMip': 2,
    'useSharedCandidateSweep': False,
    'useAdaptiveM': False,
    'useTemporalReuse': False,
    'temporalMCap': 20.0,
    'useBrickTlas': False,
    'mipPixelThreshold': 1.0,
    'useMergedTail': False,
    'tailRes': 64,
    'tailGateVoxels': 32.0,
    'logWorkStats': False,
    'logRisStats': False,
    'risStatsInterval': 256,
    'useSpatialReuse': False,
    'spatialNeighbors': 2,
    'spatialRadiusPx': 16.0,
    # 0 in the matrix BASE: the floor is itself an estimator change, so the
    # matrix isolates it (config 14) instead of baking it into every config.
    'risTargetFloor': 0.0,
    'useCompaction': False,
}

# Stats on (throttled) for tail configs: the [COST] tailRays column in the log
# is the proof the forced tail path actually ran. Counters only count - they
# do not change the estimator math, so images must still match.
TAIL_STATS = {'logWorkStats': True, 'risStatsInterval': 256}

FULL = dict(BASE,
    useSingleNeePerPath=True, footprintMip=True,
    useRIS=True, useSharedCandidateSweep=True, useAdaptiveM=True,
    useTemporalReuse=True, useBrickTlas=True, useMergedTail=True)

# (name, props, viewpoints). VP1 = near sky viewpoint, VP3 = distant.
CONFIGS = [
    ("01_ref",          BASE,                                                   [1, 3]),
    ("02_bricktlas",    dict(BASE, useBrickTlas=True),                          [1]),
    ("03_ris",          dict(BASE, useRIS=True),                                [1]),
    ("04_ris_sweep",    dict(BASE, useRIS=True, useSharedCandidateSweep=True),  [1]),
    ("05_ris_adaptm",   dict(BASE, useRIS=True, useAdaptiveM=True),             [1]),
    ("06_ris_both",     dict(BASE, useRIS=True, useSharedCandidateSweep=True,
                             useAdaptiveM=True),                                [1]),
    ("07_temporal",     dict(BASE, useRIS=True, useSharedCandidateSweep=True,
                             useAdaptiveM=True, useTemporalReuse=True),         [1]),
    ("08_singlenee",    dict(BASE, useSingleNeePerPath=True),                   [1]),
    ("09_footprint",    dict(BASE, footprintMip=True),                          [1]),
    ("10_tail4",        dict(BASE, useMergedTail=True, tailGateVoxels=4.0,
                             **TAIL_STATS),                                     [1, 3]),
    ("11_full",         FULL,                                                   [1, 3]),
    ("12_full_tail4",   dict(FULL, tailGateVoxels=4.0, **TAIL_STATS),           [1, 3]),
    # Stage C (spatial reuse) on top of the validated temporal config: static
    # camera, converged must match the reference exactly.
    ("13_spatial",      dict(BASE, useRIS=True, useSharedCandidateSweep=True,
                             useAdaptiveM=True, useTemporalReuse=True,
                             useSpatialReuse=True),                             [1]),
    # Defensive target floor alone: unbiased by construction (any positive
    # target), but the claim gets measured, not assumed.
    ("14_targetfloor",  dict(BASE, useRIS=True, useSharedCandidateSweep=True,
                             useAdaptiveM=True, risTargetFloor=0.01),           [1]),
    # Stream compaction: same estimator, different thread that shades it.
    # Phase B uses an independently-seeded RNG stream, so the 1-spp image is a
    # different noise realization - converged must still match exactly.
    ("15_compaction",   dict(BASE, useRIS=True, useSharedCandidateSweep=True,
                             useAdaptiveM=True, useCompaction=True),            [1]),
]

def render_graph_Matrix():
    g = RenderGraph("VNAMatrix")
    VolumePathTracer = createPass("VolumePathTracer", BASE)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    # Compare the LINEAR accumulated color (estimator invariance lives here).
    # The demodulated chain has its own scripted gate (WdasCloudDemod.py).
    Accum = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g

graph = render_graph_Matrix()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py / WdasSkyVNA.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props, viewpoints in CONFIGS:
    for vp in viewpoints:
        m.scene.selectViewpoint(vp)
        # updatePass rebuilds the pass (defines changed) and restarts
        # accumulation, so every config/viewpoint starts clean.
        graph.updatePass("VolumePathTracer", props)

        m.renderFrame()
        m.frameCapture.baseFilename = "{}_vp{}_1spp".format(name, vp)
        m.frameCapture.capture()

        for i in range(FRAMES_CONVERGED - 1):
            m.renderFrame()
        m.frameCapture.baseFilename = "{}_vp{}_converged".format(name, vp)
        m.frameCapture.capture()
        # Progress lands on disk, not just stdout: the window may look frozen
        # during the run, and this file is the proof of life.
        with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
            f.write("{} vp{} done ({} frames)\n".format(name, vp, FRAMES_CONVERGED))
        print("VNA-MATRIX: {} vp{} done ({} frames).".format(name, vp, FRAMES_CONVERGED))

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("ALL CAPTURES COMPLETE\n")
print("VNA-MATRIX: all captures complete -> {}".format(OUT_DIR))
print("VNA-MATRIX: now run   py scripts/VNA_Matrix_Compare.py")
