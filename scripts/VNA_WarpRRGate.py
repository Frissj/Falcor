# Minimal warp-RR gate driver. Produces ONLY the captures VNA_RadCacheScore.py
# needs to judge the redesigned warp-aware residual roulette (HANDOFF_9 5) - NOT
# the whole matrix. 5 configs vs the sweep's 15+, so it finishes in a fraction of
# the time:
#   t_p25   baseline (adopted CV config, warp-RR OFF)          @ 256 and 1024
#   t_wrr8  baseline + warp-RR at WRR_LANES                    @ 256 and 1024
#   ntb / ntb_s1 / ntb_s2   seed-only pairs -> the noise floor @ 256
#
# After it finishes, score with the EXISTING analysis (unchanged):
#     python scripts/VNA_RadCacheScore.py
# It reads {name}_vp1_w{w}.exr from vna_measurements/matrix and prints the
# t_wrr8-vs-t_p25 verdict at 256/1024 (PASS / inside-floor / REAL BIAS) plus the
# measured floor from the ntb* pairs.
#
# RESOLUTION: full res (the native window, e.g. 2560x1351) - NOT the 960x540
# Nubis path. The gate compares t_wrr8 vs t_p25, both captured here at the same
# size, so it only needs consistency within this run.
#
#   Mogwai.exe --script scripts/VNA_WarpRRGate.py
#
# SWEEP A THRESHOLD: set WRR_LANES to 8 / 16 / 24 and re-run. The capture is
# always named "t_wrr8" so the scorer's ("t_wrr8","t_p25") pair matches with no
# edit - so only ONE threshold lives in the matrix dir at a time. To keep several
# side by side, rename the "t_wrr8" row below AND add the matching pair to the
# scorer's loop.

import os
from falcor import *

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"
WARM = 192          # cache warm-up frames per config (rad + tau cache have state)
# The knob under test (the t_wrr8 capture always carries whatever is set here, so
# the existing scorer pair ("t_wrr8","t_p25") matches unchanged):
#   RESIDUAL_TAIL_Q  Lever-2: per-bounce kill prob on post-cut residual paths.
#                    0 = off. This is the current lever - caps the residual
#                    stragglers PATHLEN found. RR-unbiased, so it should PASS the
#                    gate at any value; the real test is the shade-time delta.
#   WRR_LANES        Lever-1: warp-RR threshold (set 0 - measured out already).
RESIDUAL_TAIL_Q = 0.5
WRR_LANES = 0

ACCUM_PROPS = {'enabled': True, 'precisionMode': 'Single'}
ACCUM_PROPS_OFF = {'enabled': False, 'precisionMode': 'Single'}

# Shipping config, PINNED (mirrors VNA_RadCacheSweep.py's SHIP as of 2026-07-20).
# A gate must test a FIXED config, so this is a deliberate copy - do NOT wire it
# to the live WdasSkyVNA.py, or the gate drifts with the daily driver.
SHIP = {
    'maxBounces': 64,
    'useNEE': True,
    'useSingleNeePerPath': True,
    'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
    'residualMip': 0,
    'footprintMip': True,
    'footprintScale': 1.0,
    'useRIS': True,
    'risCandidates': 2,
    'risMip': 2,
    'useSharedCandidateSweep': True,
    'useAdaptiveM': True,
    'useTemporalReuse': True,
    'temporalMCap': 20.0,
    'useSpatialReuse': False,
    'spatialNeighbors': 2,
    'spatialRadiusPx': 16.0,
    'risTargetFloor': 0.01,
    'useCompaction': True,
    'useWavefront': False,
    'useBrickTlas': True,
    'mipPixelThreshold': 1.0,
    'useMergedTail': True,
    'tailRes': 64,
    'tailGateVoxels': 32.0,
    'rouletteMinQ': 0.125,
    'rouletteStartBounce': 3,
    'useTauCache': True,
    'tauCacheRes': 64,
    'tauCacheInterval': 0,
    'radCacheRes': 64,
    'radCutBounce': 3,
    'radTrainEvery': 8,
    'radEma': 0.10,
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
}

CONFIGS = [
    # name, props, capture windows. t_p25 / t_wrr8 differ ONLY in radWarpRRLanes,
    # so the scorer's delta is the roulette alone.
    ("t_p25",  dict(SHIP, useRadCache=True, trRRThreshold=0.05, trRRMode=7, radResidualSurvival=0.25), (256, 1024)),
    ("t_wrr8", dict(SHIP, useRadCache=True, trRRThreshold=0.05, trRRMode=7, radResidualSurvival=0.25, radWarpRRLanes=WRR_LANES, radResidualTailQ=RESIDUAL_TAIL_Q), (256, 1024)),
    # Noise floor: identical estimator, seed-only changes, temporal + cache OFF.
    ("ntb",    dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0, trRRMode=0, radResidualSurvival=1.0), (256,)),
    ("ntb_s1", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0, trRRMode=0, seedOffset=7777,  radResidualSurvival=1.0), (256,)),
    ("ntb_s2", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0, trRRMode=0, seedOffset=31337, radResidualSurvival=1.0), (256,)),
]


def render_graph_WarpRRGate():
    g = RenderGraph("VNAWarpRRGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_WarpRRGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()
# Render at FULL res (the native window). No resizeFrameBuffer pin - the gate
# compares t_wrr8 vs t_p25 (both captured here, same size), so the resolution
# just has to be consistent within this run, which it is.

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props, windows in CONFIGS:
    m.scene.selectViewpoint(1)
    # Fresh pass = fresh caches (rad + tau); warm-up trains the rad cache.
    graph.updatePass("VolumePathTracer", props)
    for i in range(WARM):
        m.renderFrame()
    # Reset ACCUMULATION ONLY (recreating Accum leaves the tracer caches warm).
    # off->on with DIFFERENT dicts so neither updatePass can no-op.
    graph.updatePass("Accum", ACCUM_PROPS_OFF)
    m.renderFrame()
    graph.updatePass("Accum", ACCUM_PROPS)
    m.renderFrame()
    frame = 1
    for w in windows:
        while frame < w:
            m.renderFrame()
            frame += 1
        m.frameCapture.baseFilename = "{}_vp1_w{}".format(name, w)
        m.frameCapture.capture()
    print("VNA-WARPRR-GATE: {} done (lanes={})".format(name, WRR_LANES if name == "t_wrr8" else "-"))

print("VNA-WARPRR-GATE: complete. Now run:  python scripts/VNA_RadCacheScore.py")

# Quit Mogwai when the captures are done (same as VNA_RadCacheSweep.py:217), so
# the shortcut is fire-and-forget: launch, capture, exit, then score offline.
exit()
