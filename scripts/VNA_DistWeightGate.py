# Weighted delta-tracking gate - sweeps distWeightK for the distance sampler.
#
#   Mogwai.exe --script scripts/VNA_DistWeightGate.py
#
# WHAT IS BEING TESTED
# Softening the free-flight majorant toward the cell mean, sigmaBar_w =
# lerp(mean, majorant, K), and paying for it with a per-collision weight
# max(1, sigma/sigmaBar_w). Unbiased for ANY K (Galtier et al. 2013; Novak et al.
# 2018 STAR 5). [HOMOG] measured mean/majorant < 0.125 in 67% of traversed cells,
# so ~7 of every 8 proposed collisions are null - a softer sigmaBar deletes them
# outright. This attacks TAPS, the measured cost, unlike the 4^3 granularity fix
# which attacked cells and lost the exchange rate.
#
# THIS GATE ANSWERS TWO SEPARATE QUESTIONS. Do not conflate them.
#
#   1. CORRECTNESS (the important one). Every K is claimed unbiased, so the
#      CONVERGED captures must all match K=1.0. If w050_s4096 differs from
#      w100_s4096 by more than the noise floor, the weight is WRONG - an
#      implementation bug, not a quality trade - and the timing is irrelevant.
#      This is the test that catches a mistake in max(1, sigma/sigmaBar).
#
#   2. EFFICIENCY. At equal SAMPLES a lower K is faster and noisier. That is
#      expected and is not a result on its own. The real question is equal-TIME:
#      does the extra sample count outrun the extra variance? Read the s256
#      captures against the gpuMs from the [WORK] lines in the same run.
#
# WHY BOTH CAPTURES EXIST: a converged match proves unbiasedness; an equal-sample
# capture shows the variance cost. Either alone is misleading - a biased
# estimator can look clean at low spp, and an unbiased one can look terrible.
#
# READ THE COUNTERS TOO: [BOUNCECOST] ds taps/call should FALL as K drops. If it
# does not, the softened majorant is not reaching the sampler (mip != 0 falls
# back to the true majorant by design) and any timing difference is something
# else.
#
# LOG SEGMENTATION: script print() does NOT reach the Mogwai log. Segment on the
# [WORK] frame counter, which resets to 0 on every updatePass - one block per
# config, in CONFIGS order.

import os
from falcor import *

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\distweight"
RES = (960, 540)   # Ship resolution, pinned - same as the other gates.
WARM = 192         # updatePass recreates the pass -> rad/tau caches reset.
NOISE_SPP = 256    # Equal-sample capture: shows the variance cost.
CONV_SPP = 4096    # Converged capture: shows bias, or proves there is none.

KS = [1.0, 0.75, 0.5, 0.25]

_FIXED = {'outputSize': 'Fixed', 'fixedOutputSize': RES}

# Shipping config, PINNED (mirrors VNA_CacheAmortGate.py as of 2026-07-21).
# A gate must test a FIXED config - do NOT wire this to the live WdasSkyVNA.py.
SHIP = dict(_FIXED, **{
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
    'useRadCache': True,
    'radCacheRes': 64,
    'radCutBounce': 3,
    'radTrainEvery': 8,
    'radEma': 0.10,
    'radResidualSurvival': 0.25,
    'trRRThreshold': 0.05,
    'trRRMode': 31,
    'radWarpRRLanes': 0,
    'radResidualTailQ': 0.0,
    'radAmortPeriod': 0,     # OFF - it biases the image, and this gate is about bias.
    'distWeightK': 1.0,      # overridden per config below
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
})

ACCUM_PROPS = dict(_FIXED, enabled=True, precisionMode='Single')
ACCUM_PROPS_OFF = dict(_FIXED, enabled=False, precisionMode='Single')


def render_graph_DistWeightGate():
    g = RenderGraph("VNADistWeightGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_DistWeightGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for k in KS:
    m.scene.selectViewpoint(1)
    graph.updatePass("VolumePathTracer", dict(SHIP, distWeightK=k))
    # Warm the rad/tau caches BEFORE accumulation starts, so cache cold-start
    # never lands inside a measured window and every K converges from the same
    # kind of state.
    for i in range(WARM):
        m.renderFrame()
    # Reset accumulation only (recreating Accum leaves the tracer caches warm).
    # off -> on with DIFFERENT dicts so neither updatePass can no-op.
    graph.updatePass("Accum", ACCUM_PROPS_OFF)
    m.renderFrame()
    graph.updatePass("Accum", ACCUM_PROPS)
    m.renderFrame()
    tag = "w{:03d}".format(int(round(k * 100)))
    frame = 1
    for spp in (NOISE_SPP, CONV_SPP):
        while frame < spp:
            m.renderFrame()
            frame += 1
        m.frameCapture.baseFilename = "{}_vp1_s{}".format(tag, spp)
        m.frameCapture.capture()

# Fire-and-forget, same as the other gate drivers: launch, capture, exit.
exit()
