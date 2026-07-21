# Cache-amortization COST gate - how much of shade time is the residual?
#
#   Mogwai.exe --script scripts/VNA_CacheAmortGate.py
#
# THE PROPOSAL THIS GATES
# You already hold both Nubis structures in unbiased form (radcache ~ their MS
# field, tauCache ~ their sun grid). The only difference is that you keep
# tracking S-C after consuming the cache. The idea is to move the residual into
# TIME: most frames consume the cache only (Nubis speed), a rotating fraction of
# FRAMES pays the full residual and splats the correction back into the cache
# (radSplat, machinery already exists). The cache is then corrected by real
# measured path integrals, so a static scene converges to ground truth rather
# than to a fitted curve - consistent, not unbiased, and labelled as such.
#
# This gate measures the CEILING of that idea before any of it is built.
#
# WHY FRAMES AND NOT PIXELS - already decided by measurement
# [PATHLEN] splits 274k bounces as raw-uncut 120k (44%, mean 2.6, 0% stragglers),
# residual 96k (35%, mean 9.1, 12% stragglers), training 58k (21%). Residual +
# training is 56% of all bounces. But residual tail-RR already proved removing
# SOME of them converts terribly: it cut 13.5% of all bounces and bought 3% of
# shade, because a warp runs at max(bounces), not sum - killing long paths while
# their neighbours keep running shortens nothing. Frame-gating removes them
# UNIFORMLY: every path on a consume frame is raw-uncut, so warp max drops from
# 16+ to ~5. Pixel-gating would reproduce the tail-RR failure exactly.
#
# CONFIGS
#   ship     survival 0.25, train 8      today's unbiased estimator
#   consume  survival 0.01, train 8      residual gone, training still mixed in
#                                        (i.e. what a PIXEL-gated scheme buys)
#   floor    survival 0.01, train 64     residual gone + training cut ~8x - the
#                                        frame-gated consume-only cost FLOOR
#                                        (see FLOOR_TRAIN for why not 1e6)
#   nocache  useRadCache False           reference: no cut at all, all raw paths
#
# radResidualSurvival is clamped to [0.01, 1.0] host-side, so 0.01 is as close to
# "die at the cut" as the knob reaches. 99% of consuming paths terminate there;
# the surviving 1% carry throughput /= 0.01 = 100x compensation and will make the
# image firefly-ridden. That is EXPECTED and irrelevant - this is a timing run,
# not an image run. It also makes 'floor' a slight OVER-estimate of cost: the
# real scheme would kill 100% on a consume frame, not 99%.
#
# READ THE RESULT
#   ship -> floor on shade ms is the ceiling of the whole idea.
#     - shade drops toward raw-uncut proportions -> worth building, and that
#       delta is the prize.
#     - shade barely moves -> 56% of bounces were never the cost either, and the
#       cache-amortization family dies with every other lever.
#   Confirm the MECHANISM, not just the number: [PATHLEN] on 'floor' must show
#   essentially everything in raw-uncut at mean ~2.6 with 0% stragglers. If long
#   paths survive there, the knob did not do what this gate assumes and the
#   timing means nothing.
#
# LOG SEGMENTATION - script print() does NOT reach the Mogwai log file (verified:
# no script output appears in any log, including the older gate drivers). Segment
# instead on the [WORK] frame counter, which resets to 0 on every updatePass:
# each config is one block of 9 logged frames, warm-up is frames 0..128, measured
# is 192..512. Blocks appear in CONFIGS order, repeated ROUNDS times.
#
# INTERLEAVED. Configs run in order, twice, so clock/thermal drift shows as a
# split between rounds of the SAME config rather than as a config effect. If a
# config's two rounds disagree by more than the gap between configs, the run is
# noise - discard it, do not average it. (On the mip gate this is exactly what
# disqualified full-frame gpuMs and left shade ms usable.)

import os
from falcor import *

RES = (960, 540)   # Ship resolution (WdasSkyVNA.py _RENDER_RES), pinned.
WARM = 192         # updatePass recreates the pass -> rad/tau caches reset.
MEASURE = 384      # At risStatsInterval 64 that is 6 measured [WORK] lines each.
ROUNDS = 2

_FIXED = {'outputSize': 'Fixed', 'fixedOutputSize': RES}

# Training rate for the 'floor' config. NOT a huge number, and that is a trap
# worth spelling out: radTrainEvery has no upper clamp, so 1e6 does turn training
# off - but it turns it off during the WARM frames too, and updatePass resets the
# radcache, so the cache would never populate. The consume branch is gated on
# C.a > 0.5; with an empty cache that gate fails, the cut never fires, and every
# path runs FULL length. 'floor' would then have silently measured 'nocache'.
# 64 keeps the cache trained (1 pixel in 64, over 576 frames every cell gets many
# samples) while cutting training paths ~8x - from ~58k of the 274k bounces to
# ~7k. So 'floor' still carries ~2.6% of bounces as training; subtract that when
# reading it as the consume-only floor.
FLOOR_TRAIN = 64

# Shipping config, PINNED (mirrors VNA_WarpRRGate.py's SHIP as of 2026-07-21).
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
    'radWarpRRLanes': 0,        # measured out (HANDOFF_10 6) - off.
    'radResidualTailQ': 0.0,    # measured out (HANDOFF_10 6) - off.
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
})

CONFIGS = [
    ("ship",    dict(SHIP)),
    ("consume", dict(SHIP, radResidualSurvival=0.01)),
    ("floor",   dict(SHIP, radResidualSurvival=0.01, radTrainEvery=FLOOR_TRAIN)),
    ("nocache", dict(SHIP, useRadCache=False)),
    # The REAL lever, not a knob-combination proxy for it. risStatsInterval 63
    # instead of 64 because 64 is a multiple of the period: every logged frame
    # would be frame % 8 == 0, a CORRECTION frame, and the average would report
    # the expensive leg as if it were the amortized cost. At 63 the logged
    # frames land on consume phases (63,126,189,252,315,378 mod 8 = 7,6,5,4,3,2
    # - never 0), so this block measures the CONSUME leg. Correction frames cost
    # the same as 'ship' by construction, so:
    #     amortized(N) = ((N-1) * amort8 + ship) / N
    # Check the [PATHLEN] tag on this block says [consume] before believing it.
    ("amort8",  dict(SHIP, radAmortPeriod=8, risStatsInterval=63)),
]

ACCUM_PROPS = dict(_FIXED, enabled=True, precisionMode='Single')


def render_graph_CacheAmortGate():
    g = RenderGraph("VNACacheAmortGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_CacheAmortGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

for rnd in range(ROUNDS):
    for name, props in CONFIGS:
        m.scene.selectViewpoint(1)
        # Fresh pass = fresh rad/tau caches; the warm-up retrains the radcache
        # before measuring so cold-start never lands inside a measured window.
        # 192 frames at trainEvery 64 still deposits into every cell many times
        # over, so 'floor' consumes a genuinely populated cache - which is the
        # whole point, and what a 1e6 trainEvery would have destroyed.
        graph.updatePass("VolumePathTracer", props)
        for i in range(WARM):
            m.renderFrame()
        for i in range(MEASURE):
            m.renderFrame()

# Fire-and-forget, same as the other gate drivers: launch, measure, exit.
exit()
