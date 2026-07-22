# ds memory-boundedness gate - is sampleDistance bound on atlasTex reads?
#
#   Mogwai.exe --script scripts/VNA_DsGate.py
#
# WHY: [FRAME] (log 189) settled the attribution. ds is 14.21 ms = 81% of
# shadeMain = 47% of the whole frame - the single largest item anywhere, bigger
# than all of phase A. NEE turned out to be only 19% of shadeMain, so the target
# is unambiguous. What is NOT known is WHY ds is slow, and the candidates need
# opposite fixes:
#
#   (a) MEMORY - real atlasTex reads that the occupancy mask and brick cache
#       cannot eliminate. ds runs at occSkip 34.4% and brickHit 59.5%, both far
#       worse than phase A (63.1% / 65.2%), because scatter rays start INSIDE
#       the cloud and march toward the sun through dense medium where nothing is
#       skippable. Fix would be about tap locality / cache shape.
#
#   (b) NOT MEMORY - ALU, RayQuery traversal, or latency the SM cannot hide at
#       15% occupancy. Fix would be about instruction count or register
#       pressure, and every locality idea would be wasted effort.
#
# THE EXPERIMENT: occSkip is the one knob that separates them cleanly. It
# changes ONLY whether a tap is answered from the register-resident occupancy
# bitmask or from a real atlasTex read. Per GridVolumeSampler.slang, a clear bit
# means every voxel in the tile decodes to EXACTLY 0, so the skipped value is
# bit-identical - same cells, same collisions, same image, same work counters.
# The only thing that moves is memory traffic.
#
#   occSkip ON  -> 34.4% of ds taps never touch the atlas
#   occSkip OFF -> every ds tap is a real read
#
# READ IT AS: if turning occSkip OFF costs a lot of ms, ds IS memory-bound and
# (a) is the direction. If it barely moves, those reads were already hidden and
# ds is bound on something else - which kills the entire locality line of attack
# before anyone builds anything, which is the point.
#
# This is a RUNTIME uniform, not a define, so both configs run the same binary
# and the same shader - no recompile, no cache cold-start difference beyond the
# warm-up, and no chance of comparing two different builds.
#
# Interleaved two rounds. If a config's rounds disagree by more than the gap
# between configs, the run measured the machine and not the knob.
#
# Full window res - a memory-traffic question must be asked at the resolution
# the traffic actually happens at (WdasSkyVNA.py _RENDER_RES note).

import os
from falcor import *

WARM = 192       # updatePass recreates the pass -> rad/tau caches reset.
MEASURE = 384    # 6 [FRAME]/[WORK] lines per block at risStatsInterval 64.
ROUNDS = 2

CONFIGS = [
    ('occskip_on',  {'useOccupancySkip': True}),
    ('occskip_off', {'useOccupancySkip': False}),
]

# Shipping config, PINNED. NEE stays ON here - log 189 already priced it at
# 3.42 ms and turning it off changes what ds is asked to do.
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
    'risTargetFloor': 0.01,
    'useCompaction': True,
    'useWavefront': False,
    'useScatterSort': False,
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
    'radAmortPeriod': 0,
    'distWeightK': 1.0,
    'seedOffset': 0,
    'useOccupancySkip': True,   # overridden per config
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
}

ACCUM_PROPS = {'enabled': True, 'precisionMode': 'Single'}


def render_graph_DsGate():
    g = RenderGraph("VNADsGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_DsGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

for rnd in range(ROUNDS):
    for name, cfg in CONFIGS:
        m.scene.selectViewpoint(1)
        graph.updatePass("VolumePathTracer", dict(SHIP, **cfg))
        for i in range(WARM):
            m.renderFrame()
        for i in range(MEASURE):
            m.renderFrame()

exit()
