# Residual-mip COST gate - prices a majorant-DDA CELL against a density TAP.
#
#   Mogwai.exe --script scripts/VNA_ResidualMipGate.py
#
# WHY THIS EXISTS
# [SUBHOMOG] measured majRatio sub/brick = 0.607: descending the DDA from 8^3 to
# 4^3 cells would cut integral(sigma_bar ds), and therefore collisions, by ~39%.
# But it also roughly DOUBLES the cell count (half-width cells, ~2x crossed per
# ray), and occSkip already answers ~54% of taps from the register-resident mask
# without touching the atlas. So the trade is roughly
#     +12M range fetches  <->  -1M atlas reads + ~2M collisions' worth of ALU
# and whether that is positive depends entirely on the price of a cell against a
# tap - a number nothing in the log measures.
#
# residualMip is that price, already exposed as a knob, running the SAME trade in
# reverse: a coarser mip means fewer, wider cells (mip m cells are 8<<m voxels)
# and a looser majorant, hence more collisions. Sweep it and the hardware tells
# you the exchange rate.
#
# WHY THREE MIPS AND NOT TWO
# One A/B gives one equation in two unknowns. With cost per cell c and per tap p,
#     delta_ms = c * delta_cells + p * delta_taps
# Two configs leave c and p underdetermined; three give two independent equations
# and a solvable pair - plus mip 2 checks that the relation is actually linear. If
# the two solutions disagree wildly, cost is not a linear function of these two
# counters and the whole 4^3 prediction is built on sand - which is itself the
# result, and better learned here than after writing a traversal.
#
# WHAT IT PRICES, HONESTLY
# residualMip drives the TRANSMITTANCE walk (evalTransmittanceRQ ->
# ResidualRatioTrackingLocalDDA), i.e. the [BOUNCECOST] nee bucket - 32% of shade
# work. The distance sampler picks its mip adaptively and is untouched. So this
# measures the exchange rate on nee and ASSUMES ds obeys the same economics: same
# range texture, same atlas, same two-hop chase. Reasonable, and stated rather
# than hidden.
#
# READ THE RESULT
#   grep "RESIDMIPGATE\|\[WORK\]\|\[COST\]" on the log, then per config take
#   gpuMs, total cells and total taps (the [COST] "totals:" fields).
#     - coarser mip FASTER  -> cells cost more than the taps they buy. Finer
#       granularity is wrong and the 4^3 fix is dead despite majRatio 0.607.
#     - coarser mip SLOWER  -> taps dominate, finer is right, and 0.607 is the
#       size of the prize.
#
# INTERLEAVED, TWICE. Configs run 0,1,2,0,1,2 rather than 0,0,1,1,2,2 so clock or
# thermal drift shows up as a split between the two rounds of the SAME config
# instead of masquerading as a mip effect. If round 1 and round 2 of a config
# disagree by more than the gap between configs, the run is noise - discard it,
# do not average it.

import os
from falcor import *

RES = (960, 540)   # Ship resolution (WdasSkyVNA.py _RENDER_RES). Pinned here so
                   # the measurement never depends on window size.
WARM = 192         # updatePass recreates the pass -> rad/tau caches reset.
MEASURE = 384      # At risStatsInterval 64 that is 6 [WORK]/[COST] lines each.
ROUNDS = 2
MIPS = [0, 1, 2]

_FIXED = {'outputSize': 'Fixed', 'fixedOutputSize': RES}

# Shipping config, PINNED (mirrors VNA_WarpRRGate.py's SHIP as of 2026-07-21).
# A gate must test a FIXED config, so this is a deliberate copy - do NOT wire it
# to the live WdasSkyVNA.py, or the gate drifts with the daily driver.
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
    'trRRThreshold': 0.05,
    'trRRMode': 31,
    'radResidualSurvival': 0.25,
    'radWarpRRLanes': 0,        # measured out (HANDOFF_10 6) - off.
    'radResidualTailQ': 0.0,    # measured out (HANDOFF_10 6) - off.
    # The counters this gate exists to read.
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
})

ACCUM_PROPS = dict(_FIXED, enabled=True, precisionMode='Single')


def render_graph_ResidualMipGate():
    g = RenderGraph("VNAResidualMipGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_ResidualMipGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

print("VNA-RESIDMIPGATE: res {}x{}, mips {}, {} rounds, {} warm + {} measured each"
      .format(RES[0], RES[1], MIPS, ROUNDS, WARM, MEASURE))

for rnd in range(ROUNDS):
    for mip in MIPS:
        m.scene.selectViewpoint(1)
        # Fresh pass = fresh rad/tau caches; warm-up retrains before measuring so
        # cache cold-start never lands inside a measured window.
        graph.updatePass("VolumePathTracer", dict(SHIP, residualMip=mip))
        for i in range(WARM):
            m.renderFrame()
        # Marker BEFORE the measured frames so the [WORK]/[COST] lines that follow
        # are unambiguously attributable when grepping the log.
        print("VNA-RESIDMIPGATE: BEGIN round {} residualMip {} (cell width {} voxels)"
              .format(rnd, mip, 8 << mip))
        for i in range(MEASURE):
            m.renderFrame()
        print("VNA-RESIDMIPGATE: END   round {} residualMip {}".format(rnd, mip))

print("VNA-RESIDMIPGATE: complete. Per config read gpuMs + [COST] 'totals: cells N taps N'.")
print("VNA-RESIDMIPGATE: solve delta_ms = c*delta_cells + p*delta_taps across mip pairs;")
print("VNA-RESIDMIPGATE: c/p is the exchange rate the 4^3 decision turns on.")

# Fire-and-forget, same as the other gate drivers: launch, measure, exit.
exit()
