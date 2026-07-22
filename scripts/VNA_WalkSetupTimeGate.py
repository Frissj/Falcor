# Walk-setup hoist TIMING gate - does the confirmed mechanism buy a frame?
#
#   Mogwai.exe --script scripts/VNA_WalkSetupTimeGate.py
#   py scripts/VNA_WalkSetupTimeGate_Compare.py
#
# WHAT IS ALREADY SETTLED, SO THIS GATE DOES NOT RE-ASK IT
# The per-instance beginWalk hoist (VolumeInstanceSampler.slang, four RayQuery
# loops) is CONFIRMED to fire: log 200 measured brickCache hit 55.8 -> 65.0,
# 61.2 -> 69.2, 56.8 -> 65.8 pp across three viewpoints, with the lookup
# denominator and the cell count IDENTICAL TO THE DIGIT between arms
# (vp1 65,135,787 lookups / 86,652,354 cells in both). Same work, ~2M more
# cache hits per frame. Its image effect is below the harness noise floor:
# offA-vs-offB differed MORE than offA-vs-on on every viewpoint.
#
# So the only open question is time, and that is what this measures.
#
# WHY THIS IS INTERLEAVED, AND WHY THE WARM-UP IS SO LONG
# Two independent confounds, both measured on this machine TODAY:
#
#   1. RETRAIN RAMP. updatePass recreates the pass, so the rad/tau caches reset
#      and shadeMain climbs for ~600 frames afterwards (HANDOFF 12 2.3;
#      reproduced in logs 194/196 - gpuMs 20.7 -> 30.2 at frozen counters).
#      WARM must clear it or the measurement is taken on the ramp.
#
#   2. CLOCK DRIFT that never settles. At byte-identical work and resolution,
#      log 194 drifted +22% over 3200 frames, and recovered ~25% instantly
#      after an idle pause. Averaging does not remove a trend. So configs are
#      INTERLEAVED: with ROUNDS rounds of (off, on), drift shows up as a split
#      between rounds of the SAME config, and if that split is larger than the
#      gap between configs the run is noise and must be discarded, not averaged.
#      This is the discipline VNA_CacheAmortGate.py already uses.
#
# GROUND TRUTH IS [FRAME] wall / FPS. Not gpuMs. The GPU-side split is useful
# for attribution (main vs shadeMain) but the pass timer and the wall clock
# disagreed by >1 ms in log 194's toggle, so the wall is what decides.
#
# CACHE CONSUMPTION IS OFF (radCutBounce 0) - see the note on it below. The
# first version of this gate turned it on and lightened the image by 7.1 pp of
# dark-bin fraction, which is not something a gate may do silently.
#
# LOG SEGMENTATION - script print() does NOT reach the Mogwai log file. Segment
# on the [WORK] frame counter, which resets to 0 on every updatePass: each
# config is one block, blocks appear in CONFIGS order, repeated ROUNDS times.
# The compare script does exactly that.

import os
from falcor import *

# Frames. WARM clears the ~600-frame retrain ramp with margin; MEASURE is the
# window the compare script averages. 1280 frames per block at ~32 ms is ~41 s,
# so the whole gate is ~2 x 3 x 41 s = ~4 min.
WARM = 768
MEASURE = 512
ROUNDS = 3
VIEWPOINT = 1   # Canonical measurement camera (WdasSky.py viewpoint 1).

CONFIGS = [
    ('off', {'useSharedWalkSetup': False}),   # original per-walk preamble
    ('on',  {'useSharedWalkSetup': True}),    # per-instance hoist
]

# Shipping config, PINNED - mirrors VNA_WaveMipGate.py's SHIP as of 2026-07-22.
# Do NOT wire this to the live driver script; a gate must test a FIXED config.
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
    'tauCacheApply': True,
    'tauCacheRes': 64,
    'tauCacheInterval': 0,
    # CACHE COMPILED, CONSUMPTION OFF. radCutBounce 0 is the documented live
    # OFF switch: the pass shape and the training cost stay as they are on the
    # ship path, but no path is cut short against the cache, so the control
    # variate contributes nothing to the image.
    #
    # WHY, and this was a real mistake in the first version of this gate: the
    # amortization LIGHTENS the image, badly. Measured across this session's own
    # two gates, [LUMHIST] bin0 (darkest bin) went 33.0% with the cache off
    # (log 200) to 25.9% with it on (log 201) - 7.1 pp, larger than the 4 pp
    # "dark lift" HANDOFF 12 6.4 has open as an unexplained bias. Turning it on
    # inside a gate silently changes the picture being looked at, and it was
    # switched on here under cover of "measure what ships".
    #
    # If you want the cache-consuming variant timed instead, set radCutBounce
    # back to 3 - but then it is a DIFFERENT estimator being timed, and the two
    # numbers are not comparable.
    'useRadCache': True,
    'radCacheRes': 64,
    'radCutBounce': 0,
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
    'useWaveUniformMip': False,
    'waveMipLift': 0,
    # [FRAME] and [COST] are the whole point of the run.
    'logWorkStats': True,
    'logTrRRProbe': False,   # keep the probe OUT of main - it inflates registers.
    'logRisStats': True,
    'risStatsInterval': 64,
    'useSharedWalkSetup': True,  # overridden per config below
}

ACCUM_PROPS_OFF = {'enabled': False, 'precisionMode': 'Single'}


def render_graph_WalkSetupTimeGate():
    g = RenderGraph("VNAWalkSetupTimeGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    # Accumulation OFF: this is a timing run, and an accumulator that keeps
    # working after convergence is a constant tax on every arm equally but
    # still noise on the wall clock. No image is captured here.
    Accum = createPass("AccumulatePass", ACCUM_PROPS_OFF)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_WalkSetupTimeGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

print("VNA-WALKSETUPTIME: FULL window res, {} configs x {} rounds, {} warm + {} "
      "measured each (~4 min). Do NOT resize the window mid-run and do not "
      "alt-tab - an idle pause resets the GPU clocks and shows up as a config "
      "effect.".format(len(CONFIGS), ROUNDS, WARM, MEASURE))

for rnd in range(ROUNDS):
    for name, cfg in CONFIGS:
        m.scene.selectViewpoint(VIEWPOINT)
        # Fresh pass = fresh rad/tau caches; WARM retrains before the measured
        # window so cache cold-start never lands inside it.
        graph.updatePass("VolumePathTracer", dict(SHIP, **cfg))
        for i in range(WARM):
            m.renderFrame()
        for i in range(MEASURE):
            m.renderFrame()

print("VNA-WALKSETUPTIME: complete. Run: py scripts/VNA_WalkSetupTimeGate_Compare.py")

# Fire-and-forget, same as the other gate drivers: launch, measure, exit.
exit()
