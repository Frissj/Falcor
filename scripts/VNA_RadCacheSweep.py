# Gate sweep for the two 2026-07-20 estimator changes:
#   1. Tracker RR (trRRThreshold): weight RR on the RQ transmittance walks.
#      Textbook-unbiased; converged mean must not move.
#   2. Radiance-cache control variate (useRadCache): the deep-bounce tail's
#      mean lives in a trained world-grid cache; paths past radCutBounce only
#      estimate the residual (L - C). Unbiased for ANY cache content; the
#      residual survival p prices variance, so the sweep walks p down until
#      the gate says stop - every PASS below 0.25 is free frame time.
#
# Config isolation:
#   rc_off_rr0 : both new features off  -> vs 01_ref, detects drift from
#                everything ELSE that changed today (tau cache, q=0.125).
#   rc_off     : tracker RR on, cache off -> isolates the RR change.
#   rc_p100    : cache on, survival 1.0 -> CV bookkeeping only. The strongest
#                correctness probe: p=1 must be EXACTLY variance-neutral-or-
#                better with zero mean shift; a fail here is a CV algebra bug,
#                not a tuning problem.
#   rc_p50/25/10 : the actual cost/variance frontier.
#
# CACHE WARM-UP: unlike roulette, this estimator has state. Each config warms
# WARM frames first (training paths populate the cache, confidence crosses the
# consumption gate), then accumulation is RESET by recreating the Accum pass -
# which does not touch the tracer's caches - so both captures see the
# steady-state estimator:
#   1spp      = first frame after the reset, warm cache.
#   converged = FRAMES more frames on top.
# Mixing warm-up frames into the converged capture would ALSO be unbiased
# (plain and CV estimators share the mean), but keeping the capture pure
# steady-state makes the firefly census honest.
#
# Score with VNA_RouletteScore.py (patched to glob rc_* too).
#
#   Mogwai.exe --script scripts/VNA_RadCacheSweep.py

import os

from falcor import *

WARM = 192
FRAMES = 256
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

# Shipping config, verbatim from WdasSkyVNA.py as of 2026-07-20 (tau cache,
# q=0.125, wavefront off). Only the two new knobs vary below.
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

# SWEEP 2: the de-confounding matrix. Sweep 1 FAILed every trRR/cache config
# by -0.45..-0.86% with reproducible decile structure - but every config
# shared the temporal-reuse chain, the fused-walk RR perturbs the chain's own
# RNG stream, and identical seeds mean both runs replayed the same chain
# realization. A slow-mixing chain (M-cap 20) sitting +-0.5% off stationary
# in a 256-frame window is indistinguishable from real bias in that design.
# (rq2's clean +-0.02% comparisons never touched the chain's stream - roulette
# lives post-reservoir - which is why this never bit before.)
#
# DESIGN NOTE: temporal-off CANNOT test the suspect code directly -
# USE_FUSED_SWEEP requires USE_TEMPORAL_REUSE, so with temporal off the fused
# walk (where the bit1 RR and bit2 shrink live) is not even compiled. So:
#   - Temporal-ON rows keep the real shipping path and use WINDOW LENGTH as
#     the discriminator: each captures converged at 256 AND 1024 frames.
#     Real bias is window-invariant; a chain-realization offset shrinks as
#     the window grows past the mixing time.
#   - Two temporal-OFF rows are the textbook check for the RR math itself
#     (bit0 covers BOTH the NEE walk and the separate escape walk there).
CONFIGS = [
    # name, props, capture windows
    ("t_base", dict(SHIP, useRadCache=False, trRRThreshold=0.0,  trRRMode=0, radResidualSurvival=1.0), (256, 1024)),
    ("t_rr7",  dict(SHIP, useRadCache=False, trRRThreshold=0.05, trRRMode=7, radResidualSurvival=1.0), (256, 1024)),
    ("t_p100", dict(SHIP, useRadCache=True,  trRRThreshold=0.05, trRRMode=7, radResidualSurvival=1.0), (256, 1024)),
    ("t_p25",  dict(SHIP, useRadCache=True,  trRRThreshold=0.05, trRRMode=7, radResidualSurvival=0.25), (256, 1024)),
    ("ntb",    dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0,  trRRMode=0, radResidualSurvival=1.0), (256,)),
    # Site split (sweep 3): ntr (bit0+... old combined) FAILed -0.45% while a
    # 20M-sample simulation of the identical scheme reads -0.08% - the algebra
    # is fine, a CALL SITE's coupling is not. bit0 = NEE walk only (pure
    # linear factor); bit3 = escape walk only (feeds adaptive M + shares the
    # RIS stream). Whichever row fails names the coupling.
    ("ntrN",   dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=1, radResidualSurvival=1.0), (256,)),
    ("ntrE",   dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=8, radResidualSurvival=1.0), (256,)),
    # Sweep 4: name the escape coupling. ntrE FAILed (-0.73%) while ntrN
    # PASSed (-0.005%) - same RR code, so the difference is what the escape
    # site feeds. Two suspects, one row each:
    #   ntrE_fm: adaptive M OFF - if this passes, the M coupling is the bug.
    #   ntrE_nf: footprint mips OFF - if this passes, the coarse-mip residual
    #            factors (only the escape walk sees them) are the bug.
    ("ntrE_fm", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=8, useAdaptiveM=False, radResidualSurvival=1.0), (256,)),
    ("ntrE_nf", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=8, footprintMip=False, radResidualSurvival=1.0), (256,)),
    # Their controls: the same ablations WITHOUT RR, so each comparison is
    # ablation-vs-ablation, not ablation-vs-SHIP.
    ("ntb_fm",  dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0, trRRMode=0, useAdaptiveM=False, radResidualSurvival=1.0), (256,)),
    ("ntb_nf",  dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0, trRRMode=0, footprintMip=False, radResidualSurvival=1.0), (256,)),
    # Sweep 5: the realization test. [TRRPROBE] measured the escape walk
    # EXACTLY mean-preserving under RR (paired per pixel, +-0.02%), adaptive M
    # is excluded (ntrE_fm identical), and the radiance add is linear - every
    # mechanism is acquitted, yet ntrE reads -0.73%. Remaining hypothesis: the
    # RR coins decorrelate the downstream RNG stream in every cloudy pixel, so
    # ntrE-vs-ntb compares two different random REALIZATIONS of the scatter
    # estimator, whose 256-frame mean wobbles at this scale. Same comparison,
    # three RNG epochs: pinned at -0.73% = systematic (hunt continues with a
    # contradiction); scattered = realization variance (the gate needs paired
    # streams or many seeds for stream-decorrelating changes).
    ("ntb_s1",  dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0,  trRRMode=0, seedOffset=7777,  radResidualSurvival=1.0), (256,)),
    ("ntrE_s1", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=8, seedOffset=7777,  radResidualSurvival=1.0), (256,)),
    ("ntb_s2",  dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.0,  trRRMode=0, seedOffset=31337, radResidualSurvival=1.0), (256,)),
    ("ntrE_s2", dict(SHIP, useTemporalReuse=False, useRadCache=False, trRRThreshold=0.05, trRRMode=8, seedOffset=31337, radResidualSurvival=1.0), (256,)),
]

ACCUM_PROPS = {'enabled': True, 'precisionMode': 'Single'}
ACCUM_PROPS_OFF = {'enabled': False, 'precisionMode': 'Single'}
# HARNESS LESSON (2026-07-20, first run of this sweep): updatePass with
# IDENTICAL props silently no-opped for 4 of 6 configs, so their "1spp"
# captures were ~193-frame accumulations (caught by noise arithmetic:
# selfMSE 0.48 = 132 * (1/193 - 1/449), not by any error message). The reset
# below alternates two DIFFERENT prop dicts so recreation is unconditional.


def render_graph_RadCache():
    g = RenderGraph("VNARadCache")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_RadCache()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

# Pin the FBO to the reference resolution. The first run inherited the live
# window's 2560x1351, which made every capture incomparable with 01_ref
# (the section-5 resolution trap, again). 1080p also renders 2.6x faster.
m.resizeFrameBuffer(1920, 1080)

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props, windows in CONFIGS:
    m.scene.selectViewpoint(1)
    # Fresh pass = fresh caches (rad + tau). Warm-up trains the rad cache.
    graph.updatePass("VolumePathTracer", props)
    for i in range(WARM):
        m.renderFrame()
    # Reset ACCUMULATION ONLY: recreating Accum leaves the tracer's caches
    # warm. Off->on with DIFFERENT dicts so neither updatePass can no-op.
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
    with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
        f.write("RADCACHE2 {} done (warm {} + windows {})\n".format(name, WARM, windows))

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("RADCACHE SWEEP COMPLETE\n")
print("VNA-RADCACHE: complete.")
# Unattended batch job: close Mogwai so the run is hands-off end to end.
# (progress.txt line above is the completion receipt.)
exit()
