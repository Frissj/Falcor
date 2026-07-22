# Wave-uniform majorant mip gate - does lifting a lane's DDA mip toward its
# warp's coarsest actually raise lane occupancy, and does it keep the image?
#
#   Mogwai.exe --script scripts/VNA_WaveMipGate.py
#   py scripts/VNA_WaveMipGate_Compare.py        <- the verdict step
#
# WHAT IS BEING TESTED
# stepToNextCollision's while (t < tFar) DDA costs its warp max(trips), not
# sum(trips): 32 lanes stay locked to the slowest one. The mip sets the step
# size and is currently chosen purely per-lane, so one lane grinding at mip 0
# while its 31 neighbours coast at mip 3 makes the whole warp pay the mip-0 trip
# count. gvsWaveLiftMip raises each lane toward the coarsest level any active
# lane in its warp wants, so step size - and therefore trip count - goes
# wave-uniform by construction.
#
# WHY IT IS WORTH A GATE. analysis7.yaml read PER RANGE ranks "Active Threads
# Per Warp" first at 24.27% (4.12x range ceiling, 73% frame gain) against
# 1.36x/25.6% for the register limiter, while the SM issues at 52.6% of peak
# with long-scoreboard stalls at 1.5%. The frame is not latency-starved; it
# issues plenty and masks off 75.7% of its lanes. Both scheduling answers are
# already measured-off (useWavefront, useScatterSort) because they reorder lanes
# and the residue is INTRA-bounce march-length variance. This is the first lever
# aimed at that residue.
#
# THIS GATE ANSWERS THREE SEPARATE QUESTIONS. Do not conflate them.
#
#   1. DID THE MECHANISM FIRE? [STEPOCC] lift fraction. A warp whose lanes
#      already agree on the mip has nothing to lift, so lift ~ 0 means the lever
#      never ran and questions 2 and 3 say NOTHING about whether the idea works.
#      Read this first. Reporting a flat occupancy without checking it is the
#      [SUBHOMOG] mistake in a new place.
#
#   2. DID OCCUPANCY MOVE? [STEPOCC] shade occ against the 'off' config. This is
#      the quantity the lever exists to change, and it is the one to trust over
#      gpuMs, because the lever DELIBERATELY trades ALU loop trips for tex taps.
#      A gpuMs wash with occupancy up and a gpuMs wash with occupancy flat are
#      completely different results.
#
#   3. IS THE IMAGE STILL RIGHT? Unbiased at every lift (a coarser majorant is
#      still a bound) but NOT byte-identical - cell count drives how many
#      sampleNext1D draws the walk consumes, so the sample stream diverges. That
#      makes this an image gate, like distWeightK, not a bit-identity check.
#
# THE SHARPER CLAIM WORTH TESTING, stated so a failure is diagnostic rather than
# just red: analog delta tracking returns the medium's TRUE free-flight
# distribution for ANY bounding majorant. Not merely unbiased - distributionally
# identical. So the distance sampler's converged image AND its noise at equal
# spp should both be unchanged by the lift; only cost should move. The one
# legitimate exception is the delta-tracking TRANSMITTANCE walk, which also goes
# through stepToNextCollision and whose variance is not majorant-invariant, so a
# small noise change is allowed there. A LARGE noise change at s256 with a clean
# s4096 means the transmittance walk, not a bias.
#
# WHY THE 'lift0' CONFIG EXISTS. useWaveUniformMip=True with waveMipLift=0 is
# estimator-identical to 'off' but still pays the wave reduction instruction in
# the hottest loop in the frame. Isolating it prices the INSTRUCTION separately
# from the LEVER - the same asymmetry the occupancy-skip switch documents. If
# 'off' -> 'lift0' already costs real time, the lever has to win that back
# before it wins anything.
#
# WHY A MEASURED NOISE FLOOR AND NOT A FIXED THRESHOLD. The distWeight gate
# ended with "converged captures differ by ~10x the expected noise floor" and no
# way to tell a bug from a heavy tail, because the floor was expected rather
# than measured. Here the reference config is rendered TWICE with different
# seeds; that pair differs by noise ALONE, so it IS the floor, at this spp, on
# this scene, in this run. Every lift is judged against it.
#
# INTERLEAVED, TWICE. The perf phase runs configs 0,1,2,3,4,0,1,2,3,4 rather
# than 0,0,1,1,... so clock or thermal drift shows as a split between the two
# rounds of the SAME config instead of masquerading as a lift effect. If a
# config's two rounds disagree by more than the gap between configs, the run is
# noise - discard it, do not average it. The compare step checks this for you.
#
# LOG SEGMENTATION: script print() does NOT reach the Mogwai log. The counters
# are segmented positionally on the [WORK] frame counter, which resets to 0 on
# every VolumePathTracer updatePass - one block per config, in the order this
# file runs them. Do not reorder CONFIGS without updating the compare script.

import os
from falcor import *

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\wavemip"

# FULL WINDOW RES - no outputSize override anywhere in this file. This is not a
# preference, it is the rule the repo already states: WdasSkyVNA.py's _RENDER_RES
# note says divergence/estimator gating MUST run at full res because "960x540
# changes the warp divergence", and VNA_WarpRRGate.py - the other divergence gate
# - accordingly sets no resolution at all. A gate for a lane-occupancy lever run
# at 960x540 would be measuring a different phenomenon from the one the lever
# targets, and analysis7's 4.12x came off a full-res capture.
#
# The cost of not pinning is that the measurement now depends on window size, so
# the compare step CHECKS that every config reported the same res and fails the
# run if not. That matters: log 183 contains both 1920x1080 and 2560x1351 inside
# one session, i.e. the window really does get resized mid-run on this machine.
#
# Budget accordingly - full res is ~13.5 ms/frame against ~4 ms at 960x540, so
# the perf phase is a couple of minutes and the image phase runs in the tens.
# Set PERF_ONLY while iterating.

WARM = 192         # updatePass recreates the pass -> rad/tau caches reset.
MEASURE = 384      # At risStatsInterval 64 that is 6 [WORK]/[COST]/[STEPOCC] lines each.
ROUNDS = 2

# SPP COUNTS ARE DERIVED FROM THE RESOLUTION, NOT COPIED FROM THE 960x540 GATES.
# Getting this wrong cost a 20-minute run: the first version of this file
# inherited WARM/MEASURE/CONV_SPP verbatim from VNA_DistWeightGate.py, which is
# a 960x540 gate, and then ran them at 2560x1351 where the measured mean is
# 34.4 ms/frame - 79% of the whole run was the image phase.
#
# The correct scaling runs the OTHER way. signed_rel - the bias detector that
# actually carries the image verdict - is a mean over every pixel, so its
# standard error goes as 1/sqrt(pixels * spp). Full res has 6.7x the pixels of
# 960x540 (3.46M vs 518k), so going full res BUYS bias sensitivity; matching a
# 960x540 gate's 4096 spp means paying 8x the time for precision the pixel count
# already gave away. 512 spp here is comparable to 4096 spp there, and cheaper
# than it was at low res.
#
# Do NOT scale WARM down with them. It is not a sampling budget - it is how long
# the rad/tau caches need to retrain after updatePass recreates the pass, and an
# undertrained cache silently measures a different estimator (the radTrainEvery
# trap). 192 stays.
NOISE_SPP = 128    # Equal-sample capture: shows any variance cost.
CONV_SPP = 512     # Converged capture: shows bias, or proves there is none.

# Set True to run only the perf phase (~5 min instead of ~30). Use it while
# iterating on the lever; the image phase is what promotes it.
PERF_ONLY = False

# Perf configs, in the order the log blocks will appear. 'off' first so it is
# block 0 and every delta is read against it.
CONFIGS = [
    ('off',   {'useWaveUniformMip': False, 'waveMipLift': 0}),
    ('lift0', {'useWaveUniformMip': True,  'waveMipLift': 0}),
    ('lift1', {'useWaveUniformMip': True,  'waveMipLift': 1}),
    ('lift2', {'useWaveUniformMip': True,  'waveMipLift': 2}),
    ('lift3', {'useWaveUniformMip': True,  'waveMipLift': 3}),
]

# Image configs. The reference is rendered twice with different seeds; that pair
# is the empirical noise floor everything else is judged against. seedOffset
# only shifts gFrameCount, so both reference runs are the SAME estimator.
REF_SEED_A = 0
REF_SEED_B = 7919
IMAGE_CONFIGS = [
    ('refA', {'useWaveUniformMip': True, 'waveMipLift': 0, 'seedOffset': REF_SEED_A}),
    ('refB', {'useWaveUniformMip': True, 'waveMipLift': 0, 'seedOffset': REF_SEED_B}),
    ('img1', {'useWaveUniformMip': True, 'waveMipLift': 1, 'seedOffset': REF_SEED_A}),
    ('img2', {'useWaveUniformMip': True, 'waveMipLift': 2, 'seedOffset': REF_SEED_A}),
    ('img3', {'useWaveUniformMip': True, 'waveMipLift': 3, 'seedOffset': REF_SEED_A}),
]

# Shipping config, PINNED (mirrors VNA_DistWeightGate.py's SHIP as of
# 2026-07-22, minus its outputSize override - see the full-res note above).
# A gate must test a FIXED config - do NOT wire this to the live WdasSkyVNA.py,
# or the gate drifts with the daily driver.
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
    # 8, NOT 1e6: training off during warm means the cache never populates, the
    # C.a > 0.5 gate fails, the cut never fires, and every path runs full length
    # - measuring nocache while believing you measured ship.
    'radTrainEvery': 8,
    'radEma': 0.10,
    'radResidualSurvival': 0.25,
    'trRRThreshold': 0.05,
    'trRRMode': 31,
    'radWarpRRLanes': 0,        # measured out (HANDOFF_10 6) - off.
    'radResidualTailQ': 0.0,    # measured out (HANDOFF_10 6) - off.
    # OFF: amortization biases the image, and half this gate is about bias.
    # Also 64 % period == 0 would alias risStatsInterval so every logged frame
    # was a correction frame - the run-167 trap.
    'radAmortPeriod': 0,
    'distWeightK': 1.0,         # 1 = exact analog tracking; the lift is the only variable here.
    'seedOffset': 0,
    'useWaveUniformMip': False,  # overridden per config below
    'waveMipLift': 0,            # overridden per config below
    # The counters this gate exists to read.
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
}

# No outputSize here either. Every downstream pass must agree with the tracer's
# size, and the way to guarantee that is for nobody to override it.
ACCUM_PROPS = {'enabled': True, 'precisionMode': 'Single'}
ACCUM_PROPS_OFF = {'enabled': False, 'precisionMode': 'Single'}


def render_graph_WaveMipGate():
    g = RenderGraph("VNAWaveMipGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_WaveMipGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

print("VNA-WAVEMIPGATE: FULL window res, {} configs x {} rounds, {} warm + {} "
      "measured each. Do not resize the window mid-run."
      .format(len(CONFIGS), ROUNDS, WARM, MEASURE))

# ---------------------------------------------------------------------------
# PHASE 1 - PERF. Runs first on purpose: it is the cheap half, and if the lift
# fraction comes back ~0 the mechanism never fired and the image phase is moot.
# ---------------------------------------------------------------------------
for rnd in range(ROUNDS):
    for name, cfg in CONFIGS:
        m.scene.selectViewpoint(1)
        # Fresh pass = fresh rad/tau caches; the warm-up retrains before the
        # measured window so cache cold-start never lands inside it.
        graph.updatePass("VolumePathTracer", dict(SHIP, **cfg))
        for i in range(WARM):
            m.renderFrame()
        print("VNA-WAVEMIPGATE: BEGIN perf round {} config {}".format(rnd, name))
        for i in range(MEASURE):
            m.renderFrame()
        print("VNA-WAVEMIPGATE: END   perf round {} config {}".format(rnd, name))

# ---------------------------------------------------------------------------
# PHASE 2 - IMAGE. refA and refB are the SAME estimator with different seeds,
# so their difference is noise alone and becomes the floor for img1..img3.
# ---------------------------------------------------------------------------
if not PERF_ONLY:
    for name, cfg in IMAGE_CONFIGS:
        m.scene.selectViewpoint(1)
        graph.updatePass("VolumePathTracer", dict(SHIP, **cfg))
        for i in range(WARM):
            m.renderFrame()
        # Reset accumulation only - recreating Accum leaves the tracer caches
        # warm. off -> on with DIFFERENT dicts so neither updatePass can no-op.
        graph.updatePass("Accum", ACCUM_PROPS_OFF)
        m.renderFrame()
        graph.updatePass("Accum", ACCUM_PROPS)
        m.renderFrame()
        frame = 1
        for spp in (NOISE_SPP, CONV_SPP):
            while frame < spp:
                m.renderFrame()
                frame += 1
            m.frameCapture.baseFilename = "{}_vp1_s{}".format(name, spp)
            m.frameCapture.capture()
        print("VNA-WAVEMIPGATE: captured image config {}".format(name))

print("VNA-WAVEMIPGATE: complete. Run: py scripts/VNA_WaveMipGate_Compare.py")

# Fire-and-forget, same as the other gate drivers: launch, measure, exit.
exit()
