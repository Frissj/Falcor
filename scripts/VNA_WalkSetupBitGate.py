# Walk-setup hoist BIT-IDENTITY gate.
#
#   Mogwai.exe --script scripts/VNA_WalkSetupBitGate.py
#   py scripts/VNA_WalkSetupBitGate_Check.py
#
# WHAT IS BEING GATED (2026-07-22)
# beginWalk was widened from PER-BRICK to PER-INSTANCE in the four RayQuery
# loops of VolumeInstanceSampler.slang (evalTransmittanceRQ, sampleDistanceRQ,
# sampleCandidatesRQ, evalEscapeAndCandidatesRQ). It is a pure function of
# (gridVolume, ray) - getGrid, two 4x4 mat-vecs, one reciprocal - so rebuilding
# it on a new brick of the SAME instance recomputed a value that could not have
# changed. Nsight put GridVolumeSampler.slang:571 (`return w;`, i.e. this
# preamble) third in shadeMain at 513,710 of 9,096,776 samples = 5.6%.
#
# THIS GATE IS NOT A TIMING GATE. It answers one question: is the new code
# byte-identical to the old? Nothing else about the change is worth discussing
# until that is true, and if it is false the change is simply wrong.
#
# WHY useSharedWalkSetup IS THE A/B
# GVS_SHARED_WALK_SETUP OFF compiles the ORIGINAL call signatures - the
# convenience overloads that build a setup per walk. So OFF is the pre-change
# code exactly and ON is the new hoist, in one session, with no rebuild between
# arms and no cross-session clock drift to argue about. Both arms are the same
# estimator with the same RNG stream, so the images must match BIT FOR BIT -
# not "within noise". This is the one gate on this pass where a noise floor
# would be a bug, because there is nothing stochastic being changed.
#
# THE SECOND SIGNAL, AND IT IS THE ONE THAT SAYS THE HOIST FIRED
# A bit-identical image also results from the hoist never firing. Read
# [COST] brickCache hit in the log: ON must be HIGHER than OFF. The old code
# reset the cache on every brick (guaranteed miss at the start of each walk);
# the new code holds it across bricks of one instance, which is strictly more
# reuse. If hit% does not move, traversal is handing back candidates that
# alternate instances and the hoist is a no-op - the change is harmless but
# worthless, and no frame timer will show anything.
#
# ORDER OF CONCLUSIONS
#   1. images differ            -> change is WRONG. Stop. Do not look at time.
#   2. identical, hit% flat     -> hoist never fires. Drop it.
#   3. identical, hit% up       -> mechanism confirmed. NOW a timing A/B is
#                                  meaningful (same session, flip the toggle).
#
# THREE VIEWPOINTS on purpose: vp1 is the canonical measurement camera, vp0 and
# vp2 sit at different distances so the brick candidate stream has a different
# instance-interleaving pattern. A hoist bug that only shows when instances
# alternate would hide in a single viewpoint.

import os
from falcor import *

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\walksetup"

WARM = 256   # updatePass recreates the pass -> tau cache resets, temporal
             # reservoirs re-ramp to temporalMCap. ~8 s at this frame rate.
SPP = 320    # ~10 s of accumulation per capture at ~32 ms/frame.
             #
             # WHY SO MANY, given bit-identity is an EXACT test that converges
             # to nothing: not for settling, for COVERAGE. Each frame draws new
             # paths, so spp multiplies the number of brick candidates and
             # instance transitions the comparison actually exercises. A
             # divergence that only fires when traversal hands back two
             # instances in a particular order needs volume to be caught, and
             # 320 spp x 3.46M px x 3 viewpoints is ~3.3e9 paths against the
             # ~6.6e8 the first version tested.
             #
             # Do NOT read this number as a settling requirement and do NOT
             # reuse it for a timing A/B. Timing needs far more: the retrain
             # ramp alone is ~600 frames and gpuMs keeps drifting for thousands
             # after that (HANDOFF 12 2.3, and logs 194/196 this session).
VIEWPOINTS = (0, 1, 2)

# THE CONTROL ARM IS NOT OPTIONAL - offA and offB are the SAME config run
# twice. If they are not bit-identical to each other, this harness cannot prove
# bit-identity of anything and the off-vs-on result means nothing.
#
# Run 199 is why this exists: it compared off vs on with no control, reported
# FAIL, and the FAIL was not attributable - the `off` arm alone had already
# changed between runs 198 and 199 (7,853,163 -> 7,853,654 bytes, same config).
# The diff signature was a median of 6.10352e-05 = 2^-14 = one ULP of half
# float, confined to pixels that march the volume, with a few fireflies. That
# is accumulation noise, not a wrong transform.
#
# useRadCache is OFF below for the same reason: the radiance cache is built
# with InterlockedAdd ([RADSPLAT] deposits) and atomic ordering varies run to
# run, so with it on the image is not reproducible for a FIXED config. A
# bit-identity gate cannot run on a nondeterministic renderer. This is a
# deviation from SHIP and it is deliberate - the question here is whether the
# hoist changes the arithmetic, and the radiance cache only adds a term that
# neither arm controls.
CONFIGS = [
    ('offA', {'useSharedWalkSetup': False}),
    ('offB', {'useSharedWalkSetup': False}),
    ('on',   {'useSharedWalkSetup': True}),
]

# Shipping config, PINNED - mirrors VNA_WaveMipGate.py's SHIP as of 2026-07-22.
# A gate must test a FIXED config; do NOT wire this to the live driver script.
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
    # OFF for this gate only - see the CONFIGS note. Atomic-accumulated state
    # makes the renderer irreproducible for a fixed config, which is fatal to a
    # bit-identity test and to nothing else.
    'useRadCache': False,
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
    # brickCache hit% is the mechanism signal - this gate is useless without it.
    'logWorkStats': True,
    'logTrRRProbe': False,   # keep the probe OUT of main; see HANDOFF 12 4.4.
    'logRisStats': True,
    'risStatsInterval': 64,
    'useSharedWalkSetup': True,  # overridden per config below
}

ACCUM_PROPS = {'enabled': True, 'precisionMode': 'Single'}
ACCUM_PROPS_OFF = {'enabled': False, 'precisionMode': 'Single'}


def render_graph_WalkSetupBitGate():
    g = RenderGraph("VNAWalkSetupBitGate")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", ACCUM_PROPS)
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_WalkSetupBitGate()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

print("VNA-WALKSETUP: bit-identity gate, {} configs x {} viewpoints at {} spp."
      .format(len(CONFIGS), len(VIEWPOINTS), SPP))

for name, cfg in CONFIGS:
    for vp in VIEWPOINTS:
        m.scene.selectViewpoint(vp)
        # Fresh pass per arm AND per viewpoint, so the rad/tau caches are in the
        # same state for both arms - a warm cache carried across the flip would
        # make the two images differ for a reason that is not the change.
        graph.updatePass("VolumePathTracer", dict(SHIP, **cfg))
        for i in range(WARM):
            m.renderFrame()
        # Reset accumulation only - recreating Accum leaves the tracer caches
        # warm. off -> on with DIFFERENT dicts so neither updatePass can no-op.
        graph.updatePass("Accum", ACCUM_PROPS_OFF)
        m.renderFrame()
        graph.updatePass("Accum", ACCUM_PROPS)
        m.renderFrame()
        for frame in range(1, SPP):
            m.renderFrame()
        m.frameCapture.baseFilename = "{}_vp{}_s{}".format(name, vp, SPP)
        m.frameCapture.capture()
        print("VNA-WALKSETUP: captured {} vp{}".format(name, vp))

print("VNA-WALKSETUP: complete. Run: py scripts/VNA_WalkSetupBitGate_Check.py")

# Fire-and-forget, same as the other gate drivers: launch, measure, exit.
exit()
