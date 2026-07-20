# Minimal runner for the trRR bias hunt probes ([TRRPROBE] v4 + [TRRPROBE2]).
#
# This is NOT a gate sweep. The probes are paired per pixel inside the shader
# (RR walk vs RR-disabled reference walk, binned by the deterministic key
# exp(-coarseOpticalDepth)), so they need no image captures, no accumulation
# resets, no control rows, and no 1024-frame windows - the full
# VNA_RadCacheSweep.py matrix answers questions this run does not ask.
# One ntrE-config run writes probe lines to the Mogwai log every
# risStatsInterval frames; that log IS the output.
#
# Config = the sweep's ntrE row verbatim: temporal off, rad cache off,
# trRRThreshold 0.05, trRRMode 8 (escape-walk RR only - the convicted site).
#
#   Mogwai.exe --script scripts/VNA_TrRRProbe.py
#
# Read the result:
#   [TRRPROBE]  walk-level E-bias per det-key bin (v4 baseline: -8/-35/-23/-0.15%).
#   [TRRPROBE2] coin-level E-bias per det-key bin.
#     ~0 while the walk bins stay biased  -> coin acquitted, post-survival
#        continuation convicted (inner per-node RR interaction, FP32).
#     matches the walk bias               -> the coin itself loses mass in vivo.

import os

from falcor import *

FRAMES = 256  # risStatsInterval 64 -> 4 probe readings; the bias is systematic.
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

# Shipping config, verbatim from VNA_RadCacheSweep.py SHIP as of 2026-07-20.
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

PROBE = dict(SHIP, useTemporalReuse=False, useRadCache=False,
             trRRThreshold=0.05, trRRMode=8, radResidualSurvival=1.0)

# Phase 2: trRRMode 24 = bit3 (escape RR) + bit4 (DECORRELATED COIN).
# Run 129 verdict (2026-07-20): with the flush miscompile fixed, the walk
# reads -42%/-27% in the thick bins while the coin is per-event FAIR
# (theta*survives == sum(TrBefore) to +-1%) and negTr = 0. The mass is lost
# strictly AFTER survived coins, and the only thing a survival changes about
# the continuation is the RNG stream: a same-stream coin conditions the
# generator state on u < p, and LCG-family draws downstream are deterministic
# functions of that state. bit4 draws the coin from a counter-based hash and
# leaves sg untouched, so the survivor's continuation consumes exactly the
# draws it would have consumed without RR. If phase 2's [TRRPROBE] bins read
# ~0, the mechanism is CONFIRMED and bit4 IS the fix; if the bias persists,
# the conditioning theory dies and the hunt returns to the continuation.
# (The earlier phase-2 candidates are settled: tail never runs in this scene
# - tailRays 0.00/entry, gate 14.4 wu vs footprints <=1.8 wu - and
# footprintMip=False is a behavioral no-op here, both proven by bit-identical
# phase replays.)
PROBE_DECOIN = dict(PROBE, trRRMode=24)

# Phase 3: A/A NULL TEST. Run 131 mass accounting: bin1 is missing 4076
# T-units under RR but only ~125 T-units EVER pass through coins - the RR
# block cannot be the thief (33x too small), and the decorrelated coin
# changing nothing (-41.774 vs -41.773) says the same. The remaining
# difference between the paired walks is GENERATOR STATE: Tfull continues the
# frame's mid-stream sg, Tref starts a freshly-seeded generator. bin3's
# steady -0.215% with ~97 coins among 1.77M sky pixels was this all along
# (and v2's "generator sensitivity" ghost before that). threshold=1e-30 arms
# the probe but Tr can never cross it, so RR NEVER FIRES (CHK line must show
# coinLanes ~0) and the phase reads PURE pairing bias per bin. Subtract from
# phase 1 to get RR's true contribution; if phase 3 == phase 1, the probe's
# walk signal is all seed-sensitivity and the image-level -0.73% (which
# involves no psg) needs a different instrument.
PROBE_NULL = dict(PROBE, trRRThreshold=1e-30)


def render_graph_TrRRProbe():
    g = RenderGraph("VNATrRRProbe")
    VolumePathTracer = createPass("VolumePathTracer", PROBE)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    g.markOutput("VolumePathTracer.color")
    return g


graph = render_graph_TrRRProbe()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()
m.scene.selectViewpoint(1)

# Same reference resolution as the sweep so bin masses stay comparable with
# the v4 baseline numbers. (resizeFrameBuffer is ignored while maximized -
# section-5 trap - but the probe ratios are per-bin and resolution-robust,
# so a maximized window degrades comparability, not validity.)
m.resizeFrameBuffer(1920, 1080)

os.makedirs(OUT_DIR, exist_ok=True)

# Phase 1: mode 8, same-stream coin - must reproduce run 129's -42%/-27%
# baseline (anchor for the A/B).
for i in range(FRAMES):
    m.renderFrame()
with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("TRRPROBE phase 1 done (mode 8 same-stream coin, {} frames)\n".format(FRAMES))

# Phase 2: mode 24, decorrelated coin. The updatePass prop dict differs from
# phase 1's, so recreation cannot silently no-op (the section-6 harness trap).
graph.updatePass("VolumePathTracer", PROBE_DECOIN)
for i in range(FRAMES):
    m.renderFrame()
with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("TRRPROBE phase 2 done (mode 24 decorrelated coin, {} frames)\n".format(FRAMES))

# Phase 3: A/A null - RR armed but unfireable; reads pure pairing bias.
graph.updatePass("VolumePathTracer", PROBE_NULL)
for i in range(FRAMES):
    m.renderFrame()

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("TRRPROBE RUN COMPLETE (phase 3 A/A null, {} frames each)\n".format(FRAMES))
print("VNA-TRRPROBE: complete - grep the Mogwai log for [TRRPROBE] / [TRRPROBE2].")
# Unattended: close Mogwai so the run is hands-off end to end.
exit()
