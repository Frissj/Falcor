# Russian-roulette sweep for shadeMain path length.
#
# WHY: Nsight per-dispatch timing (traces 00_49 / 01_00) split the ~33ms frame as
#       main (phase A, Dispatch)        13.1 - 18.5 ms   ~40%
#       argsMain                          <0.01 ms         0%
#       shadeMain (phase B, ExecuteIndirect) 18.5 - 19.8 ms   ~56%   <-- largest
#       Accumulate/Composite/ToneMapper   ~1.3 ms          4%
# shadeMain is shading + NEE + the bounce loop, running for only ~13% of pixels
# (noCand 1.81M of 2.07M) yet costing more than the full-screen phase A. So the
# cost is path LENGTH, not pixel count. maxBounces is 64 and roulette was
# hardcoded to q = max(0.05, 1 - maxThp) past bounce 3; in a high-albedo cloud
# throughput stays near 1, so q sits at the 0.05 floor and paths run tens of
# bounces deep.
#
# Roulette is UNBIASED at any q - the survivor is reweighted by 1/(1-q). So every
# config below has the same mean and differs only in cost and variance. That is
# exactly what the +-0.15% converged gate is built to adjudicate: if a config
# shifts the converged mean beyond the noise floor, something is wrong with the
# reweighting, not with the choice of q.
#
# Read the results with VNA_Matrix_Compare.py against 01_ref, and watch BOTH:
#   - signedRel : should stay within +-0.15%. Bias detector.
#   - mean|d|   : WILL grow as q rises. That is the variance cost, and it is the
#                 real question - a config that halves shadeMain but needs 4x the
#                 frames to converge is not a win.
#
#   Mogwai.exe --script scripts/VNA_RouletteSweep.py

import os

from falcor import *

FRAMES = 256
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

# Shipping config, verbatim from WdasSkyVNA.py. Only the roulette knobs vary.
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
    'useBrickTlas': True,
    'mipPixelThreshold': 1.0,
    'useMergedTail': True,
    'tailRes': 64,
    'tailGateVoxels': 32.0,
    # Work stats ON so each config reports its own cells/taps alongside the EXR.
    # Interval 64 keeps it to one GPU sync per 64 frames.
    'logWorkStats': True,
    'logRisStats': True,
    'risStatsInterval': 64,
}

# SWEEP 2: bracket the knee.
#
# Sweep 1 result, judged on the GATE's metric (cloudy-region mean, mask = top-40%
# luminance of 01_ref - NOT the compare script's all-pixel signedRel, which the
# roulette tail dominates):
#     rq_baseline (q=0.05)  -0.0168%  PASS   samplerCalls 2.6
#     rq_q10      (q=0.10)  +0.0210%  PASS   samplerCalls 2.1   <-- survivor
#     rq_q20      (q=0.20)  +2.8430%  FAIL
#     rq_q30      (q=0.30)  +1.9182%  FAIL
#     rq_b1/b2/q20_b1       +1.0..1.5% FAIL
# Starting roulette earlier (b1/b2) failed at ANY floor, so that lever is dead.
# The floor knee is somewhere in 0.10 .. 0.20 - this sweep finds it.
#
# Each config captures 1spp AND converged. The 1spp capture is the one that
# actually decides adoption: the gate measures 256-frame convergence, but the
# shipping target is 1 spp, where a single firefly is visible. q10 already moved
# the |d|>100 count from 0 to 1.
CONFIGS = [
    ("rq2_baseline", dict(SHIP, rouletteMinQ=0.05,  rouletteStartBounce=3)),
    ("rq2_q075",     dict(SHIP, rouletteMinQ=0.075, rouletteStartBounce=3)),
    ("rq2_q10",      dict(SHIP, rouletteMinQ=0.10,  rouletteStartBounce=3)),
    ("rq2_q125",     dict(SHIP, rouletteMinQ=0.125, rouletteStartBounce=3)),
    ("rq2_q15",      dict(SHIP, rouletteMinQ=0.15,  rouletteStartBounce=3)),
    ("rq2_q175",     dict(SHIP, rouletteMinQ=0.175, rouletteStartBounce=3)),
]


def render_graph_Roulette():
    g = RenderGraph("VNARoulette")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g


graph = render_graph_Roulette()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props in CONFIGS:
    m.scene.selectViewpoint(1)
    # updatePass recreates the pass, which resets mFrameCount and restarts
    # accumulation - so the very next frame IS the 1-spp image. Capturing it
    # here costs one frame and gives the 1spp/converged pair from one run.
    graph.updatePass("VolumePathTracer", props)
    m.renderFrame()
    m.frameCapture.baseFilename = name + "_vp1_1spp"
    m.frameCapture.capture()
    # Continue accumulating from frame 1 to FRAMES for the converged capture.
    for i in range(FRAMES - 1):
        m.renderFrame()
    m.frameCapture.baseFilename = name + "_vp1_converged"
    m.frameCapture.capture()
    with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
        f.write("ROULETTE2 {} done (1spp + converged)\n".format(name))

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("ROULETTE2 SWEEP COMPLETE\n")
print("VNA-ROULETTE2: complete.")
# Unattended batch job: close Mogwai so the run is hands-off end to end.
exit()
