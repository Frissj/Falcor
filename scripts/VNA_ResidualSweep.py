# VNA section-2/P2 validation - residual ratio tracking mip sweep.
# Converges the single-cloud scene at viewpoint 1 with:
#   pass 0: stock RatioTrackingLocalMajorant   (the trusted baseline)
#   pass 1: ResidualRatioTracking, mip 0
#   pass 2: ResidualRatioTracking, mip 1
#   pass 3: ResidualRatioTracking, mip 2
#   pass 4: ResidualRatioTracking, mip 3
# and captures each converged EXR to vna_measurements\.
#
# Run with:
#   Mogwai.exe --script scripts/VNA_ResidualSweep.py
#
# WHAT THIS PROVES (or breaks): spec section 2 / P2 claims the coarse mean
# field is a control variate, so the estimator is unbiased at EVERY mip -
# refining LoD changes variance and cost, never the converged image. If any
# mip's converged capture FLIPs measurably against the baseline, the residual
# implementation (or the mean pyramid) has a bug. FLIP the captures against
# each other with scripts/VNA_Step2_Flip.py by editing its two patterns, or
# with ImageCompare.exe.
#
# NOTE: transmittance is roughly half the estimator (NEE shadow rays); the
# distance sampler is unchanged, so any converged difference isolates cleanly
# to the new transmittance path.

import os

from falcor import *

FRAMES_PER_CONFIG = 4096
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements"

CONFIGS = [
    ("vna_sweep_baseline", {'transmittanceEstimator': 'RatioTrackingLocalMajorant'}),
    ("vna_sweep_residual_mip0", {'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant', 'residualMip': 0}),
    ("vna_sweep_residual_mip1", {'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant', 'residualMip': 1}),
    ("vna_sweep_residual_mip2", {'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant', 'residualMip': 2}),
    ("vna_sweep_residual_mip3", {'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant', 'residualMip': 3}),
]

def render_graph_VolumePathTracer():
    g = RenderGraph("VolumePathTracer")
    VolumePathTracer = createPass("VolumePathTracer", {'maxBounces': 64, 'useNEE': True})
    g.addPass(VolumePathTracer, "VolumePathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VolumePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("AccumulatePass.output")
    g.markOutput("ToneMapper.dst")
    return g

graph = render_graph_VolumePathTracer()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_cloud.pyscene")

# Canonical viewpoints - identical to WdasCloud.py, DO NOT EDIT.
m.scene.addViewpoint(float3(-10, 73, 500),   float3(-10, 73, -43),  float3(0, 1, 0))
m.scene.addViewpoint(float3(-60, 210, 190),  float3(-30, 150, -60), float3(0, 1, 0))
m.scene.addViewpoint(float3(620, 10, 120),   float3(-10, 90, -43),  float3(0, 1, 0))
m.scene.selectViewpoint(1)

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props in CONFIGS:
    # updatePass recreates the pass with new properties, which also resets the
    # accumulator (graph recompile), so each config converges from scratch.
    graph.updatePass("VolumePathTracer", dict({'maxBounces': 64, 'useNEE': True}, **props))
    for i in range(FRAMES_PER_CONFIG):
        m.renderFrame()
    m.frameCapture.baseFilename = name
    m.frameCapture.capture()
    print("VNA: captured {} after {} frames.".format(name, FRAMES_PER_CONFIG))

print("VNA: sweep complete. FLIP each vna_sweep_residual_mip* against vna_sweep_baseline.")
print("VNA: identical converged images at every mip = P2 machinery validated.")
