# VNA section-5 Stage A validation - RIS vs reference.
# Converges the single-cloud scene at viewpoint 1 with:
#   pass 0: reference (RIS off)            -> vna_ris_off
#   pass 1: RIS on, M=8,  coarse mip 2     -> vna_ris_m8_mip2
#   pass 2: RIS on, M=4,  coarse mip 2     -> vna_ris_m4_mip2
#   pass 3: RIS on, M=8,  coarse mip 0     -> vna_ris_m8_mip0
# and additionally captures each config's FIRST frame (1 spp) before converging.
#
# Run with:
#   Mogwai.exe --script scripts/VNA_RisValidate.py
#
# PASS CRITERIA (spec section 2, P1 - the RIS identity):
#  - CONVERGED images must be identical to vna_ris_off for every M and mip.
#    Any FLIP delta beyond residual noise = bug in the RIS weights, the coarse
#    pdf, or the candidate coverage. Bias here would poison everything after.
#  - 1-SPP images are where RIS is allowed (expected) to differ: less noise in
#    lit/dense regions is the whole point. Compare 1-spp FLIP vs the converged
#    reference: RIS-on should beat RIS-off. If it does not, the target (phase x
#    coarse shadow) is not earning its cost - report the numbers, not a feeling.

import os

from falcor import *

FRAMES_CONVERGED = 4096
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements"

BASE = {'maxBounces': 64, 'useNEE': True}
CONFIGS = [
    ("vna_ris_off",     dict(BASE, useRIS=False)),
    ("vna_ris_m8_mip2", dict(BASE, useRIS=True, risCandidates=8, risMip=2)),
    ("vna_ris_m4_mip2", dict(BASE, useRIS=True, risCandidates=4, risMip=2)),
    ("vna_ris_m8_mip0", dict(BASE, useRIS=True, risCandidates=8, risMip=0)),
]

def render_graph_VolumePathTracer():
    g = RenderGraph("VolumePathTracer")
    VolumePathTracer = createPass("VolumePathTracer", BASE)
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
    # updatePass rebuilds the pass (compile-time defines changed), and the
    # graph recompile restarts accumulation, so each config starts clean.
    graph.updatePass("VolumePathTracer", props)

    m.renderFrame()
    m.frameCapture.baseFilename = name + "_1spp"
    m.frameCapture.capture()

    for i in range(FRAMES_CONVERGED - 1):
        m.renderFrame()
    m.frameCapture.baseFilename = name + "_converged"
    m.frameCapture.capture()
    print("VNA: {} captured (1spp + {} frames converged).".format(name, FRAMES_CONVERGED))

print("VNA: RIS validation captures complete.")
print("VNA: converged RIS-on MUST match vna_ris_off (unbiasedness);")
print("VNA: 1-spp RIS-on SHOULD beat vna_ris_off vs the converged reference (variance).")
