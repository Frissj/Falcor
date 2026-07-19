# VNA Step 2 - the measurement that "decides everything" (spec section 10).
# Captures, at the canonical FLIP viewpoint (index 1) of the single-cloud scene:
#   1. the 1-spp image  (what the eye gets every frame at 60 fps today)
#   2. the converged reference (accumulated REF_FRAMES frames)
# Both saved as EXR (linear HDR) + tonemapped PNG under vna_measurements\.
# Then run scripts/VNA_Step2_Flip.py to turn the pair into FLIP/MSE numbers.
#
# Run with:
#   Mogwai.exe --script scripts/VNA_Step2_Measure.py
#
# WHY THIS ORDER: the 1-spp capture happens after exactly one rendered frame,
# when the accumulator holds precisely one sample - so no reset mechanism is
# needed; the reference capture is just the same accumulator N frames later.
# The clock is paused so nothing (camera, animation) can invalidate
# accumulation between the two captures.

import os

from falcor import *

REF_FRAMES = 8192   # ~10 ms/frame -> a bit over a minute of convergence.
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements"

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
    # Mark the LINEAR accumulated output: HDR-FLIP and MSE compare linear
    # radiance, not tonemapped pixels. The tonemapped PNG is for eyeballing.
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

# Freeze time: accumulation is only valid while nothing moves.
m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

# --- 1 spp ------------------------------------------------------------------
m.renderFrame()
m.frameCapture.baseFilename = "vna_1spp"
m.frameCapture.capture()
print("VNA: captured 1-spp frame.")

# --- converged reference ----------------------------------------------------
for i in range(REF_FRAMES - 1):
    m.renderFrame()
m.frameCapture.baseFilename = "vna_ref"
m.frameCapture.capture()
print("VNA: captured reference after {} frames.".format(REF_FRAMES))
print("VNA: outputs in {}".format(OUT_DIR))
print("VNA: now run Mogwai.exe --script scripts/VNA_Step2_Flip.py for the numbers.")
