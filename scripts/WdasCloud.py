# One-shot launcher: builds the reference volumetric path tracer graph and
# loads the WDAS cloud scene. Run with:
#   Mogwai.exe --script scripts/WdasCloud.py

from falcor import *

def render_graph_VolumePathTracer():
    g = RenderGraph("VolumePathTracer")

    # No VBufferRT: the volume is not in the BVH, the pass marches the grid
    # directly from the camera.
    VolumePathTracer = createPass("VolumePathTracer", {'maxBounces': 64, 'useNEE': True})
    g.addPass(VolumePathTracer, "VolumePathTracer")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VolumePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

graph = render_graph_VolumePathTracer()
m.addGraph(graph)

# Absolute path on purpose: makes the script independent of Mogwai's working
# directory, so a desktop/taskbar shortcut works from anywhere.
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_cloud.pyscene")

# --- Canonical viewpoints for quality comparison -----------------------------
# Viewpoint 0 is the scene's own camera (added automatically by Falcor).
# These are fixed framings so a FLIP/ImageCompare run is reproducible: always
# converge from the SAME viewpoint index, or the comparison is meaningless.
#
# Grid world bounds (-217,-67.5,-298) -> (197,214,212), centre (-10,73,-43).
#
#   1 = FLIP reference. Cloud fills ~60% of frame height. Use this one.
#   2 = Wisp detail. Close on the upper edge where thin structure lives -
#       the hardest thing for any approximation to preserve.
#   3 = Side profile, low angle. Different silhouette + deeper optical path.
m.scene.addViewpoint(float3(-10, 73, 500),   float3(-10, 73, -43),  float3(0, 1, 0))
m.scene.addViewpoint(float3(-60, 210, 190),  float3(-30, 150, -60), float3(0, 1, 0))
m.scene.addViewpoint(float3(620, 10, 120),   float3(-10, 90, -43),  float3(0, 1, 0))

m.scene.selectViewpoint(1)
