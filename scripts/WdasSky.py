# One-shot launcher: reference volumetric path tracer + the instanced-sky scene.
#   Mogwai.exe --script scripts/WdasSky.py

from falcor import *

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
    g.markOutput("ToneMapper.dst")

    # Raw per-pixel work counters. Select this output in Mogwai's output
    # dropdown; hover with the pixel inspector to read exact counts.
    #   R = segments marched   G = overlap steps   B = AABB tests   A = max cover
    g.markOutput("VolumePathTracer.work")
    return g

graph = render_graph_VolumePathTracer()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# --- Canonical viewpoints ----------------------------------------------------
# Viewpoint 0 is the scene's own camera (added automatically by Falcor).
#
#   1 = SINGLE CLOUD, isolated. Frames 'core' (800,200,-900, scale 0.8) against
#       empty sky with no other instance in shot. Use this for FLIP: one cloud
#       filling the frame gives the cleanest quality signal.
#   2 = Overlap pair. Frames hero + heroWisp, the only two instances that
#       intersect. This is the shot that validates the summed-majorant path -
#       look for any seam or brightness step where their AABBs meet.
#   3 = Wide establishing shot of the whole sky (perf, not quality).
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.scene.selectViewpoint(1)
