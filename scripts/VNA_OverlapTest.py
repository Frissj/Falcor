# DIAGNOSTIC launcher: the sky, opened directly on the OVERLAP pair.
#   Mogwai.exe --script scripts/VNA_OverlapTest.py
#
# Identical to WdasSky.py except it selects viewpoint 2 (hero + heroWisp, the
# only two instances whose AABBs intersect) instead of viewpoint 1.
#
# WHY: the black screenshot was taken at this framing, but the "old sky works"
# comparison was made at the startup framing (viewpoint 1, isolated 'core').
# Different views exercise different code: viewpoint 1 never touches the
# summed-majorant overlap path, viewpoint 2 does. This isolates that variable.
#
#   black here  -> the overlap traversal path is the bug (NOT the buffer
#                  release, which is currently disabled in the sky scene).
#   correct     -> overlap is fine; the blackness tracked the buffer release.

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
    g.markOutput("VolumePathTracer.work")
    return g

m.addGraph(render_graph_VolumePathTracer())
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

# THE POINT OF THIS SCRIPT: open on the overlap pair.
m.scene.selectViewpoint(2)
