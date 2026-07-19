# DIAGNOSTIC launcher: the sky WITH the NanoVDB buffer release ON.
#   Mogwai.exe --script scripts/VNA_SkyBufReleaseTest.py
#
# Reproduction check for the only black configuration seen so far.
# Opens on viewpoint 2 (the overlap pair) - the framing of the original
# black screenshot - so this is the closest possible match to that run.

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
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky_bufrelease.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))
m.scene.selectViewpoint(2)
