# DIAGNOSTIC launcher: single cloud WITH the NanoVDB buffer release enabled.
#   Mogwai.exe --script scripts/VNA_BufReleaseTest.py
# See wdas_cloud_bufrelease_test.pyscene for the decision tree.

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
    return g

m.addGraph(render_graph_VolumePathTracer())
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_cloud_bufrelease_test.pyscene")

# Same canonical viewpoints as WdasCloud.py.
m.scene.addViewpoint(float3(-10, 73, 500),   float3(-10, 73, -43),  float3(0, 1, 0))
m.scene.addViewpoint(float3(-60, 210, 190),  float3(-30, 150, -60), float3(0, 1, 0))
m.scene.addViewpoint(float3(620, 10, 120),   float3(-10, 90, -43),  float3(0, 1, 0))
m.scene.selectViewpoint(1)
