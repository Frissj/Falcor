# ISOLATION TEST 2 of 2: Stage-A RIS ONLY.
#   Mogwai.exe --script scripts/VNA_IsolateRIS.py
#
# The sky, stock transmittance estimator, no demodulation - the ONLY change is
# resampled importance sampling of the primary scatter vertex (section 5).
#
#   black / speckled -> the RIS weights are the bug (prime suspect: the
#                       screenshot's signature is a collapsed importance weight).
#   looks right      -> RIS is innocent; suspect demodulation recombination.
#
# Opens on viewpoint 2 (the overlap pair) to match the original screenshot.

from falcor import *

def render_graph():
    g = RenderGraph("VNARisOnly")
    VolumePathTracer = createPass("VolumePathTracer", {
        'maxBounces': 64,
        'useNEE': True,
        'useRIS': True,
        'risCandidates': 4,
        'risMip': 2,
    })
    g.addPass(VolumePathTracer, "VolumePathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VolumePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

m.addGraph(render_graph())
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))
m.scene.selectViewpoint(2)
