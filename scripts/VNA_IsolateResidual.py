# ISOLATION TEST 1 of 2: residual ratio tracking ONLY.
#   Mogwai.exe --script scripts/VNA_IsolateResidual.py
#
# The sky, reference estimator in every respect EXCEPT transmittance, which
# uses the section-2/P2 analytic control-variate path. No RIS, no demodulation.
#
#   looks right -> residual tracking is innocent; suspect RIS.
#   black       -> the mean pyramid or the residual estimator is the bug.
#
# Opens on viewpoint 2 (the overlap pair) to match the original screenshot.

from falcor import *

def render_graph():
    g = RenderGraph("VNAResidualOnly")
    VolumePathTracer = createPass("VolumePathTracer", {
        'maxBounces': 64,
        'useNEE': True,
        'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
        'residualMip': 0,
        'useRIS': False,
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
