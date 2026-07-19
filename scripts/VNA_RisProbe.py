# RIS PROBE: find WHERE the RIS contribution is lost. Raw values, no thresholds.
#   Mogwai.exe --script scripts/VNA_RisProbe.py
#
# HOW TO READ IT
# Open the output dropdown and select "VolumePathTracer.risDebug", then use the
# pixel inspector (or hover) ON A BLACK CLOUD PIXEL and read channel R:
#
#   R = 0  no medium        ray hit no instance (only valid outside the cloud)
#   R = 1  no candidates    the coarse walk emitted nothing -> the coarse field
#                           (mean pyramid at risMip) is empty along this ray
#   R = 2  zero target      every candidate weighed 0 -> target collapsed
#   R = 3  zero density     the survivor's REAL density tap was 0 -> the coarse
#                           field is pointing where the cloud is not
#   R = 4  zero transmit    transmittance to the survivor estimated as 0
#   R = 5  path terminated  vertex shaded, but phase sampling/RR ended it
#   R = 6  contributed      full contribution made (this is the healthy value)
#
#   G = candidate count   B = target weight sum   A = the RIS weight W
#
# Each R value is a DIFFERENT bug with a different fix. Report R (and G/B/A) on
# a black pixel and the guilty branch is identified without further guessing.
#
# The color output is also marked, so you can flip between the image and the
# probe in the same session.

from falcor import *

def render_graph():
    g = RenderGraph("VNARisProbe")
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

    # THE POINT: raw per-pixel RIS internals.
    g.markOutput("VolumePathTracer.risDebug")
    return g

m.addGraph(render_graph())
m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))
m.scene.selectViewpoint(2)
