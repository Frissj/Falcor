from falcor import *

def render_graph_VolumePathTracer():
    g = RenderGraph("VolumePathTracer")

    # Reference volumetric path tracer. No VBufferRT: the volume is not in the
    # BVH at all, the pass marches the grid directly from the camera.
    VolumePathTracer = createPass("VolumePathTracer", {'maxBounces': 64, 'useNEE': True})
    g.addPass(VolumePathTracer, "VolumePathTracer")

    # Accumulation is what makes this a reference: let it sit for a few thousand
    # frames and the result converges to ground truth.
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VolumePathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

VolumePathTracer = render_graph_VolumePathTracer()
try: m.addGraph(VolumePathTracer)
except NameError: None
