# THE WHOLE VNA STACK, ON. One launcher that runs everything built so far,
# together, on the nine-instance sky:
#
#   section 3   cloudlet instancing            (scene: 9 instances, 1 grid)
#   section 4   segment/overlap traversal      (VolumeInstanceSampler)
#   section 2   residual ratio tracking        (analytic T_c over the mean pyramid)
#   section 5   Stage-A RIS primary scatter    (coarse candidates -> exact shade)
#   section 6   demodulated reconstruction     (accum(Lin) + accum(T) * bg)
#
#   Mogwai.exe --script scripts/WdasSkyVNA.py
#
# The reference shortcuts stay untouched on purpose: THIS graph is the fast
# path, THOSE are the ground truth it must match. If this image ever looks
# wrong, isolate the guilty stage with the validation scripts
# (VNA_ResidualSweep.py, VNA_RisValidate.py, WdasCloudDemod.py) instead of
# debugging everything at once.

from falcor import *

def render_graph_VNA():
    g = RenderGraph("VNAStack")

    VolumePathTracer = createPass("VolumePathTracer", {
        'maxBounces': 64,
        'useNEE': True,
        # Section 2/P2: analytic control-variate transmittance.
        'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
        'residualMip': 0,
        # Section 5 Stage A: RIS on the primary vertex.
        'useRIS': True,
        'risCandidates': 4,
        'risMip': 2,
    })
    g.addPass(VolumePathTracer, "VolumePathTracer")

    # Section 6: the two stochastic channels accumulate separately; background
    # is deterministic and never accumulated. Recombined AFTER accumulation.
    AccumLin = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumLin, "AccumLin")
    AccumT = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumT, "AccumT")
    Mul = createPass("Composite", {'mode': 'Multiply', 'outputFormat': 'RGBA32Float'})
    g.addPass(Mul, "Mul")
    Add = createPass("Composite", {'mode': 'Add', 'outputFormat': 'RGBA32Float'})
    g.addPass(Add, "Add")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VolumePathTracer.Lin",           "AccumLin.input")
    g.addEdge("VolumePathTracer.transmittance", "AccumT.input")
    g.addEdge("AccumT.output",                  "Mul.A")
    g.addEdge("VolumePathTracer.background",    "Mul.B")
    g.addEdge("Mul.out",                        "Add.A")
    g.addEdge("AccumLin.output",                "Add.B")
    g.addEdge("Add.out",                        "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    # Un-demodulated color through its own accumulator: flip between this and
    # ToneMapper.dst in the output dropdown - converged, they must match.
    AccumColor = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumColor, "AccumColor")
    ToneMapperRef = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperRef, "ToneMapperRef")
    g.addEdge("VolumePathTracer.color", "AccumColor.input")
    g.addEdge("AccumColor.output",      "ToneMapperRef.src")
    g.markOutput("ToneMapperRef.dst")

    # Work counters, as always.
    g.markOutput("VolumePathTracer.work")
    return g

graph = render_graph_VNA()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))
m.scene.selectViewpoint(1)
