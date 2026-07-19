# Demodulated reconstruction graph (VNA spec section 6) for the single-cloud scene.
# Run with:
#   Mogwai.exe --script scripts/WdasCloudDemod.py
#
# WHY THIS GRAPH EXISTS
# ---------------------
# The tracer now decomposes its output EXACTLY, per sample:
#
#     color = Lin + transmittance * background
#
#   Lin           in-scattered radiance (paths that scattered >= 1 time).
#                 Low-frequency inside the cloud -> safe to denoise/accumulate.
#   transmittance binary sample: did the primary flight escape unscattered?
#                 Its average over frames converges to the true transmittance T.
#                 This channel carries the sharp wisp silhouette, so it must
#                 NEVER pass through a spatial filter - temporal only.
#   background    env radiance along the primary direction. Deterministic:
#                 zero noise at frame 1, nothing to converge.
#
# Recombining AFTER accumulation, `accum(Lin) + accum(T) * bg`, gives the same
# expected image as accumulating `color` directly (that equality is the
# correctness check below), but the silhouette-against-sky term is rebuilt from
# a noise-free background and a scalar T that no spatial blur ever touched.
# When a spatial denoiser (e.g. SVGF) is added later, it slots in on the Lin
# branch ONLY, and the wisps survive by construction.

from falcor import *

def render_graph_VolumeDemod():
    g = RenderGraph("VolumeDemod")

    VolumePathTracer = createPass("VolumePathTracer", {'maxBounces': 64, 'useNEE': True})
    g.addPass(VolumePathTracer, "VolumePathTracer")

    # One accumulator per stochastic channel. background needs none.
    AccumLin = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumLin, "AccumLin")
    AccumT = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumT, "AccumT")

    # Stock composites: Mul = accum(T) * bg, Add = accum(Lin) + Mul.
    # 32F output keeps the recombination linear HDR until the tone mapper.
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

    # Correctness check: the old direct path, side by side. Flipping between
    # 'ToneMapperRef.dst' and 'ToneMapper.dst' in Mogwai's output dropdown must
    # show the SAME converged image - the decomposition is exact, so any
    # difference beyond residual noise means a bug in the split.
    AccumColor = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumColor, "AccumColor")
    ToneMapperRef = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperRef, "ToneMapperRef")
    g.addEdge("VolumePathTracer.color", "AccumColor.input")
    g.addEdge("AccumColor.output",      "ToneMapperRef.src")
    g.markOutput("ToneMapperRef.dst")

    # Raw channels, inspectable via the output dropdown.
    g.markOutput("VolumePathTracer.transmittance")
    g.markOutput("VolumePathTracer.background")
    return g

graph = render_graph_VolumeDemod()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_cloud.pyscene")

# Canonical viewpoints - identical to WdasCloud.py, DO NOT EDIT (all FLIP /
# ImageCompare numbers are only comparable within the same viewpoint index).
#   1 = FLIP reference. 2 = wisp detail. 3 = side profile.
m.scene.addViewpoint(float3(-10, 73, 500),   float3(-10, 73, -43),  float3(0, 1, 0))
m.scene.addViewpoint(float3(-60, 210, 190),  float3(-30, 150, -60), float3(0, 1, 0))
m.scene.addViewpoint(float3(620, 10, 120),   float3(-10, 90, -43),  float3(0, 1, 0))

m.scene.selectViewpoint(1)
