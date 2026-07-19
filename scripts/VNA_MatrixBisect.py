# VNA matrix follow-up: interaction bisect configs. Round 1 (b1-b4) walked
# 07_temporal -> 11_full one feature at a time and localized the 1-spp
# firefly tail; round 2 (below) confirms the remaining story. Captures land
# in the same folder as VNA_Matrix.py so VNA_Matrix_Compare.py picks them up.
#
#   Mogwai.exe --script scripts/VNA_MatrixBisect.py

import os

from falcor import *

FRAMES_CONVERGED = int(os.environ.get("VNA_MATRIX_FRAMES", "256"))
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

BASE = {
    'maxBounces': 64,
    'useNEE': True,
    'useSingleNeePerPath': False,
    'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
    'residualMip': 0,
    'footprintMip': False,
    'footprintScale': 1.0,
    'useRIS': False,
    'risCandidates': 4,
    'risMip': 2,
    'useSharedCandidateSweep': False,
    'useAdaptiveM': False,
    'useTemporalReuse': False,
    'temporalMCap': 20.0,
    'useBrickTlas': False,
    'mipPixelThreshold': 1.0,
    'useMergedTail': False,
    'tailRes': 64,
    'tailGateVoxels': 32.0,
    'logWorkStats': False,
    'logRisStats': False,
    'risStatsInterval': 256,
    'useSpatialReuse': False,
    'spatialNeighbors': 2,
    'spatialRadiusPx': 16.0,
    'risTargetFloor': 0.0,
}

RIS = dict(useRIS=True, useSharedCandidateSweep=True, useAdaptiveM=True)

CONFIGS = [
    ("b5_tail_tlas",     dict(BASE, useBrickTlas=True, useMergedTail=True)),
    ("b6_full_hugegate", dict(BASE, **RIS, useBrickTlas=True, useTemporalReuse=True,
                              footprintMip=True, useSingleNeePerPath=True,
                              useMergedTail=True, tailGateVoxels=1000000.0)),
    ("b7_tail_ris_intv", dict(BASE, **RIS, useTemporalReuse=True,
                              footprintMip=True, useSingleNeePerPath=True,
                              useMergedTail=True)),
]

def render_graph_Bisect():
    g = RenderGraph("VNABisect")
    VolumePathTracer = createPass("VolumePathTracer", BASE)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g

graph = render_graph_Bisect()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

# Canonical viewpoints - identical to WdasSky.py, DO NOT EDIT.
m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props in CONFIGS:
    m.scene.selectViewpoint(1)
    graph.updatePass("VolumePathTracer", props)

    m.renderFrame()
    m.frameCapture.baseFilename = "{}_vp1_1spp".format(name)
    m.frameCapture.capture()

    for i in range(FRAMES_CONVERGED - 1):
        m.renderFrame()
    m.frameCapture.baseFilename = "{}_vp1_converged".format(name)
    m.frameCapture.capture()
    with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
        f.write("BISECT {} done ({} frames)\n".format(name, FRAMES_CONVERGED))

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("BISECT COMPLETE\n")
print("VNA-BISECT: complete.")
