# Compaction black-cloud probe: shipping config with compaction ON vs OFF,
# short accumulation each, captured for a numeric compare (no eyeballs
# needed). Run:  Mogwai.exe --script scripts/VNA_CompactProbe.py

import os

from falcor import *

FRAMES = 256
OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements\matrix"

SHIP = {
    'maxBounces': 64,
    'useNEE': True,
    'useSingleNeePerPath': True,
    'transmittanceEstimator': 'ResidualRatioTrackingLocalMajorant',
    'residualMip': 0,
    'footprintMip': True,
    'footprintScale': 1.0,
    'useRIS': True,
    'risCandidates': 4,
    'risMip': 2,
    'useSharedCandidateSweep': True,
    'useAdaptiveM': True,
    'useTemporalReuse': True,
    'temporalMCap': 20.0,
    'useSpatialReuse': True,
    'spatialNeighbors': 2,
    'spatialRadiusPx': 16.0,
    'risTargetFloor': 0.01,
    'useBrickTlas': True,
    'mipPixelThreshold': 1.0,
    'useMergedTail': True,
    'tailRes': 64,
    'tailGateVoxels': 32.0,
    'logWorkStats': False,
    'logRisStats': True,
    'risStatsInterval': 32,
    'useCompaction': True,
}

CONFIGS = [
    ("probe3_ship_final", dict(SHIP, useSpatialReuse=False, risTargetFloor=0.0)),
]

def render_graph_Probe():
    g = RenderGraph("VNAProbe")
    VolumePathTracer = createPass("VolumePathTracer", SHIP)
    g.addPass(VolumePathTracer, "VolumePathTracer")
    Accum = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(Accum, "Accum")
    g.addEdge("VolumePathTracer.color", "Accum.input")
    g.markOutput("Accum.output")
    return g

graph = render_graph_Probe()
m.addGraph(graph)

m.loadScene(r"C:\Users\Friss\Documents\Clouds\Falcor\media\test_scenes\wdas_sky.pyscene")

m.scene.addViewpoint(float3(792, 258, -320), float3(792, 258, -934), float3(0, 1, 0))
m.scene.addViewpoint(float3(80, 190, 640),   float3(80, 170, 40),    float3(0, 1, 0))
m.scene.addViewpoint(float3(100, 40, 2000),  float3(-50, 250, -600), float3(0, 1, 0))

m.clock.pause()

os.makedirs(OUT_DIR, exist_ok=True)
m.frameCapture.outputDir = OUT_DIR

for name, props in CONFIGS:
    m.scene.selectViewpoint(1)
    graph.updatePass("VolumePathTracer", props)
    for i in range(FRAMES):
        m.renderFrame()
    m.frameCapture.baseFilename = name
    m.frameCapture.capture()
    with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
        f.write("PROBE {} done\n".format(name))

with open(os.path.join(OUT_DIR, "progress.txt"), "a") as f:
    f.write("PROBE COMPLETE\n")
print("VNA-PROBE: complete.")
