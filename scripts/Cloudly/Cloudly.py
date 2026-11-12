import os
from falcor import *

# Folder holding the CloudTex_*_mip0.bin volumes dumped from a RenderDoc capture.
# Leave empty (or point at a missing folder) to use the procedural clouds.
GAME_DATA_DIR = os.environ.get('CLOUDLY_GAME_DATA', r'C:\Users\Friss\Downloads\Capture_dxil\cloud_volumes')

def render_graph_Cloudly():
    g = RenderGraph('Cloudly')
    g.addPass(createPass('GBufferRaster', {'samplePattern': 'Center', 'sampleCount': 16}), 'GBufferRaster')
    g.addPass(createPass('Cloudly', {
        'useSceneSun': True,
        'metersPerUnit': 1.0,
        'cameraAltitude': 0.0,
        'traceDownscale': 2,
        'gameDataDir': GAME_DATA_DIR,
        'placementDensityDir': os.environ.get('CLOUDLY_PLACEMENT_DENSITY', ''),
        'placementOutputDir': os.environ.get('CLOUDLY_PLACEMENT_OUTPUT', ''),
        'placementSurface': float(os.environ.get('CLOUDLY_PLACEMENT_SURFACE', '0.5')),
        'placementSurfaceFixupStrength': float(os.environ.get('CLOUDLY_PLACEMENT_FIXUP_STRENGTH', '1')),
    }), 'Cloudly')
    g.addPass(createPass('ToneMapper', {'autoExposure': False, 'exposureCompensation': 0.0}), 'ToneMapper')
    g.addEdge('GBufferRaster.depth', 'Cloudly.depth')
    g.addEdge('GBufferRaster.diffuseOpacity', 'Cloudly.color')
    g.addEdge('Cloudly.output', 'ToneMapper.src')
    g.markOutput('ToneMapper.dst')
    return g

Cloudly = render_graph_Cloudly()
try: m.addGraph(Cloudly)
except NameError: None
