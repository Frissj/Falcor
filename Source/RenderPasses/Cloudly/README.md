# Cloudly

Volumetric clouds and sky for Falcor, modelled on the Cloudly UE plugin as used in a shipping title.
All constants were read from a RenderDoc capture of that title (D3D12, frame 21779).

## Passes

| Falcor shader | Plugin pass | Notes |
|---|---|---|
| `NoiseTextureGen.cs.slang` | — | Procedural shape noise, used only when the game volumes are not loaded. |
| `AtmosphereLut.cs.slang` | `OpticalDepthTex`, `MultiScatteringLut`, `DistantSkyLightLut` | Hillaire 2020 LUTs. Cornette-Shanks Mie, as in the plugin. |
| `SkyTrace.cs.slang` | `SkyTraceCS` | SDF stepping, 6-step geometric sun ray, 4 scattering octaves, dark edge, height AO, aerial perspective. |
| `TemporalUpscale.cs.slang` | `CheckerBoardResolveCS` | Checkerboard trace plus reprojected history. |
| `CompositeWithScreen.cs.slang` | `CompositeAtmosphereCS` | Sky, sun disk (`GetSunDiskLuminance`), aerial perspective on geometry, clouds. |

## Cloud shapes

* **Game volumes** (`gameDataDir` property). These load the plugin's own placement SDF and displacement volumes:
  `CloudTex_BASE_SDF_Tex` (512x512x64 RG8, a 200 km x 200 km x 25 km world map), `DISP1` (BC5), `DISP2` (BC4),
  `CIRRUS_SDF` (RGBA8) and `DISP_CIRRUS` (R8), each with mips 0-2. The evaluation follows `GetCloudSDF`,
  `GetCloudSDF_Cirrus` and `GetCloudDensity`: logistic SDF decode, two displacement passes with distance-band fades,
  weld against the layer floors, Hotsure near-camera detail, and `CalculateMipLevel` cone mips.
  The volumes are game assets. Dump them from your own capture and keep them out of the repository.
* **Procedural**: noise-based layers exposed as a pseudo-SDF, so the same density and lighting code applies.

## Coordinates

1 world unit = `metersPerUnit` metres. Falcor (x, y, z) = game (x, z, y). The planet centre stays straight below
the camera (the plugin's flat game map), so a camera placed at the game's camera position reproduces the game's view.
For the captured frame: position (-428.90, 1203.00, -2336.22), looking down -Z, vertical FOV 37.3 deg
(focal length 35.56 mm on a 24 mm frame). See `scripts/Cloudly/`.

## Running

```
Mogwai.exe -s scripts/Cloudly/Cloudly.py -S scripts/Cloudly/CloudlyScene.pyscene
```

`CLOUDLY_GAME_DATA` overrides the volume folder.

## Generating Base and Alto placement SDFs

See the [placement handoff](../../../docs/cloudly/README.md) for the map and sequencer index, decoded placement records, recovered shader stages, and extraction commands.

By default the pass still uses the captured volumes. Set `CLOUDLY_PLACEMENT_DENSITY` (or the
`placementDensityDir` graph property) to generate Base and Alto from `Base_density.f32` and
`Alto_density.f32` in that directory. Each file contains a 512×512×64 volume of little-endian
float32 density values, with X varying fastest, in the same texture coordinates as
`CloudTex_BASE_SDF_Tex`. Voxels with density greater than `CLOUDLY_PLACEMENT_SURFACE`
(default 0.5, graph property `placementSurface`) are inside. `CLOUDLY_GAME_DATA` still supplies
cirrus and the displacement textures; the captured Base/Alto texture is optional in this mode.

Falcor follows the recovered Base/Alto shaders. It seeds voxels where one of six axial
neighbors crosses the density threshold, propagates a nearest boundary coordinate through
jump flooding, and signs the distance using the density at the output voxel. The shader
wraps X/Y and clamps Z. `CLOUDLY_PLACEMENT_FIXUP_STRENGTH` (graph property
`placementSurfaceFixupStrength`) controls the recovered subvoxel boundary search. It ranges
from 0 (disabled) to 1 (full strength, Falcor's default). The two outputs are packed with
Cloudly's Base/Alto logistic encoding. Lower mips average 2×2×2 blocks in encoded space.
From the captured parent mips, this reproduces about 97% of the lower-mip bytes exactly,
with every byte within one code.

The original shader library contains `WriteComponentsDensityCS` and the jump-flood
kernels. The Falcor generator takes the resulting scalar density fields as input.
To reproduce a mission, those fields must be built from the mission's placement asset,
cloud part volume textures, and generation parameters. This port covers the boundary
seed, jump, and decode stages. It omits Cloudly's separate optional extreme-value blur
and normalization passes.

To reproduce the diagnostic with an existing RenderDoc volume dump, run these PowerShell
commands from the Falcor repository root. The probe needs NumPy in your Python environment.

```powershell
$env:CLOUDLY_GAME_DATA = 'C:\path\to\Capture_dxil\cloud_volumes'
$env:CLOUDLY_PLACEMENT_DENSITY = "$env:TEMP\cloudly-probe-density"
$env:CLOUDLY_PLACEMENT_OUTPUT = "$env:TEMP\cloudly-probe-generated"
py scripts/Cloudly/placement_probe.py prepare $env:CLOUDLY_GAME_DATA $env:CLOUDLY_PLACEMENT_DENSITY
& build/windows-ninja-msvc/bin/Release/Mogwai.exe -s scripts/Cloudly/Cloudly.py -S scripts/Cloudly/CloudlyScene.pyscene
py scripts/Cloudly/placement_probe.py compare $env:CLOUDLY_GAME_DATA $env:CLOUDLY_PLACEMENT_OUTPUT
```

`CLOUDLY_PLACEMENT_OUTPUT` (graph property `placementOutputDir`) optionally exports signed
float32 distances and packed mips. It must be separate from the capture directory.
`prepare` thresholds the **finished captured SDF** into a binary silhouette; the probe measures the distance reconstruction from those silhouettes.
The original pre-SDF density and subvoxel values are unavailable in that frame. For an independent comparison, skip `prepare` and supply density textures
captured before the game runs the SDF pass.

## Known differences from the capture

* Cirrus are more opaque than in the game frame.
* The sun-ray cloud shadow map, cloud AO trace, coarse trace and the plugin's tile-adaptive step counts are not
  implemented. Sun rays are always traced.
* Different tonemappers (UE filmic vs Falcor ToneMapper), and the game frame is DLSS-upscaled.
