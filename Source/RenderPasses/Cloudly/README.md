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

## Known differences from the capture

* Cirrus are more opaque than in the game frame.
* The sun-ray cloud shadow map, cloud AO trace, coarse trace and the plugin's tile-adaptive step counts are not
  implemented. Sun rays are always traced.
* Different tonemappers (UE filmic vs Falcor ToneMapper), and the game frame is DLSS-upscaled.
