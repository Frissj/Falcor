# Cloudly placement handoff

The [placement dump](cloudly-placements.json) covers the installed Windows build of Ace Combat 8. It contains 7,538 component records from 112 layouts, including 584 group records, and 10 keyframes from three sequencers. The extractor audited all 1,113 cooked map packages. Of those, 114 directly depend on a layout or sequencer, and two cinematic maps reach those assets through intermediate packages. All 112 layouts and three sequencers are reachable through the package-store graph. Four other maps export a Cloudly actor and import sky parameters. Their active layout is unresolved by this audit.

## Find a map and read its placements

1. Look up its Unreal package path in `maps`. Start with `layoutImports` and `sequencerImports`. The `cloudlyStoreDependencies` dictionary adds shortest import depths from the IoStore package graph. `namedMapReferences` records package names found in the map's name table. These links describe cooked references. Streaming, visibility, and runtime selection depend on the game.
2. Use the full asset reference to index `layouts` or `sequencers`. For example, `/Game/Cloudly/CloudlySkyParams/Ingame/Mission025/TimeLaps/CloudlySequencer_ms25_TimeLaps.CloudlySequencer_ms25_TimeLaps` uses the `Nearest` volume blend mode. The keyframes give time, layout and sky references, hidden and removed UIDs, and UID-to-array-index mappings.
3. Read each layout's `components` in stored array order. `Position`, `Rotation`, and `Scale` preserve the serialized XYZW values, and rotation XYZ is in radians. Preserve the native coordinate system until upload. The records include UV transforms, density remap, Add/Subtract/HardSubtract, KeepDensity, shear, placement layer override, and group, parent, hidden, stretch, and rain settings. `Texture` resolves the layout-local `TextureId`. Fifteen non-group records have texture ID −1.

## Continue the Falcor implementation

[Placement reconstruction](cloudly-placement-reconstruction.md) describes the recovered shader pipeline. [CloudlyPlacement.cpp](../../Source/RenderPasses/Cloudly/CloudlyPlacement.cpp) and [PlacementSdf.cs.slang](../../Source/RenderPasses/Cloudly/PlacementSdf.cs.slang) implement Base/Alto density-to-SDF generation. The [Cloudly pass README](../../Source/RenderPasses/Cloudly/README.md) gives the input file layout and probe commands.

The next stage is to resolve the selected map and keyframe, load the named cloud part textures or substitute user-provided volumes, and implement the density writer before the existing SDF stage. The placement dump supplies the per-component values. The recovered shader supplies the voxel-to-world, texture-coordinate, and density accumulation math. Exact reproduction still requires the referenced density volumes, the original CPU-side draw ordering and layer setup, and the active map's generation parameters. The authoring or baking process for the individual cloud part textures remains unknown.

Headless Mogwai ran the recovered SDF pipeline with binary density inputs derived from RenderDoc's finished textures. Mean absolute distance error within four texels of the captured surface was 0.098 texels for Base and 0.094 texels for Alto at mip 0. The [probe results](cloudly-placement-recovered-comparison.json) record per-layer and per-mip comparisons. A test of component rasterization needs pre-SDF density or the original textures and generation settings.

## Reproduce the dump from local game archives

Run from the repository root with `retoc` on PATH, `$cloudlyPaks` set to the game's Paks directory, and `$cloudlyRetocArgs` set to the Retoc global arguments needed to open the archives. Keep extracted packages and shaders in a scratch directory.

```powershell
retoc @cloudlyRetocArgs to-legacy --no-shaders --no-script-objects --no-parallel --filter .umap $cloudlyPaks ./map-export
retoc @cloudlyRetocArgs to-legacy --no-shaders --filter Live/Content/Cloudly/ --filter Live/Content/OnlineMode/Cloudly/ --filter Lit_Cloudly --filter LitCloudly --filter CloudsGenerationParamsPresets $cloudlyPaks ./placement-export
retoc @cloudlyRetocArgs list --path --package --store $cloudlyPaks | Set-Content -Encoding utf8 ./store-index.txt
py -B scripts/Cloudly/placement_dump.py --assets ./placement-export --maps ./map-export --store-index ./store-index.txt --output ./cloudly-placements.json
```

The global SM6 archive can be recovered separately. The shader extraction script uses the Windows SDK's `dxc.exe` and writes DXIL, disassembly, and the embedded preprocessed HLSL. Pass `--dxc` with the executable path if needed.

```powershell
retoc @cloudlyRetocArgs to-legacy --no-assets --no-script-objects --no-compres-shaders --filter ShaderArchive-Global-PCD3D_SM6 $cloudlyPaks ./shader-export
py -B scripts/Cloudly/placement_shader_dump.py ./shader-export/Live/Content/ShaderArchive-Global-PCD3D_SM6-PCD3D_SM6.ushaderbytecode ./placement-shaders
```
