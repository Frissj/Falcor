# Cloudly placement map reconstruction

The Windows SM6 global shader archive contains the original compute kernels. The DXIL debug records include preprocessed HLSL, which establishes the shader behavior independently of the RenderDoc frame containing the finished textures.

| Archive index | Shader |
| --- | --- |
| 13 | `WriteComponentsDensityCS` |
| 279 | `JumpFloodPassCS` |
| 280 | `InitializeTexturesCS` |
| 281 | `ExtremeValuesBlurCS` |
| 282 | `DecodeResultIntoDistanceTextureRDGCS` |
| 283 | `NormalizeSDFCS` |
| 284 | `DecodeResultIntoDistanceTextureCS` |

The cooked training layout `CVC_Training` imports named volume textures such as `VT_Cumulus_Humilis_01` and `VT_AltoCumulus_Layer_01`. The package archive also contains the default generation parameters `CGP_Default` and the default cloud volume `VT_DefaultCloud`. The component textures are shaped inputs to placement generation. Their own bake history is a separate stage.

1. **Rasterize component density.** Each output voxel center maps into world coordinates. For each component, the shader subtracts the component position, applies rotation and shear, clips to the component bounds or applies the configured stretch rule, then maps to texture UVs. It samples the red channel of the component volume and remaps density between the component's low and high thresholds. The large-part path evaluates overlapping map tiles and keeps the maximum. The optional 2D cirrus path averages four samples through the source volume.
2. **Combine components.** The writer binds up to 16 component textures per batch. Add and subtract modify the batch density. Hard subtract sets the batch density to a large negative value wherever the subtracting component is present. Each dispatch adds the batch result to the previous density volume and carries a keep-density mask.
3. **Seed and flood the surface.** A voxel is solid when its density strictly exceeds the configured level. A voxel on either side becomes a seed when one of its six axial neighbors has the opposite classification. Seeds pack X/Y/Z into 12/12/8 bits and jump flooding propagates one nearest surface coordinate per voxel. Horizontal distance uses the nearest periodic image. Z may wrap or clamp depending on the original settings.
4. **Resolve distances.** The decode signs the distance according to the voxel's density. The optional surface correction iterates up to 16 gradient steps near zero-distance seeds and accepts a result within 0.02 density units of the threshold. Separate shaders support extreme-value blur and distance normalization. The placement output uses an 8-bit logistic curve with slopes 15 for Base/Alto and 8 for Cirrus.

Falcor now implements the recovered Base/Alto seeding, flood and distance rules. It accepts scalar density volumes, uses horizontal wrap and vertical clamp, and defaults to full boundary correction. The captured mip filter is approximated by averaging 2×2×2 blocks in encoded space. The [placement dump](cloudly-placements.json) supplies the decoded component records. Matching a mission requires assembling those components using the referenced volume textures and the active generation settings. The density writer and optional blur and normalization passes remain to be implemented in Falcor.

The supplied RenderDoc capture contains the finished maps. The diagnostic inputs were derived from those maps by thresholding their decoded signs. This loses the original density gradients and can change the boundary voxels selected by the recovered shader. The comparison therefore measures reconstruction from the available silhouettes.
