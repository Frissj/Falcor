#pragma once

#include "Falcor.h"
#include <filesystem>

namespace Falcor
{
/// Builds Base/Alto in game texture coordinates from float32 density fields. Optionally
/// exports intermediate signed distances and the packed mips used by the renderer.
ref<Texture> generateCloudlyPlacement(
    RenderContext* pContext,
    const ref<Device>& pDevice,
    const std::filesystem::path& densityDir,
    const std::filesystem::path& outputDir,
    float surface,
    float surfaceFixupStrength
);
}
