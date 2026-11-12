#pragma once

#include "Falcor.h"
#include <filesystem>

namespace Falcor
{
struct CloudlyAtmosphereLayout
{
    float2 offset = float2(0.f);             ///< World X/Z translation in metres.
    float rotation = 0.f;                    ///< Degrees from +X toward +Z.
    float2 scale = float2(1.f);              ///< Horizontal scale along the rotated field's axes.
    float2 thickness = float2(2000.f, 2400.f); ///< Base/Alto depth in metres.
    float2 startHeight;
    float2 amount;
    float3 uvOffset;
    float3 uvScale;
};

/// Builds Base/Alto from float32 density files or the Atmosphere layout. Optionally
/// exports intermediate signed distances and the packed mips used by the renderer.
ref<Texture> generateCloudlyPlacement(
    RenderContext* pContext,
    const ref<Device>& pDevice,
    const std::filesystem::path& densityDir,
    const std::filesystem::path& outputDir,
    float surface,
    float surfaceFixupStrength,
    const CloudlyAtmosphereLayout* pLayout
);
}
