#include "CloudlyPlacement.h"
#include "Core/Pass/ComputePass.h"
#include <cstring>
#include <fstream>

namespace Falcor
{
namespace
{
constexpr uint32_t kWidth = 512, kHeight = 512, kDepth = 64, kMips = 3;
constexpr size_t kVoxelCount = size_t(kWidth) * kHeight * kDepth;
constexpr char kShader[] = "RenderPasses/Cloudly/PlacementSdf.cs.slang";
constexpr char kLayoutShader[] = "RenderPasses/Cloudly/AtmosphereLayout.cs.slang";

std::vector<float> readDensity(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    FALCOR_CHECK(file && file.tellg() == std::streamoff(kVoxelCount * sizeof(float)),
                 "Cloudly placement: '{}' must contain {} little-endian float32 density samples.", path.string(), kVoxelCount);
    std::vector<float> density(kVoxelCount);
    file.seekg(0);
    FALCOR_CHECK(file.read(reinterpret_cast<char*>(density.data()), kVoxelCount * sizeof(float)),
                 "Cloudly placement: could not read '{}'.", path.string());
    FALCOR_CHECK(std::all_of(density.begin(), density.end(), [](float v) { return std::isfinite(v); }),
                 "Cloudly placement: '{}' contains non-finite densities.", path.string());
    return density;
}

void writeRaw(const std::filesystem::path& path, const void* data, size_t size)
{
    std::ofstream file(path, std::ios::binary);
    FALCOR_CHECK(file.write(static_cast<const char*>(data), size), "Cloudly placement: could not write '{}'.", path.string());
}

// Inverse of decodeSdf() for the base/alto curve, with explicit UNORM8 quantization.
uint8_t encode(float distance)
{
    const float x = std::clamp((std::clamp(distance / 32.f, -1.f, 1.f) + 1.f + 1.f / 255.f) * .5f, 0.f, 1.f);
    return uint8_t(std::lround(255.f / (1.f + std::exp(-15.f * (x - .5f)))));
}
}

ref<Texture> generateCloudlyPlacement(
    RenderContext* pContext,
    const ref<Device>& pDevice,
    const std::filesystem::path& densityDir,
    const std::filesystem::path& outputDir,
    float surface,
    float surfaceFixupStrength,
    const CloudlyAtmosphereLayout* pLayout
)
{
    FALCOR_CHECK(std::isfinite(surface), "Cloudly placement: surface threshold must be finite.");
    FALCOR_CHECK(std::isfinite(surfaceFixupStrength) && surfaceFixupStrength >= 0.f && surfaceFixupStrength <= 1.f,
                 "Cloudly placement: surface fixup strength must be in [0, 1].");
    if (!outputDir.empty())
        std::filesystem::create_directories(outputDir);

    const uint3 dim(kWidth, kHeight, kDepth);
    const auto readWrite = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto createVolume = [&](ResourceFormat format, ResourceBindFlags flags, const void* data = nullptr)
    {
        return pDevice->createTexture3D(dim.x, dim.y, dim.z, format, 1, data, flags);
    };
    auto pInitialize = ComputePass::create(pDevice, kShader, "initialize");
    auto pJump = ComputePass::create(pDevice, kShader, "jumpFlood");
    auto pDecode = ComputePass::create(pDevice, kShader, "decodeDistance");
    ref<ComputePass> pRasterize;
    ref<Texture> pLayoutDensity;
    if (pLayout)
    {
        pRasterize = ComputePass::create(pDevice, kLayoutShader, "rasterize");
        pLayoutDensity = createVolume(ResourceFormat::R32Float, readWrite);
        auto cb = pRasterize->getRootVar()["LayoutCB"];
        const float angle = math::radians(std::remainder(pLayout->rotation, 360.f));
        const float2 extent(1.f / pLayout->uvScale.x, 1.f / pLayout->uvScale.y);
        cb["gDim"] = dim;
        cb["gOffset"] = float2(std::remainder(pLayout->offset.x, extent.x), std::remainder(pLayout->offset.y, extent.y));
        cb["gRotation"] = float2(std::cos(angle), std::sin(angle));
        cb["gScale"] = pLayout->scale;
        cb["gMapExtent"] = float3(extent.x, extent.y, 1.f / pLayout->uvScale.z);
        cb["gMapOffset"] = pLayout->uvOffset;
        pRasterize->getRootVar()["gLayoutDensity"] = pLayoutDensity;
    }
    ref<Texture> seeds[2] = {createVolume(ResourceFormat::R32Uint, readWrite), createVolume(ResourceFormat::R32Uint, readWrite)};
    auto pDistance = createVolume(ResourceFormat::R32Float, readWrite);

    Sampler::Desc sampler;
    sampler.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    sampler.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Clamp);
    auto pDensitySampler = pDevice->createSampler(sampler);

    std::vector<uint8_t> packed(2 * (kVoxelCount + kVoxelCount / 8 + kVoxelCount / 64));
    for (uint32_t channel = 0; channel < 2; ++channel)
    {
        const char* layer = channel == 0 ? "Base" : "Alto";
        ref<Texture> pDensity;
        if (pLayout)
        {
            auto cb = pRasterize->getRootVar()["LayoutCB"];
            cb["gLayer"] = channel;
            cb["gStartHeight"] = pLayout->startHeight[channel];
            cb["gThickness"] = pLayout->thickness[channel];
            cb["gAmount"] = pLayout->amount[channel];
            pRasterize->execute(pContext, dim);
            pDensity = pLayoutDensity;
        }
        else
        {
            const auto input = readDensity(densityDir / fmt::format("{}_density.f32", layer));
            pDensity = createVolume(ResourceFormat::R32Float, ResourceBindFlags::ShaderResource, input.data());
        }

        auto init = pInitialize->getRootVar();
        init["PlacementCB"]["gDim"] = dim;
        init["PlacementCB"]["gSurface"] = surface;
        init["gDensity"] = pDensity;
        init["gNextSeeds"] = seeds[0];
        pInitialize->execute(pContext, dim);

        uint32_t current = 0;
        auto flood = pJump->getRootVar();
        flood["PlacementCB"]["gDim"] = dim;
        for (uint32_t jump = std::max({dim.x, dim.y, dim.z}); jump; jump >>= 1)
        {
            flood["PlacementCB"]["gJump"] = jump;
            flood["gSeeds"] = seeds[current];
            flood["gNextSeeds"] = seeds[1 - current];
            pJump->execute(pContext, dim);
            current = 1 - current;
        }

        auto decode = pDecode->getRootVar();
        decode["PlacementCB"]["gDim"] = dim;
        decode["PlacementCB"]["gSurface"] = surface;
        decode["PlacementCB"]["gSurfaceFixupStrength"] = surfaceFixupStrength;
        decode["gDensitySampler"] = pDensitySampler;
        decode["gDensity"] = pDensity;
        decode["gSeeds"] = seeds[current];
        decode["gDistance"] = pDistance;
        pDecode->execute(pContext, dim);

        std::vector<float> distances(kVoxelCount);
        {
            const auto bytes = pContext->readTextureSubresource(pDistance.get(), 0);
            FALCOR_CHECK(bytes.size() == distances.size() * sizeof(float), "Cloudly placement: unexpected distance readback size.");
            std::memcpy(distances.data(), bytes.data(), bytes.size());
        }
        if (!outputDir.empty())
            writeRaw(outputDir / fmt::format("{}_distance.f32", layer), distances.data(), distances.size() * sizeof(float));

        for (size_t i = 0; i < distances.size(); ++i)
            packed[2 * i + channel] = encode(distances[i]);
    }

    // The captured lower mips match the rounded 2x2x2 average in encoded space
    // to within one byte, including across the logistic curve's transition.
    size_t srcOffset = 0;
    uint3 srcDim = dim;
    for (uint32_t mip = 1; mip < kMips; ++mip)
    {
        const size_t dstOffset = srcOffset + 2 * size_t(srcDim.x) * srcDim.y * srcDim.z;
        const uint3 dstDim = srcDim / 2u;
        for (uint32_t z = 0; z < dstDim.z; ++z)
        for (uint32_t y = 0; y < dstDim.y; ++y)
        for (uint32_t x = 0; x < dstDim.x; ++x)
        {
            uint32_t sum[2] = {};
            for (uint32_t dz = 0; dz < 2; ++dz)
            for (uint32_t dy = 0; dy < 2; ++dy)
            for (uint32_t dx = 0; dx < 2; ++dx)
            {
                const size_t src = srcOffset + 2 * (((2 * z + dz) * srcDim.y + 2 * y + dy) * srcDim.x + 2 * x + dx);
                sum[0] += packed[src];
                sum[1] += packed[src + 1];
            }
            const size_t dst = dstOffset + 2 * ((z * dstDim.y + y) * dstDim.x + x);
            for (uint32_t channel = 0; channel < 2; ++channel)
            {
                const uint32_t whole = sum[channel] / 8, remainder = sum[channel] % 8;
                packed[dst + channel] = uint8_t(whole + (remainder > 4 || (remainder == 4 && (whole & 1))));
            }
        }
        srcOffset = dstOffset;
        srcDim = dstDim;
    }

    if (!outputDir.empty())
    {
        size_t offset = 0;
        for (uint32_t mip = 0; mip < kMips; ++mip)
        {
            const size_t size = 2 * (kVoxelCount >> (3 * mip));
            writeRaw(outputDir / fmt::format("CloudTex_BASE_SDF_Tex_mip{}.bin", mip), packed.data() + offset, size);
            offset += size;
        }
    }

    auto pTexture = pDevice->createTexture3D(dim.x, dim.y, dim.z, ResourceFormat::RG8Unorm, kMips, packed.data(),
                                             ResourceBindFlags::ShaderResource);
    pTexture->setName("Cloudly::GeneratedBaseAltoSdf");
    logInfo("Cloudly: generated Base and Alto placement SDFs from '{}'.", pLayout ? "Atmosphere layout" : densityDir.string());
    return pTexture;
}
}
