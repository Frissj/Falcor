#include "Cloudly.h"
#include "Scene/Lights/Light.h"
#include <fstream>

namespace
{
const char kNoiseShader[] = "RenderPasses/Cloudly/NoiseTextureGen.cs.slang";
const char kLutShader[] = "RenderPasses/Cloudly/AtmosphereLut.cs.slang";
const char kTraceShader[] = "RenderPasses/Cloudly/SkyTrace.cs.slang";
const char kUpscaleShader[] = "RenderPasses/Cloudly/TemporalUpscale.cs.slang";
const char kCompositeShader[] = "RenderPasses/Cloudly/CompositeWithScreen.cs.slang";

// Channels.
const char kInDepth[] = "depth";
const char kInColor[] = "color";
const char kOutColor[] = "output";

// Properties.
const char kSunDirection[] = "sunDirection";
const char kSunIntensity[] = "sunIntensity";
const char kUseSceneSun[] = "useSceneSun";
const char kMetersPerUnit[] = "metersPerUnit";
const char kCameraAltitude[] = "cameraAltitude";
const char kLayerCoverage[] = "layerCoverage";
const char kLayerStartHeight[] = "layerStartHeight";
const char kTraceDownscale[] = "traceDownscale";
const char kTraceMaxSteps[] = "traceMaxSteps";
const char kWindSpeed[] = "windSpeed";
const char kWindDirection[] = "windDirection";
const char kHistoryWeight[] = "historyWeight";
const char kDrawSky[] = "drawSky";
const char kDrawSunDisk[] = "drawSunDisk";
const char kSceneAerialPerspective[] = "sceneAerialPerspective";
const char kGameDataDir[] = "gameDataDir";
const char kPlacementDensityDir[] = "placementDensityDir";
const char kPlacementOutputDir[] = "placementOutputDir";
const char kPlacementSurfaceFixupStrength[] = "placementSurfaceFixupStrength";
const char kPlacementSurface[] = "placementSurface";
const char kAtmosphereLayout[] = "atmosphereLayout";
const char kPlacementOffset[] = "placementOffset";
const char kPlacementRotation[] = "placementRotation";
const char kPlacementScale[] = "placementScale";
const char kPlacementThickness[] = "placementThickness";
const char kGameOffset[] = "gameOffset";
const char kHotsure[] = "hotsure";

/// Game cloud volumes as dumped from the capture (mip 0, raw subresource bytes).
struct GameVolumeDesc
{
    const char* name; ///< Files are <name>_mip<N>.bin.
    uint3 dim;
    ResourceFormat format;
    uint32_t mips;
};
const GameVolumeDesc kGameVolumes[] = {
    {"CloudTex_BASE_SDF_Tex", {512, 512, 64}, ResourceFormat::RG8Unorm, 3},
    {"CloudTex_DISP1_DensityTex", {256, 256, 32}, ResourceFormat::BC5Unorm, 3},
    {"CloudTex_DISP2_DensityTex", {64, 64, 16}, ResourceFormat::BC4Unorm, 3},
    {"CloudTex_CIRRUS_SDF_Tex", {1024, 1024, 2}, ResourceFormat::RGBA8Unorm, 2},
    {"CloudTex_DISP_CIRRUS_DensityTex", {512, 512, 8}, ResourceFormat::R8Unorm, 3},
};

/// Bytes of one mip of a 3D texture (block-compressed formats use 4x4 blocks).
size_t mipBytes(uint3 dim, ResourceFormat format, uint32_t mip)
{
    uint32_t w = std::max(1u, dim.x >> mip), h = std::max(1u, dim.y >> mip), d = std::max(1u, dim.z >> mip);
    uint32_t bw = getFormatWidthCompressionRatio(format), bh = getFormatHeightCompressionRatio(format);
    return size_t(div_round_up(w, bw)) * div_round_up(h, bh) * d * getFormatBytesPerBlock(format);
}

const uint32_t kBaseNoiseSize = 128;
const uint32_t kDetailNoiseSize = 32;
const uint2 kTransmittanceLutDim = uint2(256, 64); // OpticalDepthTexSize
const uint2 kMultiScatteringLutDim = uint2(32, 32);
const uint32_t kDistantSkyLightLutSize = 16;

/// Order in which the pixels of a 4x4 block are traced (Bayer), so any ds x ds sub-block is covered evenly.
const uint2 kBayer4[16] = {
    {0, 0}, {2, 2}, {2, 0}, {0, 2}, {1, 1}, {3, 3}, {3, 1}, {1, 3},
    {1, 0}, {3, 2}, {3, 0}, {1, 2}, {0, 1}, {2, 3}, {2, 1}, {0, 3},
};

uint2 subPixelForFrame(uint32_t frame, uint32_t ds)
{
    if (ds <= 1)
        return uint2(0);
    // Walk the Bayer sequence, keeping only positions inside the ds x ds block.
    uint32_t n = ds * ds;
    uint32_t want = frame % n;
    uint32_t found = 0;
    for (uint32_t k = 0;; k = (k + 1) % 16)
    {
        uint2 p = kBayer4[k];
        if (p.x < ds && p.y < ds)
        {
            if (found == want)
                return p;
            found++;
        }
    }
}

float3 dirFromElevationAzimuth(float elevationDeg, float azimuthDeg)
{
    float e = math::radians(elevationDeg);
    float a = math::radians(azimuthDeg);
    return float3(std::cos(e) * std::sin(a), std::sin(e), std::cos(e) * std::cos(a));
}
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, Cloudly>();
}

Cloudly::Cloudly(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kSunDirection)
            mParams.lightSourceDirection = value;
        else if (key == kSunIntensity)
            mParams.sunIntensity = value;
        else if (key == kUseSceneSun)
            mUseSceneSun = value;
        else if (key == kMetersPerUnit)
            mParams.metersPerUnit = value;
        else if (key == kCameraAltitude)
            mParams.cameraAltitude = value;
        else if (key == kLayerCoverage)
            mParams.layerCoverage = value;
        else if (key == kLayerStartHeight)
            mParams.cloudLayersStartHeight = value;
        else if (key == kTraceDownscale)
            mParams.traceDownscale = std::clamp((uint32_t)value, 1u, 4u);
        else if (key == kTraceMaxSteps)
            mParams.traceMaxSteps = value;
        else if (key == kWindSpeed)
            mWindSpeed = value;
        else if (key == kWindDirection)
            mWindDirection = value;
        else if (key == kHistoryWeight)
            mHistoryWeight = value;
        else if (key == kDrawSky)
            mParams.drawSky = (bool)value ? 1 : 0;
        else if (key == kDrawSunDisk)
            mParams.drawSunDisk = (bool)value ? 1 : 0;
        else if (key == kSceneAerialPerspective)
            mSceneAerialPerspective = value;
        else if (key == kGameDataDir)
            mGameDataDir = value.operator std::string();
        else if (key == kPlacementDensityDir)
            mPlacementDensityDir = value.operator std::string();
        else if (key == kPlacementOutputDir)
            mPlacementOutputDir = value.operator std::string();
        else if (key == kPlacementSurfaceFixupStrength)
            mPlacementSurfaceFixupStrength = value;
        else if (key == kPlacementSurface)
            mPlacementSurface = value;
        else if (key == kAtmosphereLayout)
            mUseAtmosphereLayout = value;
        else if (key == kPlacementOffset)
            mAtmosphereLayout.offset = value;
        else if (key == kPlacementRotation)
            mAtmosphereLayout.rotation = value;
        else if (key == kPlacementScale)
            mAtmosphereLayout.scale = value;
        else if (key == kPlacementThickness)
            mAtmosphereLayout.thickness = value;
        else if (key == kGameOffset)
            mGame.gameOffsetXY = value;
        else if (key == kHotsure)
            mGame.enableHotsure = (bool)value ? 1 : 0;
        else
            logWarning("Unknown property '{}' in Cloudly properties.", key);
    }
    mParams.lightSourceDirection = normalize(mParams.lightSourceDirection);

    Sampler::Desc wrap;
    wrap.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    wrap.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpLinearWrap = mpDevice->createSampler(wrap);

    Sampler::Desc clamp;
    clamp.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    clamp.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearClamp = mpDevice->createSampler(clamp);

    mpNoisePass = ComputePass::create(mpDevice, kNoiseShader, "main");
    mpTransmittanceLutPass = ComputePass::create(mpDevice, kLutShader, "transmittanceLut");
    mpMultiScatteringLutPass = ComputePass::create(mpDevice, kLutShader, "multiScatteringLut");
    mpDistantSkyLightPass = ComputePass::create(mpDevice, kLutShader, "distantSkyLightLut");
    mpTracePass = ComputePass::create(mpDevice, kTraceShader, "main");
    mpUpscalePass = ComputePass::create(mpDevice, kUpscaleShader, "main");
    mpCompositePass = ComputePass::create(mpDevice, kCompositeShader, "main");

    const auto uavSrv = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpBaseNoise = mpDevice->createTexture3D(kBaseNoiseSize, kBaseNoiseSize, kBaseNoiseSize, ResourceFormat::RGBA8Unorm, 1, nullptr, uavSrv);
    mpDetailNoise = mpDevice->createTexture3D(kDetailNoiseSize, kDetailNoiseSize, kDetailNoiseSize, ResourceFormat::RGBA8Unorm, 1, nullptr, uavSrv);
    mpTransmittanceLut = mpDevice->createTexture2D(kTransmittanceLutDim.x, kTransmittanceLutDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, uavSrv);
    mpMultiScatteringLut = mpDevice->createTexture2D(kMultiScatteringLutDim.x, kMultiScatteringLutDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, uavSrv);
    mpDistantSkyLightLut = mpDevice->createTexture2D(kDistantSkyLightLutSize, 1, ResourceFormat::RGBA32Float, 1, 1, nullptr, uavSrv);
    mpBaseNoise->setName("Cloudly::BaseNoise");
    mpDetailNoise->setName("Cloudly::DetailNoise");
    mpTransmittanceLut->setName("Cloudly::TransmittanceLut");
    mpMultiScatteringLut->setName("Cloudly::MultiScatteringLut");
    mpDistantSkyLightLut->setName("Cloudly::DistantSkyLightLut");

    Sampler::Desc sdf;
    sdf.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    sdf.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Clamp);
    mpSdfSampler = mpDevice->createSampler(sdf);
    sdf.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpDispSampler = mpDevice->createSampler(sdf);

    uint32_t zero = 0;
    mpDummy3D = mpDevice->createTexture3D(1, 1, 1, ResourceFormat::RGBA8Unorm, 1, &zero, ResourceBindFlags::ShaderResource);
    FALCOR_CHECK(!mUseAtmosphereLayout || mPlacementDensityDir.empty(),
                 "Cloudly: atmosphereLayout and placementDensityDir are mutually exclusive.");
    if (mUseAtmosphereLayout)
    {
        auto positive = [](float v) { return std::isfinite(v) && v > 0.f; };
        FALCOR_CHECK(std::isfinite(mAtmosphereLayout.rotation) && std::isfinite(mAtmosphereLayout.offset.x) &&
                         std::isfinite(mAtmosphereLayout.offset.y),
                     "Cloudly: placementOffset and placementRotation must be finite.");
        FALCOR_CHECK(std::isfinite(mParams.cloudLayersStartHeight.x) && std::isfinite(mParams.cloudLayersStartHeight.y) &&
                         std::isfinite(mParams.layerCoverage.x) && std::isfinite(mParams.layerCoverage.y),
                     "Cloudly: Base and Alto placement heights and amounts must be finite.");
        FALCOR_CHECK(positive(mAtmosphereLayout.scale.x) && positive(mAtmosphereLayout.scale.y) &&
                         positive(mAtmosphereLayout.thickness.x) && positive(mAtmosphereLayout.thickness.y),
                     "Cloudly: placementScale and placementThickness must be finite and positive.");
        mAtmosphereLayout.startHeight = float2(mParams.cloudLayersStartHeight.x, mParams.cloudLayersStartHeight.y);
        mAtmosphereLayout.amount = float2(mParams.layerCoverage.x, mParams.layerCoverage.y);
        mAtmosphereLayout.uvOffset = mGame.baseUVOffset;
        mAtmosphereLayout.uvScale = mGame.baseUVScale;
    }
    if (!mGameDataDir.empty())
        loadGameVolumes(mGameDataDir);
    const bool generatePlacement = mUseAtmosphereLayout || !mPlacementDensityDir.empty();
    FALCOR_CHECK(!generatePlacement || mParams.useGameVolumes,
                 "Cloudly placement generation requires gameDataDir for cirrus and displacement volumes.");
    FALCOR_CHECK(mPlacementOutputDir.empty() || generatePlacement,
                 "Cloudly placement export requires atmosphereLayout or placementDensityDir.");
    FALCOR_CHECK(mPlacementOutputDir.empty() || !std::filesystem::exists(mPlacementOutputDir) ||
                     !std::filesystem::equivalent(mPlacementOutputDir, mGameDataDir),
                 "Cloudly placement output must be separate from gameDataDir to preserve the reference volumes.");

    float3 d = mParams.lightSourceDirection;
    mSunElevationDeg = math::degrees(std::asin(std::clamp(d.y, -1.f, 1.f)));
    mSunAzimuthDeg = math::degrees(std::atan2(d.x, d.z));
}

Properties Cloudly::getProperties() const
{
    Properties props;
    props[kSunDirection] = mParams.lightSourceDirection;
    props[kSunIntensity] = mParams.sunIntensity;
    props[kUseSceneSun] = mUseSceneSun;
    props[kMetersPerUnit] = mParams.metersPerUnit;
    props[kCameraAltitude] = mParams.cameraAltitude;
    props[kLayerCoverage] = mParams.layerCoverage;
    props[kLayerStartHeight] = mParams.cloudLayersStartHeight;
    props[kTraceDownscale] = mParams.traceDownscale;
    props[kTraceMaxSteps] = mParams.traceMaxSteps;
    props[kWindSpeed] = mWindSpeed;
    props[kWindDirection] = mWindDirection;
    props[kHistoryWeight] = mHistoryWeight;
    props[kDrawSky] = mParams.drawSky != 0;
    props[kDrawSunDisk] = mParams.drawSunDisk != 0;
    props[kSceneAerialPerspective] = mSceneAerialPerspective;
    if (!mGameDataDir.empty())
        props[kGameDataDir] = mGameDataDir.string();
    if (mUseAtmosphereLayout || !mPlacementDensityDir.empty())
    {
        props[kPlacementSurface] = mPlacementSurface;
        props[kPlacementSurfaceFixupStrength] = mPlacementSurfaceFixupStrength;
    }
    if (!mPlacementDensityDir.empty())
        props[kPlacementDensityDir] = mPlacementDensityDir.string();
    if (mUseAtmosphereLayout)
    {
        props[kAtmosphereLayout] = true;
        props[kPlacementOffset] = mAtmosphereLayout.offset;
        props[kPlacementRotation] = mAtmosphereLayout.rotation;
        props[kPlacementScale] = mAtmosphereLayout.scale;
        props[kPlacementThickness] = mAtmosphereLayout.thickness;
        props[kLayerStartHeight] = float3(mAtmosphereLayout.startHeight.x, mAtmosphereLayout.startHeight.y, mParams.cloudLayersStartHeight.z);
        props[kLayerCoverage] = float3(mAtmosphereLayout.amount.x, mAtmosphereLayout.amount.y, mParams.layerCoverage.z);
    }
    if (!mPlacementOutputDir.empty())
        props[kPlacementOutputDir] = mPlacementOutputDir.string();
    props[kGameOffset] = mGame.gameOffsetXY;
    props[kHotsure] = mGame.enableHotsure != 0;
    return props;
}

bool Cloudly::loadGameVolumes(const std::filesystem::path& dir)
{
    ref<Texture> textures[5];
    for (size_t i = 0; i < 5; i++)
    {
        // Base/Alto will be supplied by the generator at graph compile time.
        if (i == 0 && (mUseAtmosphereLayout || !mPlacementDensityDir.empty()))
            continue;
        const auto& v = kGameVolumes[i];
        std::vector<uint8_t> data;
        for (uint32_t mip = 0; mip < v.mips; mip++)
        {
            auto path = dir / fmt::format("{}_mip{}.bin", v.name, mip);
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            size_t expected = mipBytes(v.dim, v.format, mip);
            size_t size = f ? (size_t)f.tellg() : 0;
            if (!f || size != expected)
            {
                logWarning("Cloudly: game volume '{}' missing or wrong size ({} bytes, expected {}). Falling back to procedural clouds.",
                           path.string(), size, expected);
                mParams.useGameVolumes = 0;
                return false;
            }
            size_t offset = data.size();
            data.resize(offset + size);
            f.seekg(0);
            f.read(reinterpret_cast<char*>(data.data() + offset), size);
        }
        textures[i] = mpDevice->createTexture3D(v.dim.x, v.dim.y, v.dim.z, v.format, v.mips, data.data(), ResourceBindFlags::ShaderResource);
        textures[i]->setName(std::string("Cloudly::") + v.name);
    }
    mpGameBaseSdf = textures[0];
    mpGameDisp1 = textures[1];
    mpGameDisp2 = textures[2];
    mpGameCirrusSdf = textures[3];
    mpGameDispCirrus = textures[4];
    mParams.useGameVolumes = 1;
    mHistoryValid = false;
    logInfo("Cloudly: loaded game cloud volumes from '{}'.", dir.string());
    return true;
}

RenderPassReflection Cloudly::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addInput(kInDepth, "Hardware depth buffer (optional; without it every pixel is background)")
        .flags(RenderPassReflection::Field::Flags::Optional);
    r.addInput(kInColor, "Scene color (optional)").flags(RenderPassReflection::Field::Flags::Optional);
    r.addOutput(kOutColor, "Scene with sky and clouds").format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess);
    return r;
}

void Cloudly::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!mPlacementGenerated && (mUseAtmosphereLayout || !mPlacementDensityDir.empty()))
    {
        mpGameBaseSdf = generateCloudlyPlacement(
            pRenderContext, mpDevice, mPlacementDensityDir, mPlacementOutputDir, mPlacementSurface, mPlacementSurfaceFixupStrength,
            mUseAtmosphereLayout ? &mAtmosphereLayout : nullptr
        );
        if (mUseAtmosphereLayout)
        {
            mParams.cloudLayersStartHeight.x = mAtmosphereLayout.startHeight.x;
            mParams.cloudLayersStartHeight.y = mAtmosphereLayout.startHeight.y;
            mParams.layerCoverage.x = mAtmosphereLayout.amount.x;
            mParams.layerCoverage.y = mAtmosphereLayout.amount.y;
        }
        mPlacementGenerated = true;
        mHistoryValid = false;
    }
    allocateTargets(compileData.defaultTexDims);
}

void Cloudly::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mHistoryValid = false;
}

void Cloudly::allocateTargets(uint2 frameDim)
{
    if (all(frameDim == mFrameDim) && mpTraceColor && mpTraceColor->getWidth() == div_round_up(frameDim.x, mParams.traceDownscale))
        return;
    mFrameDim = frameDim;
    uint2 traceDim = div_round_up(frameDim, uint2(mParams.traceDownscale));
    mParams.traceDim = traceDim;

    const auto uavSrv = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpTraceColor = mpDevice->createTexture2D(traceDim.x, traceDim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, uavSrv);
    mpTraceDistance = mpDevice->createTexture2D(traceDim.x, traceDim.y, ResourceFormat::R32Float, 1, 1, nullptr, uavSrv);
    for (int i = 0; i < 2; i++)
    {
        mpHistoryColor[i] = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, uavSrv);
        mpHistoryDistance[i] = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::R32Float, 1, 1, nullptr, uavSrv);
    }
    mHistoryValid = false;
}

void Cloudly::bindCommon(const ShaderVar& var)
{
    var["CloudlyCB"]["gParams"].setBlob(mParams);
    var["CloudlyCB"]["gCam"].setBlob(mCamera);
    var["CloudlyCB"]["gGame"].setBlob(mGame);
    bool game = mParams.useGameVolumes != 0;
    var["gGameBaseSdf"] = game ? mpGameBaseSdf : mpDummy3D;
    var["gGameDisp1"] = game ? mpGameDisp1 : mpDummy3D;
    var["gGameDisp2"] = game ? mpGameDisp2 : mpDummy3D;
    var["gGameCirrusSdf"] = game ? mpGameCirrusSdf : mpDummy3D;
    var["gGameDispCirrus"] = game ? mpGameDispCirrus : mpDummy3D;
    var["gSdfSampler"] = mpSdfSampler;
    var["gDispSampler"] = mpDispSampler;
    var["gBaseNoise"] = mpBaseNoise;
    var["gDetailNoise"] = mpDetailNoise;
    var["gTransmittanceLut"] = mpTransmittanceLut;
    var["gMultiScatteringLut"] = mpMultiScatteringLut;
    var["gDistantSkyLightLut"] = mpDistantSkyLightLut;
    var["gLinearWrap"] = mpLinearWrap;
    var["gLinearClamp"] = mpLinearClamp;
}

void Cloudly::generateNoise(RenderContext* pRenderContext)
{
    auto var = mpNoisePass->getRootVar();
    var["NoiseCB"]["gSize"] = kBaseNoiseSize;
    var["NoiseCB"]["gDetail"] = 0u;
    var["gOutput"] = mpBaseNoise;
    mpNoisePass->execute(pRenderContext, uint3(kBaseNoiseSize));

    var["NoiseCB"]["gSize"] = kDetailNoiseSize;
    var["NoiseCB"]["gDetail"] = 1u;
    var["gOutput"] = mpDetailNoise;
    mpNoisePass->execute(pRenderContext, uint3(kDetailNoiseSize));
    mNoiseDirty = false;
}

void Cloudly::updateAtmosphereLuts(RenderContext* pRenderContext)
{
    // Rebuild when the sun or any atmosphere parameter changed.
    const auto& a = mParams;
    const auto& b = mLutParams;
    bool changed = mLutsDirty || any(a.lightSourceDirection != b.lightSourceDirection) || a.sunIntensity != b.sunIntensity ||
                   a.earthRadius != b.earthRadius || a.atmMaxAltitude != b.atmMaxAltitude || a.h0Rayleigh != b.h0Rayleigh ||
                   a.h0Mie != b.h0Mie || any(a.sigmaScatRayleigh != b.sigmaScatRayleigh) || any(a.sigmaScatMie != b.sigmaScatMie) ||
                   any(a.sigmaExtMie != b.sigmaExtMie) || any(a.sigmaExtOzone != b.sigmaExtOzone) || a.miePhaseG != b.miePhaseG ||
                   a.ozoneCenterHeight != b.ozoneCenterHeight || a.ozoneHalfThickness != b.ozoneHalfThickness ||
                   any(a.groundAlbedo != b.groundAlbedo) || a.multipleScatteringScale != b.multipleScatteringScale;
    if (!changed)
        return;

    FALCOR_PROFILE(pRenderContext, "AtmosphereLUTs");
    {
        auto var = mpTransmittanceLutPass->getRootVar();
        bindCommon(var);
        var["gOutput"] = mpTransmittanceLut;
        mpTransmittanceLutPass->execute(pRenderContext, uint3(kTransmittanceLutDim, 1));
    }
    {
        auto var = mpMultiScatteringLutPass->getRootVar();
        bindCommon(var);
        var["gOutput"] = mpMultiScatteringLut;
        mpMultiScatteringLutPass->execute(pRenderContext, uint3(kMultiScatteringLutDim, 1));
    }
    {
        auto var = mpDistantSkyLightPass->getRootVar();
        bindCommon(var);
        var["gOutput"] = mpDistantSkyLightLut;
        mpDistantSkyLightPass->execute(pRenderContext, uint3(kDistantSkyLightLutSize, 1, 1));
    }
    mLutParams = mParams;
    mLutsDirty = false;
}

void Cloudly::updateCamera()
{
    const auto& cam = mpScene->getCamera()->getData();
    mCamera.invViewProj = cam.invViewProj;
    mCamera.viewProj = cam.viewProjMatNoJitter;
    mCamera.prevViewProj = cam.prevViewProjMatNoJitter;
    mCamera.posW = cam.posW;
    mCamera.cameraU = cam.cameraU;
    mCamera.cameraV = cam.cameraV;
    mCamera.cameraW = cam.cameraW;
}

void Cloudly::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOut = renderData.getTexture(kOutColor);
    if (!mpScene || !mpScene->getCamera())
    {
        pRenderContext->clearUAV(pOut->getUAV().get(), float4(0.f));
        return;
    }

    const auto& pDepth = renderData.getTexture(kInDepth);
    const auto& pColor = renderData.getTexture(kInColor);
    uint2 frameDim = uint2(pOut->getWidth(), pOut->getHeight());
    allocateTargets(frameDim);

    // Time and wind.
    auto now = std::chrono::steady_clock::now();
    float dt = mFirstFrame ? 0.f : std::min(std::chrono::duration<float>(now - mLastTime).count(), 0.1f);
    mLastTime = now;
    mFirstFrame = false;
    mTimeSec += dt;
    float3 windDir = length(mWindDirection) > 0.f ? normalize(mWindDirection) : float3(0.f);
    mParams.windOffset += windDir * mWindSpeed * dt;

    // Sun from the scene's first directional light (its direction is the light's direction of travel).
    if (mUseSceneSun)
    {
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            const auto& pLight = mpScene->getLight(i);
            if (pLight->getType() == LightType::Directional && pLight->isActive())
            {
                mParams.lightSourceDirection = -normalize(static_cast<DirectionalLight*>(pLight.get())->getWorldDirection());
                break;
            }
        }
    }

    updateCamera();
    mParams.frameIndex = mFrameIndex;

    if (mNoiseDirty)
        generateNoise(pRenderContext);
    updateAtmosphereLuts(pRenderContext);

    const uint32_t ds = mParams.traceDownscale;
    const uint2 subPixel = subPixelForFrame(mFrameIndex, ds);
    const uint32_t hasDepth = pDepth ? 1u : 0u;

    {
        FALCOR_PROFILE(pRenderContext, "SkyTrace");
        auto var = mpTracePass->getRootVar();
        bindCommon(var);
        var["TraceCB"]["gFrameDim"] = frameDim;
        var["TraceCB"]["gSubPixel"] = subPixel;
        var["TraceCB"]["gHasDepth"] = hasDepth;
        var["gDepth"] = pDepth;
        var["gOutColor"] = mpTraceColor;
        var["gOutDistance"] = mpTraceDistance;
        mpTracePass->execute(pRenderContext, uint3(mParams.traceDim, 1));
    }

    const uint32_t cur = mHistoryIndex;
    const uint32_t prev = 1 - mHistoryIndex;
    {
        FALCOR_PROFILE(pRenderContext, "TemporalUpscale");
        auto var = mpUpscalePass->getRootVar();
        bindCommon(var);
        var["UpscaleCB"]["gFrameDim"] = frameDim;
        var["UpscaleCB"]["gSubPixel"] = subPixel;
        var["UpscaleCB"]["gHistoryValid"] = mHistoryValid ? 1u : 0u;
        var["UpscaleCB"]["gHistoryWeight"] = mHistoryWeight;
        var["gTraceColor"] = mpTraceColor;
        var["gTraceDistance"] = mpTraceDistance;
        var["gHistoryColor"] = mpHistoryColor[prev];
        var["gHistoryDistance"] = mpHistoryDistance[prev];
        var["gOutColor"] = mpHistoryColor[cur];
        var["gOutDistance"] = mpHistoryDistance[cur];
        mpUpscalePass->execute(pRenderContext, uint3(frameDim, 1));
    }

    {
        FALCOR_PROFILE(pRenderContext, "Composite");
        auto var = mpCompositePass->getRootVar();
        bindCommon(var);
        var["CompositeCB"]["gFrameDim"] = frameDim;
        var["CompositeCB"]["gHasDepth"] = hasDepth;
        var["CompositeCB"]["gHasSceneColor"] = pColor ? 1u : 0u;
        var["CompositeCB"]["gSceneAerialPerspective"] = mSceneAerialPerspective ? 1u : 0u;
        var["gSceneColor"] = pColor;
        var["gDepth"] = pDepth;
        var["gClouds"] = mpHistoryColor[cur];
        var["gOutput"] = pOut;
        mpCompositePass->execute(pRenderContext, uint3(frameDim, 1));
    }

    mHistoryIndex = prev;
    mHistoryValid = true;
    mFrameIndex++;
}

void Cloudly::renderUI(Gui::Widgets& widget)
{
    bool dirtyTargets = false;

    if (auto g = widget.group("Sun", true))
    {
        g.checkbox("Use scene directional light", mUseSceneSun);
        if (!mUseSceneSun)
        {
            bool changed = g.slider("Elevation (deg)", mSunElevationDeg, -10.f, 90.f);
            changed |= g.slider("Azimuth (deg)", mSunAzimuthDeg, -180.f, 180.f);
            if (changed)
                mParams.lightSourceDirection = dirFromElevationAzimuth(mSunElevationDeg, mSunAzimuthDeg);
        }
        g.var("Sun intensity", mParams.sunIntensity, 0.f, 1e6f, 0.1f);
        g.checkbox("Draw sky", mParams.drawSky);
        g.checkbox("Draw sun disk", mParams.drawSunDisk);
        g.checkbox("Aerial perspective on scene", mSceneAerialPerspective);
    }

    if (auto g = widget.group("Layers", true))
    {
        if (mpGameBaseSdf)
        {
            bool useGame = mParams.useGameVolumes != 0;
            if (g.checkbox("Use game cloud volumes", useGame))
            {
                mParams.useGameVolumes = useGame ? 1 : 0;
                mHistoryValid = false;
            }
            g.checkbox("Hotsure detail", mGame.enableHotsure);
            g.var("Game offset XY (m)", mGame.gameOffsetXY, -1e6f, 1e6f, 10.f);
        }
        else
        {
            g.text("Game volumes not loaded (set the 'gameDataDir' property).");
        }
        if (mUseAtmosphereLayout)
        {
            g.var("Cirrus start height (m)", mParams.cloudLayersStartHeight.z, 0.f, 20000.f, 10.f);
            g.slider("Procedural cirrus coverage", mParams.layerCoverage.z, 0.f, 1.f);
        }
        else
        {
            g.var("Start heights (m)", mParams.cloudLayersStartHeight, 0.f, 20000.f, 10.f);
            g.var("Base/alto thickness (m)", mParams.layerThickness, 100.f, 12000.f, 10.f);
            g.slider("Coverage", mParams.layerCoverage, 0.f, 1.f);
        }
        g.var("Cirrus thickness (m)", mParams.cirrusThickness, 10.f, 3000.f, 1.f);
        g.var("Density (top)", mParams.cloudLayersDensity, 0.f, 2.f, 0.001f);
        g.var("Density (bottom)", mParams.cloudLayersDensityBottom, 0.f, 2.f, 0.0001f);
        g.var("Edge fade (m)", mParams.cloudLayersEdgeFadeDistance, 0.f, 20000.f, 1.f);
        g.var("Edge fade bottom (m)", mParams.cloudLayersEdgeFadeDistanceBottom, 0.f, 2000.f, 1.f);
        g.var("Sigma scat (1/m)", mParams.sigmaScatCloud, 0.f, 1.f, 0.001f);
    }

    if (mUseAtmosphereLayout)
    {
        if (auto g = widget.group("Atmosphere placement", true))
        {
            g.var("Offset X/Z (m)", mAtmosphereLayout.offset, -200000.f, 200000.f, 10.f);
            g.var("Rotation (deg)", mAtmosphereLayout.rotation, -360.f, 360.f, 1.f);
            g.var("Horizontal scale", mAtmosphereLayout.scale, 0.01f, 100.f, 0.01f);
            g.var("Base/alto start height (m)", mAtmosphereLayout.startHeight, 0.f, 20000.f, 10.f);
            g.var("Base/alto thickness (m)", mAtmosphereLayout.thickness, 100.f, 12000.f, 10.f);
            g.slider("Base/alto amount", mAtmosphereLayout.amount, 0.f, 1.f);
            if (g.button("Rebuild placement"))
            {
                mPlacementGenerated = false;
                requestRecompile();
            }
        }
    }

    if (auto g = widget.group("Shape"))
    {
        g.var("Weather scale (m)", mParams.weatherScale, 1000.f, 500000.f, 100.f);
        g.var("Base noise scale (m)", mParams.baseNoiseScale, 100.f, 100000.f, 10.f);
        g.var("Detail noise scale (m)", mParams.detailNoiseScale, 10.f, 10000.f, 1.f);
        g.slider("Detail erosion", mParams.detailErosion, 0.f, 1.f);
        g.var("SDF scale (m)", mParams.sdfScale, 10.f, 10000.f, 1.f);
        g.var("Cirrus stretch", mParams.cirrusStretch, 1.f, 20.f, 0.1f);
        g.var("Wind speed (m/s)", mWindSpeed, 0.f, 500.f, 0.5f);
        g.var("Wind direction", mWindDirection, -1.f, 1.f, 0.01f);
    }

    if (auto g = widget.group("Lighting"))
    {
        g.var("HG G1", mParams.hgG1, -0.99f, 0.99f, 0.01f);
        g.var("HG G2", mParams.hgG2, -0.99f, 0.99f, 0.01f);
        g.var("HG forward scale", mParams.phaseScaleHG, 0.f, 4.f, 0.01f);
        g.var("Backscattering scale", mParams.backscatteringScaleFactor, 0.f, 10.f, 0.01f);
        g.var("Backscattering intensity", mParams.backscatteringIntensity, 0.f, 20.f, 0.01f);
        g.var("Min G", mParams.minG, -0.99f, 0.99f, 0.01f);
        g.var("Max G", mParams.maxG, -0.99f, 0.99f, 0.01f);
        g.checkbox("Dark edge", mParams.useDarkEdge);
        g.var("Scattering octaves", mParams.cloudScatteringTimes, 1u, 8u);
        g.var("MS attenuation", mParams.msAttenuation, 0.f, 1.f, 0.01f);
        g.var("MS contribution", mParams.msContribution, 0.f, 1.f, 0.01f);
        g.var("MS eccentricity attenuation", mParams.msEccentricityAttenuationForG, 0.f, 1.f, 0.01f);
        g.var("Ambient intensity", mParams.cloudSkyAmbientLightIntensity, 0.f, 20.f, 0.01f);
        g.rgbColor("Ambient overlay color", mParams.cloudSkyAmbientLightOverlayColor);
        g.var("Ambient bottom occlusion", mParams.cloudSkyAmbientLightBottomOcclusion, 0.f, 1.f, 0.01f);
        g.var("Bottom AO start height", mParams.cloudBottomAOStartHeight, 0.f, 1.f, 0.001f);
    }

    if (auto g = widget.group("Tracing"))
    {
        uint32_t ds = mParams.traceDownscale;
        if (g.var("Trace downscale", ds, 1u, 4u))
        {
            mParams.traceDownscale = ds;
            dirtyTargets = true;
        }
        g.var("Max steps", mParams.traceMaxSteps, 16u, 2048u);
        g.var("Min step near (m)", mParams.traceMinStepSmallest, 1.f, 1000.f, 1.f);
        g.var("Min step far (m)", mParams.traceMinStepSmallestFar, 1.f, 5000.f, 1.f);
        g.var("Max trace distance (m)", mParams.cloudMaxTraceDistance, 1000.f, 2e6f, 1000.f);
        g.var("Fade begin distance (m)", mParams.cloudFadeBeginDistance, 0.f, 2e6f, 1000.f);
        g.slider("History weight", mHistoryWeight, 0.f, 0.98f);
    }

    if (auto g = widget.group("World"))
    {
        g.var("Meters per unit", mParams.metersPerUnit, 1e-4f, 1e4f, 0.01f);
        g.var("Origin altitude (m)", mParams.cameraAltitude, -1000.f, 100000.f, 1.f);
    }

    if (auto g = widget.group("Atmosphere"))
    {
        g.var("Rayleigh scattering (1/m)", mParams.sigmaScatRayleigh, 0.f, 1e-3f, 1e-7f, false, "%.3e");
        g.var("Rayleigh scale height (m)", mParams.h0Rayleigh, 100.f, 50000.f, 10.f);
        g.var("Mie scattering (1/m)", mParams.sigmaScatMie, 0.f, 1e-3f, 1e-7f, false, "%.3e");
        g.var("Mie extinction (1/m)", mParams.sigmaExtMie, 0.f, 1e-3f, 1e-7f, false, "%.3e");
        g.var("Mie scale height (m)", mParams.h0Mie, 10.f, 20000.f, 10.f);
        g.var("Mie g", mParams.miePhaseG, 0.f, 0.99f, 0.01f);
        g.var("Ozone extinction (1/m)", mParams.sigmaExtOzone, 0.f, 1e-4f, 1e-8f, false, "%.3e");
        g.rgbColor("Ground albedo", mParams.groundAlbedo);
    }

    if (dirtyTargets)
        mFrameDim = uint2(0); // Reallocated at the next execute().
}
