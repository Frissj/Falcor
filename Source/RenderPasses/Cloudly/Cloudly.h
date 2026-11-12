#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/ComputePass.h"
#include "CloudlyParams.slang"

using namespace Falcor;

/**
 * Volumetric clouds and sky modelled on the Cloudly UE plugin.
 *
 * Pipeline (Cloudly pass names in brackets):
 *   1. Noise generation, once                      [procedural detail or generated placement]
 *   2. Atmosphere LUTs when the sun moves           [OpticalDepthTex, MultiScatteringLut, DistantSkyLightLut]
 *   3. Low-res checkerboard cloud trace             [SkyTraceCS]
 *   4. Temporal upscale with history                [CheckerBoardResolveCS]
 *   5. Sky, sun disk, aerial perspective, clouds    [CompositeAtmosphereCS]
 */
class Cloudly : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(Cloudly, "Cloudly", "Cloudly-style volumetric clouds and sky.");

    static ref<Cloudly> create(ref<Device> pDevice, const Properties& props) { return make_ref<Cloudly>(pDevice, props); }

    Cloudly(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    void generateNoise(RenderContext* pRenderContext);
    void updateAtmosphereLuts(RenderContext* pRenderContext);
    void allocateTargets(uint2 frameDim);
    void bindCommon(const ShaderVar& var);
    void updateCamera();
    bool loadGameVolumes(const std::filesystem::path& dir);

    ref<Scene> mpScene;
    CloudlyParams mParams;
    CloudlyGameData mGame;
    std::filesystem::path mGameDataDir; ///< Folder with the CloudTex_*_mip0.bin volumes dumped from a capture.
    std::filesystem::path mPlacementDensityDir;
    std::filesystem::path mPlacementOutputDir;
    float mPlacementSurface = 0.5f;
    float mPlacementSurfaceFixupStrength = 1.f;
    bool mPlacementGenerated = false;
    ref<Texture> mpGameBaseSdf;
    ref<Texture> mpGameDisp1;
    ref<Texture> mpGameDisp2;
    ref<Texture> mpGameCirrusSdf;
    ref<Texture> mpGameDispCirrus;
    ref<Texture> mpDummy3D;
    ref<Sampler> mpSdfSampler;
    ref<Sampler> mpDispSampler;
    CloudlyCamera mCamera;
    bool mUseSceneSun = false;      ///< Take the sun direction from the scene's first directional light.
    bool mSceneAerialPerspective = true;
    float mWindSpeed = 0.f;         ///< m/s. The captured wind is already baked into the DISP*_UVOffset constants.
    float3 mWindDirection = float3(1.f, 0.f, 0.3f);
    float mHistoryWeight = 0.9f;
    float mSunElevationDeg = 0.f;   ///< UI helpers, derived from lightSourceDirection.
    float mSunAzimuthDeg = 0.f;

    ref<ComputePass> mpNoisePass;
    ref<ComputePass> mpTransmittanceLutPass;
    ref<ComputePass> mpMultiScatteringLutPass;
    ref<ComputePass> mpDistantSkyLightPass;
    ref<ComputePass> mpTracePass;
    ref<ComputePass> mpUpscalePass;
    ref<ComputePass> mpCompositePass;

    ref<Texture> mpBaseNoise;
    ref<Texture> mpDetailNoise;
    ref<Texture> mpTransmittanceLut;
    ref<Texture> mpMultiScatteringLut;
    ref<Texture> mpDistantSkyLightLut;
    ref<Texture> mpTraceColor;
    ref<Texture> mpTraceDistance;
    ref<Texture> mpHistoryColor[2];
    ref<Texture> mpHistoryDistance[2];
    ref<Sampler> mpLinearWrap;
    ref<Sampler> mpLinearClamp;

    uint2 mFrameDim = uint2(0);
    uint32_t mFrameIndex = 0;
    uint32_t mHistoryIndex = 0;
    bool mHistoryValid = false;
    bool mNoiseDirty = true;
    bool mLutsDirty = true;
    CloudlyParams mLutParams; ///< Parameters the LUTs were last built with.
    double mTimeSec = 0.0;
    std::chrono::steady_clock::time_point mLastTime;
    bool mFirstFrame = true;
};
