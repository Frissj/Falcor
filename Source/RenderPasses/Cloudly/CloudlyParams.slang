/***************************************************************************
 # Cloudly-style volumetric clouds for Falcor.
 #
 # Parameter names mirror the Cloudly UE plugin constant buffers
 # (CloudConstantsUniform / SkyConstantsUniform, "Data_" prefix dropped).
 # Default values are the ones read from a RenderDoc capture of a shipping
 # Cloudly title (frame 21779); see README.md for the provenance of each value.
 #
 # All distances are in metres. World units are converted with metersPerUnit.
 # The planet is a sphere of radius earthRadius whose centre lies straight below
 # the world origin; the world origin sits at altitude cameraAltitude
 # (see planetPos() in CloudlyCommon.slang).
 #
 # Shared between host and device: 16-byte packing, no bools.
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct CloudlyCamera
{
    float4x4 invViewProj;
    float4x4 viewProj;     ///< Current frame view-projection, no jitter.
    float4x4 prevViewProj; ///< Previous frame view-projection, no jitter.
    float3 posW;
    float _pad0;
    float3 cameraU;
    float _pad1;
    float3 cameraV;
    float _pad2;
    float3 cameraW;
    float _pad3;
};

struct CloudlyParams
{
    // --- Light ------------------------------------------------------------
    float3 lightSourceDirection = float3(-0.317133f, 0.797791f, 0.512793f); ///< LightSourceDirection (UE z-up (x,y,z) -> Falcor (x,z,y)), towards the sun.
    float sunIntensity = 8.4f; ///< SunIntensityWithHackScale (120000) x UnrealPreExposure (7e-5).

    // --- Planet / atmosphere (SkyConstantsUniform) --------------------------
    float earthRadius = 6360000.f;  ///< EarthRadius
    float atmMaxAltitude = 60000.f; ///< AtmMaxAltitude
    float h0Rayleigh = 8000.f;      ///< 1 / H0_RayleighRcp
    float h0Mie = 1200.f;           ///< 1 / H0_MieRcp

    float3 sigmaScatRayleigh = float3(4e-6f, 1e-5f, 2.5e-5f); ///< SigmaScatRayleigh, 1/m
    float miePhaseG = 0.8f;                                    ///< MiePhaseG

    float3 sigmaExtMie = float3(3.1e-5f); ///< SigmaExtMie, 1/m
    float ozoneCenterHeight = 20000.f;    ///< OzoneCenterHeight

    float3 sigmaScatMie = float3(1e-5f); ///< SigmaScatMie, 1/m
    float ozoneHalfThickness = 20000.f;  ///< 1 / OzoneHalfThicknessRcp

    float3 sigmaExtOzone = float3(1e-6f, 2e-6f, 0.f); ///< SigmaExtOzone, 1/m
    float multipleScatteringScale = 1.f;               ///< MultipleScatteringScale

    float3 groundAlbedo = float3(0.3f); ///< GroundAlbedo
    float sunDiskCosine = 0.999989f;    ///< SunDiskCosine

    // --- Cloud layers (CloudConstantsUniform) -------------------------------
    // x = base (cumulus), y = alto, z = cirrus.
    float3 cloudLayersStartHeight = float3(600.f, 3500.f, 12500.f); ///< CloudLayersStartHeight
    float cirrusThickness = 294.139f;                                ///< CirrusThickness

    float3 cloudLayersDensity = float3(0.299363f, 0.25f, 1.f); ///< CloudLayersDensity (density at the top of the transition)
    float sigmaScatCloud = 0.03f;                               ///< SigmaScatCloud, 1/m per unit density

    float2 cloudLayersDensityBottom = float2(0.25f, 0.0005f);              ///< CloudLayersDensityBottom
    float2 cloudLayersEdgeFadeDistanceBottom = float2(200.388f, 73.6844f); ///< CloudLayersEdgeFadeDistanceBottom

    float4 cloudLayersEdgeFadeDistance = float4(20.f, 20.f, 416.270f, 8565.83f); ///< CloudLayersEdgeFadeDistance (base, alto, cirrus SDF, cirrus 2D)

    float2 cloudLayersTransitionStartDensity = float2(0.0001f, 775.610f); ///< CloudLayersTransitionStartDensity (m above layer start)
    float2 cloudLayersTransitionEnd = float2(460.526f, 1319.69f);         ///< CloudLayersTransitionEnd

    float2 layerThickness = float2(4500.f, 2200.f); ///< Procedural shape: vertical extent of base and alto (not in the capture, the game uses SDF volumes).
    float cloudMaxHeightForBaseAltoTrace = 11887.6f; ///< CloudMaxHeightForBaseAltoTrace
    float cloudFadeBeginDistance = 237500.f;         ///< CloudFadeBeginDistance

    float cloudMaxTraceDistance = 250000.f;       ///< CloudMaxTraceDistance
    float cloudMaxTraceDistanceCirrus = 850000.f; ///< CloudMaxTraceDistanceCirrus
    float cirrusFarFadeDistance = 510000.f;       ///< CirrusFarFadeDistance
    float cirrusFarFadeMinOpacity = 0.15f;        ///< CirrusFarFadeMinOpacity

    // --- Procedural shape (replaces the game's SDF / displacement textures) ---
    float3 windOffset = float3(0.f); ///< Accumulated wind, metres.
    float weatherScale = 40000.f;    ///< World size of one tile of the coverage map.

    float baseNoiseScale = 9000.f;   ///< World size of one tile of the base shape noise.
    float detailNoiseScale = 700.f;  ///< World size of one tile of the detail noise.
    float detailErosion = 0.35f;
    float sdfScale = 1200.f; ///< Metres of pseudo-SDF depth per unit of shape value (drives edge fade and dark edge).

    float3 layerCoverage = float3(0.5f, 0.3f, 0.45f);
    float cirrusStretch = 4.f;

    // --- Tracing (SkyConstantsUniform) ---------------------------------------
    float traceMinStepSmallest = 16.f;     ///< TraceMinStepSmallest
    float traceMinStepSmallestFar = 600.f; ///< TraceMinStepSmallestFar
    float minStepSmallestFarAlpha = 6e-6f; ///< MinStepSmallestFar_Alpha
    float cloudBailOutTransmittanceLogViewRay = 3.079114f; ///< CloudBailOutTransmittanceLogViewRay

    float cloudBailOutTransmittanceLogSunRay = 3.506558f; ///< CloudBailOutTransmittanceLogSunRay
    float sunRayStepFirstLen = 20.58206f;                 ///< SunRayStepFirstLen
    float sunRayStepBase = 2.f;                           ///< SunRayStepBase
    float sunRayStepJitterVariance = 0.8f;                ///< SunRayStepJitterVariance

    float2 sunRayStepFirstLenAsSdfMinMax = float2(5.f, 25.438366f); ///< SunRayStepFirstLenAsSDF_Min / _Max
    float sunRayStepFirstLenAsSdfFraction = 2.f;                     ///< SunRayStepFirstLenAsSDF_Fraction
    uint traceMaxSteps = 384;

    // --- Cloud lighting (SkyConstantsUniform) -------------------------------
    float hgG1 = 0.2f;        ///< HGPhaseFunction_G1
    float hgG2 = 0.3f;        ///< HGPhaseFunction_G2 (used as 0.99 - G2)
    float hgG2Cirrus = 0.3f;  ///< HGPhaseFunction_G2_Cirrus
    float phaseScaleHG = 0.5f; ///< CloudPhaseFunctionScale_HG

    float phaseScaleHGCirrus = 0.5f;      ///< CloudPhaseFunctionScale_HG_Cirrus
    float phaseScale = 1.f;               ///< CloudPhaseFunctionScale
    float backscatteringScaleFactor = 2.f; ///< BackscatteringScaleFactor
    float backscatteringIntensity = 3.076923f; ///< BackscatteringIntensity

    float minG = -0.15f;     ///< MinG
    float maxG = 0.85f;      ///< MaxG
    float sunRayTrScale = 1.f; ///< SunRayTrScale
    float darkEdgeRange = 100.f; ///< DarkEdgeRange

    float2 darkEdgeSdfBias = float2(0.326352f, 0.326352f); ///< DarkEdgeSDF_Bias
    uint useDarkEdge = 1;          ///< bUseDarkEdge
    uint cloudScatteringTimes = 4; ///< CloudScatteringTimes

    float msAttenuation = 0.5f;                 ///< MS_Attenuation
    float msContribution = 0.5f;                ///< MS_Contribution
    float msEccentricityAttenuationForG = 1.f;  ///< MS_EccentricityAttenuationForG
    float slShadowOcclusionScale = 1.f;         ///< SLShadowOcclusionScale (SLShadowOcclusion = 1)

    float3 cloudSkyAmbientLightOverlayColor = float3(0.f, 0.484581f, 1.f); ///< CloudSkyAmbientLightOverlayColor (ColorMode 0)
    float cloudSkyAmbientLightIntensity = 3.718518f;                        ///< CloudSkyAmbientLightIntensity

    float cloudSkyAmbientLightBottomOcclusion = 0.88963f; ///< CloudSkyAmbientLightBottomOcclusion
    float cloudBottomAOStartHeight = 0.091358f;           ///< CloudBottomAOStartHeight
    float aerialPerspectiveStartDistance = 200.f;         ///< AerialPerspectiveStartDistance
    float metersPerUnit = 1.f;

    // --- Misc ----------------------------------------------------------------
    float cameraAltitude = 0.f; ///< Altitude (m) of the world origin above the ground.
    uint frameIndex = 0;
    uint drawSky = 1;
    uint drawSunDisk = 1;

    uint2 traceDim = uint2(0);
    uint traceDownscale = 2;
    uint useGameVolumes = 0; ///< 1: shapes come from the game SDF / displacement volumes (CloudlyGameData).
};

/**
 * Placement / displacement constants for the game cloud volumes (captured
 * CloudTex_* root parameters and DisplaceParams*). Positions are game "flat
 * map" coordinates in metres: x, y horizontal (Falcor x, z), z altitude.
 */
struct CloudlyGameData
{
    float3 baseUVOffset = float3(100000.f, 100000.f, 0.f);                    ///< CloudTex_BASE_UVOffset
    float baseTexelWorldSize = 390.625f;                                       ///< CloudTex_BASE_TexelWorldSizeInMeters
    float3 baseUVScale = float3(4.999999873689376e-06f, 4.999999873689376e-06f, 3.9999998989515007e-05f); ///< CloudTex_BASE_UVScale
    float sdfOffsetBase = 292.45703125f;                                       ///< CloudTex_SDF_Offset_BASE

    float3 disp1UVOffset = float3(100000.f, 100000.f, 0.f);
    float sdfOffsetAlto = 292.45703125f;                                       ///< CloudTex_SDF_Offset_ALTO
    float3 disp1UVScale = float3(6.774211942683905e-05f, 6.774211942683905e-05f, 0.0007225826266221702f);
    float cirrusTexelWorldSize = 195.3125f;

    float3 disp2UVOffset = float3(100280.328125f, 100000.f, 0.f);
    float sdfOffsetCirrus = 0.f;
    float3 disp2UVScale = float3(0.0010416667209938169f, 0.0010416667209938169f, 0.004166666883975267f);
    float sdfOffsetCirrusFar = 0.f;

    float3 cirrusUVOffset = float3(100000.f, 100000.f, -12500.f);
    float cirrusZeroDisplacementFade = 0.5f;                                   ///< CirrusZeroDisplacementFade
    float3 cirrusUVScale = float3(4.999999873689376e-06f, 4.999999873689376e-06f, 0.0025599999353289604f);
    float cirrusFarTransitionStart = 130000.f;                                 ///< CirrusFarTransitionStartDistance

    float3 dispCirrusUVOffset = float3(100019.8125f, 100000.f, 0.f);
    float cirrusFarTransitionEnd = 260000.f;                                   ///< CirrusFarTransitionEndDistance
    float3 dispCirrusUVScale = float3(2.8301821657805704e-05f, 2.8301821657805704e-05f, 0.001811316586099565f);
    float _pad0 = 0.f;

    // DisplaceParams1 / 2 (Base, Alto) and Cirrus.
    float4 disp1Base = float4(4451.38916f, 1665.34314f, 581.338074f, 0.f);    ///< MaxDisplacementDistance, SkipDisplacementSamplingDistance, Deflate, -
    float4 disp1BaseFade = float4(425.735321f, 3848.873779f, -200000.f, 0.f); ///< BaseSurfaceFadeStart, End, TransitionOffset, -
    float4 disp1Alto = float4(4451.38916f, 4006.250244f, 581.338074f, 0.f);
    float4 disp1AltoFade = float4(425.735321f, 3848.873779f, 0.f, 0.f);
    float4 disp2Base = float4(1027.587036f, 195.846008f, 0.f, 3.176317f);     ///< Max, Skip, Deflate, PreviousDisplacementFadeStrength
    float4 disp2BaseFade = float4(10.27587f, 30.853268f, 0.f, 0.f);
    float4 disp2Alto = float4(1027.587036f, 1175.075928f, 0.f, 0.f);
    float4 disp2AltoFade = float4(10.27587f, 30.853268f, 0.f, 0.f);
    float4 disp1Cirrus = float4(5000.f, 200000.f, 2603.712646f, 0.f);         ///< Max, Skip, Deflate, -

    float3 cloudLayersSoftRadius = float3(12.701f, 1.f, 10.f);                ///< CloudLayersSoftRadius
    float _pad1 = 0.f;
    float2 cloudLayersTransitionStart = float2(0.0001f, 319.689117f);          ///< CloudLayersTransitionStart
    float2 cloudLayersBottomBleedDisp1 = float2(1.f, 5.f);                     ///< CloudLayersBottomBleedDisp1
    float2 cloudLayersBottomBleedDisp2 = float2(1.f, 6.f);                     ///< CloudLayersBottomBleedDisp2
    float2 gameOffsetXY = float2(0.f, 0.f);                                    ///< Added to Falcor (x, z) * metersPerUnit to get game x, y.

    // HotsureEffect* (additive near-camera detail).
    float2 hotsureLayerRemap0 = float2(0.068947f, 0.f);
    float2 hotsureLayerRemap1 = float2(1.355063f, 0.f);
    float2 hotsureAnimOffsetPreScaled = float2(2.585592f, 0.f);
    float hotsureHighBaseDensityDecay = 2.f;
    float hotsureLowBaseDensityDecay = 11.22541f;
    float hotsurePositionScaleRcp = 0.026028f;
    float hotsureFrequency = 2.233694f;
    float hotsurePersistence = 1.f;
    float hotsureWhirl = 0.721341f;
    float hotsureZScale = 1.f;
    float hotsureFadeDistanceStart = 1000.f;
    float hotsureFadeDistanceEnd = 3000.f;
    uint enableHotsure = 1;

    float4 texelWorldSize = float4(57.66353225708008f, 15.f, 69.01057434082031f, 0.f); ///< DISP1, DISP2, DISP_CIRRUS texel size (m).
    float maxMip = 2.f;           ///< Highest mip loaded for the game volumes.
    float mipTraceScale = 0.75f;  ///< Cone footprint uses a trace height of frameHeight * this (the plugin traces at 0.75x).
    float2 _pad2 = float2(0.f);
};

END_NAMESPACE_FALCOR
