/***************************************************************************
 # Reference volumetric path tracer for NanoVDB grid volumes.
 **************************************************************************/
#include "VolumePathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/NumericRange.h"
#include <algorithm>
#include <cstring>
#include <execution>
#include <limits>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VolumePathTracer>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/VolumePathTracer/VolumePathTracer.cs.slang";

const ChannelList kOutputChannels = {
    // clang-format off
    { "color", "gOutputColor", "Output color (radiance)", false, ResourceFormat::RGBA32Float },
    { "work",  "gWorkDebug",   "Raw work counters: R=segments G=overlapSteps B=instSlabs A=maxCover", true, ResourceFormat::RGBA32Float },
    // Demodulated split (spec section 6). The three channels below decompose
    // 'color' EXACTLY: color = Lin + transmittance * background, per sample.
    //  - Lin: radiance of paths that scattered at least once (the cloud's own
    //    light). This is the only channel a (spatial) denoiser may touch.
    //  - transmittance: binary sample of primary-ray transmittance (did the
    //    first free flight escape without a real collision). Converges to T
    //    under TEMPORAL accumulation only; never spatially filtered, so the
    //    wisp silhouette survives reconstruction.
    //  - background: env radiance for the primary direction. Deterministic,
    //    zero noise; recombined last as accum(T) * bg.
    { "Lin",           "gOutputLin", "In-scattered radiance only (denoise this)",                     true, ResourceFormat::RGBA32Float },
    { "transmittance", "gOutputT",   "Primary transmittance sample (temporal-only convergence)",     true, ResourceFormat::RGBA32Float },
    { "background",    "gOutputBg",  "Deterministic environment radiance along the primary ray",     true, ResourceFormat::RGBA32Float },
    // Raw RIS internals, no thresholds. R = which branch ended the RIS block
    // (see kRisExit* in the shader), G = candidates generated, B = target
    // weight sum, A = the RIS weight W. Read exact values with the pixel
    // inspector; R alone identifies where the contribution is being lost.
    { "risDebug",      "gRisDebug",  "Raw RIS probe: R=exitCode G=count B=wSum A=W",                 true, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kUseNEE[] = "useNEE";
const char kUseRussianRoulette[] = "useRussianRoulette";
const char kTransmittanceEstimator[] = "transmittanceEstimator";
const char kDistanceSampler[] = "distanceSampler";
const char kResidualMip[] = "residualMip";
const char kUseRIS[] = "useRIS";
const char kRisCandidates[] = "risCandidates";
const char kRisMip[] = "risMip";
const char kUseSharedCandidateSweep[] = "useSharedCandidateSweep";
const char kFootprintMip[] = "footprintMip";
const char kFootprintScale[] = "footprintScale";
const char kUseSingleNeePerPath[] = "useSingleNeePerPath";
const char kUseAdaptiveM[] = "useAdaptiveM";
const char kUseTemporalReuse[] = "useTemporalReuse";
const char kTemporalMCap[] = "temporalMCap";
const char kUseBrickTlas[] = "useBrickTlas";
const char kUseOccupancySkip[] = "useOccupancySkip";
const char kUseBrickPrefetch[] = "useBrickPrefetch";
const char kUseScatterSort[] = "useScatterSort";
const char kMipPixelThreshold[] = "mipPixelThreshold";
const char kUseMergedTail[] = "useMergedTail";
const char kTailRes[] = "tailRes";
const char kTailGateVoxels[] = "tailGateVoxels";
const char kLogWorkStats[] = "logWorkStats";
const char kLogRisStats[] = "logRisStats";
const char kOutputSize[] = "outputSize";
const char kFixedOutputSize[] = "fixedOutputSize";
const char kRisStatsInterval[] = "risStatsInterval";
const char kUseSpatialReuse[] = "useSpatialReuse";
const char kSpatialNeighbors[] = "spatialNeighbors";
const char kSpatialRadiusPx[] = "spatialRadiusPx";
const char kRisTargetFloor[] = "risTargetFloor";
const char kRouletteMinQ[] = "rouletteMinQ";
const char kRouletteStartBounce[] = "rouletteStartBounce";
const char kUseCompaction[] = "useCompaction";
const char kUseWavefront[] = "useWavefront";
const char kWavefrontRounds[] = "wavefrontRounds";
const char kUseTauCache[] = "useTauCache";
const char kTrRRThreshold[] = "trRRThreshold";
const char kTrRRMode[] = "trRRMode";
const char kSeedOffset[] = "seedOffset";
const char kUseRadCache[] = "useRadCache";
const char kRadCacheRes[] = "radCacheRes";
const char kRadCutBounce[] = "radCutBounce";
const char kRadResidualSurvival[] = "radResidualSurvival";
const char kRadWarpRRLanes[] = "radWarpRRLanes";
const char kRadTrainEvery[] = "radTrainEvery";
const char kRadEma[] = "radEma";
const char kTauCacheRes[] = "tauCacheRes";
const char kTauCacheInterval[] = "tauCacheInterval";
} // namespace

VolumePathTracer::VolumePathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void VolumePathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kUseNEE)
            mUseNEE = value;
        else if (key == kUseRussianRoulette)
            mUseRussianRoulette = value;
        else if (key == kTransmittanceEstimator)
            mVolumeSamplerOptions.transmittanceEstimator = value;
        else if (key == kDistanceSampler)
            mVolumeSamplerOptions.distanceSampler = value;
        else if (key == kResidualMip)
            mVolumeSamplerOptions.residualMip = value;
        else if (key == kUseRIS)
            mUseRIS = value;
        else if (key == kRisCandidates)
            mRisCandidates = value;
        else if (key == kRisMip)
            mRisMip = value;
        else if (key == kUseSharedCandidateSweep)
            mUseSharedCandidateSweep = value;
        else if (key == kFootprintMip)
            mFootprintMip = value;
        else if (key == kFootprintScale)
            mFootprintScale = value;
        else if (key == kUseSingleNeePerPath)
            mUseSingleNeePerPath = value;
        else if (key == kUseAdaptiveM)
            mUseAdaptiveM = value;
        else if (key == kUseTemporalReuse)
            mUseTemporalReuse = value;
        else if (key == kTemporalMCap)
            mTemporalMCap = value;
        else if (key == kUseBrickTlas)
            mUseBrickTlas = value;
        else if (key == kUseOccupancySkip)
            mUseOccupancySkip = value;
        else if (key == kUseBrickPrefetch)
            mUseBrickPrefetch = value;
        else if (key == kUseScatterSort)
            mUseScatterSort = value;
        else if (key == kMipPixelThreshold)
            mMipPixelThreshold = value;
        else if (key == kUseMergedTail)
            mUseMergedTail = value;
        else if (key == kTailRes)
            mTailRes = value;
        else if (key == kTailGateVoxels)
            mTailGateVoxels = value;
        else if (key == kLogWorkStats)
            mLogWorkStats = value;
        else if (key == kLogRisStats)
            mLogRisStats = value;
        else if (key == kOutputSize)
            mOutputSizeSelection = value;
        else if (key == kFixedOutputSize)
            mFixedOutputSize = value;
        else if (key == kRisStatsInterval)
            mRisStatsInterval = std::max(1u, (uint32_t)value);
        else if (key == kUseSpatialReuse)
            mUseSpatialReuse = value;
        else if (key == kSpatialNeighbors)
            mSpatialNeighbors = std::clamp((uint32_t)value, 1u, 8u);
        else if (key == kSpatialRadiusPx)
            mSpatialRadiusPx = value;
        else if (key == kRisTargetFloor)
            mRisTargetFloor = value;
        else if (key == kRouletteMinQ)
            mRouletteMinQ = std::clamp((float)value, 0.f, 1.f);
        else if (key == kRouletteStartBounce)
            mRouletteStartBounce = (uint32_t)value;
        else if (key == kUseCompaction)
            mUseCompaction = value;
        else if (key == kUseWavefront)
            mUseWavefront = value;
        else if (key == kWavefrontRounds)
            mWavefrontRounds = value;
        else if (key == kUseTauCache)
            mUseTauCache = value;
        else if (key == kTrRRThreshold)
            mTrRRThreshold = std::clamp((float)value, 0.f, 0.5f);
        else if (key == kTrRRMode)
            // 31 not 15: bit4 = decorrelated RR coin (run 130 lesson - this
            // clamp silently ate mode 24 and made the bit4 A/B a no-op).
            mTrRRMode = std::clamp((uint32_t)value, 0u, 31u);
        else if (key == kSeedOffset)
            mSeedOffset = value;
        else if (key == kUseRadCache)
            mUseRadCache = value;
        else if (key == kRadCacheRes)
            mRadCacheRes = std::clamp((uint32_t)value, 8u, 512u);
        else if (key == kRadCutBounce)
            mRadCutBounce = value;
        else if (key == kRadResidualSurvival)
            mRadResidualSurvival = std::clamp((float)value, 0.01f, 1.f);
        else if (key == kRadWarpRRLanes)
            mRadWarpRRLanes = std::clamp((uint32_t)value, 0u, 32u);
        else if (key == kRadTrainEvery)
            mRadTrainEvery = std::max(1u, (uint32_t)value);
        else if (key == kRadEma)
            mRadEma = std::clamp((float)value, 0.001f, 1.f);
        else if (key == kTauCacheRes)
            mTauCacheRes = std::clamp((uint32_t)value, 8u, 512u);
        else if (key == kTauCacheInterval)
            mTauCacheInterval = value;
        else
            logWarning("Unknown property '{}' in VolumePathTracer properties.", key);
    }
}

Properties VolumePathTracer::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kUseNEE] = mUseNEE;
    props[kUseRussianRoulette] = mUseRussianRoulette;
    props[kTransmittanceEstimator] = mVolumeSamplerOptions.transmittanceEstimator;
    props[kDistanceSampler] = mVolumeSamplerOptions.distanceSampler;
    props[kResidualMip] = mVolumeSamplerOptions.residualMip;
    props[kUseRIS] = mUseRIS;
    props[kRisCandidates] = mRisCandidates;
    props[kRisMip] = mRisMip;
    props[kUseSharedCandidateSweep] = mUseSharedCandidateSweep;
    props[kFootprintMip] = mFootprintMip;
    props[kFootprintScale] = mFootprintScale;
    props[kUseSingleNeePerPath] = mUseSingleNeePerPath;
    props[kUseAdaptiveM] = mUseAdaptiveM;
    props[kUseTemporalReuse] = mUseTemporalReuse;
    props[kTemporalMCap] = mTemporalMCap;
    props[kUseBrickTlas] = mUseBrickTlas;
    props[kUseOccupancySkip] = mUseOccupancySkip;
    props[kUseBrickPrefetch] = mUseBrickPrefetch;
    props[kUseScatterSort] = mUseScatterSort;
    props[kMipPixelThreshold] = mMipPixelThreshold;
    props[kUseMergedTail] = mUseMergedTail;
    props[kTailRes] = mTailRes;
    props[kTailGateVoxels] = mTailGateVoxels;
    props[kLogWorkStats] = mLogWorkStats;
    props[kLogRisStats] = mLogRisStats;
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        props[kFixedOutputSize] = mFixedOutputSize;
    props[kRisStatsInterval] = mRisStatsInterval;
    props[kUseSpatialReuse] = mUseSpatialReuse;
    props[kSpatialNeighbors] = mSpatialNeighbors;
    props[kSpatialRadiusPx] = mSpatialRadiusPx;
    props[kRisTargetFloor] = mRisTargetFloor;
    props[kRouletteMinQ] = mRouletteMinQ;
    props[kRouletteStartBounce] = mRouletteStartBounce;
    props[kUseCompaction] = mUseCompaction;
    props[kUseWavefront] = mUseWavefront;
    props[kWavefrontRounds] = mWavefrontRounds;
    props[kUseTauCache] = mUseTauCache;
    props[kTrRRThreshold] = mTrRRThreshold;
    props[kTrRRMode] = mTrRRMode;
    props[kSeedOffset] = mSeedOffset;
    props[kUseRadCache] = mUseRadCache;
    props[kRadCacheRes] = mRadCacheRes;
    props[kRadCutBounce] = mRadCutBounce;
    props[kRadResidualSurvival] = mRadResidualSurvival;
    props[kRadWarpRRLanes] = mRadWarpRRLanes;
    props[kRadTrainEvery] = mRadTrainEvery;
    props[kRadEma] = mRadEma;
    props[kTauCacheRes] = mTauCacheRes;
    props[kTauCacheInterval] = mTauCacheInterval;
    return props;
}

RenderPassReflection VolumePathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    // Size every output to the selected internal render resolution (Default = 0
    // => full window res). The downstream chain must be set to the SAME size.
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void VolumePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass = nullptr;
    mpPassShade = nullptr;
    mpPassArgs = nullptr;
    mpPassShadeInit = nullptr;
    mpPassBounce = nullptr;
    mpPassBounceArgs = nullptr;
    mpPassBounceTail = nullptr;
    mpVolumeSampler = nullptr;
    mpEnvMapSampler = nullptr;

    if (!mpScene)
        return;

    if (mpScene->getGridVolumes().empty())
    {
        logWarning("VolumePathTracer: scene contains no grid volumes. Nothing will be rendered.");
    }

    mpVolumeSampler = std::make_unique<GridVolumeSampler>(pRenderContext, mpScene, mVolumeSamplerOptions);

    // Report the O(library) memory claim as measured fact, not assertion.
    // gridCount counts UNIQUE grids; gridVolumeCount counts instances. If the
    // cloudlet library is working, adding instances must not move gridMemory.
    {
        const auto& stats = mpScene->getSceneStats();
        const uint64_t gridBytes = stats.gridMemoryInBytes;
        const size_t volumes = mpScene->getGridVolumes().size();
        logInfo(
            "VolumePathTracer VRAM: {} grid volume instance(s) sharing {} unique grid(s), "
            "{:.1f} MB total grid memory ({:.1f} MB per unique grid, {} voxels).",
            volumes,
            stats.gridCount,
            gridBytes / (1024.0 * 1024.0),
            stats.gridCount ? (gridBytes / (1024.0 * 1024.0)) / stats.gridCount : 0.0,
            stats.gridVoxelCount
        );
        // Falcor stores each grid TWICE: the NanoVDB StructuredBuffer plus the
        // BrickedGrid textures (range + indirection + atlas). With
        // useBrickedGrid = true, GridVolumeSampler reads ONLY the textures, so
        // the NanoVDB buffer is dead VRAM at render time. Subtract the .nvdb
        // file size from the figure above to see how much is reclaimable.
        logInfo("VolumePathTracer VRAM: of which the NanoVDB buffer is unused at render time when 'Use BrickedGrid' is on.");
    }

    // The env map is the only light source this pass supports; it is what makes
    // the cloud's underside blue-grey rather than black.
    if (mpScene->getEnvMap())
    {
        mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
    }
    else
    {
        logWarning("VolumePathTracer: scene has no environment map. The image will be black.");
    }

    // ---- HW-BVH brick backend + merged tail: per-scene preprocessing. ------
    mUniqueGrids.clear();
    mInstanceGridIdx.clear();
    mInstanceMip.clear();
    mBrickBlas.clear();
    mpBrickAABBs = nullptr;
    mpBrickInstanceInfo = nullptr;
    mpBrickTlasBuffer = nullptr;
    mpBrickTlasScratch = nullptr;
    mpBrickTlas = nullptr;
    mBrickBlasesValid = false;
    mpTailTex = nullptr;
    mTailMinVoxWorld = 0.f;
    mpTauCache = nullptr;
    mpTauDir = nullptr;
    mpPassTauDir = nullptr;
    mpPassTauBuild = nullptr;
    mTauLastBuildFrame = kTauNeverBuilt;
    mpPassRadResolve = nullptr;
    mpRadAccum = nullptr;
    mpRadTex = nullptr;

    const auto& volumes = mpScene->getGridVolumes();
    for (const auto& pVolume : volumes)
    {
        const ref<Grid>& pGrid = pVolume->getDensityGrid();
        if (!pGrid)
        {
            mInstanceGridIdx.push_back(0);
            continue;
        }
        uint32_t gridIdx = uint32_t(-1);
        for (uint32_t g = 0; g < mUniqueGrids.size(); ++g)
        {
            if (mUniqueGrids[g] == pGrid)
            {
                gridIdx = g;
                break;
            }
        }
        if (gridIdx == uint32_t(-1))
        {
            gridIdx = (uint32_t)mUniqueGrids.size();
            mUniqueGrids.push_back(pGrid);
        }
        mInstanceGridIdx.push_back(gridIdx);
    }
    mInstanceMip.assign(volumes.size(), 0xffffffffu); // Force first TLAS build.

    if (!mUniqueGrids.empty())
    {
        if (mUseBrickTlas)
            buildBrickBlases(pRenderContext);
        if (mUseMergedTail)
            bakeMergedTail();
    }
}

void VolumePathTracer::buildBrickBlases(RenderContext* pRenderContext)
{
    // Occupied bricks of every unique grid, at every mip, become procedural
    // AABBs (UE HeterogeneousVolumes: sparse voxels in the BVH; UE Nanite:
    // the mip you traverse is the mip you BUILT, not a runtime branch).
    // AABBs are in SHIFTED index space - the space the samplers march in -
    // so the TLAS instance transform is gridVolume.transform * translate(minIndex).
    std::vector<RtAABB> aabbs;
    mBrickBlas.resize(mUniqueGrids.size());

    for (uint32_t g = 0; g < mUniqueGrids.size(); ++g)
    {
        const BrickedGrid& bricked = mUniqueGrids[g]->getBrickedGrid();
        if (bricked.rangeMeanData.empty())
        {
            logWarning("VolumePathTracer: grid {} has no CPU bricked data; brick TLAS disabled.", g);
            return;
        }

        for (uint32_t mip = 0; mip < 4; ++mip)
        {
            BrickBlas& blas = mBrickBlas[g][mip];
            blas.aabbOffset = (uint32_t)aabbs.size();

            const int3 dim = bricked.leafDim[mip];
            const uint64_t* rangeMean = bricked.rangeMeanData.data() + bricked.leafOffset[mip];
            const float cellSize = float(8u << mip);
            for (int z = 0; z < dim.z; ++z)
            {
                for (int y = 0; y < dim.y; ++y)
                {
                    for (int x = 0; x < dim.x; ++x)
                    {
                        const uint64_t packed = rangeMean[(z * dim.y + y) * dim.x + x];
                        const float majorant = f16tof32((uint16_t)(packed & 0xffffu));
                        if (majorant <= 0.f) continue;
                        RtAABB bb;
                        bb.min = float3(float(x), float(y), float(z)) * cellSize;
                        bb.max = bb.min + cellSize;
                        aabbs.push_back(bb);
                    }
                }
            }
            blas.aabbCount = (uint32_t)aabbs.size() - blas.aabbOffset;
        }
    }

    if (aabbs.empty())
    {
        logWarning("VolumePathTracer: no occupied bricks found; brick TLAS disabled.");
        return;
    }

    // One concatenated buffer: BLAS build input AND shader-visible AABB data
    // (the shader re-derives each candidate's [t0,t1] from it).
    mpBrickAABBs = mpDevice->createStructuredBuffer(
        sizeof(RtAABB),
        (uint32_t)aabbs.size(),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        aabbs.data(),
        false
    );
    mpBrickAABBs->setName("VolumePathTracer::mpBrickAABBs");

    // Prebuild all BLASes to size one shared scratch buffer.
    uint64_t maxScratch = 0;
    std::vector<RtAccelerationStructurePrebuildInfo> prebuilds(mUniqueGrids.size() * 4);
    std::vector<RtGeometryDesc> geomDescs(mUniqueGrids.size() * 4);
    for (uint32_t g = 0; g < mUniqueGrids.size(); ++g)
    {
        for (uint32_t mip = 0; mip < 4; ++mip)
        {
            const uint32_t idx = g * 4 + mip;
            const BrickBlas& blas = mBrickBlas[g][mip];

            RtGeometryDesc& geom = geomDescs[idx];
            geom = {};
            geom.type = RtGeometryType::ProcedurePrimitives;
            geom.flags = RtGeometryFlags::None;
            geom.content.proceduralAABBs.count = blas.aabbCount;
            geom.content.proceduralAABBs.data = mpBrickAABBs->getGpuAddress() + blas.aabbOffset * sizeof(RtAABB);
            geom.content.proceduralAABBs.stride = sizeof(RtAABB);

            RtAccelerationStructureBuildInputs inputs = {};
            inputs.kind = RtAccelerationStructureKind::BottomLevel;
            inputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;
            inputs.descCount = 1;
            inputs.geometryDescs = &geom;

            prebuilds[idx] = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), inputs);
            maxScratch = std::max(maxScratch, prebuilds[idx].scratchDataSize);
        }
    }

    ref<Buffer> pScratch = mpDevice->createBuffer(
        align_to(kAccelerationStructureByteAlignment, maxScratch), ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal
    );
    pScratch->setName("VolumePathTracer::brickBlasScratch");

    uint64_t totalBlasBytes = 0;
    for (uint32_t g = 0; g < mUniqueGrids.size(); ++g)
    {
        for (uint32_t mip = 0; mip < 4; ++mip)
        {
            const uint32_t idx = g * 4 + mip;
            BrickBlas& blas = mBrickBlas[g][mip];

            const uint64_t resultSize = align_to(kAccelerationStructureByteAlignment, prebuilds[idx].resultDataMaxSize);
            blas.pBuffer = mpDevice->createBuffer(resultSize, ResourceBindFlags::AccelerationStructure, MemoryType::DeviceLocal);
            blas.pBuffer->setName("VolumePathTracer::brickBlas");
            totalBlasBytes += resultSize;

            RtAccelerationStructure::Desc createDesc = {};
            createDesc.setKind(RtAccelerationStructureKind::BottomLevel);
            createDesc.setBuffer(blas.pBuffer, 0, resultSize);
            blas.pBlas = RtAccelerationStructure::create(mpDevice, createDesc);

            RtAccelerationStructureBuildInputs inputs = {};
            inputs.kind = RtAccelerationStructureKind::BottomLevel;
            inputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;
            inputs.descCount = 1;
            inputs.geometryDescs = &geomDescs[idx];

            RtAccelerationStructure::BuildDesc buildDesc = {};
            buildDesc.inputs = inputs;
            buildDesc.dest = blas.pBlas.get();
            buildDesc.scratchData = pScratch->getGpuAddress();

            pRenderContext->buildAccelerationStructure(buildDesc, 0, nullptr);
            // The single scratch buffer is reused serially: barrier between builds.
            pRenderContext->uavBarrier(pScratch.get());
            pRenderContext->uavBarrier(blas.pBuffer.get());
        }
    }
    pRenderContext->submit(false);

    mBrickBlasesValid = true;
    logInfo(
        "VolumePathTracer: brick BLASes built - {} unique grid(s) x 4 mips, {} AABBs total, {:.1f} MB.",
        mUniqueGrids.size(), aabbs.size(), totalBlasBytes / (1024.0 * 1024.0)
    );
}

uint32_t VolumePathTracer::selectInstanceMip(const GridVolume& volume, const float3& cameraPos, float footprintSpread, float& outFootprintWorld) const
{
    // UE Nanite projected-error selection, one decision per INSTANCE per frame:
    // error of mip m = its cell size in world units; pick the coarsest mip
    // whose projected cell stays under mMipPixelThreshold pixels at the
    // instance's NEAREST point (conservative, like GetProjectedEdgeScales'
    // min scale). Whole instances flip coherently - the anti-crack rule.
    outFootprintWorld = 0.f;
    if (footprintSpread <= 0.f) return 0;

    const AABB& bounds = volume.getBounds();
    const float3 center = bounds.center();
    const float radius = length(bounds.extent()) * 0.5f;
    const float dist = std::max(0.f, length(center - cameraPos) - radius);
    outFootprintWorld = footprintSpread * dist;
    if (dist <= 0.f) return 0; // Camera inside the bounds: finest structure.

    // World size of one fine voxel for this instance: smallest basis length of
    // the index->world transform (conservative under non-uniform scale).
    const float4x4& xform = volume.getData().transform;
    const float voxWorld = std::min(
        std::min(length(xform.getCol(0).xyz()), length(xform.getCol(1).xyz())), length(xform.getCol(2).xyz())
    );
    if (voxWorld <= 0.f) return 0;

    const float footprint = footprintSpread * dist * mMipPixelThreshold;
    const float cell0 = 8.f * voxWorld;
    if (footprint <= cell0) return 0;
    const int mip = (int)std::floor(std::log2(footprint / cell0));
    return (uint32_t)std::clamp(mip, 0, 3);
}

void VolumePathTracer::updateBrickTlas(RenderContext* pRenderContext, const uint2 targetDim)
{
    FALCOR_ASSERT(mBrickBlasesValid);

    const auto& volumes = mpScene->getGridVolumes();
    const auto& pCamera = mpScene->getCamera();
    const float3 cameraPos = pCamera ? pCamera->getPosition() : float3(0.f);

    // Physical footprint spread (world width one pixel spans per unit
    // distance) - used for LoD selection regardless of the residual-mip
    // feature toggle; it is a camera fact, not an option.
    float spread = 0.f;
    if (pCamera && pCamera->getFocalLength() > 0.f)
    {
        const float tanHalfFovY = pCamera->getFrameHeight() * 0.5f / pCamera->getFocalLength();
        spread = 2.f * tanHalfFovY / float(targetDim.y);
    }

    // Select per-instance mips; rebuild only when the selection changed.
    // Footprints are recorded every frame so the [LOD] log reflects what the
    // selection actually saw, not a reconstruction.
    mLastSpread = spread;
    mInstanceFootprint.assign(volumes.size(), 0.f);
    bool changed = (mpBrickTlas == nullptr);
    std::vector<uint32_t> newMips(volumes.size());
    for (size_t i = 0; i < volumes.size(); ++i)
    {
        float footprint = 0.f;
        newMips[i] = volumes[i]->getDensityGrid() ? selectInstanceMip(*volumes[i], cameraPos, spread, footprint) : 0;
        mInstanceFootprint[i] = footprint;
        changed |= (newMips[i] != mInstanceMip[i]);
    }
    if (!changed) return;
    mInstanceMip = newMips;

    // Instance descs + shader-side instance info.
    struct BrickTlasInstanceInfo
    {
        uint32_t gridVolumeIdx;
        uint32_t aabbOffset;
    };
    std::vector<RtInstanceDesc> instanceDescs;
    std::vector<BrickTlasInstanceInfo> instanceInfos;
    instanceDescs.reserve(volumes.size());
    instanceInfos.reserve(volumes.size());

    for (size_t i = 0; i < volumes.size(); ++i)
    {
        if (!volumes[i]->getDensityGrid()) continue;
        const uint32_t gridIdx = mInstanceGridIdx[i];
        const BrickBlas& blas = mBrickBlas[gridIdx][mInstanceMip[i]];
        if (blas.aabbCount == 0 || !blas.pBlas) continue;

        // Shifted-index -> world: gridVolume.transform composed with the
        // +minIndex shift the sampler subtracts (see transformRayToIndexSpace).
        const int3 minIndex = mUniqueGrids[gridIdx]->getMinIndex();
        const float4x4 xform = mul(volumes[i]->getData().transform, math::matrixFromTranslation(float3(minIndex)));

        RtInstanceDesc desc = {};
        desc.setTransform(xform);
        desc.instanceID = (uint32_t)i;
        desc.instanceMask = 0xff;
        desc.instanceContributionToHitGroupIndex = 0;
        desc.flags = RtGeometryInstanceFlags::None;
        desc.accelerationStructure = blas.pBlas->getGpuAddress();
        instanceDescs.push_back(desc);

        BrickTlasInstanceInfo info;
        info.gridVolumeIdx = (uint32_t)i;
        info.aabbOffset = blas.aabbOffset;
        instanceInfos.push_back(info);
    }
    if (instanceDescs.empty()) return;

    if (!mpBrickInstanceInfo || mpBrickInstanceInfo->getElementCount() < (uint32_t)instanceInfos.size())
    {
        mpBrickInstanceInfo = mpDevice->createStructuredBuffer(
            sizeof(BrickTlasInstanceInfo),
            (uint32_t)instanceInfos.size(),
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            instanceInfos.data(),
            false
        );
        mpBrickInstanceInfo->setName("VolumePathTracer::mpBrickInstanceInfo");
    }
    else
    {
        mpBrickInstanceInfo->setBlob(instanceInfos.data(), 0, instanceInfos.size() * sizeof(BrickTlasInstanceInfo));
    }

    RtAccelerationStructureBuildInputs inputs = {};
    inputs.kind = RtAccelerationStructureKind::TopLevel;
    inputs.descCount = (uint32_t)instanceDescs.size();
    inputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;

    if (!mpBrickTlasScratch)
    {
        // Size prebuild for the MAXIMUM instance count: a mip flip can change
        // how many instances carry non-empty BLASes between frames, and the
        // scratch/result buffers are cached across those changes.
        RtAccelerationStructureBuildInputs maxInputs = inputs;
        maxInputs.descCount = (uint32_t)volumes.size();
        mBrickTlasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), maxInputs);
        mpBrickTlasScratch = mpDevice->createBuffer(
            align_to(kAccelerationStructureByteAlignment, mBrickTlasPrebuild.scratchDataSize),
            ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal
        );
        mpBrickTlasScratch->setName("VolumePathTracer::mpBrickTlasScratch");
    }

    if (!mpBrickTlas)
    {
        mpBrickTlasBuffer =
            mpDevice->createBuffer(mBrickTlasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, MemoryType::DeviceLocal);
        mpBrickTlasBuffer->setName("VolumePathTracer::mpBrickTlasBuffer");

        RtAccelerationStructure::Desc createDesc = {};
        createDesc.setKind(RtAccelerationStructureKind::TopLevel);
        createDesc.setBuffer(mpBrickTlasBuffer, 0, mBrickTlasPrebuild.resultDataMaxSize);
        mpBrickTlas = RtAccelerationStructure::create(mpDevice, createDesc);
    }
    else
    {
        pRenderContext->uavBarrier(mpBrickTlasBuffer.get());
        pRenderContext->uavBarrier(mpBrickTlasScratch.get());
    }

    RtAccelerationStructure::BuildDesc buildDesc = {};
    buildDesc.inputs = inputs;
    buildDesc.dest = mpBrickTlas.get();
    buildDesc.scratchData = mpBrickTlasScratch->getGpuAddress();

    GpuMemoryHeap::Allocation allocation =
        mpDevice->getUploadHeap()->allocate(inputs.descCount * sizeof(RtInstanceDesc), sizeof(RtInstanceDesc));
    std::memcpy(allocation.pData, instanceDescs.data(), inputs.descCount * sizeof(RtInstanceDesc));
    buildDesc.inputs.instanceDescs = allocation.getGpuAddress();
    mpDevice->getUploadHeap()->release(allocation);

    pRenderContext->buildAccelerationStructure(buildDesc, 0, nullptr);
    pRenderContext->uavBarrier(mpBrickTlasBuffer.get());
}

void VolumePathTracer::bakeMergedTail()
{
    // UE Nanite Assemblies lesson 11: above the fine structure, the union of
    // all instances is ONE merged object. Bake a world-space grid of
    // conservative majorants of the SUMMED extinction field (valid bound:
    // max of sum <= sum of per-instance maxima over each cell) plus an
    // approximate summed mean as the control variate. The majorant MUST be
    // conservative for correctness; the mean only moves variance.
    const auto& volumes = mpScene->getGridVolumes();

    constexpr float kFloatMax = std::numeric_limits<float>::max();
    AABB worldBounds;
    float minVoxWorld = kFloatMax;
    for (const auto& pVolume : volumes)
    {
        if (!pVolume->getDensityGrid()) continue;
        worldBounds.include(pVolume->getBounds());
        const float4x4& xform = pVolume->getData().transform;
        const float voxWorld = std::min(
            std::min(length(xform.getCol(0).xyz()), length(xform.getCol(1).xyz())), length(xform.getCol(2).xyz())
        );
        if (voxWorld > 0.f) minVoxWorld = std::min(minVoxWorld, voxWorld);
    }
    if (!worldBounds.valid() || minVoxWorld == kFloatMax) return;

    const float3 extent = worldBounds.extent();
    const float maxExtent = std::max(extent.x, std::max(extent.y, extent.z));
    if (maxExtent <= 0.f) return;

    const float cell = maxExtent / float(std::max(mTailRes, 8u));
    const uint3 dim = uint3(
        std::max(1u, (uint32_t)std::ceil(extent.x / cell)),
        std::max(1u, (uint32_t)std::ceil(extent.y / cell)),
        std::max(1u, (uint32_t)std::ceil(extent.z / cell))
    );

    mTailOrigin = worldBounds.minPoint;
    mTailCellSize = float3(cell);
    mTailDim = dim;
    mTailMinVoxWorld = minVoxWorld;

    // Per-instance bake inputs (host copies of the coarse pyramids at mip 3:
    // 64-voxel cells, the cheapest level that still bounds correctly).
    struct BakeSource
    {
        const BrickedGrid* bricked;
        float4x4 invTransform;
        int3 minIndex;
        float densityScale;
        AABB bounds;
    };
    std::vector<BakeSource> sources;
    for (const auto& pVolume : volumes)
    {
        const ref<Grid>& pGrid = pVolume->getDensityGrid();
        if (!pGrid || pGrid->getBrickedGrid().rangeMeanData.empty()) continue;
        BakeSource src;
        src.bricked = &pGrid->getBrickedGrid();
        src.invTransform = pVolume->getData().invTransform;
        src.minIndex = pGrid->getMinIndex();
        src.densityScale = pVolume->getData().densityScale;
        src.bounds = pVolume->getBounds();
        sources.push_back(src);
    }
    if (sources.empty()) return;

    std::vector<uint32_t> texels((size_t)dim.x * dim.y * dim.z, 0u);

    auto zRange = NumericRange<int>(0, (int)dim.z);
    std::for_each(
        std::execution::par, zRange.begin(), zRange.end(),
        [&](int z)
        {
            for (uint32_t y = 0; y < dim.y; ++y)
            {
                for (uint32_t x = 0; x < dim.x; ++x)
                {
                    const float3 cellMin = mTailOrigin + float3(float(x), float(y), float(z)) * cell;
                    const float3 cellMax = cellMin + cell;
                    // Non-const: AABB::overlaps() is a non-const member in Falcor.
                    AABB cellBounds(cellMin, cellMax);

                    float sumMaj = 0.f;
                    float sumMean = 0.f;
                    for (const BakeSource& src : sources)
                    {
                        if (!cellBounds.overlaps(src.bounds)) continue;

                        // Transform the cell's corners into shifted index
                        // space, pad 1 fine voxel for the stochastic taps.
                        float3 iMin(kFloatMax), iMax(-kFloatMax);
                        for (int c = 0; c < 8; ++c)
                        {
                            const float3 corner(
                                (c & 1) ? cellMax.x : cellMin.x, (c & 2) ? cellMax.y : cellMin.y, (c & 4) ? cellMax.z : cellMin.z
                            );
                            const float3 ipos = mul(src.invTransform, float4(corner, 1.f)).xyz() - float3(src.minIndex);
                            iMin = min(iMin, ipos);
                            iMax = max(iMax, ipos);
                        }
                        iMin -= 1.f;
                        iMax += 1.f;

                        // Walk covering mip-3 cells (64 voxels wide).
                        const int3 dimM = src.bricked->leafDim[3];
                        const uint64_t* rangeMean = src.bricked->rangeMeanData.data() + src.bricked->leafOffset[3];
                        const int3 c0 = int3(
                            std::clamp((int)std::floor(iMin.x / 64.f), 0, dimM.x - 1),
                            std::clamp((int)std::floor(iMin.y / 64.f), 0, dimM.y - 1),
                            std::clamp((int)std::floor(iMin.z / 64.f), 0, dimM.z - 1)
                        );
                        const int3 c1 = int3(
                            std::clamp((int)std::floor(iMax.x / 64.f), 0, dimM.x - 1),
                            std::clamp((int)std::floor(iMax.y / 64.f), 0, dimM.y - 1),
                            std::clamp((int)std::floor(iMax.z / 64.f), 0, dimM.z - 1)
                        );
                        // Reject cells whose index box lies fully outside the grid.
                        if (iMax.x < 0.f || iMax.y < 0.f || iMax.z < 0.f) continue;
                        if (iMin.x > dimM.x * 64.f || iMin.y > dimM.y * 64.f || iMin.z > dimM.z * 64.f) continue;

                        float instMaj = 0.f;
                        float instMeanSum = 0.f;
                        int covered = 0;
                        for (int cz = c0.z; cz <= c1.z; ++cz)
                        {
                            for (int cy = c0.y; cy <= c1.y; ++cy)
                            {
                                for (int cx = c0.x; cx <= c1.x; ++cx)
                                {
                                    // Packed texel: f16 majorant | f16 minorant << 16 | f16 mean << 32.
                                    const uint64_t packed = rangeMean[(cz * dimM.y + cy) * dimM.x + cx];
                                    instMaj = std::max(instMaj, f16tof32((uint16_t)(packed & 0xffffu)));
                                    instMeanSum += f16tof32((uint16_t)((packed >> 32) & 0xffffu));
                                    covered++;
                                }
                            }
                        }
                        if (covered == 0) continue;
                        sumMaj += src.densityScale * instMaj;
                        sumMean += src.densityScale * (instMeanSum / covered);
                    }

                    // Mean may never exceed the conservative majorant (the
                    // shader clamps too; keeping the bake consistent avoids
                    // wasting residual budget).
                    sumMean = std::min(sumMean, sumMaj);
                    const uint32_t packed = (uint32_t)f32tof16(sumMaj) | ((uint32_t)f32tof16(sumMean) << 16);
                    texels[((size_t)z * dim.y + y) * dim.x + x] = packed;
                }
            }
        }
    );

    mpTailTex = mpDevice->createTexture3D(dim.x, dim.y, dim.z, ResourceFormat::RG16Float, 1, texels.data(), ResourceBindFlags::ShaderResource);
    mpTailTex->setName("VolumePathTracer::mpTailTex");

    logInfo(
        "VolumePathTracer: merged tail baked - {}x{}x{} cells of {:.1f} world units, gate at footprint > {:.1f} world units.",
        dim.x, dim.y, dim.z, cell, mTailGateVoxels * mTailMinVoxWorld
    );
}

void VolumePathTracer::buildTauCache(RenderContext* pRenderContext)
{
    // GPU bake, unlike the tail's CPU bake: tauBuildMain calls the exact
    // coarseOpticalDepth the live path calls, so a cache texel is semantically
    // identical to the walk it replaces - no second implementation to drift.
    FALCOR_ASSERT(mpPassTauDir && mpPassTauBuild);

    // World bounds of the union of grid volumes (tail-bake convention: cell
    // size from the longest axis, per-axis counts from the extents).
    constexpr float kFloatMax = std::numeric_limits<float>::max();
    AABB worldBounds;
    for (const auto& pVolume : mpScene->getGridVolumes())
    {
        if (!pVolume->getDensityGrid()) continue;
        worldBounds.include(pVolume->getBounds());
    }
    if (!worldBounds.valid()) return;
    const float3 extent = worldBounds.extent();
    const float maxExtent = std::max(extent.x, std::max(extent.y, extent.z));
    if (maxExtent <= 0.f) return;

    const float cell = maxExtent / float(std::max(mTauCacheRes, 8u));
    const uint3 dim = uint3(
        std::max(1u, (uint32_t)std::ceil(extent.x / cell)),
        std::max(1u, (uint32_t)std::ceil(extent.y / cell)),
        std::max(1u, (uint32_t)std::ceil(extent.z / cell))
    );

    mTauOrigin = worldBounds.minPoint;
    mTauCellSize = float3(cell);
    mTauInvExtent = float3(1.f) / (float3(dim) * cell);
    mTauDim = dim;

    if (!mpTauCache || mpTauCache->getWidth() != dim.x || mpTauCache->getHeight() != dim.y || mpTauCache->getDepth() != dim.z)
    {
        mpTauCache = mpDevice->createTexture3D(
            dim.x, dim.y, dim.z, ResourceFormat::R16Float, 1, nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        mpTauCache->setName("VolumePathTracer::mpTauCache");
    }
    if (!mpTauDir)
    {
        mpTauDir = mpDevice->createStructuredBuffer(
            sizeof(float4), 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
        );
        mpTauDir->setName("VolumePathTracer::mpTauDir");
    }
    if (!mpTauSampler)
    {
        Sampler::Desc sd;
        sd.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        sd.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpTauSampler = mpDevice->createSampler(sd);
    }

    // Both build passes bind only what they reference (reflection strips the
    // rest of the module), so this stays minimal on purpose.
    auto bindBuild = [&](const ShaderVar& var)
    {
        if (auto v = var.findMember("gTauDirBuf"); v.isValid())
            v = mpTauDir;
        if (auto v = var.findMember("gTauCacheUav"); v.isValid())
            v = mpTauCache;
        if (auto tcb = var.findMember("TauCB"); tcb.isValid())
        {
            tcb["gTauOrigin"] = mTauOrigin;
            tcb["gTauCellSize"] = mTauCellSize;
            tcb["gTauInvExtent"] = mTauInvExtent;
            tcb["gTauDim"] = mTauDim;
            tcb["gTauEnable"] = 0u; // Unused by the build kernels.
        }
    };

    bindBuild(mpPassTauDir->getRootVar());
    mpPassTauDir->execute(pRenderContext, uint3(1, 1, 1));
    pRenderContext->uavBarrier(mpTauDir.get());

    bindBuild(mpPassTauBuild->getRootVar());
    mpPassTauBuild->execute(pRenderContext, uint3(dim));
    pRenderContext->uavBarrier(mpTauCache.get());

    mTauLastBuildFrame = mFrameCount;
    logInfo(
        "VolumePathTracer: tau cache baked - {}x{}x{} cells of {:.1f} world units ({} walks).",
        dim.x, dim.y, dim.z, cell, dim.x * dim.y * dim.z
    );
}

void VolumePathTracer::prepareProgram(RenderContext* pRenderContext)
{
    FALCOR_ASSERT(mpScene && mpVolumeSampler);

    auto makeDesc = [&](const char* entry)
    {
        ProgramDesc d;
        d.addShaderModules(mpScene->getShaderModules());
        d.addShaderLibrary(kShaderFile).csEntry(entry);
        d.addTypeConformances(mpScene->getTypeConformances());
        return d;
    };
    ProgramDesc desc = makeDesc("main");

    DefineList defines;
    defines.add(mpScene->getSceneDefines());
    defines.add(mpSampleGenerator->getDefines());
    defines.add(mpVolumeSampler->getDefines());
    defines.add("MAX_BOUNCES", std::to_string(mMaxBounces));
    defines.add("USE_NEE", mUseNEE ? "1" : "0");
    defines.add("USE_SINGLE_NEE", mUseSingleNeePerPath ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", mUseRussianRoulette ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mpEnvMapSampler ? "1" : "0");
    defines.add("USE_RIS", mUseRIS ? "1" : "0");
    defines.add("USE_ADAPTIVE_M", (mUseRIS && mUseAdaptiveM) ? "1" : "0");
    defines.add("USE_TEMPORAL_REUSE", (mUseRIS && mUseTemporalReuse) ? "1" : "0");
    // Stage C rides the Stage B history buffer, so it requires temporal on.
    defines.add("USE_SPATIAL_REUSE", (mUseRIS && mUseTemporalReuse && mUseSpatialReuse) ? "1" : "0");
    defines.add("SPATIAL_NEIGHBORS", std::to_string(mSpatialNeighbors));
    defines.add("USE_RIS_STATS", (mUseRIS && mLogRisStats) ? "1" : "0");
    defines.add("USE_WORK_STATS", mLogWorkStats ? "1" : "0");
    // Enables GridVolumeSampler's internal cell/tap counters - the marching
    // cost that a ~470 ms frame was measured to hide from every other counter.
    defines.add("GRID_VOLUME_SAMPLER_STATS", mLogWorkStats ? "1" : "0");
    defines.add("USE_SHARED_CANDIDATE_SWEEP", mUseSharedCandidateSweep ? "1" : "0");
    defines.add("USE_FOOTPRINT_MIP", mFootprintMip ? "1" : "0");
    // HW-BVH brick backend + merged tail (UE ports; VNA_UE_SOURCE_LESSONS.md).
    // The TLAS define is only enabled when the BLASes actually built, so a
    // scene without bricked grids degrades to the interval backend instead of
    // binding a null acceleration structure.
    defines.add("USE_BRICK_TLAS", (mUseBrickTlas && mBrickBlasesValid) ? "1" : "0");
    defines.add("USE_MERGED_TAIL", (mUseMergedTail && mpTailTex) ? "1" : "0");
    // Lever-2 prefetch is COMPILE-TIME: the entry costs ~9 registers whether
    // or not a runtime flag reads it (run 137: gpuMs 26 -> 38 with identical
    // counters = register pressure). The checkbox rebuilds the program.
    defines.add("GVS_BRICK_PREFETCH", mUseBrickPrefetch ? "1" : "0");
    // Stream compaction: only meaningful on the RIS path (the reference path
    // stays a single fused kernel, byte-identical ground truth).
    const bool compactionCompiled = mUseRIS && mUseCompaction;
    defines.add("USE_COMPACTION", compactionCompiled ? "1" : "0");
    // Sun-tau cache: compiled only when it can actually be built (env light
    // present), so a no-envmap scene never references an unbindable texture.
    // The RUNTIME switch is gTauEnable in TauCB, not this define.
    defines.add("USE_TAU_CACHE", (mUseRIS && mUseTauCache && mpEnvMapSampler) ? "1" : "0");
    // Radiance-cache control variate: fused shadeMain backend only (the
    // reference path in main stays untouched, and the wavefront backend is
    // measured-off anyway).
    const bool radCacheCompiled = compactionCompiled && mUseRadCache;
    defines.add("USE_RAD_CACHE", radCacheCompiled ? "1" : "0");
    // Size the sampler's per-thread arrays to the scene, not to a fixed 16.
    // These arrays dominate register/scratch usage, and this pass measured as
    // spill-bound rather than work-bound, so oversizing them costs real time.
    const uint32_t instanceCount = mpScene ? (uint32_t)mpScene->getGridVolumes().size() : 1u;
    defines.add("MAX_INSTANCES", std::to_string(std::max(1u, instanceCount)));
    defines.add("RIS_CANDIDATES", std::to_string(mRisCandidates));
    defines.add("RIS_MIP", std::to_string(mRisMip));
    // Placeholders so the shader compiles before execute() sets the real values.
    defines.add("is_valid_gWorkDebug", "0");
    defines.add("is_valid_gOutputLin", "0");
    defines.add("is_valid_gOutputT", "0");
    defines.add("is_valid_gOutputBg", "0");
    defines.add("is_valid_gRisDebug", "0");

    mpPass = ComputePass::create(mpDevice, desc, defines, true);

    // Compaction pipeline: same module, two more entry points. Created from
    // the same DefineList so all three kernels agree on every feature define.
    mpPassShade = nullptr;
    mpPassArgs = nullptr;
    if (compactionCompiled)
    {
        mpPassShade = ComputePass::create(mpDevice, makeDesc("shadeMain"), defines, true);
        mpPassArgs = ComputePass::create(mpDevice, makeDesc("argsMain"), defines, true);
    }

    // Lever-1b scatter-queue counting sort (same module + DefineList). Pure
    // scheduling - image-identical (shadeMain seeds by pixel) - so it rides
    // the compaction pipeline unconditionally and a checkbox A/Bs it.
    mpPassScatterClass = nullptr;
    mpPassScatterOffset = nullptr;
    mpPassScatterSort = nullptr;
    if (compactionCompiled)
    {
        mpPassScatterClass = ComputePass::create(mpDevice, makeDesc("scatterClassMain"), defines, true);
        mpPassScatterOffset = ComputePass::create(mpDevice, makeDesc("scatterOffsetMain"), defines, true);
        mpPassScatterSort = ComputePass::create(mpDevice, makeDesc("scatterSortMain"), defines, true);
    }

    // Wavefront phase B (per-bounce requeue). Same module + DefineList as
    // every other kernel; selection between fused shadeMain and this pipeline
    // is purely which passes execute() dispatches, so the A/B is a checkbox.
    mpPassShadeInit = nullptr;
    mpPassBounce = nullptr;
    mpPassBounceArgs = nullptr;
    mpPassBounceTail = nullptr;
    if (compactionCompiled && mUseWavefront)
    {
        mpPassShadeInit = ComputePass::create(mpDevice, makeDesc("shadeInitMain"), defines, true);
        mpPassBounce = ComputePass::create(mpDevice, makeDesc("bounceMain"), defines, true);
        mpPassBounceArgs = ComputePass::create(mpDevice, makeDesc("bounceArgsMain"), defines, true);
        mpPassBounceTail = ComputePass::create(mpDevice, makeDesc("bounceTailMain"), defines, true);
    }

    // Sun-tau cache build passes (same module + DefineList, so the bake's
    // coarseOpticalDepth agrees with the live path on every feature define).
    // A fresh program means a fresh bake: defines can change the field.
    mpPassTauDir = nullptr;
    mpPassTauBuild = nullptr;
    mTauLastBuildFrame = kTauNeverBuilt;
    if (mUseRIS && mUseTauCache && mpEnvMapSampler)
    {
        mpPassTauDir = ComputePass::create(mpDevice, makeDesc("tauDirMain"), defines, true);
        mpPassTauBuild = ComputePass::create(mpDevice, makeDesc("tauBuildMain"), defines, true);
    }

    mpPassRadResolve = nullptr;
    if (radCacheCompiled)
        mpPassRadResolve = ComputePass::create(mpDevice, makeDesc("radResolveMain"), defines, true);

    // RIS stat buffers. Always created (and always bound) so the shader's UAV
    // is never null, regardless of whether logging is currently enabled.
    if (!mpRisStats)
    {
        mpRisStats = mpDevice->createStructuredBuffer(
            sizeof(uint32_t),
            kRisStatSlots,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr
        );
        mpRisStatsReadback = mpDevice->createBuffer(kRisStatSlots * sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack);
    }
    if (!mpGpuTimer)
        mpGpuTimer = GpuTimer::create(mpDevice);
    if (!mpGpuTimerA)
        mpGpuTimerA = GpuTimer::create(mpDevice);

    // Bind resources that never change for the lifetime of the program.
    // Plain bindShaderData is still correct for the SCENE block: this pass
    // never traces against the scene's TLAS. The brick TLAS is our own,
    // pass-owned structure, bound separately in execute() each frame.
    // findMember guards: the shade/args entry points reference different
    // subsets of the globals, and reflection strips what a program never
    // touches.
    auto bindStatics = [&](const ShaderVar& var)
    {
        if (auto v = var.findMember("gScene"); v.isValid())
            mpScene->bindShaderData(v);
        mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler)
        {
            if (auto v = var.findMember("gParams"); v.isValid())
                mpEnvMapSampler->bindShaderData(v["envMapSampler"]);
        }
    };
    bindStatics(mpPass->getRootVar());
    if (mpPassShade)
        bindStatics(mpPassShade->getRootVar());
    if (mpPassShadeInit)
        bindStatics(mpPassShadeInit->getRootVar());
    if (mpPassBounce)
        bindStatics(mpPassBounce->getRootVar());
    if (mpPassBounceTail)
        bindStatics(mpPassBounceTail->getRootVar());
    if (mpPassTauDir)
        bindStatics(mpPassTauDir->getRootVar());
    if (mpPassTauBuild)
        bindStatics(mpPassTauBuild->getRootVar());
}

void VolumePathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            if (Texture* pDst = renderData.getTexture(it.name).get())
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // Lazy structure builds: options can be enabled after scene load, and the
    // program defines depend on whether the structures actually exist - so a
    // successful build forces a program rebuild to pick the defines up.
    if (mUseBrickTlas && !mBrickBlasesValid && !mUniqueGrids.empty())
    {
        buildBrickBlases(pRenderContext);
        if (mBrickBlasesValid) mpPass = nullptr;
    }
    if (mUseMergedTail && !mpTailTex && !mUniqueGrids.empty())
    {
        bakeMergedTail();
        if (mpTailTex) mpPass = nullptr;
    }

    if (!mpPass)
        prepareProgram(pRenderContext);
    FALCOR_ASSERT(mpPass);

    // Sun-tau cache: bake on first use, then only on the interval (0 = never
    // again - the scene and env are static). The bake is two dispatches and
    // ~5M DDA cells at the default 64-cell res, vs 39.5M cells per frame for
    // the live walks it replaces.
    if (mpPassTauBuild)
    {
        const bool needBuild = (mTauLastBuildFrame == kTauNeverBuilt) ||
                               (mTauCacheInterval > 0 && mFrameCount - mTauLastBuildFrame >= mTauCacheInterval);
        if (needBuild)
            buildTauCache(pRenderContext);
    }

    // Radiance-cache resources (see RadCB in the shader). Same grid-geometry
    // convention as the tau cache: cell from the longest axis, per-axis counts
    // from the extents. Created empty (confidence 0), so consumption stays
    // inert until training paths have populated cells.
    if (mpPassRadResolve && !mpRadAccum && mpScene)
    {
        AABB wb;
        for (const auto& pVolume : mpScene->getGridVolumes())
        {
            if (!pVolume->getDensityGrid()) continue;
            wb.include(pVolume->getBounds());
        }
        const float3 ext = wb.valid() ? wb.extent() : float3(0.f);
        const float maxExt = std::max(ext.x, std::max(ext.y, ext.z));
        if (maxExt > 0.f)
        {
            const float cell = maxExt / float(std::max(mRadCacheRes, 8u));
            const uint3 dim = uint3(
                std::max(1u, (uint32_t)std::ceil(ext.x / cell)),
                std::max(1u, (uint32_t)std::ceil(ext.y / cell)),
                std::max(1u, (uint32_t)std::ceil(ext.z / cell))
            );
            mRadOrigin = wb.minPoint;
            mRadCellSize = float3(cell);
            mRadInvExtent = float3(1.f) / (float3(dim) * cell);
            mRadDim = dim;

            const uint32_t cells = dim.x * dim.y * dim.z;
            mpRadAccum = mpDevice->createStructuredBuffer(
                sizeof(uint32_t), cells * 4u, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
                MemoryType::DeviceLocal
            );
            mpRadAccum->setName("VolumePathTracer::mpRadAccum");
            pRenderContext->clearUAV(mpRadAccum->getUAV().get(), uint4(0, 0, 0, 0));

            mpRadTex = mpDevice->createTexture3D(
                dim.x, dim.y, dim.z, ResourceFormat::RGBA16Float, 1, nullptr,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
            mpRadTex->setName("VolumePathTracer::mpRadTex");
            pRenderContext->clearUAV(mpRadTex->getUAV().get(), float4(0.f));

            if (!mpTauSampler)
            {
                Sampler::Desc sd;
                sd.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
                sd.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
                mpTauSampler = mpDevice->createSampler(sd);
            }

            logInfo(
                "VolumePathTracer: radiance cache {}x{}x{} cells of {:.1f} world units (train 1-in-{}, cut bounce {}, residual p {:.2f}).",
                dim.x, dim.y, dim.z, cell, mRadTrainEvery, mRadCutBounce, mRadResidualSurvival
            );
        }
    }

    // Tell the shaders which optional outputs are actually connected.
    mpPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));
    if (mpPassShade)
        mpPassShade->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));
    if (mpPassShadeInit)
        mpPassShadeInit->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));
    if (mpPassBounce)
        mpPassBounce->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Render at the selected internal resolution, not the window size. All
    // per-pixel state (reservoirs at mReservoirDim, footprintSpread which uses
    // targetDim.y) keys off this, so lowering it both quarters the pixel cost
    // AND activates the footprint LoD (finer footprints -> coarser mips) - the
    // compounding win the script's resolution note describes.
    const uint2 targetDim = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, renderData.getDefaultTextureDims());
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Footprint spread: world-space width one pixel spans per unit distance.
    // ALWAYS uploaded - it is a camera fact that drives the merged-tail gate
    // and the TLAS LoD; the residual-mip feature keys off its own define
    // (USE_FOOTPRINT_MIP), not off this value.
    float footprintSpread = 0.f;
    if (mpScene->getCamera())
    {
        const auto& pCamera = mpScene->getCamera();
        const float focalLength = pCamera->getFocalLength();
        if (focalLength > 0.f)
        {
            const float tanHalfFovY = pCamera->getFrameHeight() * 0.5f / focalLength;
            footprintSpread = 2.f * tanHalfFovY / float(targetDim.y) * mFootprintScale;
        }
    }

    // HW-BVH brick backend: refresh the TLAS once per frame (side effect kept
    // OUT of the bind lambda below, which runs once per kernel).
    if (mUseBrickTlas && mBrickBlasesValid)
        updateBrickTlas(pRenderContext, targetDim);

    // Stage B temporal reservoir history: (re)create + clear on resize or
    // first use; ping-pong bindings happen in the bind lambda.
    const bool temporalCompiled = mUseRIS && mUseTemporalReuse;
    const bool temporalActive = temporalCompiled && mMaxBounces > 0;
    if (temporalCompiled)
    {
        const uint32_t pixelCount = targetDim.x * targetDim.y;
        if (!mpReservoir[0] || any(mReservoirDim != targetDim))
        {
            for (uint32_t i = 0; i < 2; ++i)
            {
                mpReservoir[i] = mpDevice->createStructuredBuffer(
                    32, pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                );
                mpReservoir[i]->setName(i == 0 ? "VolumePathTracer::mpReservoir0" : "VolumePathTracer::mpReservoir1");
                pRenderContext->clearUAV(mpReservoir[i]->getUAV().get(), uint4(0));
            }
            mReservoirDim = targetDim;
            mPrevCamValid = false; // History cleared: do not reproject into it.
        }
    }

    // Stream compaction: queue of RIS survivors + indirect args for the dense
    // shade dispatch. Worst case every pixel scatters, so size to pixel count.
    const bool compactionActive = mUseRIS && mUseCompaction && mpPassShade && mpPassArgs;
    if (compactionActive)
    {
        const uint32_t pixelCount = targetDim.x * targetDim.y;
        if (!mpScatterQueue || mpScatterQueue->getElementCount() < pixelCount)
        {
            mpScatterQueue = mpDevice->createStructuredBuffer(
                16, pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
            );
            mpScatterQueue->setName("VolumePathTracer::mpScatterQueue");
        }
        if (!mpScatterCount)
        {
            mpScatterCount = mpDevice->createStructuredBuffer(
                sizeof(uint32_t), 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
            );
            mpScatterCount->setName("VolumePathTracer::mpScatterCount");
        }
        if (!mpDispatchArgs)
        {
            mpDispatchArgs = mpDevice->createBuffer(
                3 * sizeof(uint32_t), ResourceBindFlags::IndirectArg | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal
            );
            mpDispatchArgs->setName("VolumePathTracer::mpDispatchArgs");
        }
        pRenderContext->clearUAV(mpScatterCount->getUAV().get(), uint4(0, 0, 0, 0));

        // Lever-1b sort buffers: sorted queue mirror + per-entry class + a
        // 8-uint histogram/cursor block. Only when the sort will actually run.
        if (mUseScatterSort && mpPassScatterSort)
        {
            if (!mpScatterQueueSorted || mpScatterQueueSorted->getElementCount() < pixelCount)
            {
                mpScatterQueueSorted = mpDevice->createStructuredBuffer(
                    16, pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                );
                mpScatterQueueSorted->setName("VolumePathTracer::mpScatterQueueSorted");
            }
            if (!mpScatterClass || mpScatterClass->getElementCount() < pixelCount)
            {
                mpScatterClass = mpDevice->createStructuredBuffer(
                    sizeof(uint32_t), pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                );
                mpScatterClass->setName("VolumePathTracer::mpScatterClass");
            }
            if (!mpScatterClassCount)
            {
                mpScatterClassCount = mpDevice->createStructuredBuffer(
                    sizeof(uint32_t), 8, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                );
                mpScatterClassCount->setName("VolumePathTracer::mpScatterClassCount");
            }
            pRenderContext->clearUAV(mpScatterClassCount->getUAV().get(), uint4(0, 0, 0, 0));
        }

        // Wavefront path-state pool. Sized to pixelCount like the scatter
        // queue (worst case every pixel scatters). Stride comes from
        // REFLECTION - PathState embeds SampleGenerator and the NEE reservoir,
        // and a hand-maintained byte count would drift the first time either
        // struct changes.
        if (mUseWavefront && mpPassBounce)
        {
            const auto stateVar = mpPassBounce->getRootVar()["gPathState"];
            if (!mpPathState || mpPathState->getElementCount() < pixelCount)
            {
                mpPathState = mpDevice->createStructuredBuffer(
                    stateVar, pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                );
                mpPathState->setName("VolumePathTracer::mpPathState");
                logInfo(
                    "VolumePathTracer: wavefront path-state pool {} paths x {} B = {:.1f} MB.",
                    pixelCount, mpPathState->getStructSize(), mpPathState->getSize() / (1024.0 * 1024.0)
                );
            }
            for (int q = 0; q < 2; ++q)
            {
                if (!mpPathQueue[q] || mpPathQueue[q]->getElementCount() < pixelCount)
                {
                    mpPathQueue[q] = mpDevice->createStructuredBuffer(
                        sizeof(uint32_t), pixelCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
                        MemoryType::DeviceLocal
                    );
                    mpPathQueue[q]->setName(fmt::format("VolumePathTracer::mpPathQueue[{}]", q));
                }
                if (!mpPathCount[q])
                {
                    mpPathCount[q] = mpDevice->createStructuredBuffer(
                        sizeof(uint32_t), 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal
                    );
                    mpPathCount[q]->setName(fmt::format("VolumePathTracer::mpPathCount[{}]", q));
                }
            }
            if (!mpBounceArgs)
            {
                mpBounceArgs = mpDevice->createBuffer(
                    3 * sizeof(uint32_t), ResourceBindFlags::IndirectArg | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal
                );
                mpBounceArgs->setName("VolumePathTracer::mpBounceArgs");
            }
        }
    }

    // Per-frame bindings, shared by the fused/phase-A kernel and the shade
    // kernel. Every loose global goes through findMember: the two programs
    // reference different subsets and reflection strips the rest.
    auto bindFrame = [&](const ShaderVar& var)
    {
        var["CB"]["gFrameCount"] = mFrameCount + mSeedOffset;
        var["CB"]["gFrameDim"] = targetDim;
        var["CB"]["gFootprintSpread"] = footprintSpread;
        var["CB"]["gOccSkipEnable"] = mUseOccupancySkip ? 1u : 0u;
        var["CB"]["gPrevViewProj"] = mPrevViewProj;
        var["CB"]["gPrevCamPos"] = mPrevCamPos;
        var["CB"]["gTemporalEnable"] = (temporalActive && mPrevCamValid) ? 1u : 0u;
        var["CB"]["gTemporalMCap"] = mTemporalMCap;
        var["CB"]["gSpatialRadiusPx"] = mSpatialRadiusPx;
        var["CB"]["gRisTargetFloor"] = mRisTargetFloor;
        var["CB"]["gRouletteMinQ"] = mRouletteMinQ;
        var["CB"]["gRouletteStartBounce"] = mRouletteStartBounce;

        if (auto trcb = var.findMember("TrRRCB"); trcb.isValid())
        {
            trcb["gTrRRThreshold"] = mTrRRThreshold;
            trcb["gTrRRMode"] = mTrRRMode;
            // Per-frame salt for the bit4 decorrelated coin stream (hashed on
            // GPU with ray bits + coin ordinal, so the raw counter is enough).
            trcb["gTrRRCoinSeed"] = mFrameCount;
        }

        if (mUseBrickTlas && mBrickBlasesValid && mpBrickTlas)
        {
            if (auto v = var.findMember("gBrickTlas"); v.isValid())
                v.setAccelerationStructure(mpBrickTlas);
            if (auto v = var.findMember("gBrickTlasInstances"); v.isValid())
                v = mpBrickInstanceInfo;
            if (auto v = var.findMember("gBrickAABBs"); v.isValid())
                v = mpBrickAABBs;
        }


        if (temporalCompiled)
        {
            if (auto v = var.findMember("gReservoirPrev"); v.isValid())
                v = mpReservoir[mReservoirFrame & 1];
            if (auto v = var.findMember("gReservoirCurr"); v.isValid())
                v = mpReservoir[(mReservoirFrame + 1) & 1];
        }

        // Sun-tau cache: baked tau grid + constants. gTauEnable is the
        // runtime A/B switch; both paths stay compiled (measurement
        // discipline: same-session toggle, no recompile).
        {
            const bool tauUsable = mpTauCache != nullptr && mTauLastBuildFrame != kTauNeverBuilt;
            if (auto v = var.findMember("gTauTex"); v.isValid())
                v = mpTauCache;
            if (auto v = var.findMember("gTauSmp"); v.isValid())
                v = mpTauSampler;
            if (auto tcb = var.findMember("TauCB"); tcb.isValid())
            {
                tcb["gTauOrigin"] = mTauOrigin;
                tcb["gTauCellSize"] = mTauCellSize;
                tcb["gTauInvExtent"] = mTauInvExtent;
                tcb["gTauDim"] = mTauDim;
                tcb["gTauEnable"] = (tauUsable && mTauCacheApply) ? 1u : 0u;
            }
        }

        // Radiance-cache CV: resolved cache + deposit accumulators + knobs.
        // Every knob is a CB value, so the estimator config sweeps live -
        // gRadCutBounce = 0 is the runtime OFF switch.
        if (mpRadAccum && mpRadTex)
        {
            if (auto v = var.findMember("gRadTex"); v.isValid())
                v = mpRadTex;
            if (auto v = var.findMember("gRadSmp"); v.isValid())
                v = mpTauSampler;
            if (auto v = var.findMember("gRadAccum"); v.isValid())
                v = mpRadAccum;
            if (auto rcb = var.findMember("RadCB"); rcb.isValid())
            {
                rcb["gRadOrigin"] = mRadOrigin;
                rcb["gRadCutBounce"] = mRadCutBounce;
                rcb["gRadCellSize"] = mRadCellSize;
                rcb["gRadResidualSurvival"] = mRadResidualSurvival;
                rcb["gRadWarpRRLanes"] = mRadWarpRRLanes;
                rcb["gRadInvExtent"] = mRadInvExtent;
                rcb["gRadTrainEvery"] = mRadTrainEvery;
                rcb["gRadDim"] = mRadDim;
                rcb["gRadEma"] = mRadEma;
            }
        }

        // Merged coarse tail: baked summed-field grid + its constants.
        if (mUseMergedTail && mpTailTex)
        {
            if (auto v = var.findMember("gTailTex"); v.isValid())
                v = mpTailTex;
            if (auto tcb = var.findMember("TailCB"); tcb.isValid())
            {
                tcb["gTailOrigin"] = mTailOrigin;
                tcb["gTailFootprintGate"] = mTailGateVoxels * mTailMinVoxWorld;
                tcb["gTailCellSize"] = mTailCellSize;
                tcb["gTailEnabled"] = 1u;
                tcb["gTailDim"] = mTailDim;
            }
        }

        for (auto channel : kOutputChannels)
        {
            if (auto v = var.findMember(channel.texname); v.isValid())
                v = renderData.getTexture(channel.name);
        }

        if (auto v = var.findMember("gRisStats"); v.isValid())
            v = mpRisStats;

        if (compactionActive)
        {
            if (auto v = var.findMember("gScatterQueue"); v.isValid())
                v = mpScatterQueue;
            if (auto v = var.findMember("gScatterCount"); v.isValid())
                v = mpScatterCount;
        }
    };

    // Roulette sweep driver (see the member note). Runs the PT Enhanced
    // 6.2.4 experiment unattended: schedule = baseline, then minQ
    // {0.10, 0.15, 0.30} at the current start bounce, then {0.05, 0.15, 0.30}
    // at start bounce 2. Each step gets mSweepFramesPerStep frames and one
    // forced [SWEEP] readback on its last frame.
    bool sweepLogFrame = false;
    if (mSweepActive && mFrameCount >= mSweepStartFrame)
    {
        static constexpr float kSweepQ[6] = {0.10f, 0.15f, 0.30f, 0.05f, 0.15f, 0.30f};
        static constexpr uint32_t kSweepB[6] = {3u, 3u, 3u, 2u, 2u, 2u};
        constexpr uint32_t kSweepSteps = 7; // baseline + 6 table entries.

        const uint32_t rel = mFrameCount - mSweepStartFrame;
        const uint32_t step = rel / mSweepFramesPerStep;
        if (step >= kSweepSteps)
        {
            mRouletteMinQ = mSweepSavedMinQ;
            mRouletteStartBounce = mSweepSavedStartBounce;
            mSweepActive = false;
            mOptionsChanged = true;
            logInfo("[SWEEP] done - restored minQ {:.2f} startBounce {}.", mRouletteMinQ, mRouletteStartBounce);
        }
        else
        {
            const float q = (step == 0) ? mSweepSavedMinQ : kSweepQ[step - 1];
            const uint32_t b = (step == 0) ? mSweepSavedStartBounce : kSweepB[step - 1];
            if (q != mRouletteMinQ || b != mRouletteStartBounce)
            {
                mRouletteMinQ = q;
                mRouletteStartBounce = b;
                // Step boundary: reset accumulation so each step's image can
                // be judged for variance on its own.
                mOptionsChanged = true;
            }
            sweepLogFrame = (rel % mSweepFramesPerStep) == mSweepFramesPerStep - 1;
        }
    }

    // Log the RIS branch histogram every mRisStatsInterval frames. The readback
    // needs a full GPU sync, so it is throttled rather than run every frame.
    const bool logThisFrame =
        ((mLogWorkStats || (mUseRIS && mLogRisStats)) && mpRisStats && (mFrameCount % mRisStatsInterval == 0)) ||
        (sweepLogFrame && mpRisStats);
    if (logThisFrame)
        pRenderContext->clearUAV(mpRisStats->getUAV().get(), uint4(0, 0, 0, 0));

    // Time the dispatches on the GPU. A CPU timer here would measure command
    // submission, not the work, and would report near-zero regardless of cost.
    // The timer brackets the WHOLE phase-B pipeline - fused: three dispatches;
    // wavefront: init + maxBounces rounds of (args, clear, bounce) - so gpuMs
    // stays the whole-pipeline number the log has always reported.
    const bool timeThisFrame = mLogWorkStats && mpGpuTimer && logThisFrame;
    if (timeThisFrame)
        mpGpuTimer->begin();

    // Nested phase-A timer: [WORK] prints gpuMs split main/shade so the
    // main-vs-shadeMain attribution comes from the log, not from a stale
    // Nsight ratio. Independent timestamp pairs, so nesting is safe.
    if (timeThisFrame && mpGpuTimerA)
        mpGpuTimerA->begin();
    bindFrame(mpPass->getRootVar());
    mpPass->execute(pRenderContext, uint3(targetDim, 1));
    if (timeThisFrame && mpGpuTimerA)
        mpGpuTimerA->end();

    if (compactionActive)
    {
        // Queue count -> indirect args (one thread), then the dense shade
        // dispatch. Falcor inserts the UAV barriers between dispatches.
        auto argsVar = mpPassArgs->getRootVar();
        argsVar["gScatterCount"] = mpScatterCount;
        argsVar["gDispatchArgs"] = mpDispatchArgs;
        mpPassArgs->execute(pRenderContext, uint3(1, 1, 1));

        // The rad-cache CV lives in the fused shadeMain only; the wavefront
        // backend (measured-off anyway) does not implement the cut.
        const bool wavefrontActive = mUseWavefront && !mUseRadCache && mpPassShadeInit && mpPassBounce && mpPassBounceArgs &&
                                     mpPassBounceTail && mpPathState;
        if (!wavefrontActive)
        {
            // Lever-1b: classify -> offsets -> scatter, then shade reads the
            // SORTED queue. Image-identical (per-pixel seeding); the sort only
            // changes which paths share a warp. Three count-sized-or-smaller
            // dispatches; Falcor inserts the UAV barriers.
            const bool sortActive = mUseScatterSort && mpPassScatterClass && mpScatterQueueSorted;
            if (sortActive)
            {
                auto classVar = mpPassScatterClass->getRootVar();
                bindFrame(classVar);
                classVar["gScatterClass"] = mpScatterClass;
                classVar["gScatterClassCount"] = mpScatterClassCount;
                mpPassScatterClass->executeIndirect(pRenderContext, mpDispatchArgs.get());

                auto offVar = mpPassScatterOffset->getRootVar();
                offVar["gScatterClassCount"] = mpScatterClassCount;
                mpPassScatterOffset->execute(pRenderContext, uint3(1, 1, 1));

                auto sortVar = mpPassScatterSort->getRootVar();
                bindFrame(sortVar);
                sortVar["gScatterClass"] = mpScatterClass;
                sortVar["gScatterClassCount"] = mpScatterClassCount;
                sortVar["gScatterQueueSorted"] = mpScatterQueueSorted;
                mpPassScatterSort->executeIndirect(pRenderContext, mpDispatchArgs.get());
            }

            auto shadeVar = mpPassShade->getRootVar();
            bindFrame(shadeVar);
            if (sortActive)
                shadeVar["gScatterQueue"] = mpScatterQueueSorted;
            mpPassShade->executeIndirect(pRenderContext, mpDispatchArgs.get());
        }
        else
        {
            // Wavefront phase B. shadeInitMain shades the RIS vertex for every
            // queued pixel and parks survivors; then up to maxBounces rounds of
            // (count -> args, clear next count, advance one bounce). Dead
            // rounds cost one empty ExecuteIndirect - no CPU readback, no
            // sync. All GPU-GPU dependencies are UAV barriers Falcor already
            // inserts between dispatches touching the same resources.
            pRenderContext->clearUAV(mpPathCount[0]->getUAV().get(), uint4(0, 0, 0, 0));

            auto initVar = mpPassShadeInit->getRootVar();
            bindFrame(initVar);
            initVar["gPathState"] = mpPathState;
            initVar["gPathQueueOut"] = mpPathQueue[0];
            initVar["gPathCountOut"] = mpPathCount[0];
            mpPassShadeInit->executeIndirect(pRenderContext, mpDispatchArgs.get());

            auto bounceVar = mpPassBounce->getRootVar();
            bindFrame(bounceVar);
            bounceVar["gPathState"] = mpPathState;
            auto bounceArgsVar = mpPassBounceArgs->getRootVar();
            bounceArgsVar["gBounceArgs"] = mpBounceArgs;

            // K dense requeue rounds, then ONE tail dispatch that loops the
            // survivors to death. v1 ran all 64 rounds: ~192 serializing sync
            // points per frame, most of them advancing a near-empty queue -
            // measured MUCH slower than the fused loop. Requeueing pays only
            // while populations keep warps dense; the thin tail is cheaper
            // divergent than re-dispatched.
            uint32_t cur = 0;
            const uint32_t rounds = std::min(mWavefrontRounds, mMaxBounces);
            for (uint32_t b = 0; b < rounds; ++b)
            {
                const uint32_t next = 1u - cur;
                bounceArgsVar["gPathCountIn"] = mpPathCount[cur];
                mpPassBounceArgs->execute(pRenderContext, uint3(1, 1, 1));
                pRenderContext->clearUAV(mpPathCount[next]->getUAV().get(), uint4(0, 0, 0, 0));

                bounceVar["gPathQueueIn"] = mpPathQueue[cur];
                bounceVar["gPathCountIn"] = mpPathCount[cur];
                bounceVar["gPathQueueOut"] = mpPathQueue[next];
                bounceVar["gPathCountOut"] = mpPathCount[next];
                mpPassBounce->executeIndirect(pRenderContext, mpBounceArgs.get());
                cur = next;
            }

            if (rounds < mMaxBounces)
            {
                bounceArgsVar["gPathCountIn"] = mpPathCount[cur];
                mpPassBounceArgs->execute(pRenderContext, uint3(1, 1, 1));

                auto tailVar = mpPassBounceTail->getRootVar();
                bindFrame(tailVar);
                tailVar["gPathState"] = mpPathState;
                tailVar["gPathQueueIn"] = mpPathQueue[cur];
                tailVar["gPathCountIn"] = mpPathCount[cur];
                mpPassBounceTail->executeIndirect(pRenderContext, mpBounceArgs.get());
            }
        }

        // Fold this frame's training deposits into the resolved cache and
        // clear the accumulators. Runs after phase B so deposits are complete;
        // consumption reads the resolved texture next frame.
        if (mpPassRadResolve && mpRadAccum && mpRadTex)
        {
            // Manual bind: this kernel touches only RadCB + the two rad
            // resources. bindFrame would poke CB members the program does not
            // reflect and throw.
            auto rv = mpPassRadResolve->getRootVar();
            rv["gRadAccum"] = mpRadAccum;
            rv["gRadResolveUav"] = mpRadTex;
            if (auto rcb = rv.findMember("RadCB"); rcb.isValid())
            {
                rcb["gRadOrigin"] = mRadOrigin;
                rcb["gRadCutBounce"] = mRadCutBounce;
                rcb["gRadCellSize"] = mRadCellSize;
                rcb["gRadResidualSurvival"] = mRadResidualSurvival;
                rcb["gRadWarpRRLanes"] = mRadWarpRRLanes;
                rcb["gRadInvExtent"] = mRadInvExtent;
                rcb["gRadTrainEvery"] = mRadTrainEvery;
                rcb["gRadDim"] = mRadDim;
                rcb["gRadEma"] = mRadEma;
            }
            mpPassRadResolve->execute(pRenderContext, uint3(mRadDim));
        }
    }

    if (timeThisFrame)
    {
        mpGpuTimer->end();
        mpGpuTimer->resolve();
        if (mpGpuTimerA)
            mpGpuTimerA->resolve();
    }

    if (logThisFrame)
    {
        pRenderContext->copyResource(mpRisStatsReadback.get(), mpRisStats.get());
        pRenderContext->submit(true); // Wait: the counters must be complete before reading.

        const uint32_t* s = static_cast<const uint32_t*>(mpRisStatsReadback->map());
        if (s)
        {
            const uint64_t pixels = (uint64_t)targetDim.x * targetDim.y;
            const uint32_t inMedium = s[8];
            const double avgCand = inMedium > 0 ? double(s[7]) / inMedium : 0.0;
            // One line, every count raw. Whichever exit dominates IS the bug:
            // each corresponds to a different failure with a different fix.
            if (mUseRIS && mLogRisStats)
            {
                logInfo(
                    "[RIS] frame {} pixels {} inMedium {} | noMedium {} noCand {} zeroTarget {} zeroDensity {} "
                    "zeroTrans {} terminated {} contributed {} | avgCandidates {:.2f} avgM {:.2f} temporalMerged {}",
                    mFrameCount, pixels, inMedium,
                    s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                    avgCand, pixels > 0 ? double(s[25]) / pixels : 0.0, s[26]
                );
            }

            if (mLogWorkStats)
            {
                if (mpGpuTimer)
                    mLastGpuMs = mpGpuTimer->getElapsedTime();
                if (mpGpuTimerA)
                    mLastGpuMsA = mpGpuTimerA->getElapsedTime();

                // [DIVERGE] Why only ~23% of lane-slots retire useful work.
                // laneUtil is computed the same way Nsight computes Active
                // Threads Per Warp, but from marching cells instead of
                // instructions - if the two disagree, distrust this probe.
                //   idleFrac HIGH  -> miss/hit imbalance, compaction wins, and
                //                     idleFrac is the CEILING on that win.
                //   idleFrac LOW   -> tail-length variance inside the medium;
                //                     compaction cannot pack it away and the
                //                     divergence is intrinsic to delta tracking.
                // A previous "busyUtil" column here was REMOVED: it divided
                // frame-total work by frame-total waveMax scaled by an AVERAGE
                // busy-lane count. That identity only holds per-warp, so mixing
                // fully-idle warps (0 waveMax, 32 idle lanes) into the average
                // produced utilisations above 1.0 - impossible, and a reminder
                // that any ratio built from two independently-aggregated sums
                // needs the per-unit identity checked first.
                //
                // Note laneUtil is ALREADY restricted to marching warps: idle
                // warps contribute 0 to both numerator and denominator. So it
                // reads as "of the 32 slots in a warp that marches, what
                // fraction did useful work" - and 1/laneUtil is the ceiling on
                // any intra-warp packing win.
                const double dvWarps = double(s[33]);
                const double dvLanes = double(s[35]);
                const double dvMaxSum = double(s[32]);
                const double lanesPerWarp = dvWarps > 0.0 ? dvLanes / dvWarps : 0.0;
                const double critical = dvMaxSum * lanesPerWarp;
                const double laneUtil = critical > 0.0 ? double(s[31]) / critical : 0.0;
                const double idleFrac = dvLanes > 0.0 ? double(s[34]) / dvLanes : 0.0;
                const double idleWarps = double(s[36]);
                const double marchWarps = dvWarps - idleWarps;
                const double idleWarpFrac = dvWarps > 0.0 ? idleWarps / dvWarps : 0.0;
                const double busyLanesPerMarchWarp = marchWarps > 0.0 ? double(s[37]) / marchWarps : 0.0;
                logInfo(
                    "[DIVERGE] frame {} warps {} lanes/warp {:.1f} | laneUtil {:.3f} idleFrac {:.3f} "
                    "| idleWarpFrac {:.3f} marchWarps {} busyLanes/marchWarp {:.1f} "
                    "| work(cells+taps+cands): total {} avgPerLane {:.1f} avgWaveMax {:.1f}",
                    mFrameCount, s[33], lanesPerWarp, laneUtil, idleFrac,
                    idleWarpFrac, (uint32_t)marchWarps, busyLanesPerMarchWarp,
                    s[31], dvLanes > 0.0 ? double(s[31]) / dvLanes : 0.0,
                    dvWarps > 0.0 ? dvMaxSum / dvWarps : 0.0
                );

                // Per-loop SIMD occupancy. THIS is the branch-divergence test:
                // occ = lane-iterations / iteration-slots burned. 1.0 means the
                // warp never runs a step with idle lanes; 0.2 means four fifths
                // of every step is wasted on lanes that already left the loop.
                // A low occ here with the high laneUtil above is the signature
                // of divergence the whole-kernel probe is blind to.
                const double rqOcc = s[39] > 0 ? double(s[38]) / (double(s[39]) * 32.0) : 0.0;
                const double ddaOcc = s[41] > 0 ? double(s[40]) / (double(s[41]) * 32.0) : 0.0;
                logInfo(
                    "[LOOPOCC] frame {} | rqTraversal occ {:.3f} laneIters {} warpIters {} "
                    "| coarseDDA occ {:.3f} laneIters {} warpIters {}",
                    mFrameCount, rqOcc, s[38], s[39], ddaOcc, s[40], s[41]
                );

                // shadeMain path-length divergence. Unlike main, every lane
                // holds a real path, so LOW occupancy here is directly the win
                // available to per-bounce requeueing (wavefront scheduling):
                // bounceOcc low + workOcc low = warps idle behind their longest
                // path; both near 1.0 = roulette keeps warps coherent and the
                // restructuring is not worth building.
                const double sbOcc = s[43] > 0 ? double(s[42]) / (double(s[43]) * 32.0) : 0.0;
                const double swOcc = s[45] > 0 ? double(s[44]) / (double(s[45]) * 32.0) : 0.0;
                logInfo(
                    "[SHADEOCC] frame {} | bounces occ {:.3f} laneSum {} warpMaxSum {} "
                    "| march work occ {:.3f} laneSum {} warpMaxSum {} | warpRRkills {} | queueClasses {}/{}/{}/{}",
                    mFrameCount, sbOcc, s[42], s[43], swOcc, s[44], s[45], s[72], s[73], s[74], s[75], s[76]
                );

                // Escape-walk E-bias probe (v4), binned by the DETERMINISTIC
                // key exp(-coarseOpticalDepth) - never by either walk's own
                // realization, which selection-biases the bins (v2/v3 defect).
                // The thick bins are where RR fires; v1's global sum hid them
                // under the sky's T~1 mass. Active only while trRRThreshold > 0
                // and mode bit3 is set.
                if (s[47] + s[49] + s[51] + s[53] > 0)
                {
                    auto bias = [&](int b) {
                        return s[47 + 2 * b] > 0 ? (double(s[46 + 2 * b]) / double(s[47 + 2 * b]) - 1.0) * 100.0 : 0.0;
                    };
                    logInfo(
                        "[TRRPROBE] frame {} | E-bias by det-key bin (Tdet=exp(-coarseTau)): Tdet<0.01 {:+.3f}% (ref {}) | 0.01-0.1 {:+.3f}% (ref {}) "
                        "| 0.1-0.5 {:+.3f}% (ref {}) | Tdet>0.5 {:+.3f}% (ref {})",
                        mFrameCount, bias(0), s[47], bias(1), s[49], bias(2), s[51], bias(3), s[53]
                    );
                }

                // [TRRPROBE2] Coin-level E-preservation, same det-key bins.
                // Per-event unbiasedness demands threshold*survives ==
                // sum(TrBefore), i.e. survives == sumBefore/64 in slot units.
                // coinBias ~ 0 while the walk bin reads -35% acquits the coin
                // and convicts the post-survival continuation (inner per-node
                // RR at GridVolumeSampler.slang:658 interaction, FP32);
                // coinBias matching the walk bias convicts the coin itself.
                if (s[54] + s[57] + s[60] + s[63] > 0)
                {
                    auto coinBias = [&](int b) {
                        const double before = double(s[55 + 3 * b]) / 64.0;
                        return before > 0 ? (double(s[56 + 3 * b]) / before - 1.0) * 100.0 : 0.0;
                    };
                    logInfo(
                        "[TRRPROBE2] frame {} | coin E-bias by det-key bin: Tdet<0.01 {:+.3f}% (coins {} surv {}) | 0.01-0.1 {:+.3f}% (coins {} surv {}) "
                        "| 0.1-0.5 {:+.3f}% (coins {} surv {}) | Tdet>0.5 {:+.3f}% (coins {} surv {})",
                        mFrameCount,
                        coinBias(0), s[54], s[56], coinBias(1), s[57], s[59],
                        coinBias(2), s[60], s[62], coinBias(3), s[63], s[65]
                    );
                    // Self-checks: survGtCoins and refCoins MUST be 0 - any
                    // nonzero invalidates every [TRRPROBE2] reading above.
                    // coinLanes is the RR exposure (tail-gated walks never
                    // reach the RR site: run 127 showed the 1080p-vs-maximized
                    // resolution change moved most thick pixels onto the tail
                    // path and the walk bias "vanished" for lack of exposure).
                    logInfo(
                        "[TRRPROBE2-CHK] frame {} | survGtCoins {} (must 0) | refCoins {} (must 0) | coinLanes {} | negTr armed {} ref {}",
                        mFrameCount, s[66], s[67], s[68], s[69], s[70]
                    );
                }

                // One line per sweep step, greppable as a table. warpMaxSum is
                // the number the sweep exists to move (the warp critical path);
                // occ is the derived utilisation; gpuMs is same-session here so
                // it is actually comparable across steps for once.
                if (sweepLogFrame)
                {
                    const uint32_t step = (mFrameCount - mSweepStartFrame) / mSweepFramesPerStep;
                    logInfo(
                        "[SWEEP] step {}/7 minQ {:.2f} startB {} | bounces occ {:.3f} laneSum {} warpMaxSum {} "
                        "| work occ {:.3f} | shadeCells+taps {} | gpuMs {:.3f}",
                        step + 1, mRouletteMinQ, mRouletteStartBounce,
                        sbOcc, s[42], s[43], swOcc, uint64_t(s[44]), mLastGpuMs
                    );
                }

                const double px = double(pixels);
                // Per-pixel averages: these are the units that explain the ms.
                // Multiply a per-pixel count by ~2M pixels to see the true
                // operation volume, and compare against gpuMs to find the
                // dominant term. No thresholds, no interpretation baked in.
                logInfo(
                    "[WORK] frame {} res {}x{} gpuMs {:.3f} (main {:.3f} + shade {:.3f}) | per-pixel: samplerCalls {:.1f} instSlabs {:.1f} "
                    "brickCands {:.1f} segments {:.1f} coarseWalks {:.2f} coarseCells {:.1f} sweepCells {:.1f} "
                    "sweepTaps {:.2f} | totals: instSlabs {} brickCands {} coarseCells {}",
                    mFrameCount, targetDim.x, targetDim.y, mLastGpuMs, mLastGpuMsA, std::max(0.0, (double)mLastGpuMs - mLastGpuMsA),
                    s[15] / px, s[9] / px, s[29] / px, s[10] / px, s[11] / px, s[12] / px, s[13] / px, s[14] / px,
                    s[9], s[29], s[12]
                );

                // Phase attribution of the sampler-internal marching. 'cells'
                // are majorant-DDA steps (range-tex fetch each), 'taps' are
                // real stochastic density fetches (the memory-bound op). This
                // is the cost the plain [WORK] counters were blind to; the
                // dominant bucket here is the optimization target.
                const uint64_t totalCells = (uint64_t)s[16] + s[18] + s[20] + s[13];
                const uint64_t totalTaps = (uint64_t)s[17] + s[19] + s[21] + s[14] + s[23];
                logInfo(
                    "[COST] frame {} | per-pixel cells/taps: escapeT {:.1f}/{:.1f} candGen {:.1f}/{:.1f} "
                    "shadeNEE {:.1f}/{:.1f} sweep {:.1f}/{:.1f} | overlap steps {:.2f} lookups {:.2f} "
                    "| tailRays {:.2f}/entry | totals: cells {} taps {} "
                    "| brickCache hit {:.1f}% ({}/{}) promote {:.1f}% ({}) | occSkip {:.1f}% ({})",
                    mFrameCount,
                    s[16] / px, s[17] / px, s[18] / px, s[19] / px,
                    s[20] / px, s[21] / px, s[13] / px, s[14] / px,
                    s[22] / px, s[23] / px,
                    s[15] > 0 ? double(s[24]) / s[15] : 0.0,
                    totalCells, totalTaps,
                    // Coherence of the brick-coherent tap path. The change is
                    // only worth keeping if this is high: a hit skips both the
                    // rangeMean and indirection loads AND removes the dependent
                    // hop into atlasTex. A low rate means revert, not tune.
                    (s[27] + s[28]) > 0 ? 100.0 * double(s[27]) / double(s[27] + s[28]) : 0.0,
                    s[27], s[27] + s[28],
                    // Lever-2 payoff: the share of ex-misses now answered by the
                    // prefetch entry (register moves; loads were issued a cell
                    // early, off the critical path). s[28] still counts a
                    // promote as a miss for hit-rate continuity, so the promote
                    // share is reported against the same denominator.
                    (s[27] + s[28]) > 0 ? 100.0 * double(s[71]) / double(s[27] + s[28]) : 0.0,
                    s[71],
                    // Occupancy-mask payoff (HANDOFF_6 6.3): the share of taps
                    // answered from the register-resident bitmask with no
                    // atlasTex read. totalTaps still counts these - they are
                    // real taps, just no longer memory-bound - so this is the
                    // fraction of the most-stalled load in the frame that is
                    // simply not issued any more.
                    totalTaps > 0 ? 100.0 * double(s[30]) / double(totalTaps) : 0.0,
                    s[30]
                );

                // [HOMOG] Uniformity of the cells the primary candidate sweep
                // actually traversed - the gauge that decides whether uniform
                // interior boxes can be collapsed into one analytic segment.
                // Histogram of sigma_mean/sigma_majorant in 8 bins over [0,1].
                // RIGHT-loaded (mass at ratio -> 1) = near-uniform = homogenize;
                // LEFT-loaded (mass near 0) = turbulent = homogenization only
                // adds bias. "homogenizable" = share of traversed cells with
                // ratio >= 0.75 (bins 6-7); that is the fraction of candGen work
                // an analytic segment could replace.
                {
                    uint64_t hb[8];
                    uint64_t hbTotal = 0;
                    for (int k = 0; k < 8; ++k) { hb[k] = s[77 + k]; hbTotal += hb[k]; }
                    const double hd = hbTotal > 0 ? double(hbTotal) : 1.0;
                    const double homogenizable = 100.0 * double(hb[6] + hb[7]) / hd;
                    const double nearConst = 100.0 * double(hb[7]) / hd;
                    logInfo(
                        "[HOMOG] frame {} | sweptCells {} | ratio(mean/maj) hist %: "
                        "[.00-.125] {:.1f} [.125-.25] {:.1f} [.25-.375] {:.1f} [.375-.50] {:.1f} "
                        "[.50-.625] {:.1f} [.625-.75] {:.1f} [.75-.875] {:.1f} [.875-1.0] {:.1f} "
                        "| homogenizable(>=0.75) {:.1f}% | near-const(>=0.875) {:.1f}%",
                        mFrameCount, hbTotal,
                        100.0 * double(hb[0]) / hd, 100.0 * double(hb[1]) / hd,
                        100.0 * double(hb[2]) / hd, 100.0 * double(hb[3]) / hd,
                        100.0 * double(hb[4]) / hd, 100.0 * double(hb[5]) / hd,
                        100.0 * double(hb[6]) / hd, 100.0 * double(hb[7]) / hd,
                        homogenizable, nearConst
                    );
                }

                // What the projected-error LoD actually decided this frame:
                // per-pixel footprint growth, the tail gate in the same units,
                // and per-instance "mip@footprint" - so LoD questions are
                // answered by the log, not by trigonometry.
                if (mUseBrickTlas && mBrickBlasesValid && !mInstanceFootprint.empty())
                {
                    std::string perInstance;
                    for (size_t i = 0; i < mInstanceFootprint.size(); ++i)
                    {
                        perInstance += fmt::format("{}@{:.1f}{}", mInstanceMip[i], mInstanceFootprint[i], i + 1 < mInstanceFootprint.size() ? " " : "");
                    }
                    logInfo(
                        "[LOD] frame {} | spread {:.5f} wu/dist | voxel {:.2f} wu, mip0 cell {:.2f} wu | tail gate {:.1f} wu "
                        "| per-instance mip@footprintWu: {}",
                        mFrameCount, mLastSpread, mTailMinVoxWorld, 8.f * mTailMinVoxWorld,
                        mTailGateVoxels * mTailMinVoxWorld, perInstance
                    );
                }
            }
            mpRisStatsReadback->unmap();
        }
    }

    // Cache this frame's camera for next frame's reprojection, and flip the
    // reservoir ping-pong AFTER the dispatch consumed the bindings.
    if (mpScene->getCamera())
    {
        mPrevViewProj = mpScene->getCamera()->getViewProjMatrixNoJitter();
        mPrevCamPos = mpScene->getCamera()->getPosition();
        mPrevCamValid = true;
    }
    mReservoirFrame++;

    mFrameCount++;
}

void VolumePathTracer::renderUI(Gui::Widgets& widget)
{
    bool rebuild = false;

    // Internal render resolution. Changing it changes reflect() output dims, so
    // the graph must recompile. Remember to set the SAME size on the downstream
    // Accum/Composite/ToneMapper passes.
    if (widget.dropdown("Render size", mOutputSizeSelection))
        requestRecompile();
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
    {
        if (widget.var("Fixed render size (px)", mFixedOutputSize, 32u, 16384u))
            requestRecompile();
    }
    widget.tooltip("Internal cloud render resolution (Nubis-style low-res + upscale). Lower = faster and softer; upscaled to the window at present with no stretch.", true);

    rebuild |= widget.var("Max bounces", mMaxBounces, 0u, 1024u);
    widget.tooltip("Maximum number of volume scattering events per path.\nClouds need many bounces to look right.", true);

    rebuild |= widget.checkbox("Use NEE", mUseNEE);
    widget.tooltip("Next-event estimation to the environment map.", true);

    if (mUseNEE)
    {
        rebuild |= widget.checkbox("Single NEE per path (MegaLights budget)", mUseSingleNeePerPath);
        widget.tooltip(
            "ONE reservoir-picked vertex does NEE per path - one shadow ray per\n"
            "pixel instead of one per vertex - weighted by vertex count.\n"
            "Unbiased (converged image must match per-vertex NEE exactly);\n"
            "trades the dominant [COST] bucket for noise that temporal\n"
            "accumulation absorbs. The MegaLights fixed-budget shape.",
            true
        );
    }

    rebuild |= widget.checkbox("Use Russian roulette", mUseRussianRoulette);

    rebuild |= widget.checkbox("Log work profile", mLogWorkStats);
    widget.tooltip("Writes a [WORK] line: real GPU ms for this pass plus the per-pixel operation\ncounts that time is spent on. Costs a GPU sync on logged frames only.", true);
    // Governs BOTH the [WORK] and [RIS] lines, so it lives outside the RIS group.
    widget.var("Log interval (frames)", mRisStatsInterval, 1u, 600u);
    widget.tooltip("Frames between log lines. Each logged frame costs a full GPU sync:\nat interval 1 the stall alone was measured to turn a 24 ms pass into a\n77 ms frame. Keep this high unless single-frame counters are needed.", true);
    if (mLogWorkStats)
        widget.text("Last GPU ms: " + std::to_string(mLastGpuMs));

    if (auto group = widget.group("RIS (section 5, Stage A)", true))
    {
        rebuild |= group.checkbox("Use RIS", mUseRIS);
        group.tooltip(
            "Resampled importance sampling of the PRIMARY scatter vertex:\n"
            "M candidates that are REAL delta-tracking collisions, weighted by a\n"
            "cheap lighting target, one survivor shaded exactly.\n"
            "Unbiased: the converged image must match the reference exactly;\n"
            "only 1-spp noise may change.",
            true
        );
        if (mUseRIS)
        {
            rebuild |= group.var("Candidates (M)", mRisCandidates, 1u, 16u);
            rebuild |= group.checkbox("Adaptive M (from transmittance)", mUseAdaptiveM);
            group.tooltip(
                "Per-pixel candidate budget M = clamp(ceil(Mmax * (1 - T)), 1, Mmax),\n"
                "decided from the escape term's transmittance before the candidate RNG.\n"
                "Unbiased (the reservoir weight divides by the M actually used).\n"
                "Measured: 82-88% of fixed-M processes escape - this stops paying them.",
                true
            );
            rebuild |= group.checkbox("Shared candidate sweep", mUseSharedCandidateSweep);
            group.tooltip(
                "ON: all M delta-tracking processes share ONE majorant-DDA traversal\n"
                "(one range-tex fetch per cell instead of M). Distribution-identical\n"
                "to M independent walks; toggle OFF to A/B cost and convergence.",
                true
            );
            rebuild |= group.checkbox("Temporal reuse (Stage B)", mUseTemporalReuse);
            group.tooltip(
                "Merge last frame's surviving scatter vertex into this frame's\n"
                "reservoir (t-shift + confidence-capped weight, re-validated with a\n"
                "real density tap and bounded transmittance march). Effective M\n"
                "grows over frames for the cost of ~one extra bounded march.\n"
                "Gate: converged image must match Stage A exactly.",
                true
            );
            if (mUseTemporalReuse)
            {
                group.var("Temporal M cap (x budget)", mTemporalMCap, 1.f, 100.f);
                group.tooltip("History confidence cap as a multiple of the current per-pixel\ncandidate budget. Longer memory = smoother but slower reaction.", true);
                rebuild |= group.checkbox("Spatial reuse (Stage C)", mUseSpatialReuse);
                group.tooltip(
                    "Merge prev-frame reservoirs from Gaussian-offset NEIGHBOR pixels\n"
                    "through the exact Stage B shift/guard/weight. Effective M grows\n"
                    "spatially AND temporally; fresh candidate work can then shrink.\n"
                    "Gate: converged image must match Stage A exactly (matrix config 13).",
                    true
                );
                if (mUseSpatialReuse)
                {
                    rebuild |= group.var("Spatial neighbors", mSpatialNeighbors, 1u, 8u);
                    group.var("Spatial sigma (px)", mSpatialRadiusPx, 1.f, 64.f);
                    group.tooltip("Gaussian sigma of the neighbor offsets. ~16 px matches the\nclassic 30 px uniform disk (ReSTIR PT Enhanced, Fig. 7).", true);
                }
            }
            group.var("Tracker RR threshold (0=off)", mTrRRThreshold, 0.f, 0.5f);
            group.tooltip(
                "Weight RR on the RQ transmittance trackers: below this, survive\n"
                "w.p. Tr/threshold reweighted to the threshold (unbiased). Dead\n"
                "tracker => NEE walk ends, fused walk commit-shrinks to the\n"
                "farthest candidate. Live CB value - A/B in one session.",
                true
            );
            group.var("Target floor (x isotropic)", mRisTargetFloor, 0.f, 1.f);
            group.var("Roulette min q", mRouletteMinQ, 0.f, 1.f);
            group.tooltip(
                "Russian-roulette survival floor: q = max(this, 1 - maxThroughput).\n"
                "shadeMain (the bounce loop) is the largest single item in the frame\n"
                "and its cost is path LENGTH; in a high-albedo cloud throughput stays\n"
                "near 1 so q sits at this floor. Unbiased at any value - raising it\n"
                "shortens paths and trades cost for variance.",
                true
            );
            group.var("Roulette start bounce", mRouletteStartBounce, 0u, 64u);
            group.tooltip("Roulette applies only past this bounce index. Lower = shorter paths.", true);

            if (!mSweepActive)
            {
                if (group.button("Run roulette sweep"))
                {
                    mSweepSavedMinQ = mRouletteMinQ;
                    mSweepSavedStartBounce = mRouletteStartBounce;
                    mSweepStartFrame = mFrameCount + 1; // Controller waits until reached (no wrap).
                    mSweepActive = true;
                    if (!mLogWorkStats)
                    {
                        // The sweep reads the shadeMain divergence counters,
                        // which only compile under work stats.
                        mLogWorkStats = true;
                        mpPass = nullptr;
                        logInfo("[SWEEP] work stats were off - enabling them for the sweep (recompiles once).");
                    }
                    logInfo(
                        "[SWEEP] starting: baseline ({:.2f}, {}) then minQ {{0.10, 0.15, 0.30}} @ startB {} "
                        "and {{0.05, 0.15, 0.30}} @ startB 2, {} frames per step.",
                        mSweepSavedMinQ, mSweepSavedStartBounce, mSweepSavedStartBounce, mSweepFramesPerStep
                    );
                }
                group.tooltip(
                    "ReSTIR PT Enhanced 6.2.4 experiment, unattended: steps the two\n"
                    "roulette CB values through 7 configs, ~1 [SWEEP] log line each,\n"
                    "then restores. Same session, no recompiles between steps.\n"
                    "Watch: warpMaxSum down + bounces occ up = the tail is roulette-\n"
                    "tunable and per-bounce requeueing is NOT worth building.\n"
                    "Judge the image per step too - this knob trades variance.",
                    true
                );
                group.var("Sweep frames/step", mSweepFramesPerStep, 16u, 600u);
            }
            else
            {
                const uint32_t step =
                    mFrameCount >= mSweepStartFrame ? std::min((mFrameCount - mSweepStartFrame) / mSweepFramesPerStep + 1, 7u) : 1u;
                group.text(fmt::format("Sweep running: step {}/7 (minQ {:.2f}, startB {})", step, mRouletteMinQ, mRouletteStartBounce));
                if (group.button("Cancel sweep"))
                {
                    mRouletteMinQ = mSweepSavedMinQ;
                    mRouletteStartBounce = mSweepSavedStartBounce;
                    mSweepActive = false;
                    mOptionsChanged = true;
                }
            }
            group.tooltip(
                "Defensive floor on the RIS target, relative to a fully-lit isotropic\n"
                "vertex. Bounds the L/Lhat firefly mechanism (matrix: isolated 800-2200x\n"
                "pixels at 1 spp). Unbiased for any value; 0 = raw target.",
                true
            );
            rebuild |= group.checkbox("Compacted shading (two-phase)", mUseCompaction);
            group.tooltip(
                "Stream compaction (ReSTIR PT Enhanced 6.2.2 / UE): per-pixel phase A\n"
                "queues the ~13% of pixels whose reservoir scattered; an indirect\n"
                "phase B shades one thread per REAL path in dense waves instead of\n"
                "87%-idle warps. Estimator-identical (matrix config 15).",
                true
            );
            if (mUseCompaction)
            {
                rebuild |= group.checkbox("Wavefront bounces (per-bounce requeue)", mUseWavefront);
                group.tooltip(
                    "[SHADEOCC] measured the fused bounce loop at 0.271 lane occupancy:\n"
                    "the average path dies at 8 bounces, the average warp's LONGEST runs\n"
                    "29, and 27 of 32 lanes idle behind it. This advances every live path\n"
                    "ONE bounce per dispatch and requeues survivors, keeping warps dense\n"
                    "(ceiling ~2.6x on shadeMain's marching).\n\n"
                    "BYTE-IDENTICAL estimator: the per-bounce body is the fused loop body\n"
                    "verbatim and the RNG state rides in PathState, never re-seeded.\n"
                    "A/B against the fused loop with this checkbox - converged images\n"
                    "must be BIT-identical, stronger than the +-0.15% gate.",
                    true
                );
                if (mUseWavefront)
                {
                    group.var("Wavefront rounds", mWavefrontRounds, 0u, 64u);
                    group.tooltip(
                        "Dense requeue rounds before the tail finisher loops survivors\n"
                        "to death. Each round costs args + clear + dispatch with\n"
                        "serializing barriers, so it only pays while the queue is fat\n"
                        "(avg path dies at 8). LIVE knob, no recompile - sweep it and\n"
                        "watch gpuMs same-session. 0 = tail-only (~fused schedule),\n"
                        "64 = pure wavefront (measured much slower: barrier overhead).",
                        true
                    );
                }
            }
            rebuild |= group.checkbox("Radiance cache CV (final gather)", mUseRadCache);
            group.tooltip(
                "Volumetric final gather, unbiased: the deep-bounce tail's MEAN\n"
                "moves into a world-grid cache trained by 1-in-N paths; consuming\n"
                "paths take the cache at the cut vertex and continue only to\n"
                "estimate the residual (L - C). Roulette past q=0.125 FAILED the\n"
                "matrix gate because it deleted real energy; the same kill rates\n"
                "on residuals delete a term with mean ~0. Unbiased for ANY cache\n"
                "content - staleness/coarseness cost residual variance only.\n"
                "Give the cache ~2-5 s to train after enabling (confidence gate).",
                true
            );
            if (mUseRadCache)
            {
                group.var("Cut bounce (0=off, live)", mRadCutBounce, 0u, 16u);
                group.tooltip("Bounce index where the CV applies. 0 = runtime off switch\nfor same-session A/B - no recompile.", true);
                group.var("Residual survival p", mRadResidualSurvival, 0.01f, 1.f);
                group.var("Warp-RR lanes (0=off)", mRadWarpRRLanes, 0u, 32u);
                group.tooltip(
                    "Lever 1 (2026-07-20): when fewer than this many lanes of a\n"
                    "shadeMain wave are alive, residual paths roulette per bounce\n"
                    "with p = alive/lanes (1/p compensated - unbiased). Attacks\n"
                    "the 0.229 bounce occupancy: warps idling behind their longest\n"
                    "paths, which after the cut carry only mean-zero residual.\n"
                    "ESTIMATOR CHANGE - gate via the matrix sweep before adopting.",
                    true
                );
                group.tooltip("Survival probability past the cut. 1.0 = CV bookkeeping only\n(pure variance reduction, no cost win). Lower = shorter paths,\nmore residual variance. The matrix prices it.", true);
                group.var("Train 1-in-N", mRadTrainEvery, 1u, 64u);
                group.tooltip("Training pixels run FULL paths (deposit, never consume), so the\ncache never learns from itself.", true);
                group.var("Cache EMA", mRadEma, 0.001f, 1.f);
                if (group.var("Cache res", mRadCacheRes, 8u, 512u))
                {
                    mpRadAccum = nullptr; // Recreate next frame.
                    mpRadTex = nullptr;
                }
            }
            rebuild |= group.var("Coarse mip", mRisMip, 0u, 3u);
            group.tooltip("Mean-pyramid mip for the target's shadow walks.\nCoarser = cheaper, cruder 'is this candidate lit' guess.", true);
            rebuild |= group.checkbox("Log branch histogram", mLogRisStats);
            group.tooltip("Writes a [RIS] line to the Mogwai log showing which branch\nended the RIS block for every pixel. Costs a GPU sync per logged frame.", true);
        }
    }

    if (auto group = widget.group("LoD + acceleration (UE lessons)", true))
    {
        rebuild |= group.checkbox("Brick TLAS (HW-BVH)", mUseBrickTlas);
        group.tooltip(
            "UE HeterogeneousVolumes port: occupied bricks are procedural AABBs in\n"
            "per-(grid,mip) BLASes; a TLAS instances the mip chosen per grid-volume\n"
            "from projected error. Traversal is an inline RayQuery - no interval\n"
            "lists, no spill. All backends are unbiased: converged images must be\n"
            "identical to the interval backend.",
            true
        );
        group.checkbox("Occupancy tap skip", mUseOccupancySkip);
        group.tooltip(
            "HANDOFF_6 6.3 / UE Nanite Foliage brick mask. A tap landing in an\n"
            "all-zero BC4 tile returns the cached minorant from a register-\n"
            "resident bitmask instead of reading atlasTex - the load that\n"
            "profiles at LONG SCOREBOARD 83-86%.\n\n"
            "EXACT: a clear bit means every voxel in the tile decodes to exactly\n"
            "0, so the skipped result is bit-identical. Toggling this must not\n"
            "change the image or any work counter - only occSkip and gpuMs.\n\n"
            "Flip it WITHIN one session and compare [COST]. Cross-session gpuMs\n"
            "here has twice drifted ~25% with identical work counters, which is\n"
            "bigger than the effect being measured.",
            true
        );
        rebuild |= group.checkbox("Brick prefetch", mUseBrickPrefetch);
        group.tooltip(
            "Lever 2 (2026-07-20): speculative next-brick prefetch in the DDA\n"
            "walks. The marching loop is a serial chain of dependent loads\n"
            "(LONG SCOREBOARD 30% in the Nsight capture, SM 50% / VRAM 12% =\n"
            "latency-bound); the next cell's address is pure ALU, so its\n"
            "range/indirection/occupancy loads go in flight while the current\n"
            "cell's value is still outstanding. EXACT: promotes reuse the same\n"
            "texels; a wasted prefetch is dead loads, never wrong data.\n\n"
            "COMPILE-TIME toggle (rebuilds the program): run 137 measured the\n"
            "runtime-flag version at +12ms with identical counters - the ~9\n"
            "registers of prefetch entry cost occupancy whether or not the\n"
            "flag reads them. Same-session A/B still holds: flipping this\n"
            "recompiles without restarting the process.",
            true
        );
        group.checkbox("Scatter-queue sort", mUseScatterSort);
        group.tooltip(
            "Lever 1b (2026-07-20): counting sort of the shade queue by last\n"
            "frame's transmittance hint, so warp-mates get similar expected\n"
            "path lengths ([SHADEOCC] bounce occ 0.229 = warps idling behind\n"
            "their longest path). IMAGE-IDENTICAL: shadeMain seeds by pixel,\n"
            "so slot order never enters the estimator - this is scheduling,\n"
            "same class as compaction. Flip within one session; watch\n"
            "[SHADEOCC] bounces occ / warpMaxSum and gpuMs.",
            true
        );

        if (mUseBrickTlas)
        {
            group.var("Mip pixel threshold", mMipPixelThreshold, 0.125f, 16.f);
            group.tooltip(
                "Pixels one acceleration-structure cell may span before the next\n"
                "coarser mip's BLAS is instanced (1 = cell ~ pixel, Nanite's rule).\n"
                "Takes effect on the next TLAS refresh - no shader rebuild.",
                true
            );
        }

        rebuild |= group.checkbox("Tau shadow cache", mUseTauCache);
        group.tooltip(
            "Nubis3 (SIGGRAPH 2023) / UE transmittance-volume port: the RIS\n"
            "target's per-candidate shadow walk becomes ONE trilinear fetch of\n"
            "precomputed optical depth toward the dominant env direction.\n\n"
            "The walk it replaces measured 39.5M DDA cells/frame at 0.342 SIMD\n"
            "occupancy - 3.6x the warp-slots of the entire RayQuery traversal\n"
            "([LOOPOCC] 2026-07-20). UNBIASED: tau only shapes which candidate\n"
            "wins and the estimator divides the target back out, so cache\n"
            "coarseness costs selection variance, never the converged image.\n\n"
            "This checkbox COMPILES the feature; A/B with the runtime toggle\n"
            "below (no recompile, same-session per measurement discipline).",
            true
        );
        if (mUseTauCache)
        {
            group.checkbox("Apply tau cache (A/B)", mTauCacheApply);
            group.tooltip("Runtime switch (CB uniform). OFF = live DDA walk, same session.", true);
            if (group.var("Tau cache res", mTauCacheRes, 8u, 512u))
                mTauLastBuildFrame = kTauNeverBuilt;
            group.tooltip("Longest-axis cell count. Changing it rebakes next frame.", true);
            if (group.var("Tau rebuild interval", mTauCacheInterval, 0u, 600u))
                mTauLastBuildFrame = kTauNeverBuilt;
            group.tooltip("Frames between rebakes. 0 = bake once (static scene + env).", true);
        }

        rebuild |= group.checkbox("Merged coarse tail", mUseMergedTail);
        group.tooltip(
            "UE Nanite Assemblies lesson: far rays march ONE world-space grid of\n"
            "conservative summed majorants instead of per-instance structures.\n"
            "Real density taps still hit the true fields - exactly unbiased.",
            true
        );
        if (mUseMergedTail)
        {
            group.var("Tail resolution", mTailRes, 16u, 256u);
            group.tooltip("Longest-axis cell count of the tail grid. Rebake on scene reload.", true);
            group.var("Tail gate (fine voxels)", mTailGateVoxels, 1.f, 1024.f);
            group.tooltip(
                "A ray uses the tail when its pixel footprint at the cloud entry\n"
                "spans at least this many fine voxels. Applied next frame\n"
                "(the gate is recomputed from this value at bind time).",
                true
            );
        }

        rebuild |= group.checkbox("Footprint-driven residual mip", mFootprintMip);
        group.tooltip(
            "UE Nanite projected-error rule applied to the residual transmittance\n"
            "estimator: per instance and segment, pick the coarsest mean/range mip\n"
            "whose cell size stays under the pixel footprint at that distance.\n"
            "Unbiased at every mip - converged image must not change (gate:\n"
            "VNA_ResidualSweep.py). Applies to the primary ray's escape term.",
            true
        );
        if (mFootprintMip)
        {
            group.var("Footprint scale", mFootprintScale, 0.125f, 8.f);
            group.tooltip("1 = coarse cell size matches one pixel. Larger = coarser cells\n(fewer DDA steps, looser ranges = more residual taps).", true);
        }
    }

    if (mpVolumeSampler)
    {
        if (auto group = widget.group("Volume sampler", true))
        {
            if (mpVolumeSampler->renderUI(group))
            {
                // Keep the pass-level copy in sync so getProperties() reflects
                // what the UI selected (the sampler owns the live options).
                mVolumeSamplerOptions = mpVolumeSampler->getOptions();
                rebuild = true;
            }
        }
    }

    // All of these are compile-time constants in the shader, so changing any of
    // them requires rebuilding the program and restarting accumulation.
    if (rebuild)
    {
        mpPass = nullptr;
        mOptionsChanged = true;
    }
}
