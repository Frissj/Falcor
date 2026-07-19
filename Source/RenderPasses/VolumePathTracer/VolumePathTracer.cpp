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
    { "work",  "gWorkDebug",   "Raw work counters: R=segments G=overlapSteps B=aabbTests A=maxCover", true, ResourceFormat::RGBA32Float },
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
const char kUseBrickTlas[] = "useBrickTlas";
const char kMipPixelThreshold[] = "mipPixelThreshold";
const char kUseMergedTail[] = "useMergedTail";
const char kTailRes[] = "tailRes";
const char kTailGateVoxels[] = "tailGateVoxels";
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
        else if (key == kUseBrickTlas)
            mUseBrickTlas = value;
        else if (key == kMipPixelThreshold)
            mMipPixelThreshold = value;
        else if (key == kUseMergedTail)
            mUseMergedTail = value;
        else if (key == kTailRes)
            mTailRes = value;
        else if (key == kTailGateVoxels)
            mTailGateVoxels = value;
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
    props[kUseBrickTlas] = mUseBrickTlas;
    props[kMipPixelThreshold] = mMipPixelThreshold;
    props[kUseMergedTail] = mUseMergedTail;
    props[kTailRes] = mTailRes;
    props[kTailGateVoxels] = mTailGateVoxels;
    return props;
}

RenderPassReflection VolumePathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void VolumePathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass = nullptr;
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
        if (bricked.rangeData.empty())
        {
            logWarning("VolumePathTracer: grid {} has no CPU bricked data; brick TLAS disabled.", g);
            return;
        }

        for (uint32_t mip = 0; mip < 4; ++mip)
        {
            BrickBlas& blas = mBrickBlas[g][mip];
            blas.aabbOffset = (uint32_t)aabbs.size();

            const int3 dim = bricked.leafDim[mip];
            const uint32_t* range = bricked.rangeData.data() + bricked.leafOffset[mip];
            const float cellSize = float(8u << mip);
            for (int z = 0; z < dim.z; ++z)
            {
                for (int y = 0; y < dim.y; ++y)
                {
                    for (int x = 0; x < dim.x; ++x)
                    {
                        const uint32_t packed = range[(z * dim.y + y) * dim.x + x];
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
        if (!pGrid || pGrid->getBrickedGrid().rangeData.empty()) continue;
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
                        const uint32_t* range = src.bricked->rangeData.data() + src.bricked->leafOffset[3];
                        const uint16_t* mean = src.bricked->meanData.data() + src.bricked->leafOffset[3];
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
                                    const uint32_t packed = range[(cz * dimM.y + cy) * dimM.x + cx];
                                    instMaj = std::max(instMaj, f16tof32((uint16_t)(packed & 0xffffu)));
                                    instMeanSum += f16tof32(mean[(cz * dimM.y + cy) * dimM.x + cx]);
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

void VolumePathTracer::prepareProgram(RenderContext* pRenderContext)
{
    FALCOR_ASSERT(mpScene && mpVolumeSampler);

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    desc.addTypeConformances(mpScene->getTypeConformances());

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

    // Bind resources that never change for the lifetime of the program.
    // Plain bindShaderData is still correct for the SCENE block: this pass
    // never traces against the scene's TLAS. The brick TLAS is our own,
    // pass-owned structure, bound separately in execute() each frame.
    auto var = mpPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    mpSampleGenerator->bindShaderData(var);
    if (mpEnvMapSampler)
        mpEnvMapSampler->bindShaderData(var["gParams"]["envMapSampler"]);
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

    // Tell the shader which optional outputs are actually connected.
    mpPass->getProgram()->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    auto var = mpPass->getRootVar();

    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFrameDim"] = targetDim;

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
    var["CB"]["gFootprintSpread"] = footprintSpread;

    // HW-BVH brick backend: refresh the TLAS (per-instance mip = projected
    // error; rebuild only when a selection changed) and bind its resources.
    if (mUseBrickTlas && mBrickBlasesValid)
    {
        updateBrickTlas(pRenderContext, targetDim);
        if (mpBrickTlas)
        {
            var["gBrickTlas"].setAccelerationStructure(mpBrickTlas);
            var["gBrickTlasInstances"] = mpBrickInstanceInfo;
            var["gBrickAABBs"] = mpBrickAABBs;
        }
    }

    // Merged coarse tail: bind the baked summed-field grid + its constants.
    if (mUseMergedTail && mpTailTex)
    {
        var["gTailTex"] = mpTailTex;
        var["TailCB"]["gTailOrigin"] = mTailOrigin;
        var["TailCB"]["gTailFootprintGate"] = mTailGateVoxels * mTailMinVoxWorld;
        var["TailCB"]["gTailCellSize"] = mTailCellSize;
        var["TailCB"]["gTailEnabled"] = 1u;
        var["TailCB"]["gTailDim"] = mTailDim;
    }

    for (auto channel : kOutputChannels)
        var[channel.texname] = renderData.getTexture(channel.name);

    var["gRisStats"] = mpRisStats;

    // Log the RIS branch histogram every mRisStatsInterval frames. The readback
    // needs a full GPU sync, so it is throttled rather than run every frame.
    const bool logThisFrame = (mLogWorkStats || (mUseRIS && mLogRisStats)) && mpRisStats && (mFrameCount % mRisStatsInterval == 0);
    if (logThisFrame)
        pRenderContext->clearUAV(mpRisStats->getUAV().get(), uint4(0, 0, 0, 0));

    // Time the dispatch on the GPU. A CPU timer here would measure command
    // submission, not the work, and would report near-zero regardless of cost.
    const bool timeThisFrame = mLogWorkStats && mpGpuTimer && logThisFrame;
    if (timeThisFrame)
        mpGpuTimer->begin();

    mpPass->execute(pRenderContext, uint3(targetDim, 1));

    if (timeThisFrame)
    {
        mpGpuTimer->end();
        mpGpuTimer->resolve();
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
                    "zeroTrans {} terminated {} contributed {} | avgCandidates {:.2f} avgM {:.2f}",
                    mFrameCount, pixels, inMedium,
                    s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                    avgCand, pixels > 0 ? double(s[25]) / pixels : 0.0
                );
            }

            if (mLogWorkStats)
            {
                if (mpGpuTimer)
                    mLastGpuMs = mpGpuTimer->getElapsedTime();

                const double px = double(pixels);
                // Per-pixel averages: these are the units that explain the ms.
                // Multiply a per-pixel count by ~2M pixels to see the true
                // operation volume, and compare against gpuMs to find the
                // dominant term. No thresholds, no interpretation baked in.
                logInfo(
                    "[WORK] frame {} res {}x{} gpuMs {:.3f} | per-pixel: samplerCalls {:.1f} aabbTests {:.1f} "
                    "segments {:.1f} coarseWalks {:.2f} coarseCells {:.1f} sweepCells {:.1f} sweepTaps {:.2f} "
                    "| totals: aabb {} coarseCells {}",
                    mFrameCount, targetDim.x, targetDim.y, mLastGpuMs,
                    s[15] / px, s[9] / px, s[10] / px, s[11] / px, s[12] / px, s[13] / px, s[14] / px,
                    s[9], s[12]
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
                    "| tailRays {:.2f}/entry | totals: cells {} taps {}",
                    mFrameCount,
                    s[16] / px, s[17] / px, s[18] / px, s[19] / px,
                    s[20] / px, s[21] / px, s[13] / px, s[14] / px,
                    s[22] / px, s[23] / px,
                    s[15] > 0 ? double(s[24]) / s[15] : 0.0,
                    totalCells, totalTaps
                );

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

    mFrameCount++;
}

void VolumePathTracer::renderUI(Gui::Widgets& widget)
{
    bool rebuild = false;

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
    widget.tooltip("1 = every frame (the default while frame times are pathological).\nRaise it once the renderer is fast again.", true);
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
