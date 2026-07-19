/***************************************************************************
 # Reference volumetric path tracer for NanoVDB grid volumes.
 #
 # Brute-force ground truth: no ReSTIR, no reuse, no LoD. Pair with
 # AccumulatePass to converge a frame, and use that image to validate
 # faster (resampled) estimators against.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Core/API/GpuTimer.h"
#include "Core/API/Raytracing.h"
#include "Core/API/RtAccelerationStructure.h"
#include "RenderGraph/RenderPass.h"
#include "Rendering/Volumes/GridVolumeSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Utils/Sampling/SampleGenerator.h"
#include <array>
#include <vector>

using namespace Falcor;

class VolumePathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VolumePathTracer, "VolumePathTracer", "Reference volumetric path tracer for grid volumes.");

    static ref<VolumePathTracer> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VolumePathTracer>(pDevice, props);
    }

    VolumePathTracer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    void parseProperties(const Properties& props);
    void prepareProgram(RenderContext* pRenderContext);

    // HW-BVH brick backend (VNA section 4; see VNA_UE_SOURCE_LESSONS.md).
    void buildBrickBlases(RenderContext* pRenderContext);
    void updateBrickTlas(RenderContext* pRenderContext, const uint2 targetDim);
    uint32_t selectInstanceMip(const GridVolume& volume, const float3& cameraPos, float footprintSpread) const;

    // Merged coarse tail (VNA section 3).
    void bakeMergedTail();

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<GridVolumeSampler> mpVolumeSampler;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    /// Sampler options survive here across setScene() (the sampler object is
    /// recreated there) and are settable from Python, so estimator sweeps -
    /// e.g. validating that residual ratio tracking converges to the same
    /// image at every mip - can be scripted instead of clicked.
    GridVolumeSampler::Options mVolumeSamplerOptions;

    // Options.
    uint32_t mMaxBounces = 64u;    ///< Cloud multiple scattering needs many bounces; this is a reference renderer.
    bool mUseNEE = true;           ///< Next-event estimation to the environment.
    bool mUseRussianRoulette = true;
    /// UE MegaLights lesson: ONE reservoir-picked vertex does NEE per path
    /// (one shadow ray per pixel), weighted by vertex count - unbiased, fixed
    /// budget. OFF by default: per-vertex NEE is the reference estimator, and
    /// [COST] measured it as the dominant bucket (shadeNEE ~43 cells/30 taps
    /// per pixel), which is exactly what this trades against variance.
    bool mUseSingleNeePerPath = false;

    // VNA section 5, Stage A (RIS on the primary scatter vertex).
    // OFF by default: the plain path is the reference ground truth.
    bool mUseRIS = false;          ///< Enable resampled importance sampling of the primary vertex.
    /// Candidates per pixel (M). Each is a full delta-tracking process; with the
    /// shared sweep all M share one traversal (one range-tex fetch per DDA cell
    /// instead of M), so only the real-tap cost stays linear in M.
    uint32_t mRisCandidates = 4u;
    /// Generate the M candidates in ONE shared majorant-DDA sweep instead of M
    /// independent walks. Distribution-identical (each process keeps its own
    /// exponential chain and acceptance draws); only traversal cost is shared.
    /// Kept toggleable so cost and convergence can be A/B'd against the M-walk
    /// variant with a single switch.
    bool mUseSharedCandidateSweep = true;
    /// Footprint-driven residual-mip selection (UE Nanite projected-error rule:
    /// a voxel level's error is its cell size in world units; pick the level
    /// whose cells match the pixel footprint). Runtime-only - no shader rebuild.
    /// Unbiased at every mip, so this may change cost and noise, never the
    /// converged image (VNA_ResidualSweep.py is the gate).
    bool mFootprintMip = false;
    /// Scales the footprint used for mip selection (1 = cell size ~ pixel).
    float mFootprintScale = 1.f;

    // ---- HW-BVH brick backend (VNA section 4, UE port) ----------------------
    /// Traverse per-brick procedural AABBs with inline ray queries instead of
    /// the interval machinery. Per-(grid,mip) BLASes are built once; a TLAS
    /// instances the mip selected per grid-volume from projected error, so LoD
    /// is a property of the acceleration structure and a whole instance flips
    /// level coherently (the anti-crack rule).
    bool mUseBrickTlas = true;
    /// Pixels one acceleration-structure cell may span before the next coarser
    /// mip is selected (1 = cell ~ pixel, the Nanite balance point).
    float mMipPixelThreshold = 1.f;

    /// Unique grids referenced by the scene's grid volumes, in discovery order.
    std::vector<ref<Grid>> mUniqueGrids;
    /// Per grid-volume instance: index into mUniqueGrids.
    std::vector<uint32_t> mInstanceGridIdx;
    /// Per grid-volume instance: currently selected mip (drives TLAS rebuild).
    std::vector<uint32_t> mInstanceMip;

    struct BrickBlas
    {
        ref<Buffer> pBuffer;                  ///< Backing memory for the built BLAS.
        ref<RtAccelerationStructure> pBlas;   ///< The BLAS object.
        uint32_t aabbOffset = 0;              ///< First AABB in mpBrickAABBs.
        uint32_t aabbCount = 0;               ///< Number of brick AABBs at this mip.
    };
    /// Per unique grid: one BLAS per mip (4). Bricks at mip m are (8 << m)
    /// voxels wide; their AABBs live in SHIFTED index space.
    std::vector<std::array<BrickBlas, 4>> mBrickBlas;
    ref<Buffer> mpBrickAABBs;                 ///< All grids' brick AABBs, all mips, concatenated (RtAABB layout).
    ref<Buffer> mpBrickInstanceInfo;          ///< Per-TLAS-instance {gridVolumeIdx, aabbOffset}.
    ref<Buffer> mpBrickTlasBuffer;            ///< TLAS backing memory.
    ref<Buffer> mpBrickTlasScratch;           ///< TLAS build scratch.
    ref<RtAccelerationStructure> mpBrickTlas; ///< The TLAS.
    RtAccelerationStructurePrebuildInfo mBrickTlasPrebuild{};
    bool mBrickBlasesValid = false;

    // ---- Merged coarse tail (VNA section 3, UE Nanite Assemblies lesson) ----
    /// Far rays march ONE world-space grid of conservative summed majorants +
    /// approximate summed means instead of per-instance structures. Real taps
    /// still hit the true fields, so the estimator stays exactly unbiased.
    bool mUseMergedTail = true;
    /// Longest-axis resolution of the tail grid.
    uint32_t mTailRes = 64;
    /// Ray uses the tail when its pixel footprint at the cloud-region entry
    /// spans at least this many FINE voxels (of the smallest-voxel instance).
    float mTailGateVoxels = 32.f;
    ref<Texture> mpTailTex;
    float3 mTailOrigin = float3(0.f);
    float3 mTailCellSize = float3(1.f);
    uint3 mTailDim = uint3(0);
    /// Smallest fine-voxel world size over all instances, captured at bake so
    /// the gate (mTailGateVoxels * this) tracks UI changes without a rebake.
    float mTailMinVoxWorld = 0.f;
    /// Mip of the coarse mean field used for the TARGET's shadow estimate (0..3).
    /// Candidates no longer come from this field (they are real delta-tracking
    /// collisions), so this only decides how cheap/crude the "is this candidate
    /// lit" guess is. Measured at 159M cells/frame at mip 0 - the single largest
    /// cost in the pass - so it defaults coarse. Being crude here costs variance
    /// only, never correctness.
    uint32_t mRisMip = 2u;

    // RIS diagnostics. The shader tallies which branch ended the RIS block for
    // every pixel; this is read back and written to the log as a histogram, so
    // the answer is a text line rather than a colour to squint at.
    // Slots 16..23 are the [COST] phase buckets: the marching done INSIDE
    // GridVolumeSampler (DDA cells + real density taps) attributed to the
    // escape term, candidate generation, and shading/NEE - the work the
    // original [WORK] line was measured to be blind to.
    static const uint32_t kRisStatSlots = 25;
    ref<Buffer> mpRisStats;         ///< Device-local counters (atomics).
    ref<Buffer> mpRisStatsReadback; ///< CPU-visible copy.
    bool mLogRisStats = true;       ///< Log the histogram while RIS is on.
    /// Frames between log lines. 1 = every frame, which is the useful default
    /// while the frame time is pathological: at 2 fps a 60-frame interval is
    /// one line every 30 seconds. The readback costs a GPU sync per logged
    /// frame - irrelevant next to a 500 ms frame, so raise this only once the
    /// renderer is fast again.
    uint32_t mRisStatsInterval = 1;

    /// Work profiling: real GPU milliseconds for this pass plus the operation
    /// counts that time is spent on. On by default - a renderer whose cost is
    /// unexplained is not one you can optimise.
    bool mLogWorkStats = true;
    ref<GpuTimer> mpGpuTimer;
    double mLastGpuMs = 0.0;

    uint32_t mFrameCount = 0;
    bool mOptionsChanged = false;
};
