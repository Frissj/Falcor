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
    uint32_t selectInstanceMip(const GridVolume& volume, const float3& cameraPos, float footprintSpread, float& outFootprintWorld) const;

    // Merged coarse tail (VNA section 3).
    void bakeMergedTail();

    // Sun-tau cache build (Nubis3/UE port; see TauCB in the shader).
    void buildTauCache(RenderContext* pRenderContext);

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    // Stream compaction pipeline (phase A = mpPass with USE_COMPACTION=1):
    // argsMain turns the survivor count into indirect args, shadeMain shades
    // one thread per queued path in dense waves. Null when compaction is off.
    ref<ComputePass> mpPassShade;
    ref<ComputePass> mpPassArgs;
    ref<Buffer> mpScatterQueue;  ///< RIS survivors (16 B each), sized to pixel count.
    ref<Buffer> mpScatterCount;  ///< Single uint: queued path count.
    ref<Buffer> mpDispatchArgs;  ///< uint3 indirect dispatch args for shadeMain.
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
    // Runtime A/B for the 6.3 occupancy skip. Toggle in the UI and compare
    // [COST] occSkip / gpuMs within ONE session - cross-session gpuMs on this
    // machine drifts ~25% with identical work counters.
    bool mUseOccupancySkip = true;
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

    // Stage B temporal reservoir history: ping-pong per-pixel reservoirs
    // (32 B each) plus last frame's camera for reprojection. Cleared on
    // resize, scene change, or (re)enabling the feature - a cleared reservoir
    // has M = 0 and is ignored by the shader.
    ref<Buffer> mpReservoir[2];
    uint32_t mReservoirFrame = 0;
    uint2 mReservoirDim = uint2(0);
    float4x4 mPrevViewProj = float4x4::identity();
    float3 mPrevCamPos = float3(0.f);
    bool mPrevCamValid = false;

    // [LOD] log state: what the projected-error selection actually decided,
    // captured by updateBrickTlas so claims about LoD behaviour come from the
    // log, not from arithmetic.
    float mLastSpread = 0.f;                    ///< World footprint growth per unit distance.
    std::vector<float> mInstanceFootprint;      ///< Per instance: footprint (wu) at its nearest point.
    /// Transmittance-adaptive candidate budget: M per pixel =
    /// clamp(ceil(kRisM * (1 - Tfull)), 1, kRisM), decided from the escape
    /// term's transmittance BEFORE the candidate RNG runs - so it is exactly
    /// unbiased (the reservoir weight divides by the M actually used).
    /// Measured motivation: 82-88% of fixed-M candidate processes escape.
    bool mUseAdaptiveM = false;
    /// Stage B: temporal reservoir reuse of the primary scatter vertex
    /// (t-shift + confidence-capped merge; the principled form of UE
    /// MegaLights' temporal feedback). OFF by default: Stage A alone is the
    /// validated baseline; VNA_RisValidate.py gates this before it is trusted.
    bool mUseTemporalReuse = false;
    /// Confidence cap: previous M clamped to this multiple of the current
    /// per-pixel candidate budget. Bounds stale-history influence (variance,
    /// not bias) and the reservoir's memory length.
    float mTemporalMCap = 20.f;
    /// Stage C: spatial reuse of prev-frame reservoirs at Gaussian offsets
    /// around the reprojected pixel. Same ratio-space merge, shift, Jacobian
    /// and deterministic domain guard as Stage B - one estimator, more
    /// sources. Requires temporal reuse (it reads the same history buffer).
    /// OFF by default: the matrix (VNA_Matrix.py config 13) gates it.
    bool mUseSpatialReuse = false;
    /// Spatial neighbors per pixel (compile-time loop bound).
    uint32_t mSpatialNeighbors = 2u;
    /// Gaussian sigma of the neighbor offsets, in pixels (live, no rebuild).
    float mSpatialRadiusPx = 16.f;
    /// Stream compaction (ReSTIR PT Enhanced 6.2.2 / UE classify->compact->
    /// dense-dispatch): split the fused kernel at the reservoir selection.
    /// Phase A (per pixel) generates candidates + merges reservoirs and
    /// queues the ~13% of pixels that scatter; phase B shades them one
    /// thread per real path in dense waves. Estimator-identical: matrix
    /// config 15 gates converged-identity. Requires RIS.
    bool mUseCompaction = false;
    /// Defensive RIS target floor, relative to a fully-lit isotropic vertex
    /// (1/4pi). Bounds the L/Lhat firefly mechanism measured in the matrix
    /// (isolated 800-2200x pixels at 1 spp). Unbiased for any value, but it
    /// IS an estimator change, so the conservative default is 0 (raw target);
    /// the everything-on shortcut sets 0.01 explicitly per house policy.
    float mRisTargetFloor = 0.f;
    /// Russian-roulette survival floor and start bounce (see gRouletteMinQ in
    /// the shader). shadeMain - the bounce loop - is the largest single item in
    /// the frame (Nsight: 18.5-19.8ms of ~33ms), and its cost is path length.
    /// Roulette is unbiased at any q, so these only trade cost against
    /// variance. Defaults reproduce the previous hardcoded 0.05f / bounce > 3.
    float mRouletteMinQ = 0.05f;
    uint32_t mRouletteStartBounce = 3;
    /// Roulette sweep driver (the ReSTIR PT Enhanced 6.2.4 experiment): one
    /// button steps (minQ, startBounce) through a fixed schedule, holds each
    /// for mSweepFramesPerStep frames, force-logs a [SWEEP] line with the
    /// shadeMain divergence counters on each step's last frame, then restores
    /// the original values. CB values only - no recompile between steps, so
    /// the whole comparison is one session, per measurement discipline.
    /// Accumulation is reset at every step boundary so each step's image is
    /// judgeable on its own.
    bool mSweepActive = false;
    uint32_t mSweepStartFrame = 0;
    uint32_t mSweepFramesPerStep = 64;
    float mSweepSavedMinQ = 0.05f;
    uint32_t mSweepSavedStartBounce = 3;
    /// Mip of the coarse mean field used for the TARGET's shadow estimate (0..3).
    /// Candidates no longer come from this field (they are real delta-tracking
    /// collisions), so this only decides how cheap/crude the "is this candidate
    /// lit" guess is. Measured at 159M cells/frame at mip 0 - the single largest
    /// cost in the pass - so it defaults coarse. Being crude here costs variance
    /// only, never correctness.
    uint32_t mRisMip = 2u;
    /// Sun-tau cache (Nubis3 / UE transmittance-volume port, see TauCB in the
    /// shader): replace the target's per-candidate coarseOpticalDepth walk -
    /// [LOOPOCC] measured it at 39.5M cells/frame at 0.342 SIMD occupancy,
    /// 3.6x the warp-slots of the entire RayQuery traversal - with one
    /// trilinear fetch of tau baked toward the dominant env direction.
    /// Unbiased: target-only, so any staleness/coarseness is variance.
    /// mUseTauCache compiles the feature (resources + build passes);
    /// mTauCacheApply is the RUNTIME A/B switch (CB uniform, no recompile),
    /// same discipline as mUseOccupancySkip.
    bool mUseTauCache = true;
    bool mTauCacheApply = true;
    /// Cache cells along the longest world axis (tail-bake convention).
    /// Nubis shipped 256x256x32 for a whole sky; this shapes a target only.
    uint32_t mTauCacheRes = 64;
    /// Rebuild every N frames; 0 = bake once (scene and env are static).
    uint32_t mTauCacheInterval = 0;
    ref<Texture> mpTauCache;             ///< R16F 3D tau grid (UAV+SRV).
    ref<Sampler> mpTauSampler;           ///< Linear clamp.
    ref<Buffer> mpTauDir;                ///< float4[1], dominant light direction.
    ref<ComputePass> mpPassTauDir;       ///< tauDirMain.
    ref<ComputePass> mpPassTauBuild;     ///< tauBuildMain.
    float3 mTauOrigin = float3(0.f);
    float3 mTauCellSize = float3(0.f);
    float3 mTauInvExtent = float3(0.f);
    uint3 mTauDim = uint3(0);
    uint32_t mTauLastBuildFrame = kTauNeverBuilt;
    static const uint32_t kTauNeverBuilt = 0xffffffffu;

    // RIS diagnostics. The shader tallies which branch ended the RIS block for
    // every pixel; this is read back and written to the log as a histogram, so
    // the answer is a text line rather than a colour to squint at.
    // Slots 16..23 are the [COST] phase buckets: the marching done INSIDE
    // GridVolumeSampler (DDA cells + real density taps) attributed to the
    // escape term, candidate generation, and shading/NEE - the work the
    // original [WORK] line was measured to be blind to.
    // 27..28 = brick-cache hits/misses, 29 = brick candidates, 30 = occupancy tap skips,
    // 31..35 = divergence probe (laneWork sum, waveMax sum, warps, idle lanes, active lanes),
    // 36..37 = fully idle warps, busy lanes within marching warps,
    // 38..41 = per-loop SIMD occupancy (RQ traversal sum/max, coarse-DDA sum/max),
    // 42..45 = shadeMain divergence (bounce sum/max, marching-work sum/max).
    static const uint32_t kRisStatSlots = 46;
    ref<Buffer> mpRisStats;         ///< Device-local counters (atomics).
    ref<Buffer> mpRisStatsReadback; ///< CPU-visible copy.
    bool mLogRisStats = false;      ///< Log the histogram while RIS is on.
    /// Frames between log lines. The readback costs a full GPU sync
    /// (submit-and-wait) per logged frame: at interval 1 the GPU never has a
    /// second frame in flight, and a ~24 ms pass was measured to become a
    /// ~77 ms wall-clock frame (13 fps) from the stall alone. The interval-1
    /// era was only defensible when the frame itself was ~500 ms.
    uint32_t mRisStatsInterval = 60;

    /// Work profiling: real GPU milliseconds for this pass plus the operation
    /// counts that time is spent on. Off by default for the shipping path:
    /// enabling it recompiles the sampler with per-step counters AND turns on
    /// the throttled readback sync above. Flip it (UI or script) to explain
    /// cost; leave it off to measure shipping fps.
    bool mLogWorkStats = false;
    ref<GpuTimer> mpGpuTimer;
    double mLastGpuMs = 0.0;

    uint32_t mFrameCount = 0;
    bool mOptionsChanged = false;
};
