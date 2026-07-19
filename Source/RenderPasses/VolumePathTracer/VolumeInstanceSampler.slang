/***************************************************************************
 # Multi-instance grid volume sampler (VNA spec section 3 + 4).
 #
 # Falcor's stock GridVolumeSampler only ever samples gScene.getGridVolume(0).
 # This sampler traverses ALL grid volumes in the scene, so a sky can be built
 # from many instances of a small cloudlet library (memory O(library)).
 #
 # Method: intersect the ray against every instance's world AABB, split the ray
 # at the interval boundaries, then march segment by segment:
 #  - Segment covered by ONE instance: use the stock sampler's hierarchical-DDA
 #    local-majorant tracking (the fast path that makes a dense cloud ~10 ms).
 #  - Segment covered by SEVERAL overlapping instances: run one null-collision
 #    process against the SUM of the covering instances' global majorants, with
 #    total density = sum of per-instance densities. Summing majorants keeps the
 #    bound valid, so this stays unbiased (spec section 4, point 4).
 #
 # Restarting the tracking at segment boundaries is statistically free: free
 # flight distances are memoryless (exponential), so no bias is introduced.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Math.Ray;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGeneratorInterface;
import Rendering.Volumes.GridVolumeSampler;
// Slang imports are not transitive: GridVolumeSampler imports the params
// module internally, but the TransmittanceEstimator enum NAME is only visible
// here if we import it ourselves (used by the estimator dispatch below).
import Rendering.Volumes.GridVolumeSamplerParams;

// ---- VNA section 5 Stage A (RIS) compile-time configuration -----------------
// Guarded defaults so this module keeps compiling for hosts that predate RIS.
#ifndef RIS_CANDIDATES
#define RIS_CANDIDATES 8
#endif
#ifndef RIS_MIP
#define RIS_MIP 2
#endif

// ---- VNA section 4: HW-BVH brick backend (UE HeterogeneousVolumes port) -----
// Occupied bricks of every unique grid, at 4 mip granularities, live as
// procedural AABBs in per-(grid,mip) BLASes; a TLAS instances the mip the host
// selected per grid-volume from projected error (UE: LoD is a property of the
// acceleration structure, and a whole instance flips level coherently).
// Traversal state is a RayQuery plus a handful of scalars - no interval lists,
// no per-thread arrays, nothing to spill.
#ifndef USE_BRICK_TLAS
#define USE_BRICK_TLAS 0
#endif

// Footprint-driven residual-mip selection (UE projected-error rule). Its own
// define because footprintSpread is always populated on primary rays.
#ifndef USE_FOOTPRINT_MIP
#define USE_FOOTPRINT_MIP 0
#endif

// ---- VNA section 3: merged coarse tail (UE Nanite Assemblies lesson) --------
// Above the fine structure, the union of ALL instances is baked into ONE
// world-space grid of conservative summed majorants + approximate summed means.
// Far rays march that single grid; real density taps still hit the true
// per-instance fields, so the estimator stays unbiased - the tail is a control
// variate + majorant provider, never a density substitute.
#ifndef USE_MERGED_TAIL
#define USE_MERGED_TAIL 0
#endif

#if USE_BRICK_TLAS
RaytracingAccelerationStructure gBrickTlas;

/** Per-TLAS-instance data, indexed by CandidateInstanceIndex(). */
struct BrickTlasInstanceInfo
{
    uint gridVolumeIdx; ///< Scene grid-volume index (transform/albedo/anisotropy source).
    uint aabbOffset;    ///< First AABB of this instance's (grid, mip) in gBrickAABBs.
};
StructuredBuffer<BrickTlasInstanceInfo> gBrickTlasInstances;

/** Brick AABBs in SHIFTED index space; same memory the BLAS was built from. */
struct BrickAABB
{
    float3 minPos;
    float3 maxPos;
};
StructuredBuffer<BrickAABB> gBrickAABBs;
#endif // USE_BRICK_TLAS

#if USE_MERGED_TAIL
/** World-axis-aligned tail cells. R = conservative majorant of the SUMMED
    extinction (densityScale folded in at bake), G = approximate summed mean
    (the control variate; any value keeps the estimator unbiased, the mean
    minimises variance).
*/
Texture3D<float2> gTailTex;

cbuffer TailCB
{
    float3 gTailOrigin;       ///< World position of tail cell (0,0,0) corner.
    float gTailFootprintGate; ///< World-space footprint above which a ray uses the tail.
    float3 gTailCellSize;     ///< World size of one tail cell (per axis).
    uint gTailEnabled;        ///< 0 disables the tail path entirely.
    uint3 gTailDim;           ///< Tail grid resolution.
    float _tailPad0;
}
#endif // USE_MERGED_TAIL

struct VolumeInstanceSampler
{
    // Upper bound on instances considered along one ray. Instances beyond this
    // are ignored (clamped, not crashed).
    //
    // THIS IS A PERFORMANCE-CRITICAL CONSTANT, not just a limit. Every local
    // array below is sized by it, per thread: ids/t0/t1, bounds[2N], and the
    // coarse walk's ipos/idir/ri (3 x float3 x N). At N=16 that is well over a
    // kilobyte of per-thread state, which spills to scratch memory and destroys
    // occupancy. Measured: halving the instruction count made the frame SLOWER
    // (537 -> 635 ms), the signature of a spill-bound shader. So the host sets
    // this from the actual scene instance count instead of a fixed 16.
#ifndef MAX_INSTANCES
#define MAX_INSTANCES 16
#endif
    static const uint kMaxInstances = MAX_INSTANCES;

    // Raw work counters. These accumulate across every call made through one
    // instance of this struct, so declaring it once per pixel in main() gives
    // per-pixel totals for the whole path, NEE shadow rays included.
    uint statRays;          ///< sampleDistance + evalTransmittance calls.
    uint statAabbTests;     ///< Instance AABB intersection tests.
    uint statHitRays;       ///< Rays that intersected at least one instance.
    uint statSegments;      ///< Segments marched (single-cover + overlap).
    uint statOverlapSegs;   ///< Segments that took the slow summed-majorant path.
    uint statOverlapSteps;  ///< Null-collision steps taken in overlap segments.
    uint statOverlapLookups;///< Density lookups done in overlap segments.
    uint statMaxCover;      ///< Most instances covering any one segment.
    // RIS work counters (section 5). These are the operations the RIS path adds
    // on top of the reference estimator, counted so the cost can be ATTRIBUTED
    // rather than guessed at: multiply by instances to get texture fetches.
    uint statCoarseWalks;   ///< coarseOpticalDepth calls (one per candidate, each a full shadow-direction walk).
    uint statCoarseCells;   ///< DDA cells stepped across ALL coarse walks. The dominant RIS cost if it dominates.
    uint statCandCells;     ///< Majorant DDA cells stepped by the shared candidate sweep.
    uint statRealTaps;      ///< Real (raw-VDB) density taps made by the shared candidate sweep.
    uint statTailRays;      ///< Public entries that took the merged-tail path (subset of statRays).

    /** Footprint-driven residual-mip selection (UE Nanite lesson: LoD level is
        chosen by PROJECTED ERROR, where a voxel level's error is simply its
        cell size in world units - see VNA_UE_SOURCE_LESSONS.md).

        footprintSpread = world-space width the pixel footprint grows per unit
        distance along the ray (host: 2*tan(fovY/2)/resY * user scale). 0
        disables the feature and every call uses the compile-time kResidualMip.

        Correctness: residual ratio tracking is unbiased at EVERY mip, and this
        choice is deterministic geometry (independent of the sample stream), so
        it can only move cost/variance, never the converged image - which is
        exactly what scripts/VNA_ResidualSweep.py verifies.

        Cost model, stated honestly: FINER cells (low mip) = more cells stepped
        but tighter [min,max] ranges = fewer residual taps; COARSER cells (high
        mip) = fewer cells but looser ranges = more taps. The footprint rule
        picks cell size ~ pixel size, the same balance point Nanite ships.
    */
    float footprintSpread;

    /** Residual mip for one instance/segment: coarsest level whose cell size
        stays below the pixel footprint at segment entry distance tAtEntry.
        Compile-time gated: footprintSpread is now always populated for the
        primary ray (the tail gate and the TLAS LoD need it as a plain camera
        fact), so this feature keys off its own define rather than spread == 0.
    */
    int selectResidualMip(const GridVolume gridVolume, const float3 rayDirW, const float tAtEntry)
    {
#if !USE_FOOTPRINT_MIP
        return GridVolumeSampler::kResidualMip;
#endif
        if (footprintSpread <= 0.f) return GridVolumeSampler::kResidualMip;
        // Voxels per world unit along this ray, for THIS instance's transform
        // (per-instance scale handled exactly - the assembly-instancing case).
        const float3 idirW = mul(gridVolume.data.invTransform, float4(rayDirW, 0.f)).xyz;
        const float voxPerWorld = length(idirW);
        const float footprintVox = footprintSpread * max(tAtEntry, 0.f) * voxPerWorld;
        // Level-0 cells are 8 voxels wide; each mip doubles that (dim = 8 << mip).
        return clamp(int(floor(log2(max(footprintVox, 1e-3f) / 8.f))), 0, 3);
    }

    struct DistanceSample
    {
        float t;        ///< Sampled scatter distance along the ray.
        float3 thp;     ///< Throughput (single-scattering albedo of the chosen instance).
        uint instance;  ///< Index of the grid volume that caused the scatter event.
    };

    /** Per-ray list of instance intersection intervals.
    */
    struct IntervalList
    {
        uint count;
        uint ids[kMaxInstances];
        float t0[kMaxInstances];
        float t1[kMaxInstances];
    };

    /** Intersect the ray against every instance AABB and collect the intervals.
    */
    [mutating] IntervalList gatherIntervals(const Ray ray)
    {
        IntervalList list;
        list.count = 0;

        // NOTE: statRays is counted at the PUBLIC entries (evalTransmittance /
        // sampleDistance* / sampleCandidates*), not here - the RayQuery backend
        // never gathers intervals, and entry counts must stay comparable
        // across backends.
        const uint volumeCount = min(gScene.getGridVolumeCount(), kMaxInstances);
        statAabbTests += volumeCount;
        for (uint i = 0; i < volumeCount; ++i)
        {
            const GridVolume gridVolume = gScene.getGridVolume(i);
            if (!gridVolume.hasDensityGrid()) continue;

            const AABB bounds = gridVolume.getBounds();
            float2 nearFar;
            intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, nearFar);
            nearFar.x = max(nearFar.x, ray.tMin);
            nearFar.y = min(nearFar.y, ray.tMax);
            if (nearFar.x >= nearFar.y) continue;

            list.ids[list.count] = i;
            list.t0[list.count] = nearFar.x;
            list.t1[list.count] = nearFar.y;
            list.count++;
        }
        if (list.count > 0) statHitRays++;
        return list;
    }

    /** Global (whole-grid) majorant of one instance: densityScale * max voxel value.
    */
    float instanceGlobalMajorant(const GridVolume gridVolume, const Grid densityGrid)
    {
        return gridVolume.data.densityScale * densityGrid.getMaxValue();
    }

    /** Density of one instance at a world-space position (stochastically interpolated).
    */
    float instanceDensity(const GridVolume gridVolume, const Grid densityGrid, const float3 posW, const float3 u)
    {
        float3 ipos = mul(gridVolume.data.invTransform, float4(posW, 1.f)).xyz;
        ipos -= densityGrid.getMinIndex(); // Bricked-grid index offset, mirrors GridVolumeSampler.
        return gridVolume.data.densityScale * densityGrid.lookupStochasticIndexTex(ipos, u);
    }

    /** Sort the segment boundaries of an interval list into 'bounds' (ascending).
        Returns the number of boundaries. Insertion sort: n <= 32, tiny.
    */
    uint buildBoundaries(const IntervalList list, out float bounds[2 * kMaxInstances])
    {
        uint n = 0;
        for (uint i = 0; i < list.count; ++i)
        {
            bounds[n++] = list.t0[i];
            bounds[n++] = list.t1[i];
        }
        for (uint i = 1; i < n; ++i)
        {
            const float key = bounds[i];
            int j = int(i) - 1;
            while (j >= 0 && bounds[j] > key)
            {
                bounds[j + 1] = bounds[j];
                j--;
            }
            bounds[j + 1] = key;
        }
        return n;
    }

    /** Bitmask of instances (indices into 'list') covering the segment midpoint.
    */
    uint coveringMask(const IntervalList list, const float mid)
    {
        uint mask = 0;
        for (uint i = 0; i < list.count; ++i)
        {
            if (list.t0[i] <= mid && mid <= list.t1[i]) mask |= (1u << i);
        }
        return mask;
    }

    /** Transmittance along a ray through all instances (public entry).

        Backend dispatch, in priority order:
         1. Merged tail - far rays march ONE world-space summed grid
            (UE Nanite Assemblies: instances only exist at the fine end).
         2. Brick TLAS - RayQuery over per-brick procedural AABBs. The
            transmittance over disjoint bricks is a PRODUCT of independent
            per-brick factors, so traversal order is irrelevant and no
            sorting or interval state exists at all.
         3. Interval machinery - the legacy backend, kept compilable for
            one-toggle A/B.
        Every backend estimates the same quantity unbiasedly: converged images
        must be identical, only cost and variance may differ.
    */
    [mutating] float evalTransmittance<S : ISampleGenerator>(const Ray ray, inout S sg)
    {
        statRays++;
#if USE_MERGED_TAIL
        if (tailGate(ray))
        {
            statTailRays++;
            const IntervalList tailList = gatherIntervals(ray);
            if (tailList.count == 0) return 1.f;
            return evalTransmittanceTail(tailList, ray, sg);
        }
#endif
#if USE_BRICK_TLAS
        return evalTransmittanceRQ(ray, sg);
#else
        return evalTransmittanceIntervals(ray, sg);
#endif
    }

    [mutating] float evalTransmittanceIntervals<S : ISampleGenerator>(const Ray ray, inout S sg)
    {
        const IntervalList list = gatherIntervals(ray);
        if (list.count == 0) return 1.f;

        float bounds[2 * kMaxInstances];
        const uint nBounds = buildBoundaries(list, bounds);

        GridVolumeSampler gvs = {};
        float Tr = 1.f;

        for (uint s = 0; s + 1 < nBounds; ++s)
        {
            const float a = bounds[s];
            const float b = bounds[s + 1];
            if (b - a <= 1e-6f) continue;

            const uint mask = coveringMask(list, 0.5f * (a + b));
            if (mask == 0) continue;

            statSegments++;
            statMaxCover = max(statMaxCover, countbits(mask));

            if ((mask & (mask - 1)) == 0) // Single instance: fast DDA path.
            {
                const uint slot = firstbitlow(mask);
                const GridVolume gridVolume = gScene.getGridVolume(list.ids[slot]);
                // Static dispatch on the estimator chosen in the UI: residual
                // ratio tracking (VNA section 2 / P2, analytic control field)
                // vs plain ratio tracking. Both unbiased; kTransmittanceEstimator
                // is a compile-time constant so there is no runtime branch.
                if (GridVolumeSampler::kTransmittanceEstimator == TransmittanceEstimator::ResidualRatioTrackingLocalMajorant)
                    Tr *= gvs.evalTransmittanceResidualRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, float2(a, b), sg,
                        selectResidualMip(gridVolume, ray.dir, a));
                else
                    Tr *= gvs.evalTransmittanceRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, float2(a, b), sg);
            }
            else // Overlap: ratio tracking against the summed global majorant.
            {
                statOverlapSegs++;
                Tr *= evalTransmittanceOverlap(list, mask, ray, a, b, sg);
            }

            if (Tr <= 0.f) return 0.f;
        }
        return Tr;
    }

    [mutating] float evalTransmittanceOverlap<S : ISampleGenerator>(const IntervalList list, const uint mask, const Ray ray, const float a, const float b, inout S sg)
    {
        float majSum = 0.f;
        for (uint i = 0; i < list.count; ++i)
        {
            if ((mask & (1u << i)) == 0) continue;
            const GridVolume gridVolume = gScene.getGridVolume(list.ids[i]);
            Grid densityGrid;
            gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
            majSum += instanceGlobalMajorant(gridVolume, densityGrid);
        }
        if (majSum <= 0.f) return 1.f;
        const float invMajSum = 1.f / majSum;

        float Tr = 1.f;
        float t = a;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajSum;
            if (t >= b) return Tr;

            statOverlapSteps++;

            const float3 posW = ray.origin + t * ray.dir;
            const float3 u = sampleNext3D(sg);
            float dSum = 0.f;
            for (uint i = 0; i < list.count; ++i)
            {
                if ((mask & (1u << i)) == 0) continue;
                const GridVolume gridVolume = gScene.getGridVolume(list.ids[i]);
                Grid densityGrid;
                gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
                statOverlapLookups++;
                dSum += instanceDensity(gridVolume, densityGrid, posW, u);
            }

            // The max guards against non-bounding majorants from interpolation,
            // same rare-firefly guard as the stock ratio tracker.
            Tr *= max(0.f, 1.f - dSum * invMajSum);
            if (Tr < 0.1f) // Russian roulette.
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return Tr;
    }

    /** Sample a scattering distance along a ray through all instances (public
        entry). Backend dispatch mirrors evalTransmittance; see there.
        \return true if a real scatter event happened; ds identifies where and in which instance.
    */
    [mutating] bool sampleDistance<S : ISampleGenerator>(const Ray ray, inout S sg, out DistanceSample ds)
    {
        statRays++;
#if USE_MERGED_TAIL
        if (tailGate(ray))
        {
            statTailRays++;
            ds = {};
            const IntervalList tailList = gatherIntervals(ray);
            if (tailList.count == 0) return false;
            return sampleDistanceTail(tailList, ray, sg, ds);
        }
#endif
#if USE_BRICK_TLAS
        return sampleDistanceRQ(ray, sg, ds);
#else
        const IntervalList list = gatherIntervals(ray);
        return sampleDistanceIntervals(list, ray, sg, ds);
#endif
    }

    /** As sampleDistance, but with the instance intervals supplied by the caller.
        Under the brick-TLAS backend the list is unused (hardware traverses);
        it exists so interval-backend callers can gather once and reuse.
    */
    [mutating] bool sampleDistanceWithList<S : ISampleGenerator>(const IntervalList list, const Ray ray, inout S sg, out DistanceSample ds)
    {
        statRays++;
#if USE_BRICK_TLAS
        return sampleDistanceRQ(ray, sg, ds);
#else
        return sampleDistanceIntervals(list, ray, sg, ds);
#endif
    }

    [mutating] bool sampleDistanceIntervals<S : ISampleGenerator>(const IntervalList list, const Ray ray, inout S sg, out DistanceSample ds)
    {
        ds = {};

        if (list.count == 0) return false;

        float bounds[2 * kMaxInstances];
        const uint nBounds = buildBoundaries(list, bounds);

        GridVolumeSampler gvs = {};

        for (uint s = 0; s + 1 < nBounds; ++s)
        {
            const float a = bounds[s];
            const float b = bounds[s + 1];
            if (b - a <= 1e-6f) continue;

            const uint mask = coveringMask(list, 0.5f * (a + b));
            if (mask == 0) continue;

            statSegments++;
            statMaxCover = max(statMaxCover, countbits(mask));

            if ((mask & (mask - 1)) == 0) // Single instance: fast DDA path.
            {
                const uint slot = firstbitlow(mask);
                const GridVolume gridVolume = gScene.getGridVolume(list.ids[slot]);
                GridVolumeSampler::DistanceSample gds;
                if (gvs.sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, float2(a, b), sg, gds))
                {
                    ds.t = gds.t;
                    ds.thp = gds.thp;
                    ds.instance = list.ids[slot];
                    return true;
                }
            }
            else // Overlap: delta tracking against the summed global majorant.
            {
                statOverlapSegs++;
                if (sampleDistanceOverlap(list, mask, ray, a, b, sg, ds)) return true;
            }
        }
        return false;
    }

    // =========================================================================
    // VNA section 5, Stage A - coarse-field (p-hat) utilities.
    //
    // TWO DIFFERENT COARSE FIELDS, TWO DIFFERENT JOBS. Conflating them is what
    // produced a black cloud once already; the distinction is load-bearing.
    //
    //  PROPOSAL (where to put candidates) = MAJORANT, from rangeTex.
    //    An importance-sampling proposal must COVER the integrand: it may never
    //    be small where the real density is large. The majorant bounds density
    //    by construction, so it covers everywhere. The MEAN does NOT: a wisp's
    //    density far exceeds its neighbourhood average, so a mean-based
    //    proposal systematically under-samples exactly the dense, bright
    //    structures - measured as avgCandidates 1.8/8 and 94% of survivors
    //    landing on zero real density.
    //
    //  TARGET / SHADOW ESTIMATE (which candidate to keep) = MEAN, from meanTex.
    //    Here we want the best GUESS of actual optical depth and actual local
    //    density, and the mean is exactly that. Being wrong here costs variance
    //    only - never correctness - which is the P1 licence (spec section 2).
    //
    // CANDIDATE GENERATION = M real delta-tracking processes in ONE sweep.
    // Candidates must be REAL collisions (measured: majorant-CDF candidates
    // landed on zero density 87% of the time). M independent walks are
    // distribution-correct but re-march the same ray M times; the shared sweep
    // below runs all M processes against a single majorant-DDA traversal - one
    // range-tex fetch per cell instead of M - while each process keeps its own
    // exponential free-flight chain and its own acceptance draws, so the joint
    // distribution is IDENTICAL to M independent walks. This is the UE shape
    // (one register-resident walk, no per-thread arrays beyond M floats); see
    // VNA_UE_SOURCE_LESSONS.md.
    // =========================================================================

    static const uint kRisM = RIS_CANDIDATES;
    static const int kRisMip = RIS_MIP;

    /** Result of the shared candidate sweep: up to kRisM REAL collisions.
        Each entry is a genuine scatter event in the real medium; processes
        whose walk escaped simply contribute no entry (the escape/background
        term is estimated separately by the caller).
    */
    struct CandidateSet
    {
        uint count;             ///< Number of real-collision candidates emitted (<= kRisM).
        float t[kRisM];         ///< Scatter distances (world units along the ray).
        uint inst[kRisM];       ///< Instance that caused each collision (albedo/anisotropy source).
    };

    /** TARGET field of one instance: the local mean density (cell-constant). */
    float instanceSigmaMean(const GridVolume gridVolume, const Grid densityGrid, const float3 iposAtT)
    {
        return gridVolume.data.densityScale * densityGrid.lookupIndexLocalMeanTex(int3(iposAtT), kRisMip);
    }

    /** Total coarse optical depth along a ray (all instances, all segments).
        Analytic: one DDA walk, zero RNG. Used for the RIS target's shadow term,
        so it uses the MEAN field - the best cheap ESTIMATE of true optical
        depth. (The majorant would systematically overestimate and make every
        candidate look shadowed.)
    */
    [mutating] float coarseOpticalDepth(const Ray ray)
    {
        statCoarseWalks++;

        const IntervalList list = gatherIntervals(ray);
        if (list.count == 0) return 0.f;

        float bounds[2 * kMaxInstances];
        const uint nBounds = buildBoundaries(list, bounds);

        // Per-instance index-space rays, transformed once for the whole walk.
        GridVolumeSampler gvs = {};
        float3 ipos[kMaxInstances];
        float3 idir[kMaxInstances];
        float3 ri[kMaxInstances];
        for (uint i = 0; i < list.count; ++i)
        {
            const GridVolume gv = gScene.getGridVolume(list.ids[i]);
            Grid grid;
            gScene.getGrid(gv.getDensityGrid(), grid);
            gvs.transformRayToIndexSpace(gv, grid, ray.origin, ray.dir, ipos[i], idir[i]);
            ri[i] = 1.f / idir[i];
        }

        float tau = 0.f;
        for (uint s = 0; s + 1 < nBounds; ++s)
        {
            const float a = bounds[s];
            const float b = bounds[s + 1];
            if (b - a <= 1e-6f) continue;
            const uint mask = coveringMask(list, 0.5f * (a + b));
            if (mask == 0) continue;

            statSegments++;

            float t = a;
            while (t < b)
            {
                // Cell exit = nearest coarse-cell boundary over covering instances.
                float tExit = b;
                float sigma = 0.f;
                for (uint i = 0; i < list.count; ++i)
                {
                    if ((mask & (1u << i)) == 0) continue;
                    const GridVolume gv = gScene.getGridVolume(list.ids[i]);
                    Grid grid;
                    gScene.getGrid(gv.getDensityGrid(), grid);
                    const float3 p = ipos[i] + idir[i] * t;
                    tExit = min(tExit, gvs.stepDDA(ipos[i], ri[i], p, kRisMip));
                    sigma += instanceSigmaMean(gv, grid, p);
                }
                // Forward-progress guard. stepDDA's half-voxel dilation makes
                // each instance's exit strictly greater than t, but the MIN over
                // several instances (each in its own index space) can land on or
                // below t numerically. Without this the loop spins forever on a
                // zero-length cell and the dispatch never completes.
                if (tExit <= t) tExit = min(b, t + max(1e-4f, abs(t) * 1e-5f));
                statCoarseCells++;
                tau += sigma * (tExit - t);
                t = tExit;
            }
        }
        return tau;
    }

    /** Draw up to kRisM REAL primary-scatter candidates: M independent
        delta-tracking processes sharing ONE traversal of the ray.

        Process state is tauLeft[j] - the remaining tentative optical depth to
        process j's next tentative collision (exponentially distributed,
        memoryless). Inside each constant-majorant region [t, tExit) process j
        crosses at tc = t + tauLeft[j]/sigma while tc < tExit; every crossing
        takes ONE real density tap and accepts with p = d/sigma (delta
        tracking); a rejection draws a fresh exponential and continues from tc.
        Carrying tauLeft across cells and segments is the textbook piecewise-
        constant free-flight inversion: no restart, no bias, and empty regions
        (sigma = 0) consume nothing. Each process keeps its OWN exponential
        chain and its OWN acceptance draws, so the joint distribution is
        IDENTICAL to M independent sampleDistanceWithList walks - only the
        traversal (DDA stepping + range-tex fetches) is shared, which is
        exactly the part M independent walks duplicated.

        Processes still alive when the sweep ends escaped the medium: they
        emit nothing, and the caller's separately-estimated escape/background
        term carries that radiance (no double counting).
    */
    /** numProcesses (<= kRisM) is the per-pixel candidate budget - possibly
        transmittance-adaptive (see the pass). Processes are contiguous
        [0, numProcesses); arrays stay kRisM-sized so register layout is
        budget-independent.
    */
    [mutating] void sampleCandidatesSharedSweep<S : ISampleGenerator>(const IntervalList list, const Ray ray, const uint numProcesses, inout S sg, out CandidateSet cands)
    {
        statRays++;
#if USE_MERGED_TAIL
        if (tailGate(ray))
        {
            statTailRays++;
            sampleCandidatesTail(list, ray, numProcesses, sg, cands);
            return;
        }
#endif
#if USE_BRICK_TLAS
        sampleCandidatesRQ(ray, numProcesses, sg, cands);
#else
        sampleCandidatesSweepIntervals(list, ray, numProcesses, sg, cands);
#endif
    }

    [mutating] void sampleCandidatesSweepIntervals<S : ISampleGenerator>(const IntervalList list, const Ray ray, const uint numProcesses, inout S sg, out CandidateSet cands)
    {
        cands = {};
        cands.count = 0;
        if (list.count == 0) return;

        const uint nProc = min(numProcesses, kRisM);

        float bounds[2 * kMaxInstances];
        const uint nBounds = buildBoundaries(list, bounds);

        GridVolumeSampler gvs = {};

        float tauLeft[kRisM];
        uint aliveMask = (nProc >= 32) ? 0xffffffffu : ((1u << nProc) - 1u);
        for (uint j = 0; j < kRisM; ++j) tauLeft[j] = -log(1.f - sampleNext1D(sg));

        for (uint s = 0; s + 1 < nBounds && aliveMask != 0; ++s)
        {
            const float a = bounds[s];
            const float b = bounds[s + 1];
            if (b - a <= 1e-6f) continue;
            const uint mask = coveringMask(list, 0.5f * (a + b));
            if (mask == 0) continue;

            statSegments++;
            statMaxCover = max(statMaxCover, countbits(mask));

            if ((mask & (mask - 1)) == 0) // Single instance: local-majorant DDA.
            {
                const uint slot = firstbitlow(mask);
                const uint volumeId = list.ids[slot];
                const GridVolume gridVolume = gScene.getGridVolume(volumeId);
                Grid densityGrid;
                gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);

                float3 ipos, idir;
                gvs.transformRayToIndexSpace(gridVolume, densityGrid, ray.origin, ray.dir, ipos, idir);
                const float3 ri = 1.f / idir;

                // Same adaptive mip heuristic as the stock stepToNextCollision:
                // climb while cells are empty, drop after any collision. The
                // level only picks which (valid) majorant bounds the cell, so
                // it moves cost, never correctness.
                float level = GridVolumeSampler::kDDAMipStart;

                float t = a;
                while (t < b && aliveMask != 0)
                {
                    statCandCells++;
                    const int mip = int(round(level));
                    const float3 pos = ipos + idir * t;
                    const float tExit = min(gvs.stepDDA(ipos, ri, pos, mip), b);
                    const float sigma = gridVolume.data.densityScale * densityGrid.lookupIndexLocalMajorantTex(pos, mip);
                    bool collidedInCell = false;

                    if (sigma > 0.f)
                    {
                        const float invSigma = 1.f / sigma;
                        for (uint j = 0; j < kRisM; ++j)
                        {
                            if ((aliveMask & (1u << j)) == 0) continue;
                            float tj = t;
                            while (true)
                            {
                                const float dtau = (tExit - tj) * sigma;
                                if (tauLeft[j] > dtau) { tauLeft[j] -= dtau; break; }
                                tj += tauLeft[j] * invSigma;
                                collidedInCell = true;
                                statRealTaps++;
                                const float3 posW = ray.origin + tj * ray.dir;
                                const float d = instanceDensity(gridVolume, densityGrid, posW, sampleNext3D(sg));
                                if (sampleNext1D(sg) * sigma < d)
                                {
                                    // Real collision: emit and retire process j.
                                    cands.t[cands.count] = tj;
                                    cands.inst[cands.count] = volumeId;
                                    cands.count++;
                                    aliveMask &= ~(1u << j);
                                    break;
                                }
                                // Null collision: fresh free flight from tj.
                                tauLeft[j] = -log(1.f - sampleNext1D(sg));
                            }
                        }
                    }

                    level = collidedInCell
                        ? max(0.f, level - GridVolumeSampler::kDDAMipDownSpeed)
                        : min(3.f, level + GridVolumeSampler::kDDAMipUpSpeed);
                    t = tExit;
                }
            }
            else // Overlap: summed global majorant, same per-process chains.
            {
                float majSum = 0.f;
                for (uint i = 0; i < list.count; ++i)
                {
                    if ((mask & (1u << i)) == 0) continue;
                    const GridVolume gv = gScene.getGridVolume(list.ids[i]);
                    Grid grid;
                    gScene.getGrid(gv.getDensityGrid(), grid);
                    majSum += instanceGlobalMajorant(gv, grid);
                }
                statOverlapSegs++;
                if (majSum <= 0.f) continue;
                const float invMajSum = 1.f / majSum;

                for (uint j = 0; j < kRisM; ++j)
                {
                    if ((aliveMask & (1u << j)) == 0) continue;
                    float tj = a;
                    while (true)
                    {
                        const float dtau = (b - tj) * majSum;
                        if (tauLeft[j] > dtau) { tauLeft[j] -= dtau; break; }
                        tj += tauLeft[j] * invMajSum;
                        statOverlapSteps++;

                        const float3 posW = ray.origin + tj * ray.dir;
                        const float3 u = sampleNext3D(sg);
                        float densities[kMaxInstances];
                        float dSum = 0.f;
                        for (uint i = 0; i < list.count; ++i)
                        {
                            densities[i] = 0.f;
                            if ((mask & (1u << i)) == 0) continue;
                            const GridVolume gv = gScene.getGridVolume(list.ids[i]);
                            Grid grid;
                            gScene.getGrid(gv.getDensityGrid(), grid);
                            statOverlapLookups++;
                            densities[i] = instanceDensity(gv, grid, posW, u);
                            dSum += densities[i];
                        }

                        if (sampleNext1D(sg) * majSum < dSum)
                        {
                            // Real collision: attribute by density, like the
                            // reference overlap sampler.
                            uint chosen = firstbitlow(mask);
                            float pick = sampleNext1D(sg) * dSum;
                            for (uint i = 0; i < list.count; ++i)
                            {
                                if ((mask & (1u << i)) == 0) continue;
                                chosen = i;
                                pick -= densities[i];
                                if (pick <= 0.f) break;
                            }
                            cands.t[cands.count] = tj;
                            cands.inst[cands.count] = list.ids[chosen];
                            cands.count++;
                            aliveMask &= ~(1u << j);
                            break;
                        }
                        tauLeft[j] = -log(1.f - sampleNext1D(sg));
                    }
                }
            }
        }
    }

    [mutating] bool sampleDistanceOverlap<S : ISampleGenerator>(const IntervalList list, const uint mask, const Ray ray, const float a, const float b, inout S sg, out DistanceSample ds)
    {
        ds = {};

        float majSum = 0.f;
        for (uint i = 0; i < list.count; ++i)
        {
            if ((mask & (1u << i)) == 0) continue;
            const GridVolume gridVolume = gScene.getGridVolume(list.ids[i]);
            Grid densityGrid;
            gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
            majSum += instanceGlobalMajorant(gridVolume, densityGrid);
        }
        if (majSum <= 0.f) return false;
        const float invMajSum = 1.f / majSum;

        float t = a;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajSum;
            if (t >= b) return false;

            statOverlapSteps++;

            const float3 posW = ray.origin + t * ray.dir;
            const float3 u = sampleNext3D(sg);
            float densities[kMaxInstances];
            float dSum = 0.f;
            for (uint i = 0; i < list.count; ++i)
            {
                densities[i] = 0.f;
                if ((mask & (1u << i)) == 0) continue;
                const GridVolume gridVolume = gScene.getGridVolume(list.ids[i]);
                Grid densityGrid;
                gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
                statOverlapLookups++;
                densities[i] = instanceDensity(gridVolume, densityGrid, posW, u);
                dSum += densities[i];
            }

            // Real collision with probability dSum / majSum.
            if (sampleNext1D(sg) * majSum < dSum)
            {
                // Attribute the event to one instance with probability d_i / dSum,
                // so per-instance albedo/anisotropy stay correct in overlaps.
                float pick = sampleNext1D(sg) * dSum;
                uint chosen = 0;
                for (uint i = 0; i < list.count; ++i)
                {
                    if ((mask & (1u << i)) == 0) continue;
                    chosen = i;
                    pick -= densities[i];
                    if (pick <= 0.f) break;
                }
                ds.t = t;
                ds.thp = gScene.getGridVolume(list.ids[chosen]).data.albedo;
                ds.instance = list.ids[chosen];
                return true;
            }
            // Null collision: keep marching.
        }
        return false;
    }

#if USE_BRICK_TLAS
    // =========================================================================
    // HW-BVH brick backend (VNA section 4; UE HeterogeneousVolumes +
    // Nanite-foliage lesson: sparse voxels ARE the acceleration structure,
    // traversal state fits in registers, LoD was chosen when the host picked
    // which mip's BLAS to instance).
    // =========================================================================

    /** Slab-intersect the candidate's object-space ray (shifted-index space)
        with its brick AABB. Object-space t equals world-space t by DXR
        construction, so the result feeds the world-space marchers directly.
    */
    static float2 intersectBrickAABB(const float3 origin, const float3 dir, const BrickAABB bb, const float tMin, const float tMax)
    {
        const float3 invDir = 1.f / dir;
        const float3 p0 = (bb.minPos - origin) * invDir;
        const float3 p1 = (bb.maxPos - origin) * invDir;
        const float3 tn = min(p0, p1);
        const float3 tf = max(p0, p1);
        return float2(
            max(max(tn.x, max(tn.y, tn.z)), tMin),
            min(min(tf.x, min(tf.y, tf.z)), tMax));
    }

    /** Transmittance via RayQuery over the brick TLAS.

        The transmittance factors of disjoint bricks (and of different
        instances, overlapping or not) are INDEPENDENT, so the total is a
        plain product in whatever order traversal delivers candidates: no
        sorting, no boundaries, no per-thread arrays. Restarting the tracker
        per brick is statistically free (free flights are memoryless).
    */
    [mutating] float evalTransmittanceRQ<S : ISampleGenerator>(const Ray ray, inout S sg)
    {
        GridVolumeSampler gvs = {};
        float Tr = 1.f;

        RayDesc rd;
        rd.Origin = ray.origin;
        rd.TMin = ray.tMin;
        rd.Direction = ray.dir;
        rd.TMax = ray.tMax;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(gBrickTlas, RAY_FLAG_NONE, 0xff, rd);
        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

            const BrickTlasInstanceInfo info = gBrickTlasInstances[q.CandidateInstanceIndex()];
            const BrickAABB bb = gBrickAABBs[info.aabbOffset + q.CandidatePrimitiveIndex()];
            const float2 tt = intersectBrickAABB(q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(), bb, ray.tMin, ray.tMax);
            if (tt.x >= tt.y) continue;

            statSegments++;
            const GridVolume gridVolume = gScene.getGridVolume(info.gridVolumeIdx);
            if (GridVolumeSampler::kTransmittanceEstimator == TransmittanceEstimator::ResidualRatioTrackingLocalMajorant)
                Tr *= gvs.evalTransmittanceResidualRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, tt, sg,
                    selectResidualMip(gridVolume, ray.dir, tt.x));
            else
                Tr *= gvs.evalTransmittanceRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, tt, sg);

            if (Tr <= 0.f) return 0.f;
        }
        return Tr;
    }

    /** Distance sampling via RayQuery.

        Correctness: real (post-thinning) collision events of different bricks
        and different instances are independent Poisson processes, so the true
        first collision of the union is the MIN of per-brick first collisions.
        Bricks can therefore be processed in ANY order: run in-brick delta
        tracking per candidate, keep the earliest real collision, and commit
        it so the hardware prunes bricks that start beyond it. This also makes
        the summed-majorant overlap machinery obsolete by construction.
    */
    [mutating] bool sampleDistanceRQ<S : ISampleGenerator>(const Ray ray, inout S sg, out DistanceSample ds)
    {
        ds = {};
        GridVolumeSampler gvs = {};
        float bestT = FLT_MAX;

        RayDesc rd;
        rd.Origin = ray.origin;
        rd.TMin = ray.tMin;
        rd.Direction = ray.dir;
        rd.TMax = ray.tMax;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(gBrickTlas, RAY_FLAG_NONE, 0xff, rd);
        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

            const BrickTlasInstanceInfo info = gBrickTlasInstances[q.CandidateInstanceIndex()];
            const BrickAABB bb = gBrickAABBs[info.aabbOffset + q.CandidatePrimitiveIndex()];
            const float2 tt = intersectBrickAABB(q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(), bb, ray.tMin, min(ray.tMax, bestT));
            if (tt.x >= tt.y) continue;

            statSegments++;
            const GridVolume gridVolume = gScene.getGridVolume(info.gridVolumeIdx);
            GridVolumeSampler::DistanceSample gds;
            if (gvs.sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, tt, sg, gds) && gds.t < bestT)
            {
                bestT = gds.t;
                ds.t = gds.t;
                ds.thp = gds.thp;
                ds.instance = info.gridVolumeIdx;
                // Shrink the query interval: bricks entirely beyond the current
                // earliest real collision can never win (min property) and get
                // culled by the RT cores - UE's AcceptHitAndEndSearch shape.
                q.CommitProceduralPrimitiveHit(bestT);
            }
        }
        return bestT < FLT_MAX;
    }

    /** M candidate scatter events via ONE RayQuery traversal.

        Per process j the estimate is the same min-over-bricks construction as
        sampleDistanceRQ, with its own independent RNG draws, so the joint
        distribution equals M independent sampleDistanceRQ calls. The shared
        part is the traversal and the per-brick setup. Pruning commits the MAX
        over the per-process bests: a brick starting beyond every process's
        current best cannot improve any of them.
    */
    [mutating] void sampleCandidatesRQ<S : ISampleGenerator>(const Ray ray, const uint numProcesses, inout S sg, out CandidateSet cands)
    {
        cands = {};
        cands.count = 0;
        GridVolumeSampler gvs = {};

        const uint nProc = min(numProcesses, kRisM);
        float bestT[kRisM];
        uint bestInst[kRisM];
        for (uint j = 0; j < kRisM; ++j)
        {
            bestT[j] = FLT_MAX;
            bestInst[j] = 0;
        }

        RayDesc rd;
        rd.Origin = ray.origin;
        rd.TMin = ray.tMin;
        rd.Direction = ray.dir;
        rd.TMax = ray.tMax;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(gBrickTlas, RAY_FLAG_NONE, 0xff, rd);
        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

            float allBest = 0.f;
            for (uint j = 0; j < nProc; ++j) allBest = max(allBest, bestT[j]);

            const BrickTlasInstanceInfo info = gBrickTlasInstances[q.CandidateInstanceIndex()];
            const BrickAABB bb = gBrickAABBs[info.aabbOffset + q.CandidatePrimitiveIndex()];
            const float2 tt = intersectBrickAABB(q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(), bb, ray.tMin, min(ray.tMax, allBest));
            if (tt.x >= tt.y) continue;

            statSegments++;
            const GridVolume gridVolume = gScene.getGridVolume(info.gridVolumeIdx);

            for (uint j = 0; j < nProc; ++j)
            {
                if (tt.x >= bestT[j]) continue;
                GridVolumeSampler::DistanceSample gds;
                if (gvs.sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, float2(tt.x, min(tt.y, bestT[j])), sg, gds) &&
                    gds.t < bestT[j])
                {
                    bestT[j] = gds.t;
                    bestInst[j] = info.gridVolumeIdx;
                }
            }

            float newMax = 0.f;
            for (uint j = 0; j < nProc; ++j) newMax = max(newMax, bestT[j]);
            // Only commit once every process has a real collision (else the
            // max is FLT_MAX and there is nothing safe to prune against).
            if (newMax < ray.tMax) q.CommitProceduralPrimitiveHit(newMax);
        }

        for (uint j = 0; j < nProc; ++j)
        {
            if (bestT[j] < FLT_MAX)
            {
                cands.t[cands.count] = bestT[j];
                cands.inst[cands.count] = bestInst[j];
                cands.count++;
            }
        }
    }

    /** FUSED escape transmittance + M candidate processes, ONE traversal.

        evalTransmittanceRQ and sampleCandidatesRQ walk the SAME primary ray
        through the SAME bricks back to back, paying BVH traversal and brick
        setup twice for identical geometry. This walks it once: per brick the
        ratio/residual tracker advances Tr AND every candidate process
        delta-tracks its own chain. No hit is committed - transmittance needs
        the full ray anyway - so candidate pruning degrades to the per-process
        interval clamp, which is strictly cheaper than a second traversal.

        Estimator-identical to the two-pass form: per-brick transmittance
        factors are independent (memoryless free flights), each candidate
        process keeps its own independent draws, and the two estimators never
        share randomness beyond the sampler stream position. Both marginals
        equal their standalone versions exactly.
    */
    [mutating] float evalEscapeAndCandidatesRQ<S : ISampleGenerator>(const Ray ray, const uint numProcesses, inout S sg, out CandidateSet cands)
    {
        cands = {};
        cands.count = 0;
        statRays++;

#if USE_MERGED_TAIL
        // Same tail dispatch as the unfused entries: far rays march the
        // summed grid for BOTH the escape term and the candidates.
        if (tailGate(ray))
        {
            statTailRays++;
            const IntervalList tailList = gatherIntervals(ray);
            if (tailList.count == 0) return 1.f;
            sampleCandidatesTail(tailList, ray, numProcesses, sg, cands);
            return evalTransmittanceTail(tailList, ray, sg);
        }
#endif

        GridVolumeSampler gvs = {};

        const uint nProc = min(numProcesses, kRisM);
        float bestT[kRisM];
        uint bestInst[kRisM];
        for (uint j = 0; j < kRisM; ++j)
        {
            bestT[j] = FLT_MAX;
            bestInst[j] = 0;
        }

        float Tr = 1.f;

        RayDesc rd;
        rd.Origin = ray.origin;
        rd.TMin = ray.tMin;
        rd.Direction = ray.dir;
        rd.TMax = ray.tMax;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(gBrickTlas, RAY_FLAG_NONE, 0xff, rd);
        while (q.Proceed())
        {
            if (q.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;

            const BrickTlasInstanceInfo info = gBrickTlasInstances[q.CandidateInstanceIndex()];
            const BrickAABB bb = gBrickAABBs[info.aabbOffset + q.CandidatePrimitiveIndex()];
            const float2 tt = intersectBrickAABB(q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(), bb, ray.tMin, ray.tMax);
            if (tt.x >= tt.y) continue;

            statSegments++;
            const GridVolume gridVolume = gScene.getGridVolume(info.gridVolumeIdx);

            // Escape term: identical per-brick factor to evalTransmittanceRQ.
            // Tr hitting zero stops the tracker but NOT the traversal - the
            // candidate processes still need the remaining bricks.
            if (Tr > 0.f)
            {
                if (GridVolumeSampler::kTransmittanceEstimator == TransmittanceEstimator::ResidualRatioTrackingLocalMajorant)
                    Tr *= gvs.evalTransmittanceResidualRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, tt, sg,
                        selectResidualMip(gridVolume, ray.dir, tt.x));
                else
                    Tr *= gvs.evalTransmittanceRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, tt, sg);
            }

            // Candidates: identical per-brick min construction to
            // sampleCandidatesRQ, clamped per process instead of committed.
            for (uint j = 0; j < nProc; ++j)
            {
                if (tt.x >= bestT[j]) continue;
                GridVolumeSampler::DistanceSample gds;
                if (gvs.sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, float2(tt.x, min(tt.y, bestT[j])), sg, gds) &&
                    gds.t < bestT[j])
                {
                    bestT[j] = gds.t;
                    bestInst[j] = info.gridVolumeIdx;
                }
            }
        }

        for (uint j = 0; j < nProc; ++j)
        {
            if (bestT[j] < FLT_MAX)
            {
                cands.t[cands.count] = bestT[j];
                cands.inst[cands.count] = bestInst[j];
                cands.count++;
            }
        }
        return max(Tr, 0.f);
    }
#endif // USE_BRICK_TLAS

#if USE_MERGED_TAIL
    // =========================================================================
    // Merged coarse tail (VNA section 3; UE Nanite Assemblies lesson 11:
    // instancing exists only at the fine end - the coarse end is ONE merged
    // object). One world-space grid stores conservative majorants and means of
    // the SUMMED instance field; far rays march it in a single DDA. Real
    // density taps still evaluate the true per-instance fields, so every
    // estimator below is exactly unbiased; the bake quality only moves cost
    // and variance.
    // =========================================================================

    /** Deterministic geometry gate: use the tail when the pixel footprint at
        the cloud-region entry exceeds gTailFootprintGate. Deterministic =>
        this is an estimator switch, never a bias source.
    */
    bool tailGate(const Ray ray)
    {
        if (gTailEnabled == 0 || footprintSpread <= 0.f) return false;
        const float3 bbMin = gTailOrigin;
        const float3 bbMax = gTailOrigin + gTailCellSize * float3(gTailDim);
        const float3 invDir = 1.f / ray.dir;
        const float3 p0 = (bbMin - ray.origin) * invDir;
        const float3 p1 = (bbMax - ray.origin) * invDir;
        const float3 tn = min(p0, p1);
        const float3 tf = max(p0, p1);
        const float tEnter = max(max(tn.x, max(tn.y, tn.z)), ray.tMin);
        const float tExit = min(min(tf.x, min(tf.y, tf.z)), ray.tMax);
        return tEnter < tExit && footprintSpread * max(tEnter, 0.f) > gTailFootprintGate;
    }

    /** Ray/tail-grid overlap in world t. */
    float2 tailSpan(const Ray ray)
    {
        const float3 bbMin = gTailOrigin;
        const float3 bbMax = gTailOrigin + gTailCellSize * float3(gTailDim);
        const float3 invDir = 1.f / ray.dir;
        const float3 p0 = (bbMin - ray.origin) * invDir;
        const float3 p1 = (bbMax - ray.origin) * invDir;
        const float3 tn = min(p0, p1);
        const float3 tf = max(p0, p1);
        return float2(
            max(max(tn.x, max(tn.y, tn.z)), ray.tMin),
            min(min(tf.x, min(tf.y, tf.z)), ray.tMax));
    }

    /** Advance to the next tail-cell boundary after world distance t. */
    float tailStep(const Ray ray, const float t)
    {
        const float3 co = (ray.origin - gTailOrigin) / gTailCellSize;
        const float3 cd = ray.dir / gTailCellSize;
        const float3 invCd = 1.f / cd;
        const float3 p = co + cd * t;
        const float3 cellMin = floor(p);
        const float3 next = (cellMin + select(cd >= 0.f, 1.f, 0.f) - co) * invCd;
        // Half-cell dilation on the boundary in the degenerate flat case is
        // unnecessary here: the forward-progress guard at the call sites
        // handles exact-boundary starts.
        return min(next.x, min(next.y, next.z));
    }

    /** Tail cell index at world distance t (clamped). */
    int3 tailCell(const Ray ray, const float t)
    {
        const float3 p = (ray.origin + ray.dir * t - gTailOrigin) / gTailCellSize;
        return clamp(int3(floor(p)), int3(0), int3(gTailDim) - 1);
    }

    /** Sum of REAL densities over instances covering distance t; the true
        summed field the tail's majorant bounds.
    */
    [mutating] float tailSumDensity(const IntervalList list, const float t, const float3 posW, const float3 u)
    {
        const uint mask = coveringMask(list, t);
        float dSum = 0.f;
        for (uint i = 0; i < list.count; ++i)
        {
            if ((mask & (1u << i)) == 0) continue;
            const GridVolume gv = gScene.getGridVolume(list.ids[i]);
            Grid grid;
            gScene.getGrid(gv.getDensityGrid(), grid);
            statOverlapLookups++;
            dSum += instanceDensity(gv, grid, posW, u);
        }
        return dSum;
    }

    /** Residual ratio tracking against the tail: control = per-cell summed
        mean (analytic), residual bound = max(maj - mean, mean) since the
        conservative per-cell minorant is 0. Real taps hit the true summed
        field, so the result is unbiased for any bake quality.
    */
    [mutating] float evalTransmittanceTail<S : ISampleGenerator>(const IntervalList list, const Ray ray, inout S sg)
    {
        const float2 span = tailSpan(ray);
        if (span.x >= span.y) return 1.f;

        float tauC = 0.f;
        float Tr = 1.f;

        float t = span.x;
        while (t < span.y)
        {
            gvsStatCell();
            const int3 cell = tailCell(ray, t + 1e-5f);
            float tExit = min(tailStep(ray, t), span.y);
            // Forward-progress guard for exact-boundary starts.
            if (tExit <= t) tExit = min(span.y, t + max(1e-4f, abs(t) * 1e-5f));

            const float2 cellData = gTailTex[cell];
            const float sigmaMaj = cellData.x;
            const float sigmaC = min(cellData.y, sigmaMaj);
            const float bound = max(sigmaMaj - sigmaC, sigmaC);

            tauC += sigmaC * (tExit - t);

            if (bound > 0.f)
            {
                const float invBound = 1.f / bound;
                float tt = t;
                while (true)
                {
                    tt -= log(1.f - sampleNext1D(sg)) * invBound;
                    if (tt >= tExit) break;
                    gvsStatTap();
                    const float3 posW = ray.origin + tt * ray.dir;
                    const float d = tailSumDensity(list, tt, posW, sampleNext3D(sg));
                    // max(0,..) is the same conservative guard as the stock
                    // tracker for the rare tap that exceeds the baked majorant.
                    Tr *= max(0.f, 1.f - (d - sigmaC) * invBound);
                }
            }
            t = tExit;

            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return exp(-tauC) * Tr;
    }

    /** Delta tracking against the tail majorant with real summed-field taps
        and density-proportional instance attribution. Exactly unbiased.
    */
    [mutating] bool sampleDistanceTail<S : ISampleGenerator>(const IntervalList list, const Ray ray, inout S sg, out DistanceSample ds)
    {
        ds = {};
        const float2 span = tailSpan(ray);
        if (span.x >= span.y) return false;

        float tauLeft = -log(1.f - sampleNext1D(sg));

        float t = span.x;
        while (t < span.y)
        {
            gvsStatCell();
            const int3 cell = tailCell(ray, t + 1e-5f);
            float tExit = min(tailStep(ray, t), span.y);
            if (tExit <= t) tExit = min(span.y, t + max(1e-4f, abs(t) * 1e-5f));

            const float sigmaMaj = gTailTex[cell].x;
            if (sigmaMaj > 0.f)
            {
                const float invSigma = 1.f / sigmaMaj;
                float tj = t;
                while (true)
                {
                    const float dtau = (tExit - tj) * sigmaMaj;
                    if (tauLeft > dtau)
                    {
                        tauLeft -= dtau;
                        break;
                    }
                    tj += tauLeft * invSigma;
                    gvsStatTap();

                    const float3 posW = ray.origin + tj * ray.dir;
                    const float3 u = sampleNext3D(sg);
                    const uint mask = coveringMask(list, tj);
                    float densities[kMaxInstances];
                    float dSum = 0.f;
                    for (uint i = 0; i < list.count; ++i)
                    {
                        densities[i] = 0.f;
                        if ((mask & (1u << i)) == 0) continue;
                        const GridVolume gv = gScene.getGridVolume(list.ids[i]);
                        Grid grid;
                        gScene.getGrid(gv.getDensityGrid(), grid);
                        statOverlapLookups++;
                        densities[i] = instanceDensity(gv, grid, posW, u);
                        dSum += densities[i];
                    }

                    if (sampleNext1D(sg) * sigmaMaj < dSum)
                    {
                        uint chosen = 0;
                        float pick = sampleNext1D(sg) * dSum;
                        for (uint i = 0; i < list.count; ++i)
                        {
                            if ((mask & (1u << i)) == 0) continue;
                            chosen = i;
                            pick -= densities[i];
                            if (pick <= 0.f) break;
                        }
                        ds.t = tj;
                        ds.thp = gScene.getGridVolume(list.ids[chosen]).data.albedo;
                        ds.instance = list.ids[chosen];
                        return true;
                    }
                    tauLeft = -log(1.f - sampleNext1D(sg));
                }
            }
            t = tExit;
        }
        return false;
    }

    /** M candidate processes sharing one tail DDA - the tail counterpart of
        sampleCandidatesSweepIntervals, same independence argument.
    */
    [mutating] void sampleCandidatesTail<S : ISampleGenerator>(const IntervalList list, const Ray ray, const uint numProcesses, inout S sg, out CandidateSet cands)
    {
        cands = {};
        cands.count = 0;
        if (list.count == 0) return;

        const float2 span = tailSpan(ray);
        if (span.x >= span.y) return;

        const uint nProc = min(numProcesses, kRisM);
        float tauLeft[kRisM];
        uint aliveMask = (nProc >= 32) ? 0xffffffffu : ((1u << nProc) - 1u);
        for (uint j = 0; j < kRisM; ++j) tauLeft[j] = -log(1.f - sampleNext1D(sg));

        float t = span.x;
        while (t < span.y && aliveMask != 0)
        {
            statCandCells++;
            const int3 cell = tailCell(ray, t + 1e-5f);
            float tExit = min(tailStep(ray, t), span.y);
            if (tExit <= t) tExit = min(span.y, t + max(1e-4f, abs(t) * 1e-5f));

            const float sigmaMaj = gTailTex[cell].x;
            if (sigmaMaj > 0.f)
            {
                const float invSigma = 1.f / sigmaMaj;
                for (uint j = 0; j < kRisM; ++j)
                {
                    if ((aliveMask & (1u << j)) == 0) continue;
                    float tj = t;
                    while (true)
                    {
                        const float dtau = (tExit - tj) * sigmaMaj;
                        if (tauLeft[j] > dtau)
                        {
                            tauLeft[j] -= dtau;
                            break;
                        }
                        tj += tauLeft[j] * invSigma;
                        statRealTaps++;

                        const float3 posW = ray.origin + tj * ray.dir;
                        const float3 u = sampleNext3D(sg);
                        const uint mask = coveringMask(list, tj);
                        float densities[kMaxInstances];
                        float dSum = 0.f;
                        for (uint i = 0; i < list.count; ++i)
                        {
                            densities[i] = 0.f;
                            if ((mask & (1u << i)) == 0) continue;
                            const GridVolume gv = gScene.getGridVolume(list.ids[i]);
                            Grid grid;
                            gScene.getGrid(gv.getDensityGrid(), grid);
                            statOverlapLookups++;
                            densities[i] = instanceDensity(gv, grid, posW, u);
                            dSum += densities[i];
                        }

                        if (sampleNext1D(sg) * sigmaMaj < dSum)
                        {
                            uint chosen = 0;
                            float pick = sampleNext1D(sg) * dSum;
                            for (uint i = 0; i < list.count; ++i)
                            {
                                if ((mask & (1u << i)) == 0) continue;
                                chosen = i;
                                pick -= densities[i];
                                if (pick <= 0.f) break;
                            }
                            cands.t[cands.count] = tj;
                            cands.inst[cands.count] = list.ids[chosen];
                            cands.count++;
                            aliveMask &= ~(1u << j);
                            break;
                        }
                        tauLeft[j] = -log(1.f - sampleNext1D(sg));
                    }
                }
            }
            t = tExit;
        }
    }
#endif // USE_MERGED_TAIL
};
