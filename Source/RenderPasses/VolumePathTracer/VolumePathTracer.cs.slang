/***************************************************************************
 # Reference volumetric path tracer for NanoVDB grid volumes.
 #
 # Brute-force, unbiased-by-construction: null-collision distance sampling
 # through ALL grid volume instances in the scene (VolumeInstanceSampler),
 # Henyey-Greenstein scattering with per-instance anisotropy, and next-event
 # estimation to the environment with ratio-tracked shadow transmittance.
 #
 # There is deliberately no reuse, no LoD and no resampling here. This is the
 # ground-truth image that faster estimators get validated against.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

// Provides is_valid(name) for optional render-pass I/O.
#define is_valid(name) (is_valid_##name != 0)

import Scene.Scene;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Volumes.PhaseFunction;
// Imported directly (imports are not transitive) for the gGvsStat* per-thread
// work counters that attribute the marching cost done inside the sampler.
import Rendering.Volumes.GridVolumeSampler;
import VolumeInstanceSampler;

cbuffer CB
{
    uint gFrameCount;
    uint2 gFrameDim;
    /// World-space pixel-footprint growth per unit distance
    /// (2*tan(fovY/2)/resY * user scale). 0 = footprint-driven residual mip off.
    float gFootprintSpread;
    /// Stage B temporal reuse: previous frame's camera.
    float4x4 gPrevViewProj;
    float3 gPrevCamPos;
    uint gTemporalEnable; ///< 0 until the host has valid history buffers + camera.
    /// Confidence cap: prev M is clamped to this multiple of the CURRENT
    /// candidate budget, bounding how long stale history can dominate.
    float gTemporalMCap;
    /// Stage C spatial reuse: Gaussian sigma (in pixels) of the neighbor
    /// offsets drawn around the reprojected pixel. ReSTIR PT Enhanced found
    /// Gaussian sigma 16 matches (and slightly beats) the classic 30 px disk.
    /// NOTE: spatial reuse currently FAILS the converged-identity gate
    /// (best measured +0.52%..+1.0% with sigmaT reweights; -5.6% without) -
    /// ship configs keep it OFF until the cross-ray sigmaT problem is solved.
    float gSpatialRadiusPx;
    /// Defensive floor on the RIS target, relative to a fully-lit isotropic
    /// vertex (1/4pi). Bounds the L/Lhat mismatch ratio: measured fireflies
    /// (single pixels at 800-2200x the mean) are candidates whose coarse
    /// target said "dark" (backscatter phase x floored visibility ~ 1e-6)
    /// while the survivor's real NEE found light. Any strictly positive
    /// target is unbiased, so flooring trades a little selection quality in
    /// dark regions for a bounded weight spread.
    float gRisTargetFloor;

    /// Russian-roulette survival floor and start bounce for the continuation
    /// loop. Nsight per-dispatch timing puts shadeMain (phase B: shading, NEE
    /// and the bounce loop) at 18.5-19.8ms of a ~33ms frame - the single
    /// largest item in the pass, despite only ~13% of pixels enqueueing a path.
    /// The cost is path LENGTH: in a high-albedo cloud throughput stays near 1,
    /// so q = max(floor, 1 - maxThp) sits at its floor and paths run tens of
    /// bounces deep. Roulette is unbiased at ANY q, so both are a pure
    /// cost/variance trade - raise the floor or start earlier to shorten the
    /// tail, and let the validation matrix price the variance.
    /// CB values, not defines: sweeping them must not recompile the shader.
    float gRouletteMinQ;
    uint  gRouletteStartBounce;
}

#ifndef USE_SPATIAL_REUSE
#define USE_SPATIAL_REUSE 0
#endif
#ifndef SPATIAL_NEIGHBORS
#define SPATIAL_NEIGHBORS 0
#endif
// Stream compaction (ReSTIR PT Enhanced 6.2.2 / the UE classify->compact->
// dense-dispatch pattern). Phase A ("main") runs per pixel: escape estimate,
// candidate generation, reservoir merges - the work EVERY pixel owes - and
// appends the ~13% of pixels whose reservoir selected a real scatter vertex
// to a queue. Phase B ("shadeMain") is an indirect dispatch with one thread
// per QUEUED PATH: shading, NEE and the bounce loop run in dense waves
// instead of warps that are 87% idle lanes (measured: noCand 1.807M of
// 2.07M pixels). Estimator-identical by construction - the same vertices
// get shaded with the same math; only the thread that shades them changes.
#ifndef USE_COMPACTION
#define USE_COMPACTION 0
#endif
// Fused escape+candidate traversal. Derived, not host-fed: the two TLAS
// RayQuery passes per pixel walked the SAME primary ray (measured 3.5
// samplerCalls and 14.4 AABB tests per pixel), so when every prerequisite
// is compiled in, one traversal serves both. Temporal history is required
// because the adaptive-M budget switches to LAST frame's transmittance
// (MegaLights-style feedback, stored in the reservoir pad) - that is what
// breaks the Tfull -> M -> candidates dependency chain.
#define USE_FUSED_SWEEP (USE_RIS && USE_BRICK_TLAS && USE_SHARED_CANDIDATE_SWEEP && USE_TEMPORAL_REUSE && MAX_BOUNCES > 0)

/** Snapshot accessors for the sampler-internal work counters. Compile to 0
    when GRID_VOLUME_SAMPLER_STATS is off, so all bucket math folds away.
*/
uint gvsCellsNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatCells;
#else
    return 0;
#endif
}
uint gvsTapsNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatTaps;
#else
    return 0;
#endif
}
/** Brick-cache coherence. The whole value of the brick-coherent tap path rests
    on consecutive taps landing in the same 8^3 brick; hits/(hits+misses) is that
    assumption made measurable rather than believed. A low hit rate means the
    cache is dead weight and should be reverted, not tuned.
*/
uint gvsBrickHitsNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatBrickHits;
#else
    return 0;
#endif
}
uint gvsBrickMissesNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatBrickMisses;
#else
    return 0;
#endif
}

/** Resource-bearing parameters must live in a ParameterBlock, not as bare
    globals. EnvMapSampler holds a SamplerState and a Texture2D; declaring it at
    global scope makes Slang lay it out as loose root parameters, which fails at
    dispatch. This mirrors how PathTracer.slang holds its EnvMapSampler.
*/
struct VolumePathTracerParams
{
    EnvMapSampler envMapSampler;
};

ParameterBlock<VolumePathTracerParams> gParams;

RWTexture2D<float4> gOutputColor;

/** Demodulated split (spec section 6): color = Lin + T * bg, exactly.
    Lin is what a denoiser is allowed to blur; T is the sharp wisp channel
    (temporal accumulation only); bg is deterministic and noise-free.
*/
RWTexture2D<float4> gOutputLin;
RWTexture2D<float4> gOutputT;
RWTexture2D<float4> gOutputBg;

/** Raw RIS probe. R = exit code (below), G = candidate count, B = target
    weight sum, A = RIS weight W. No thresholds, no normalisation.
*/
RWTexture2D<float4> gRisDebug;

/** GPU-side tallies, read back and logged by the host.
    [0..6] = RIS exit-code histogram
    [7]    = total candidates emitted
    [8]    = pixels whose primary ray actually entered the medium
    WORK counters (this is the "where do the milliseconds go" data):
    [9]  = instance AABB tests      [10] = segments marched
    [11] = coarse shadow walks      [12] = coarse cells stepped (shadow walks)
    [13] = candidate-sweep majorant DDA cells stepped
    [14] = candidate-sweep real density taps
    [15] = sampler entry calls (sampleDistance + evalTransmittance)
    COST buckets: the marching done INSIDE GridVolumeSampler, attributed per
    phase by snapshotting its per-thread counters around each call region.
    This is the work the old [WORK] line was blind to (measured: a ~470 ms
    frame with every counted term near-flat).
    [16] = escape-term (Tfull) DDA cells   [17] = escape-term real taps
    [18] = candidate-phase DDA cells       [19] = candidate-phase real taps
    [20] = shading/march DDA cells (NEE + continuation + reference path)
    [21] = shading/march real taps
    [22] = overlap null-collision steps    [23] = overlap density lookups
    [24] = entries that took the merged-tail path (backend attribution:
           tail = s[24], RayQuery/intervals = s[15] - s[24])
    [25] = sum of per-pixel candidate budgets actually used (adaptive M:
           s[25]/pixels = average M; equals kRisM when the feature is off)
    [26] = pixels whose reservoir merged a previous-frame sample (Stage B
           temporal reuse activity; 0 when history is invalid or disabled)
*/
RWStructuredBuffer<uint> gRisStats;

// Where the RIS block stopped. Each value is a DIFFERENT bug with a different
// fix, which is why this is measured rather than assumed.
static const uint kRisExitNoMedium    = 0; ///< Ray missed every instance, or maxBounces == 0.
static const uint kRisExitNoCandidate = 1; ///< All M delta-tracking walks escaped: no real collision on this ray (legitimate for thin regions).
static const uint kRisExitZeroTarget  = 2; ///< Real collisions found but every weight was 0: target collapsed.
static const uint kRisExitZeroDensity = 3; ///< UNUSED since candidates became real collisions - must stay 0 in the log.
static const uint kRisExitZeroTrans   = 4; ///< UNUSED since the transmittance factor cancels analytically - must stay 0.
static const uint kRisExitTerminated  = 5; ///< Vertex shaded, but phase sampling / RR ended the path.
static const uint kRisExitContributed = 6; ///< Full contribution made.

/** Raw per-pixel work counters. Not normalised, not thresholded: read exact
    values with Mogwai's pixel inspector, or scale them for a heatmap.
      R = segments marched          G = overlap null-collision steps
      B = instance AABB tests       A = max instances covering one segment
*/
RWTexture2D<float4> gWorkDebug;

static const uint kMaxBounces = MAX_BOUNCES;
static const bool kUseNEE = USE_NEE;
static const bool kUseRussianRoulette = USE_RUSSIAN_ROULETTE;
static const bool kUseEnvLight = USE_ENV_LIGHT;

/** UE MegaLights lesson (spec section 9: "a fixed per-pixel ray budget is the
    shippable shape"): instead of one NEE shadow ray at EVERY path vertex -
    measured as the dominant [COST] bucket (shadeNEE ~43 cells / ~30 taps per
    pixel) - keep ONE uniformly reservoir-sampled vertex per path and trace ONE
    shadow ray at path end, weighted by the vertex count.

    Unbiased: for vertices v_1..v_N, E[N * NEE(v_K)] with K uniform equals
    sum_k NEE(v_k) exactly. The trade is variance for a fixed budget, which
    temporal accumulation absorbs - the same bet MegaLights ships.
*/
static const bool kSingleNeePerPath = USE_SINGLE_NEE;

/** Transmittance-adaptive candidate count (our extension of the MegaLights
    fixed-budget idea; UE's own next step would be temporal feedback, which for
    this renderer is Stage B reservoir reuse).

    Measured: 82-88% of candidate processes march to ESCAPE (avgCandidates
    0.48 at M=4), i.e. most of the dominant candGen bucket is spent proving
    that nearly-transparent rays scatter nothing. The escape term already
    computes Tfull BEFORE candidates run - a fresher signal than any
    reprojected history - so M is chosen per pixel from it:
        M = clamp(ceil(Mmax * (1 - Tfull)), 1, Mmax).

    Unbiased: M depends only on draws made BEFORE the candidate RNG, and the
    scatter estimator is unbiased conditioned on ANY M >= 1 (the reservoir
    weight divides by the M actually used), so the total expectation is exact.
    M never reaches 0: a stochastic Tfull == 1 does not prove empty space.
*/
static const bool kUseAdaptiveM = USE_ADAPTIVE_M;

/** Stage B: temporal reservoir reuse (VNA spec section 5; the principled form
    of UE MegaLights' temporal feedback). Last frame's surviving primary
    scatter vertex is shifted onto this frame's ray (t-shift: t' = projection
    of its world position onto the new ray; Jacobian = dot(prevDir, newDir))
    and merged into the current reservoir in RATIO SPACE - the same space the
    fresh candidates' weights live in (w = p-hat/p = Lhat, the pdf-cancelled
    form) and the same space the stored W is expressed in:
        w_prev = Lhat(t') * W_prev * M_prev_capped * J.
    The implicit (sigmaT)_new/(sigmaT)_prev shift ratio is exactly 1 for a
    static camera and is the standard shift approximation under motion.
    (History note: multiplying sigma*T into w_prev - mixing canonical-UCW
    units into ratio space - double-counted the cancelled factor; history
    then never won proportionally to its confidence while M_total still grew,
    darkening every contribution by ~1/(1+cap). Measured and fixed.)
    The estimator keeps its universal form W = wSum / (M_total * Lhat_sel).
    Empty previous reservoirs contribute confidence only (M adds, wSum does
    not) - they are M genuine zero-weight candidates, so this stays unbiased.
    Domain guard for shifted samples: a DETERMINISTIC coarse-mean tap (zero
    mean = vertex left the cloud), never a stochastic test.
    MIS note: confidence weights (M-capping) approximate the generalized
    balance heuristic, the same approximation every shipped ReSTIR (including
    Volumetric ReSTIR) uses; exact pairwise MIS is the documented follow-up.
*/
static const bool kUseTemporalReuse = USE_TEMPORAL_REUSE;

struct TemporalReservoir
{
    float3 pos; ///< Survivor's world position (re-expressed on the ray that kept it).
    float W;    ///< Unbiased contribution weight of the survivor (0 = empty reservoir).
    float M;    ///< Confidence: total candidates this reservoir stands for.
    uint inst;  ///< Survivor's instance (albedo/anisotropy source).
    float _pad0;
    float _pad1;
};

RWStructuredBuffer<TemporalReservoir> gReservoirPrev; ///< Last frame's reservoirs (read).
RWStructuredBuffer<TemporalReservoir> gReservoirCurr; ///< This frame's reservoirs (written).

#if USE_COMPACTION
/** One queued path = one RIS survivor. 16 bytes: everything else the shade
    phase needs (primary ray, albedo, anisotropy) is recomputed from pixel
    and instance id, which is cheaper than hauling it through memory.
*/
struct ScatterEntry
{
    uint pixel;  ///< Packed x | y<<16.
    float t;     ///< Selected scatter distance on the primary ray.
    uint inst;   ///< Grid volume instance of the survivor.
    float W;     ///< Unbiased contribution weight of the reservoir.
};
RWStructuredBuffer<ScatterEntry> gScatterQueue;
RWStructuredBuffer<uint> gScatterCount; ///< [0] = number of queued paths.
#endif

/** Reservoir over path vertices for the single-NEE estimator: vertex k
    survives with probability 1/k, giving a uniform pick without knowing the
    path length in advance. Stores everything one deferred NEE needs.
*/
struct NeeVertexReservoir
{
    uint count;       ///< Vertices seen so far (the estimator's N).
    float3 pos;       ///< Scatter position of the kept vertex.
    float3 wi;        ///< Direction toward the previous vertex (away from pos).
    float g;          ///< Phase anisotropy at the kept vertex.
    float3 thp;       ///< Path throughput AT the kept vertex (pre-continuation).
};

/** VNA spec section 5, Stage A: resampled importance sampling (RIS) of the
    PRIMARY scatter vertex. Off by default - the plain path is the reference
    ground truth and must stay untouched (handoff rule).

    With RIS on, the pixel estimate splits into two independently estimated,
    non-overlapping terms:
      escape term   T * bg      - T estimated by the transmittance estimator
                                  (continuous value, lower variance than the
                                  binary escape event of the plain path);
      scatter term  RIS         - M cheap candidates from the coarse field
                                  (exact analytic pdfs), one survivor shaded
                                  with the REAL raw-VDB field. Unbiased for any
                                  target by the RIS identity (spec section 2, P1).
*/
static const bool kUseRIS = USE_RIS;

/** Radiance arriving from direction 'dir' when a ray escapes the medium.
*/
float3 evalEnvironment(const float3 dir)
{
    if (!kUseEnvLight)
        return float3(0.f);
    return gScene.envMap.eval(dir);
}

/** Next-event estimation: connect a scatter vertex to the environment.

    Importance-samples the env map, then estimates transmittance along the
    shadow ray by ratio tracking through all instances. The phase function is
    evaluated for the sampled direction and divided by the env-map pdf.
*/
float3 evalNEE<S : ISampleGenerator>(
    const float3 pos,
    const float3 wi,
    const HenyeyGreensteinPhaseFunction phase,
    inout VolumeInstanceSampler volumeSampler,
    inout S sg
)
{
    if (!kUseEnvLight)
        return float3(0.f);

    EnvMapSample lightSample;
    if (!gParams.envMapSampler.sample(sampleNext2D(sg), lightSample))
        return float3(0.f);
    if (lightSample.pdf <= 0.f)
        return float3(0.f);

    // Phase function value for the light direction.
    // Convention: both directions point away from the scatter point.
    const float3 fp = phase.eval(wi, lightSample.dir);
    if (all(fp <= 0.f))
        return float3(0.f);

    // Shadow ray through the medium. FLT_MAX so it leaves every volume entirely
    // rather than being clipped by the camera far plane.
    Ray shadowRay = Ray(pos, lightSample.dir, 0.f, FLT_MAX);
    const float Tr = volumeSampler.evalTransmittance(shadowRay, sg);
    if (Tr <= 0.f)
        return float3(0.f);

    return fp * lightSample.Le * Tr / lightSample.pdf;
}

/** Target weight Lhat for the RIS reservoir: cheap lighting guess at a
    candidate primary-scatter vertex. Shared by fresh candidates and by
    temporally reused samples so both are weighed under the SAME target.
*/
float evalTargetLhat(
    const uint inst,
    const float t,
    const Ray ray,
    const float3 wi,
    const bool haveTargetLight,
    const EnvMapSample targetLight,
    inout VolumeInstanceSampler volumeSampler
)
{
    if (!haveTargetLight) return 1.f;
    const float gj = gScene.getGridVolume(inst).data.anisotropy;
    const float3 fp = HenyeyGreensteinPhaseFunction(gj).eval(wi, targetLight.dir);
    const float3 posJ = ray.origin + t * ray.dir;
    const float tauShadow = volumeSampler.coarseOpticalDepth(Ray(posJ, targetLight.dir, 0.f, FLT_MAX));
    // Floored: deep points are lit by multiple scattering, which this
    // single-scatter target cannot see. Without the floor exp(-tau)
    // underflows to 0 and the target collapses.
    const float kTargetMinVisibility = 1e-3f;
    const float lhat = (fp.x + fp.y + fp.z) * (1.f / 3.f) * max(exp(-tauShadow), kTargetMinVisibility);
    // Defensive mixture floor (see gRisTargetFloor): the visibility floor
    // above bounds the SHADOW side, but the PHASE side can still push Lhat
    // to ~1e-6 on backscatter vertices. W divides by Lhat(selected) and the
    // shading multiplies by L/Lhat, so an underestimated target is the one
    // mechanism that produces unbounded weights. Floor relative to the
    // fully-lit isotropic value 1/(4pi).
    return max(lhat, gRisTargetFloor * M_1_4PI);
}

/** Shade one scatter vertex and set up the continuation ray.

    Shared by the reference path and the RIS primary path (VNA spec section 5,
    Stage A) so both run byte-identical vertex shading: NEE, phase sampling,
    Russian roulette. Returns false if the path terminates here.
*/
bool scatterAtVertex<S : ISampleGenerator>(
    const float3 scatterPos,
    const float3 wi,
    const float g,
    const uint bounce,
    inout Ray ray,
    inout float3 throughput,
    inout float3 radiance,
    inout NeeVertexReservoir neeReservoir,
    inout VolumeInstanceSampler volumeSampler,
    inout S sg
)
{
    const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(g);

    if (kUseNEE)
    {
        if (kSingleNeePerPath)
        {
            // MegaLights-shaped budget: defer NEE. Keep this vertex with
            // probability 1/count (streaming uniform reservoir); the single
            // shadow ray is traced once, at path end.
            neeReservoir.count++;
            if (neeReservoir.count == 1 || sampleNext1D(sg) * float(neeReservoir.count) < 1.f)
            {
                neeReservoir.pos = scatterPos;
                neeReservoir.wi = wi;
                neeReservoir.g = g;
                neeReservoir.thp = throughput;
            }
        }
        else
        {
            radiance += throughput * evalNEE(scatterPos, wi, phase, volumeSampler, sg);
        }
    }

    // Sample a new direction from the phase function.
    float3 wo;
    float pdf;
    float3 weight;
    if (!phase.sample(wi, wo, pdf, weight, sg))
        return false;
    throughput *= weight;

    if (kUseRussianRoulette && bounce > gRouletteStartBounce)
    {
        const float maxThp = max(throughput.x, max(throughput.y, throughput.z));
        const float q = max(gRouletteMinQ, 1.f - maxThp);
        if (sampleNext1D(sg) < q)
            return false;
        throughput /= (1.f - q);
    }

    if (all(throughput <= 0.f))
        return false;

    ray = Ray(scatterPos, wo, 0.f, FLT_MAX);
    return true;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim))
        return;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    VolumeInstanceSampler volumeSampler = {};

    // Primary ray. Extend tMax past the camera far plane: the far plane is a
    // rasterization concept and would otherwise clip the volume.
    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);
    ray.tMax = FLT_MAX;

    float3 radiance = float3(0.f);
    float3 throughput = float3(1.f);

    // Section 6 demodulation. The environment along the PRIMARY direction is a
    // deterministic lookup, so it is evaluated once here, noise-free. primaryT
    // holds an unbiased estimate of the primary transmittance: on the plain
    // path it is the binary escape outcome of the first free flight; on the
    // RIS path it is the continuous ratio-tracked value (same expectation,
    // lower variance). Either way it only ever needs TEMPORAL averaging.
    // Everything else the path gathers is in-scattered radiance.
    const float3 primaryBg = evalEnvironment(ray.dir);
    float primaryT = 0.f;

    uint bounce = 0;
    bool traceLoop = true;

    // Single-NEE-per-path reservoir (UE MegaLights lesson). Resolved once,
    // right before the outputs are written, so every path exit is covered.
    NeeVertexReservoir neeReservoir = {};

    // Cost-bucket state: attribute the sampler-internal marching per phase.
    uint cellsMark = gvsCellsNow();
    uint tapsMark = gvsTapsNow();
    uint cellsT = 0, tapsT = 0;
    uint cellsCand = 0, tapsCand = 0;

    // RIS probe state (written to gRisDebug at the end).
    uint risExit = kRisExitNoMedium;
    float risCount = 0.f;
    uint risMUsed = 0; ///< Candidate budget actually used this pixel (adaptive M diagnostics).
    uint risTemporalMerged = 0; ///< 1 if a previous-frame sample merged into this pixel's reservoir.
    float risWSum = 0.f;
    float risW = 0.f;

    if (kUseRIS)
    {
        // ---- Escape/background term: continuous transmittance estimate. ----
        // Replaces the plain path's binary escape event with the (residual)
        // ratio-tracked value: same expectation, strictly lower variance, and
        // it feeds the section-6 T channel directly.
        // The primary ray is the one place the pixel footprint is well defined,
        // so footprint-driven residual-mip selection (UE projected-error rule)
        // applies here and is reset immediately after: NEE shadow rays measure
        // distance from the vertex, not the camera, and would misuse it.
#if !USE_FUSED_SWEEP
        volumeSampler.footprintSpread = gFootprintSpread;
        const float Tfull = volumeSampler.evalTransmittance(ray, sg);
        volumeSampler.footprintSpread = 0.f;
        primaryT = Tfull;
        radiance += Tfull * primaryBg;

        cellsT = gvsCellsNow() - cellsMark;
        tapsT = gvsTapsNow() - tapsMark;
        cellsMark = gvsCellsNow();
        tapsMark = gvsTapsNow();
#endif

        // ---- Scatter term: RIS over the primary vertex. ----
        //
        // CANDIDATES ARE REAL COLLISIONS. Each candidate comes from a full
        // null-collision (delta-tracking) walk, so it is by construction a
        // genuine scatter event in the real medium. The previous design drew
        // candidates from the majorant CDF, which places points anywhere in an
        // occupied 8^3 brick - and measurement showed 87% of survivors landed
        // on ZERO real density (that ratio IS the null-collision rate; delta
        // tracking survives it by CONTINUING to march, a single RIS pick does
        // not). Generating candidates this way removes that failure entirely
        // rather than compensating for it.
        //
        // THE ALGEBRA COLLAPSES. Delta tracking samples distance with
        //     p(t) = sigma_t(t) * T(t)
        // and the true integrand is f(t) = T(t) * sigma_t(t) * albedo * L(t).
        // Choosing the target p-hat(t) = p(t) * Lhat(t) makes the RIS weight
        //     w = p-hat/p = Lhat
        // so p CANCELS - it is never evaluated - and the final estimate is
        //     f(y)/p-hat(y) * (1/M) sum(w) = albedo * L(y)/Lhat(y) * (1/M) sum(w)
        // i.e. just albedo * (sum(w) / (M * w_sel)), with L supplied by the
        // normal vertex shading. No pdf, no transmittance estimate to the
        // vertex, no huge 1/pdf weights - all sources of the fireflies.
        traceLoop = false;
        if (kMaxBounces > 0)
        {
            // One shared target light direction per pixel. It only shapes the
            // TARGET (which candidate wins); the surviving vertex does its own
            // fresh NEE, so no correlation enters the result.
            EnvMapSample targetLight = {};
            bool haveTargetLight = false;
            if (kUseEnvLight)
            {
                haveTargetLight = gParams.envMapSampler.sample(sampleNext2D(sg), targetLight) && targetLight.pdf > 0.f;
            }

            const float3 wi = -ray.dir;

            // All M candidate walks travel the SAME primary ray, so the instance
            // intervals are identical for every one of them: gather once. Doing
            // it per walk cost 130 AABB tests/pixel (271M/frame) - measured as
            // the largest work term in the pass.
            const VolumeInstanceSampler::IntervalList primaryList = volumeSampler.gatherIntervals(ray);

            // ---- Candidate generation: M real delta-tracking collisions. ----
#if USE_FUSED_SWEEP
            // MegaLights temporal feedback: the budget comes from LAST
            // frame's transmittance at this pixel (reservoir pad), so escape
            // and candidates can share one traversal below. Any M in
            // [1, kRisM] is unbiased - the reservoir weight divides by the M
            // actually used - so a stale hint moves cost, never the mean.
            // Cleared/invalid history (T hint 0) uses the full budget.
            uint mUsed = VolumeInstanceSampler::kRisM;
            if (kUseAdaptiveM && gTemporalEnable != 0)
            {
                const float tHint = gReservoirPrev[pixel.y * gFrameDim.x + pixel.x]._pad0;
                if (tHint > 0.f)
                    mUsed = clamp(uint(ceil(float(VolumeInstanceSampler::kRisM) * saturate(1.f - tHint))), 1u, VolumeInstanceSampler::kRisM);
            }
#else
            // Per-pixel candidate budget from the transmittance just computed
            // (see kUseAdaptiveM above). The reservoir weight divides by this
            // exact value, which is what keeps per-pixel M unbiased.
            const uint mUsed = kUseAdaptiveM
                ? clamp(uint(ceil(float(VolumeInstanceSampler::kRisM) * saturate(1.f - Tfull))), 1u, VolumeInstanceSampler::kRisM)
                : VolumeInstanceSampler::kRisM;
#endif
            risMUsed = mUsed;

            // Primary-ray footprint applies here exactly as it did for the
            // escape term: it drives the merged-tail gate and the residual-mip
            // choice inside the marchers. Reset after - the target's shadow
            // walks and the survivor's NEE measure from the vertex, not the
            // camera.
            volumeSampler.footprintSpread = gFootprintSpread;
            VolumeInstanceSampler::CandidateSet cands;
#if USE_FUSED_SWEEP
            // ONE traversal for the escape term AND the M candidate chains
            // (they used to walk the same primary ray twice). Estimator-
            // identical to the two-pass form; matrix/probe gates the claim.
            const float Tfull = volumeSampler.evalEscapeAndCandidatesRQ(ray, mUsed, sg, cands);
            primaryT = Tfull;
            radiance += Tfull * primaryBg;
#elif USE_SHARED_CANDIDATE_SWEEP
            // One traversal, M processes (UE lesson: share the walk, keep the
            // chains independent). Distribution identical to the M-walk loop
            // below - VNA_RisValidate.py is the gate for that claim.
            volumeSampler.sampleCandidatesSharedSweep(primaryList, ray, mUsed, sg, cands);
#else
            // Reference variant: M independent full walks along the same ray.
            // Kept selectable for A/B of both cost and correctness.
            cands = {};
            cands.count = 0;
            for (uint j = 0; j < mUsed; ++j)
            {
                VolumeInstanceSampler::DistanceSample ds;
                // Escaped candidates are not scatter events: weight 0. The
                // escape term above already accounts for that radiance, so
                // there is no double counting.
                if (!volumeSampler.sampleDistanceWithList(primaryList, ray, sg, ds)) continue;
                cands.t[cands.count] = ds.t;
                cands.inst[cands.count] = ds.instance;
                cands.count++;
            }
#endif
            volumeSampler.footprintSpread = 0.f;

            // ---- Streaming reservoir over the candidates. ----
            float wSum = 0.f;
            float selW = 0.f;
            VolumeInstanceSampler::DistanceSample selDs = {};
            uint nReal = cands.count;

            for (uint j = 0; j < cands.count; ++j)
            {
                // Lhat: the cheap lighting guess. Any strictly positive value is
                // unbiased (spec section 2, P1) - it only decides which real
                // collision is kept.
                const float w = evalTargetLhat(cands.inst[j], cands.t[j], ray, wi, haveTargetLight, targetLight, volumeSampler);

                wSum += w;
                if (w > 0.f && sampleNext1D(sg) * wSum < w)
                {
                    selDs.t = cands.t[j];
                    selDs.instance = cands.inst[j];
                    // Same throughput rule as the reference sampler: the chosen
                    // instance's single-scattering albedo.
                    selDs.thp = gScene.getGridVolume(cands.inst[j]).data.albedo;
                    selW = w;
                }
            }

            // ---- Fresh-stage unbiased contribution weight (Stage A). ----
            // W = (1/M sum w) / Lhat(y) over the fresh candidates only. The
            // reuse chain below REPLACES W step by step; it never divides by
            // a pooled confidence again (that constant-MIS normalization was
            // the measured -1.4%/-4% darkening: sources that cannot cover the
            // whole ray still swallowed 1/mTotal of everyone's mass).
            float W = selW > 0.f ? wSum / (float(mUsed) * selW) : 0.f;
            float confTotal = float(mUsed);
#if USE_TEMPORAL_REUSE
            if (gTemporalEnable != 0)
            {
                // Fetch location: the previous pixel that saw our representative
                // point. The choice only decides WHICH history we try to reuse -
                // the merge weight below re-validates it - so a heuristic here
                // can cost variance, never correctness.
                float tRep = selDs.t;
                if (selW <= 0.f && primaryList.count > 0)
                {
                    float tEnter = FLT_MAX;
                    float tExit = 0.f;
                    for (uint i = 0; i < primaryList.count; ++i)
                    {
                        tEnter = min(tEnter, primaryList.t0[i]);
                        tExit = max(tExit, primaryList.t1[i]);
                    }
                    tRep = 0.5f * (tEnter + tExit);
                }

                if (tRep > 0.f)
                {
                    const float4 prevClip = mul(gPrevViewProj, float4(ray.origin + tRep * ray.dir, 1.f));
                    if (prevClip.w > 0.f)
                    {
                        const float2 prevUv = (prevClip.xy / prevClip.w) * float2(0.5f, -0.5f) + 0.5f;
                        const int2 prevPixCenter = int2(floor(prevUv * float2(gFrameDim)));

                        // ---- Stage B/C reuse chain: sequential EXACT pairwise MIS. ----
                        // Source 0 is the temporal reprojection, sources 1..K
                        // are prev-frame reservoirs at Gaussian offsets. Each
                        // source merges as a textbook TWO-technique GRIS step
                        // against the current reservoir, with the balance
                        // heuristic evaluated pointwise: m_src(y) + m_cur(y)
                        // = 1 for the SAME y, so no mass is lost when a
                        // source cannot cover part of the ray. Confidence
                        // only weights the m's; it never divides the
                        // estimate. Temporal validated at -0.09% (p7/p8).
                        // SPATIAL sources still fail the gate (sigmaT ratio;
                        // see the CB note) and stay OFF in ship configs.
                        const uint kMergeSources = 1 + (USE_SPATIAL_REUSE != 0 ? SPATIAL_NEIGHBORS : 0);
                        for (uint src = 0; src < kMergeSources; ++src)
                        {
                            int2 prevPix = prevPixCenter;
#if USE_SPATIAL_REUSE
                            if (src > 0)
                            {
                                // Gaussian offset via Box-Muller: matches the
                                // ReSTIR-PT-Enhanced finding that a Gaussian
                                // (sigma ~16 px) selects more compatible
                                // neighbors than a uniform disk.
                                const float2 u = sampleNext2D(sg);
                                const float rad = gSpatialRadiusPx * sqrt(-2.f * log(max(u.x, 1e-6f)));
                                const float ang = M_2PI * u.y;
                                prevPix += int2(round(float2(rad * cos(ang), rad * sin(ang))));
                            }
#endif
                            // Off-screen source: skip entirely. A pixel that
                            // does not exist is NOT a zero-weight candidate,
                            // so it must not add confidence either.
                            if (any(prevPix < 0) || any(prevPix >= int2(gFrameDim)))
                                continue;

                            const TemporalReservoir prev = gReservoirPrev[prevPix.y * gFrameDim.x + prevPix.x];
                            const float srcCap = (src == 0) ? gTemporalMCap : 2.f;
                            const float cSrc = min(prev.M, srcCap * float(mUsed));
                            if (cSrc <= 0.f)
                                continue;

                            // ---- Forward shift: source sample onto our ray. ----
                            // The t-map is affine (t_c = const + t_i * jac), so
                            // jac serves both directions exactly (1/jac back).
                            float wSrc = 0.f;
                            float lhatPrev = 0.f;
                            float tNew = 0.f;
                            float jac = 0.f;
                            float3 dirSrc = float3(0.f);
                            const float3 srcVec = prev.pos - gPrevCamPos;
                            const float srcDist = length(srcVec);
                            if (srcDist > 0.f)
                            {
                                dirSrc = srcVec / srcDist;
                                jac = dot(dirSrc, ray.dir);
                            }
                            // jac floor: below it the affine map is too
                            // oblique to be a useful bijection; the source is
                            // treated as unable to generate here (m_src = 0
                            // AND q_rev = 0 below - support stays consistent).
                            const bool mapOk = jac > 0.1f;

                            if (prev.W > 0.f && mapOk)
                            {
                                tNew = dot(prev.pos - ray.origin, ray.dir);
                                if (tNew > ray.tMin)
                                {
                                    // Domain guard: DETERMINISTIC coarse-mean tap
                                    // at the shifted point - zero means the vertex
                                    // no longer sits in cloud; the partition hands
                                    // that mass back to the current sample instead
                                    // of losing it.
                                    const float3 posNew = ray.origin + tNew * ray.dir;
                                    const uint mask = volumeSampler.coveringMask(primaryList, tNew);
                                    float meanSigma = 0.f;
                                    for (uint i = 0; i < primaryList.count; ++i)
                                    {
                                        if ((mask & (1u << i)) == 0) continue;
                                        const GridVolume gvI = gScene.getGridVolume(primaryList.ids[i]);
                                        Grid gridI;
                                        gScene.getGrid(gvI.getDensityGrid(), gridI);
                                        const float3 iposI = mul(gvI.data.invTransform, float4(posNew, 1.f)).xyz - gridI.getMinIndex();
                                        meanSigma += volumeSampler.instanceSigmaMean(gvI, gridI, iposI);
                                    }
                                    if (meanSigma > 0.f)
                                    {
                                        lhatPrev = evalTargetLhat(prev.inst, tNew, ray, wi, haveTargetLight, targetLight, volumeSampler);
                                        if (lhatPrev > 0.f)
                                        {
                                            // Exact 2-way balance at the shifted
                                            // point y': source density is
                                            // c_i*qSrc/jac (its own-domain target
                                            // mapped into ours), current density
                                            // is confTotal*Lhat_c(y').
                                            const Ray srcRay = Ray(gPrevCamPos, dirSrc, 0.f, FLT_MAX);
                                            const float qSrc = evalTargetLhat(prev.inst, srcDist, srcRay, -dirSrc, haveTargetLight, targetLight, volumeSampler);
                                            const float a = cSrc * qSrc / jac;
                                            const float b = confTotal * lhatPrev;
                                            const float mSrc = (a + b) > 0.f ? a / (a + b) : 0.f;

                                            // sigma-T reweight for CROSS-RAY shifts:
                                            // stored W's live in the ratio space of
                                            // their OWN ray; a spatial shift needs
                                            // sigmaT_c(y')/sigmaT_i(y_i) (measured
                                            // -2.2%/neighbor when omitted). Both
                                            // halves from the DETERMINISTIC coarse
                                            // fields. Best measured residual:
                                            // +0.52% converged at +4.4 ms - still
                                            // above the gate, so spatial ships OFF;
                                            // kept for iteration. (The amortized
                                            // _pad1-certificate variant measured
                                            // WORSE, +1.0%: its sigma sum lacks the
                                            // covering-mask restriction this
                                            // numerator uses - see HANDOFF_4.)
                                            float shiftRatio = 1.f;
#if USE_SPATIAL_REUSE
                                            if (src > 0)
                                            {
                                                float sigmaSrc = 0.f;
                                                for (uint gi = 0; gi < gScene.getGridVolumeCount(); ++gi)
                                                {
                                                    const GridVolume gvS = gScene.getGridVolume(gi);
                                                    Grid gridS;
                                                    gScene.getGrid(gvS.getDensityGrid(), gridS);
                                                    const float3 iposS = mul(gvS.data.invTransform, float4(prev.pos, 1.f)).xyz - gridS.getMinIndex();
                                                    sigmaSrc += volumeSampler.instanceSigmaMean(gvS, gridS, iposS);
                                                }
                                                const float odC = volumeSampler.coarseOpticalDepth(Ray(ray.origin, ray.dir, 0.f, tNew));
                                                const float odS = volumeSampler.coarseOpticalDepth(Ray(gPrevCamPos, dirSrc, 0.f, srcDist));
                                                const float num = meanSigma * exp(-odC);
                                                const float den = sigmaSrc * exp(-odS);
                                                shiftRatio = den > 1e-8f ? clamp(num / den, 0.f, 8.f) : 0.f;
                                            }
#endif
                                            // GRIS resampling weight of the shifted
                                            // sample in OUR domain:
                                            //   w = m * Lhat_c(y') * W_prev * jac.
                                            wSrc = mSrc * lhatPrev * prev.W * jac * shiftRatio;
                                        }
                                    }
                                }
                            }

                            // ---- Our current sample, defended pairwise. ----
                            // Same partition evaluated at OUR sample x_c: the
                            // source's density there comes from the exact
                            // affine inverse (reverse shift). If the source
                            // cannot reach x_c (bad map, behind its camera,
                            // zero target), m_cur = 1 and nothing is lost.
                            float wCur = 0.f;
                            if (selW > 0.f && W > 0.f)
                            {
                                float qRev = 0.f;
                                if (mapOk)
                                {
                                    const float tSrc = (selDs.t - dot(gPrevCamPos - ray.origin, ray.dir)) / jac;
                                    if (tSrc > 0.f)
                                    {
                                        const Ray srcRay = Ray(gPrevCamPos, dirSrc, 0.f, FLT_MAX);
                                        qRev = evalTargetLhat(selDs.instance, tSrc, srcRay, -dirSrc, haveTargetLight, targetLight, volumeSampler);
                                    }
                                }
                                const float aC = mapOk ? cSrc * qRev / jac : 0.f;
                                const float bC = confTotal * selW;
                                const float mCur = (aC + bC) > 0.f ? bC / (aC + bC) : 0.f;
                                wCur = mCur * selW * W;
                            }

                            // ---- One streaming 2-way merge step. ----
                            const float wTot = wCur + wSrc;
                            if (wTot > 0.f)
                            {
                                if (wSrc > 0.f && sampleNext1D(sg) * wTot < wSrc)
                                {
                                    selDs.t = tNew;
                                    selDs.instance = prev.inst;
                                    selDs.thp = gScene.getGridVolume(prev.inst).data.albedo;
                                    selW = lhatPrev;
                                    risTemporalMerged = 1;
                                }
                                W = wTot / selW;
                            }
                            confTotal += cSrc;
                        }
                    }
                }
            }
#endif // USE_TEMPORAL_REUSE

            // candGen bucket = candidate generation + target weighting +
            // temporal merge: everything that SELECTS the primary vertex.
            cellsCand = gvsCellsNow() - cellsMark;
            tapsCand = gvsTapsNow() - tapsMark;
            cellsMark = gvsCellsNow();
            tapsMark = gvsTapsNow();

            risCount = float(nReal);
            risWSum = wSum;
            risExit = (nReal == 0) ? kRisExitNoCandidate : kRisExitZeroTarget;

            // W already holds the final unbiased contribution weight: the
            // fresh-stage value, refined in place by each exact pairwise
            // merge step above. No pooled-confidence division exists anymore.

#if USE_TEMPORAL_REUSE
            if (gTemporalEnable != 0)
            {
                // Persist this pixel's merged reservoir for the next frame.
                // Empty (W == 0) reservoirs still carry confidence.
                TemporalReservoir cur;
                cur.pos = ray.origin + selDs.t * ray.dir;
                cur.W = W;
                cur.M = confTotal;
                cur.inst = selDs.instance;
                // T hint for next frame's fused-sweep candidate budget
                // (MegaLights temporal feedback).
                cur._pad0 = primaryT;
                cur._pad1 = 0.f;
                gReservoirCurr[pixel.y * gFrameDim.x + pixel.x] = cur;
            }
#endif

            if (selW > 0.f)
            {
                risW = W;
#if USE_COMPACTION
                // Phase A ends here: append the survivor and let the dense
                // shade dispatch do the rest. Plain per-thread atomic append:
                // the wave-aggregated form (prefix count + one atomic per
                // wave) collapsed to one surviving entry per wave on this
                // driver - the measured symptom was a regular grid of lit
                // dots on a black cloud. ~266k appends/frame is noise next
                // to the marching cost; correctness wins.
                risExit = kRisExitContributed;
                uint slot = 0;
                InterlockedAdd(gScatterCount[0], 1u, slot);
                ScatterEntry entry;
                entry.pixel = pixel.x | (pixel.y << 16);
                entry.t = selDs.t;
                entry.inst = selDs.instance;
                entry.W = W;
                gScatterQueue[slot] = entry;
#else
                const GridVolume gv = gScene.getGridVolume(selDs.instance);
                // ds.thp carries the instance albedo, exactly as the reference
                // path uses it.
                throughput *= selDs.thp * W;

                const float3 scatterPos = ray.origin + selDs.t * ray.dir;
                risExit = kRisExitTerminated;
                if (scatterAtVertex(scatterPos, wi, gv.data.anisotropy, 0, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
                {
                    // Continuation bounces run the standard loop below.
                    bounce = 1;
                    traceLoop = true;
                }
                risExit = kRisExitContributed;
#endif
            }
        }
    }

    while (traceLoop)
    {
        VolumeInstanceSampler::DistanceSample ds;
        const bool scattered = volumeSampler.sampleDistance(ray, sg, ds);

        if (!scattered)
        {
            // Ray escaped the medium: pick up the environment.
            // bounce == 0 here means the path NEVER scattered (bounce only
            // increments after a real collision), i.e. this is the primary
            // transmittance event: throughput is still exactly 1 and the
            // pickup is exactly 1 * primaryBg.
            if (bounce == 0)
                primaryT = 1.f;
            radiance += throughput * evalEnvironment(ray.dir);
            break;
        }

        // Real scattering event. ds.thp carries the chosen instance's albedo.
        throughput *= ds.thp;

        const float3 scatterPos = ray.origin + ds.t * ray.dir;
        const float3 wi = -ray.dir; // Points away from the scatter point.

        if (bounce >= kMaxBounces)
            break;

        // Per-instance anisotropy (spec section 3: rim vs core lobe weights).
        const float g = gScene.getGridVolume(ds.instance).data.anisotropy;

        if (!scatterAtVertex(scatterPos, wi, g, bounce, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
            break;

        bounce++;
    }

    // Resolve the deferred single NEE: one shadow ray for the whole path,
    // weighted by the number of vertices it stands in for. Placed here so
    // every way the path can end (RIS termination, loop break, escape) is
    // covered exactly once.
    if (kSingleNeePerPath && kUseNEE && neeReservoir.count > 0)
    {
        const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(neeReservoir.g);
        radiance += neeReservoir.thp * evalNEE(neeReservoir.pos, neeReservoir.wi, phase, volumeSampler, sg) * float(neeReservoir.count);
    }

    gOutputColor[pixel] = float4(radiance, 1.f);

    // Demodulated outputs. Lin is derived by subtraction rather than being
    // accumulated separately in the loop: the primary-escape contribution is
    // exactly primaryT * primaryBg (throughput was still 1), so the
    // decomposition color = Lin + T * bg is EXACT per sample, not a model.
    if (is_valid(gOutputLin))
        gOutputLin[pixel] = float4(radiance - primaryT * primaryBg, 1.f);
    // T replicated into rgb so a stock Multiply composite (T * bg) works
    // channelwise without a custom pass.
    if (is_valid(gOutputT))
        gOutputT[pixel] = float4(primaryT, primaryT, primaryT, 1.f);
    if (is_valid(gOutputBg))
        gOutputBg[pixel] = float4(primaryBg, 1.f);

    if (is_valid(gRisDebug))
        gRisDebug[pixel] = float4(float(risExit), risCount, risWSum, risW);

#if USE_RIS_STATS
    // Wave-reduce before touching memory: a per-pixel atomic on 7 addresses
    // from millions of threads would serialise the whole dispatch. Counting
    // within the wave first makes this ~32x fewer atomics and stays EXACT
    // (no sampling, no estimation).
    for (uint c = 0; c <= kRisExitContributed; ++c)
    {
        const uint n = WaveActiveCountBits(risExit == c);
        if (WaveIsFirstLane() && n > 0)
            InterlockedAdd(gRisStats[c], n);
    }
    {
        const uint nCand = WaveActiveSum(uint(risCount));
        if (WaveIsFirstLane() && nCand > 0)
            InterlockedAdd(gRisStats[7], nCand);
        const uint nMedium = WaveActiveCountBits(risExit != kRisExitNoMedium);
        if (WaveIsFirstLane() && nMedium > 0)
            InterlockedAdd(gRisStats[8], nMedium);
        const uint nM = WaveActiveSum(risMUsed);
        if (WaveIsFirstLane() && nM > 0)
            InterlockedAdd(gRisStats[25], nM);
        const uint nT = WaveActiveCountBits(risTemporalMerged != 0);
        if (WaveIsFirstLane() && nT > 0)
            InterlockedAdd(gRisStats[26], nT);
    }
#endif

#if USE_WORK_STATS
    // Work profile. Same wave-reduce trick. These counts are what the GPU time
    // is actually spent on, so dividing the measured pass ms by them tells you
    // which operation to attack - instead of guessing at the hot spot.
    {
        // Everything after the last mark is shading/marching: reference-path
        // distance sampling, NEE shadow rays, continuation bounces.
        const uint cellsShade = gvsCellsNow() - cellsMark;
        const uint tapsShade = gvsTapsNow() - tapsMark;

        const uint a = WaveActiveSum(volumeSampler.statAabbTests);
        const uint b = WaveActiveSum(volumeSampler.statSegments);
        const uint c = WaveActiveSum(volumeSampler.statCoarseWalks);
        const uint d = WaveActiveSum(volumeSampler.statCoarseCells);
        const uint e = WaveActiveSum(volumeSampler.statCandCells);
        const uint f = WaveActiveSum(volumeSampler.statRealTaps);
        const uint g = WaveActiveSum(volumeSampler.statRays);
        const uint h0 = WaveActiveSum(cellsT);
        const uint h1 = WaveActiveSum(tapsT);
        const uint h2 = WaveActiveSum(cellsCand);
        const uint h3 = WaveActiveSum(tapsCand);
        const uint h4 = WaveActiveSum(cellsShade);
        const uint h5 = WaveActiveSum(tapsShade);
        const uint h6 = WaveActiveSum(volumeSampler.statOverlapSteps);
        const uint h7 = WaveActiveSum(volumeSampler.statOverlapLookups);
        const uint h8 = WaveActiveSum(volumeSampler.statTailRays);
        // Wave-reduce like every counter above. Reading these INSIDE the
        // WaveIsFirstLane branch counted lane 0 only and undercounted by the wave
        // width - that is the entire "brick cache only covers 3% of taps" result;
        // it was a counting bug, not a coverage problem. Wave ops must be issued
        // with all lanes active, so they belong here, not below.
        const uint h9 = WaveActiveSum(gvsBrickHitsNow());
        const uint h10 = WaveActiveSum(gvsBrickMissesNow());
        if (WaveIsFirstLane())
        {
            if (a > 0) InterlockedAdd(gRisStats[9], a);
            if (b > 0) InterlockedAdd(gRisStats[10], b);
            if (c > 0) InterlockedAdd(gRisStats[11], c);
            if (d > 0) InterlockedAdd(gRisStats[12], d);
            if (e > 0) InterlockedAdd(gRisStats[13], e);
            if (f > 0) InterlockedAdd(gRisStats[14], f);
            if (g > 0) InterlockedAdd(gRisStats[15], g);
            if (h0 > 0) InterlockedAdd(gRisStats[16], h0);
            if (h1 > 0) InterlockedAdd(gRisStats[17], h1);
            if (h2 > 0) InterlockedAdd(gRisStats[18], h2);
            if (h3 > 0) InterlockedAdd(gRisStats[19], h3);
            if (h4 > 0) InterlockedAdd(gRisStats[20], h4);
            if (h5 > 0) InterlockedAdd(gRisStats[21], h5);
            if (h6 > 0) InterlockedAdd(gRisStats[22], h6);
            if (h7 > 0) InterlockedAdd(gRisStats[23], h7);
            if (h8 > 0) InterlockedAdd(gRisStats[24], h8);
            if (h9 > 0) InterlockedAdd(gRisStats[27], h9);
            if (h10 > 0) InterlockedAdd(gRisStats[28], h10);
        }
    }
#endif

    if (is_valid(gWorkDebug))
    {
        gWorkDebug[pixel] = float4(
            float(volumeSampler.statSegments),
            float(volumeSampler.statOverlapSteps),
            float(volumeSampler.statAabbTests),
            float(volumeSampler.statMaxCover)
        );
    }
}

#if USE_COMPACTION
/** Phase B: dense shading of queued paths. One thread per REAL path - waves
    are fully occupied doing shading/NEE/bounce marching instead of carrying
    ~87% no-scatter lanes (the measured noCand fraction).

    RNG: a fresh stream, decorrelated from phase A by a large seed offset.
    Phase A's draws already decided everything up to and including the
    survivor; phase B only needs INDEPENDENT draws for NEE/phase/bounces.
    Different random numbers, identical distributions => converged-identical
    (matrix config 15 is the gate).
*/
[numthreads(64, 1, 1)]
void shadeMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint queueIdx = dispatchThreadId.x;
    if (queueIdx >= gScatterCount[0])
        return;

    const ScatterEntry entry = gScatterQueue[queueIdx];
    const uint2 pixel = uint2(entry.pixel & 0xffffu, entry.pixel >> 16);

    // Seed by PIXEL, not queue index: queue order is nondeterministic (wave
    // append order), and per-pixel seeding keeps the whole frame reproducible
    // regardless of it.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount + 0x9e3779b9u);
    VolumeInstanceSampler volumeSampler = {};

    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);
    ray.tMax = FLT_MAX;
    const float3 wi = -ray.dir;

    float3 radiance = float3(0.f);
    const GridVolume gv = gScene.getGridVolume(entry.inst);
    // Same throughput rule as the fused path: instance albedo x reservoir W.
    float3 throughput = gv.data.albedo * entry.W;

    NeeVertexReservoir neeReservoir = {};
    uint bounce = 0;
    bool traceLoop = false;

    const float3 scatterPos = ray.origin + entry.t * ray.dir;
    if (scatterAtVertex(scatterPos, wi, gv.data.anisotropy, 0, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
    {
        bounce = 1;
        traceLoop = true;
    }

    // Continuation bounces: byte-identical logic to the fused loop, minus the
    // bounce==0 primary-transmittance case, which cannot occur here (phase A
    // owns the primary escape term).
    while (traceLoop)
    {
        VolumeInstanceSampler::DistanceSample ds;
        const bool scattered = volumeSampler.sampleDistance(ray, sg, ds);

        if (!scattered)
        {
            radiance += throughput * evalEnvironment(ray.dir);
            break;
        }

        throughput *= ds.thp;

        const float3 pos = ray.origin + ds.t * ray.dir;
        const float3 wiB = -ray.dir;

        if (bounce >= kMaxBounces)
            break;

        const float g = gScene.getGridVolume(ds.instance).data.anisotropy;
        if (!scatterAtVertex(pos, wiB, g, bounce, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
            break;

        bounce++;
    }

    // Deferred single NEE for the whole path, exactly as the fused kernel.
    if (kSingleNeePerPath && kUseNEE && neeReservoir.count > 0)
    {
        const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(neeReservoir.g);
        radiance += neeReservoir.thp * evalNEE(neeReservoir.pos, neeReservoir.wi, phase, volumeSampler, sg) * float(neeReservoir.count);
    }

    // Phase A wrote the background/escape terms; this adds the scatter terms.
    // One queue entry per pixel => single writer, plain read-modify-write.
    gOutputColor[pixel] += float4(radiance, 0.f);
    if (is_valid(gOutputLin))
        gOutputLin[pixel] += float4(radiance, 0.f);

#if USE_WORK_STATS
    // Shade-phase work lands in the same [WORK]/[COST] slots as the fused
    // kernel's shading bucket, so the log stays comparable across modes.
    {
        const uint a = WaveActiveSum(volumeSampler.statAabbTests);
        const uint b = WaveActiveSum(volumeSampler.statSegments);
        const uint g2 = WaveActiveSum(volumeSampler.statRays);
        const uint h4 = WaveActiveSum(gvsCellsNow());
        const uint h5 = WaveActiveSum(gvsTapsNow());
        const uint h6 = WaveActiveSum(volumeSampler.statOverlapSteps);
        const uint h7 = WaveActiveSum(volumeSampler.statOverlapLookups);
        const uint h8 = WaveActiveSum(volumeSampler.statTailRays);
        // Wave-reduce like every counter above. Reading these INSIDE the
        // WaveIsFirstLane branch counted lane 0 only and undercounted by the wave
        // width - that is the entire "brick cache only covers 3% of taps" result;
        // it was a counting bug, not a coverage problem. Wave ops must be issued
        // with all lanes active, so they belong here, not below.
        const uint h9 = WaveActiveSum(gvsBrickHitsNow());
        const uint h10 = WaveActiveSum(gvsBrickMissesNow());
        if (WaveIsFirstLane())
        {
            if (a > 0) InterlockedAdd(gRisStats[9], a);
            if (b > 0) InterlockedAdd(gRisStats[10], b);
            if (g2 > 0) InterlockedAdd(gRisStats[15], g2);
            if (h4 > 0) InterlockedAdd(gRisStats[20], h4);
            if (h5 > 0) InterlockedAdd(gRisStats[21], h5);
            if (h6 > 0) InterlockedAdd(gRisStats[22], h6);
            if (h7 > 0) InterlockedAdd(gRisStats[23], h7);
            if (h8 > 0) InterlockedAdd(gRisStats[24], h8);
            if (h9 > 0) InterlockedAdd(gRisStats[27], h9);
            if (h10 > 0) InterlockedAdd(gRisStats[28], h10);
        }
    }
#endif
}

/** Queue count -> indirect dispatch args for shadeMain (groups of 64). */
RWByteAddressBuffer gDispatchArgs;

[numthreads(1, 1, 1)]
void argsMain()
{
    const uint n = gScatterCount[0];
    gDispatchArgs.Store3(0, uint3((n + 63u) / 64u, 1u, 1u));
}
#endif // USE_COMPACTION
