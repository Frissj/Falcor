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
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.HashUtils;
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
    uint gOccSkipEnable;    ///< 6.3 occupancy tap skip on/off. Pushed into the sampler per thread.
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
uint gvsHomogBinNow(uint k)
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsHomogBins[k];
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

/** Taps answered from the per-brick occupancy mask without reading atlasTex
    (HANDOFF_6 6.3). This is the payoff metric for the mask: skips/taps is the
    fraction of the frame's most-stalled load that is no longer issued. If it is
    near zero the bricks are internally dense and the mask is dead weight.
*/
uint gvsBrickPromotesNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatBrickPromotes;
#else
    return 0;
#endif
}
uint gvsTapSkipsNow()
{
#if GRID_VOLUME_SAMPLER_STATS
    return gGvsStatTapSkips;
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
    [27] = brick-cache hits          [28] = brick-cache misses
    [29] = brick-TLAS procedural candidates (real brick AABBs tested by the
           RayQuery). THIS is the candidate count, not [9]: [9] is
           `+= getGridVolumeCount()` per gatherIntervals call and is a constant
           multiple of the gather count, blind to the brick TLAS and to LoD mip.
           Measured 2026-07-20: forcing 6 of 9 instances to mip 1-2 left [9]
           pinned at 14.6/px, which is why [27] had to exist before any claim
           about coarser bricks (HANDOFF_6 6.1) can be made.
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

/** Sun-tau cache: the Nubis3 lighting trick (SIGGRAPH 2023, slides 127-130),
    also UE's DIM_USE_TRANSMITTANCE_VOLUME path (HeterogeneousVolumes
    RayMarchingUtils.ush ComputeTransmittance: cache lookup FIRST, ray march
    LAST). Replaces the per-candidate coarseOpticalDepth shadow walk - measured
    at 39.5M DDA cells/frame at 0.342 SIMD occupancy, 115.5M warp-slots, 3.6x
    the cost of the entire RayQuery traversal ([LOOPOCC] 2026-07-20) - with ONE
    trilinear fetch of precomputed optical depth toward the dominant light.

    UNBIASED BY CONSTRUCTION: tau only feeds the RIS TARGET Lhat, and the
    estimator divides Lhat back out (w = phat/p = Lhat, see the algebra note in
    VolumeInstanceSampler.slang). Any approximation here - coarse grid, single
    direction, stale by a few frames - moves candidate-selection VARIANCE,
    never the converged image. Nubis and UE both defend image quality with
    their caches; ours only has to pick good candidates.

    The cache is built along ONE direction (the luminance-weighted mean of 64
    env-map importance samples) while the actual per-pixel target light varies.
    Same argument: direction mismatch = variance only. Nubis got a 40% frame
    reduction AND better shadows from the equivalent trade.
*/
#ifndef USE_TAU_CACHE
#define USE_TAU_CACHE 0
#endif

cbuffer TauCB
{
    float3 gTauOrigin;    ///< World-space min corner of the cache grid.
    uint   gTauEnable;    ///< Runtime A/B toggle (CB, not define: same-session A/B per measurement discipline).
    float3 gTauCellSize;  ///< World size of one cache voxel.
    float  _tauPad0;
    float3 gTauInvExtent; ///< 1 / (gTauDim * gTauCellSize): world -> uvw.
    float  _tauPad1;
    uint3  gTauDim;       ///< Cache resolution.
    uint   _tauPad2;
}
Texture3D<float> gTauTex;               ///< Baked tau (optical depth to volume exit toward gTauDirBuf[0]).
SamplerState gTauSmp;                   ///< Linear clamp.
RWTexture3D<float> gTauCacheUav;        ///< tauBuildMain output (same texture, UAV view).
RWStructuredBuffer<float4> gTauDirBuf;  ///< [0].xyz = dominant light direction, written by tauDirMain.

/** In-scattered radiance cache + control-variate residual (volumetric final
    gather, the ReSTIR-FG / Nubis lesson in house style).

    WHY: the deep-bounce tail is the last big cost (avg path 8 bounces, tail to
    64) and every scheduling attack on it is measured dead. Roulette past
    q=0.125 FAILS the matrix gate because the tail carries real energy - so
    the tail's MEAN moves into a cache C, and the path only estimates the
    RESIDUAL (L - C):

        L = C + E[S - C]  for any C     (S = the path's suffix from the cut)

    At the cut vertex: radiance += thp*C; survive w.p. p; on survival
    thp /= p and radiance -= thp*C, then the vertex proceeds normally so its
    continued suffix S accrues with the boosted throughput. Expectation is
    thp*C + (thp/p)*E[S - C]*p = thp*E[S]: UNBIASED FOR ANY CACHE CONTENT,
    stale, coarse, or empty. Quality of C moves only the residual's variance -
    which is exactly the knob the gate prices. The aggressive kill rates that
    failed on raw paths are safe on residuals because the residual's mean is
    ~0 when C is good.

    POPULATION (NRC-style split): 1-in-gRadTrainEvery pixels are TRAINING
    paths - they skip consumption (so the cache never learns from itself) and
    at death deposit their measured suffix S/thp_cut at the cut vertex.
    Deposits accumulate in fixed-point atomics; radResolveMain folds them into
    the resolved texture with an EMA so the cache converges over frames and
    tracks slow lighting change. Confidence rides in .a; consumption only
    fires where confidence > 0.5, so an empty cache degrades to the plain
    estimator, not to garbage.
*/
#ifndef USE_RAD_CACHE
#define USE_RAD_CACHE 0
#endif

cbuffer RadCB
{
    float3 gRadOrigin;          ///< World min corner of the cache grid.
    uint   gRadCutBounce;       ///< Bounce index where the CV applies (0 = off).
    float3 gRadCellSize;        ///< World size of one cell.
    float  gRadResidualSurvival;///< p: residual survival probability at the cut.
    float3 gRadInvExtent;       ///< world -> uvw.
    uint   gRadTrainEvery;      ///< 1-in-N pixels are training paths.
    uint3  gRadDim;             ///< Grid resolution.
    float  gRadEma;             ///< Resolve blend factor per frame.
    /// Lever-1 warp-aware residual roulette: when fewer than this many lanes
    /// of a shadeMain wave are still alive, RESIDUAL paths (post-cut, cache
    /// consumed) roulette per bounce with survival p = alive/lanes and 1/p
    /// compensation. The warp critical path (0.229 bounce occupancy = warps
    /// idling behind their longest path) shortens exactly where the surviving
    /// work matters least: a residual path's expected remaining contribution
    /// is mean-zero-priced by the CV. Unbiased for any p measurable at coin
    /// time (same argument as adaptive M); p depending on OTHER lanes is
    /// fine - it is independent of this path's future randomness.
    /// 0 = off. ESTIMATOR CHANGE: gate via the matrix sweep before adopting.
    uint   gRadWarpRRLanes;

    /// Unconditional post-cut residual roulette: per-bounce KILL probability
    /// applied to residual paths (0 = off). The residual is mean-zero (S - C),
    /// so this is unbiased and caps the residual stragglers [PATHLEN] found to
    /// set the shade critical path - the deep in-scatter Nubis simply omits.
    /// Not warp-gated (unlike gRadWarpRRLanes). ESTIMATOR CHANGE: gate first.
    float  gRadResidualTailQ;
}
RWStructuredBuffer<uint> gRadAccum;   ///< 4 uints/cell: fixed-point RGB sums + sample count.
Texture3D<float4> gRadTex;            ///< Resolved cache (rgb = L_in estimate, a = confidence).
RWTexture3D<float4> gRadResolveUav;   ///< radResolveMain's view of the same texture.
SamplerState gRadSmp;                 ///< Linear clamp.

static const float kRadFxScale = 1024.f; ///< Fixed-point scale for the atomic sums.

/** Deposit one suffix sample at a world position (training paths, at death). */
void radSplat(const float3 pos, const float3 L)
{
    const float3 q = (pos - gRadOrigin) / gRadCellSize;
    if (any(q < 0.f) || any(q >= float3(gRadDim)))
        return;
    const uint3 c = uint3(q);
    const uint idx = ((c.z * gRadDim.y + c.y) * gRadDim.x + c.x) * 4u;
    // Clamp keeps one firefly from saturating a cell's 32-bit sum; the cache
    // is a smooth control, outliers belong to the residual anyway.
    const float3 Lc = clamp(L, 0.f, 4096.f);
    InterlockedAdd(gRadAccum[idx + 0], uint(Lc.r * kRadFxScale));
    InterlockedAdd(gRadAccum[idx + 1], uint(Lc.g * kRadFxScale));
    InterlockedAdd(gRadAccum[idx + 2], uint(Lc.b * kRadFxScale));
    InterlockedAdd(gRadAccum[idx + 3], 1u);
}

/** Training-path selector. Hash, not stripes: neighbours in a wave must not
    all be training (deposit atomics would hotspot) nor all consuming.
*/
bool radIsTraining(const uint2 pixel)
{
    if (gRadTrainEvery <= 1u)
        return true;
    const uint h = (pixel.x * 9781u) ^ (pixel.y * 6271u) ^ (gFrameCount * 26699u);
    return (h % gRadTrainEvery) == 0u;
}

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
    // Cached tau (one trilinear fetch) vs live DDA walk: see the TauCB note.
    // Inside USE_TAU_CACHE both paths stay compiled and gTauEnable picks at
    // runtime, so the A/B is same-session; the define only exists so a
    // no-envmap scene never references an unbindable texture.
    float tauShadow;
#if USE_TAU_CACHE
    if (gTauEnable != 0)
        tauShadow = gTauTex.SampleLevel(gTauSmp, (posJ - gTauOrigin) * gTauInvExtent, 0.f);
    else
        tauShadow = volumeSampler.coarseOpticalDepth(Ray(posJ, targetLight.dir, 0.f, FLT_MAX));
#else
    tauShadow = volumeSampler.coarseOpticalDepth(Ray(posJ, targetLight.dir, 0.f, FLT_MAX));
#endif
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

// CTA SHAPE IS A PERF KNOB, NOT A LAYOUT CHOICE. This kernel has no
// groupshared, no SV_GroupThreadID, no barriers, and its only wave ops are
// wave-scoped (32-lane) stat reductions, so any multiple of 32 is correct and
// produces an identical image. The shape is tuned, and here is what it buys:
//
// The driver allocates ~92 B/thread of shared memory for the RayQuery inline-RT
// traversal stack. Nothing declares it - there is no groupshared anywhere in
// this pass or its includes, and the compiled DXIL has zero addrspace(3)
// (verified by disassembling the shader cache, 2026-07-20). It is a fixed
// per-thread cost of using RayQuery at all, so per-CTA smem = threads * 92 B:
//     16x16 = 256 thr -> 23552 B/CTA   (what Nsight reported)
//      8x8  =  64 thr ->  5888 B/CTA
// Proof it is the traversal stack and not the indexable temp arrays: shadeMain
// has ZERO allocas yet still gets 76 B/thread; main has 69 allocas and gets 92.
// smem tracks threads and RayQuery, not array count.
//
// That mattered because analysis.yaml (multi-pass capture) shows BOTH launch
// stalls live on this kernel: register allocation 22.78% and shared-memory
// allocation 11.21%. A smaller CTA cuts smem/CTA directly, and also lets the
// hardware hit finer register-count targets instead of the hard 2-CTA wall the
// 128-register cap imposes at 256 threads.
//
// MEASURED (analysis2.yaml, 8x8 vs analysis.yaml, 16x16):
//   smem launch stall   11.21% -> 0.43%   (10,607,172 -> 383,346 cycles)
//   register stall      22.78% -> 28.05%  (now the SOLE limiter)
//   CS warps/cycle      30.20% -> 31.22%  (unchanged, and correctly so)
//
// Occupancy did NOT move and could not have. While registers bind, total
// resident warps = 65536 / (regs_per_thread * 32) and CTA SHAPE CANCELS:
// 16x16 gives 2 CTAs * 8 warps, 8x8 gives 8 CTAs * 2 warps, both = 16 warps.
// The reshape removes the shared-memory co-constraint; it does not and cannot
// raise occupancy on its own.
//
// WHAT IT DOES BUY: a much finer register step function. At 256 threads every
// value from 128 down to 81 gave the same 16 warps (which is why four separate
// register-cutting attempts all returned <=3%). At 64 threads:
//     128 regs -> 16 warps      96 regs -> 20 warps
//     112 regs -> 18 warps      80 regs -> 24 warps
// So incremental register cuts now pay incrementally. Do not carry over the old
// "nothing between 128 and 81 changes anything" rule - it was a property of the
// 256-thread CTA, not of the kernel.
//
// Frame-time effect is UNMEASURED. The cross-capture cycle delta is not
// trustworthy (the two ranges swept different amounts of raster work: EarlyZ
// samples 7.50M vs 3.46M) and cumulative CS warp latency moved only -2.75%.
// Same-session A/B is the only trustworthy comparison on this machine.
//
// REVERT TO 16x16 IF IT REGRESSES. Smaller CTAs mean more of them and more
// scheduling overhead. Worth trying 8x4 = 32 (one wave per CTA) as well.
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gvsSetOccSkipEnable(gOccSkipEnable);
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
        // Escape call site: RR here is governed by mode bit3, not bit0 - the
        // escape value couples into adaptive M and the RIS stream, so the
        // bias investigation gates it separately from the NEE walk.
        visSetTrRRSite(8u);
#if USE_WORK_STATS
        // [TRRPROBE2] Fresh coin telemetry for exactly this walk; later walks
        // in the frame may also fire coins but are never flushed.
        visResetTrRRCoinStats();
#endif
        const float Tfull = volumeSampler.evalTransmittance(ray, sg);
        visSetTrRRSite(1u);
#if USE_WORK_STATS
        // [TRRPROBE2] Snapshot the armed walk's telemetry IMMEDIATELY, before
        // any other code can touch the statics. Run 127 logged surv > coins -
        // impossible per the increment logic - so the flush now uses these
        // locals, and the check slots below test the invariants directly
        // instead of trusting the pipeline.
        const uint probeCoins = visTrRRCoinCount();
        const uint probeSurv = visTrRRCoinSurvives();
        const float probeBefore = visTrRRCoinSumBefore();
        const uint probeNegFull = visTrRRNegTr();
#endif

#if USE_WORK_STATS
        // [TRRPROBE] Direct walk-level bias measurement (slots 46..53). The
        // sweep bisection pinned the -0.73% on this call site but could not
        // separate "the walk's E[T] moves under RR" from "a downstream
        // consumer couples nonlinearly" - and simulation cannot resolve it
        // (the product estimator is too heavy-tailed for sample means).
        // So: run the SAME walk again with RR hard-disabled (site mask 0) on
        // a decorrelated RNG, and accumulate both estimates. The ratio of the
        // two frame sums is E-bias of the walk itself, paired per pixel,
        // with no RIS, adaptive M, or image pipeline in the loop.
        if (gTrRRThreshold > 0.f && (gTrRRMode & 8u) != 0u)
        {
            // PROBE V3 (kept): the reference walk must use the SAME generator
            // family as the real walk. V2 used TinyUniform (LCG) and bin3 -
            // where T > 0.5 and RR cannot fire, so any difference is NOT RR -
            // read a consistent -0.44%. Same family, decorrelated seed
            // constant.
            SampleGenerator psg = SampleGenerator(pixel, gFrameCount + 0x51ed270bu);
            // Canary: the statics were snapshotted above, so reset them and
            // let the reference walk run - it uses site mask 0, so its coin
            // count after the walk must be IDENTICALLY zero. A nonzero slot 67
            // means the mask gating is broken and every probe reading with it.
            visResetTrRRCoinStats();
            visSetTrRRSite(0u); // mask 0: RR cannot fire regardless of mode.
            const float Tref = volumeSampler.evalTransmittance(ray, psg);
            visSetTrRRSite(1u);
            const uint refCoins = visTrRRCoinCount();
            const uint refNeg = visTrRRNegTr();
            // PROBE V4: bin by a DETERMINISTIC key, never by either walk's own
            // outcome. V2/V3 binned by Tref, which is a random variable: a
            // straddler pixel enters the thick bin exactly on the realizations
            // where its ref walk read HIGH, so E[Tref | bin] is inflated by
            // selection while the RR side is unconditioned - every v2/v3 bin
            // reading (including bin3's "impossible" -0.38%) is invalid by
            // construction. The key here is exp(-coarseOpticalDepth(view ray)):
            // analytic mean-field DDA, zero RNG, identical for both walks and
            // across frames, so bin membership is a fixed pixel property and
            // both estimators are unconditioned within every bin.
            // NOTE Tdet is only a proxy for E[T] (Jensen: true E[T] >=
            // exp(-mean tau) in heterogeneous media) - fine, the key only
            // needs to be deterministic and thickness-monotone, not exact.
            // COST: one extra coarse DDA walk in probe-armed rows only; it
            // also bumps gvs cell counters, so [LOOPOCC]/cost buckets in
            // probe-armed rows are not representative (gpuMs already wasn't).
            const float Tdet = exp(-volumeSampler.coarseOpticalDepth(ray));
            // Bins by Tdet: [0,0.01) [0.01,0.1) [0.1,0.5) [0.5,inf).
            // Fixed-point scales are 16x more conservative than v3's: the
            // walked T inside a bin is no longer range-bounded by the bin
            // edges (straddlers + Jensen), so each scale is sized so overflow
            // would need an absurd average T for that bin's key range
            // (e.g. bin0 wraps only if 3.5M px keyed Tdet<0.01 average
            // T>0.3). Ratios are computed within a bin, so scale cancels.
            // fx are MULTIPLES OF 20 (not powers of two) so threshold-valued
            // outputs land on exact integers: the RR walk's survivors carry
            // atoms at exactly theta = 0.05, and uint(0.05*2048 + 0.5) = 102
            // vs true 102.4 would inject a systematic -0.4% on the RR side
            // under the per-lane flush below. theta*fx = 200/100/25/13 exact.
            uint bin;
            float fx;
            if (Tdet < 0.01f) { bin = 0; fx = 4000.f; }
            else if (Tdet < 0.1f) { bin = 1; fx = 2000.f; }
            else if (Tdet < 0.5f) { bin = 2; fx = 500.f; }
            else { bin = 3; fx = 260.f; }
            // [TRRPROBE2] slots 54..65: coin telemetry from the armed walk,
            // same det-key bins. sumBefore is Tr/threshold per coin (each term
            // in (0,1)) at x64 fixed point: worst case ~13 coins/walk (one per
            // brick candidate) x 3.5M px x 64 stays far under 2^32. The
            // E-preservation check "threshold*survives == sum(TrBefore)" is
            // scale-free on the CPU side (survives vs sumBefore/64).
            // Per-lane atomics for the walk sums too - the probe now contains
            // ZERO wave ops. Run 131's mass accounting (coins touch <=125
            // T-units of bin1 while 4076 are missing) proves the deficit is
            // not the RR block, so before chasing deeper the last instance of
            // the wave pattern that provably miscompiled in runs 127/128 is
            // removed from the evidence chain. Per-lane rounding at these fx
            // is ~0.001% on multi-million-unit bin sums (error ~ sqrt(N)*0.3
            // units) and treats both sides identically; theta-atoms are exact
            // by the fx choice above.
            InterlockedAdd(gRisStats[46 + 2 * bin], uint(Tfull * fx + 0.5f));
            InterlockedAdd(gRisStats[47 + 2 * bin], uint(Tref * fx + 0.5f));

            // [TRRPROBE2] slots 54..65 + [TRRPROBE2-CHK] 66..70, flushed with
            // PER-LANE atomics, deliberately NO wave ops. Run 128 printed
            // per-bin surv > coins while the per-lane invariant check read 0
            // violations - set-theoretically impossible if the wave-summed
            // flush were faithful, so the WaveActiveSum/WaveIsFirstLane
            // combination inside the unrolled loop is itself under suspicion
            // (reconvergence miscompile shape). Coins are sparse (~140k lanes
            // at 1080p), so per-lane atomics cost nothing measurable here and
            // have no wave semantics to miscompile. If per-lane readings
            // restore surv <= coins, the wave path was guilty; sRR/sRef above
            // stay wave-summed (dense, every lane) and get cross-checked by
            // the image-level gate instead.
            //   66 = lanes violating surv <= coins (must be 0),
            //   67 = reference-walk coins (must be 0; site-mask canary),
            //   68 = lanes that fired at least one coin (RR exposure),
            //   69/70 = armed/reference walks whose Tr went negative (a
            //        residual factor outside [0,2]; such walks die WITHOUT RR
            //        compensation).
            if (probeCoins > 0u)
            {
                InterlockedAdd(gRisStats[54 + 3 * bin], probeCoins);
                InterlockedAdd(gRisStats[55 + 3 * bin], uint(probeBefore * 64.f + 0.5f));
                InterlockedAdd(gRisStats[56 + 3 * bin], probeSurv);
                InterlockedAdd(gRisStats[68], 1u);
            }
            if (probeSurv > probeCoins) InterlockedAdd(gRisStats[66], 1u);
            if (refCoins > 0u) InterlockedAdd(gRisStats[67], refCoins);
            if (probeNegFull > 0u) InterlockedAdd(gRisStats[69], probeNegFull);
            if (refNeg > 0u) InterlockedAdd(gRisStats[70], refNeg);
        }
#endif
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

            // LIVE-STATE DIGEST (register cut, see the numthreads note on main).
            // primaryList is 27 values (9 ids + 9 t0 + 9 t1). Its two LATE
            // consumers - the tRep fallback and the temporal-shift domain guard -
            // sit past the RayQuery traversals, which forced all 27 to stay live
            // across the hottest stretch of the kernel. The fallback only ever
            // wanted min(t0)/max(t1), so collapse it to two floats here and let
            // the late shift path re-gather (deterministic: gatherIntervals reads
            // only the Ray, no RNG, so this is image-identical).
            //
            // This is NOT the trade that regressed 25% in coarseOpticalDepth.
            // That one recomputed 81 floats per (cell, instance) INSIDE the
            // marching loop, moving work into an SM-bound inner loop. This adds
            // 9 slab tests on a cold branch and one 9-iteration min/max per
            // pixel, to shorten a 27-value live range spanning ~200 lines.
            uint primaryCount = primaryList.count;
            float primaryEnter = FLT_MAX;
            float primaryExit = 0.f;
            for (uint pi = 0; pi < primaryList.count; ++pi)
            {
                primaryEnter = min(primaryEnter, primaryList.t0[pi]);
                primaryExit = max(primaryExit, primaryList.t1[pi]);
            }

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
                if (selW <= 0.f && primaryCount > 0)
                {
                    // Digest computed at gather time - identical value, but it
                    // does not pin the 27-value interval list live to here.
                    tRep = 0.5f * (primaryEnter + primaryExit);
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
                                    // Re-gather rather than hold primaryList live
                                    // across the traversals above. Same ray, no
                                    // RNG, so this list is bit-identical to the
                                    // primary one; it costs 9 slab tests on a
                                    // branch that is already three conditions deep.
                                    const VolumeInstanceSampler::IntervalList shiftList = volumeSampler.gatherIntervals(ray);
                                    const uint mask = volumeSampler.coveringMask(shiftList, tNew);
                                    float meanSigma = 0.f;
                                    for (uint i = 0; i < shiftList.count; ++i)
                                    {
                                        if ((mask & (1u << i)) == 0) continue;
                                        const GridVolume gvI = gScene.getGridVolume(shiftList.ids[i]);
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
        const uint a2 = WaveActiveSum(volumeSampler.statBrickCandidates);
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
        const uint h11 = WaveActiveSum(gvsTapSkipsNow());
        const uint h10 = WaveActiveSum(gvsBrickMissesNow());
        const uint h12 = WaveActiveSum(gvsBrickPromotesNow());

        // [HOMOG] mean/majorant histogram (slots 77..84). Source is the gvs
        // majorant-DDA counter (the LIVE candGen traversal), NOT the dormant
        // shared-candidate sweep. Explicit per-bin wave-sums, NOT an [unroll]'d
        // loop: WaveActiveSum + WaveIsFirstLane + [unroll] + InterlockedAdd
        // miscompiled here before (HANDOFF_9 6), dropping adds. One line per bin.
        const uint hg0 = WaveActiveSum(gvsHomogBinNow(0));
        const uint hg1 = WaveActiveSum(gvsHomogBinNow(1));
        const uint hg2 = WaveActiveSum(gvsHomogBinNow(2));
        const uint hg3 = WaveActiveSum(gvsHomogBinNow(3));
        const uint hg4 = WaveActiveSum(gvsHomogBinNow(4));
        const uint hg5 = WaveActiveSum(gvsHomogBinNow(5));
        const uint hg6 = WaveActiveSum(gvsHomogBinNow(6));
        const uint hg7 = WaveActiveSum(gvsHomogBinNow(7));

        // ---- DIVERGENCE PROBE (slots 31-35) ----
        // Nsight's largest rule on this pass is Active Threads Per Warp: only
        // 22.8% of lane-slots retire useful work (4.39x, against 1.38x for the
        // register limiter). It cannot say WHY, and the two candidate
        // mechanisms need opposite responses:
        //   (a) MISS/HIT IMBALANCE - sky lanes finish instantly while cloud
        //       lanes march thousands of cells. FIXABLE: enqueue only the
        //       marching pixels and dispatch them packed, reusing the queue
        //       that already feeds shadeMain.
        //   (b) TAIL-LENGTH VARIANCE among cloud lanes - a heterogeneous
        //       medium legitimately terminates rays at different depths.
        //       NOT fixable by compaction; intrinsic to delta tracking.
        // idleLanes is the discriminator: if most of the waste is lanes doing
        // literally zero marching it is (a); if every lane is busy but for
        // wildly different durations it is (b).
        //
        // waveMax is the warp's critical path - a warp holds its slots until
        // its SLOWEST lane finishes - so sum(work) / (lanes * sum(waveMax)) is
        // the fraction of lane-slots doing useful work, in OUR units and
        // directly comparable to Nsight's 22.8%. Two independent measurements
        // agreeing is what makes this trustworthy; if they disagree the probe
        // is wrong, not Nsight.
        //
        // Wave ops belong HERE with all lanes active, never inside the
        // WaveIsFirstLane branch (see the counting bug noted above).
        // Work proxy must be non-zero on the SHIPPING path. cells alone is not:
        // GridVolumeSampler.slang:71 says gGvsStatCells counts majorant-DDA
        // steps only, and "the non-DDA estimators only tap", so the brick-TLAS
        // RayQuery path this config actually runs would report ~0 cells and the
        // probe would read as 100% idle - a blind probe that looks like a
        // finding. Summing cells + taps + brick candidates covers every backend;
        // the units are mixed, but divergence only needs RELATIVE per-lane
        // duration, not an absolute cost model.
        const uint laneWork = gvsCellsNow() + gvsTapsNow() + volumeSampler.statBrickCandidates;
        const uint dvSum = WaveActiveSum(laneWork);
        const uint dvMax = WaveActiveMax(laneWork);
        const uint dvIdle = WaveActiveCountBits(laneWork == 0);
        const uint dvLanes = WaveActiveCountBits(true);

        // PER-LOOP SIMD occupancy (slots 38-41). laneUtil above measures load
        // BALANCE across the whole kernel and came back 0.76 with 31.2/32 lanes
        // busy - which rules out idle lanes and imbalance, but NOT lanes taking
        // different branches at the same instruction. These two catch that:
        // sum(iters) / (max(iters) * 32) per loop. The coarse DDA is the prime
        // suspect - the comment on it records ~13.6% of all SM samples with
        // BRANCH 31% / SELECT 36% / NOINST 38% stalls.
        const uint rqSum = WaveActiveSum(volumeSampler.statRqIters);
        const uint rqMax = WaveActiveMax(volumeSampler.statRqIters);
        const uint ddaSum = WaveActiveSum(volumeSampler.statDdaIters);
        const uint ddaMax = WaveActiveMax(volumeSampler.statDdaIters);

        if (WaveIsFirstLane())
        {
            if (dvSum > 0) InterlockedAdd(gRisStats[31], dvSum);
            if (dvMax > 0) InterlockedAdd(gRisStats[32], dvMax);
            InterlockedAdd(gRisStats[33], 1);
            if (dvIdle > 0) InterlockedAdd(gRisStats[34], dvIdle);
            InterlockedAdd(gRisStats[35], dvLanes);

            // Slots 36/37 settle the clustering question by MEASUREMENT rather
            // than inference. The first run reported idleFrac 0.783 with
            // laneUtil 0.760, which are only mutually consistent if the idle
            // lanes sit in WHOLLY idle warps (evenly-spread idle lanes would
            // cap laneUtil at ~0.22). That was arithmetic, not data. A warp
            // with dvMax == 0 marched nothing at all, so slot 36 counts fully
            // idle warps directly, and slot 37 counts busy lanes only in warps
            // that DID march - giving the true average occupancy of a marching
            // warp instead of one derived from an inconsistency.
            if (dvMax == 0)
                InterlockedAdd(gRisStats[36], 1);
            else
                InterlockedAdd(gRisStats[37], dvLanes - dvIdle);

            if (rqSum > 0) InterlockedAdd(gRisStats[38], rqSum);
            if (rqMax > 0) InterlockedAdd(gRisStats[39], rqMax);
            if (ddaSum > 0) InterlockedAdd(gRisStats[40], ddaSum);
            if (ddaMax > 0) InterlockedAdd(gRisStats[41], ddaMax);

            if (hg0 > 0) InterlockedAdd(gRisStats[77], hg0);
            if (hg1 > 0) InterlockedAdd(gRisStats[78], hg1);
            if (hg2 > 0) InterlockedAdd(gRisStats[79], hg2);
            if (hg3 > 0) InterlockedAdd(gRisStats[80], hg3);
            if (hg4 > 0) InterlockedAdd(gRisStats[81], hg4);
            if (hg5 > 0) InterlockedAdd(gRisStats[82], hg5);
            if (hg6 > 0) InterlockedAdd(gRisStats[83], hg6);
            if (hg7 > 0) InterlockedAdd(gRisStats[84], hg7);

            if (a > 0) InterlockedAdd(gRisStats[9], a);
            if (a2 > 0) InterlockedAdd(gRisStats[29], a2);
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
            if (h11 > 0) InterlockedAdd(gRisStats[30], h11);
            if (h12 > 0) InterlockedAdd(gRisStats[71], h12);
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
    gvsSetOccSkipEnable(gOccSkipEnable);
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

#if USE_RAD_CACHE
    // Control-variate state (see the RadCB note). Training paths snapshot at
    // the cut and deposit their measured suffix at death; consuming paths take
    // the cache's mean there and continue only to estimate the residual.
    const bool radTraining = gRadCutBounce != 0u && radIsTraining(pixel);
    bool radHasCut = false;
    bool radIsResidual = false; ///< Path consumed the cache at the cut; its remaining bounces estimate the mean-zero residual.
    float radRRComp = 1.f; ///< Cumulative 1/p this path has taken from warp-RR. Capped so a survivor's throughput cannot compound into a firefly (HANDOFF_9 5 fix 2).
    float3 radCutPos = float3(0.f);
    float3 radCutThp = float3(0.f);
    float3 radCutRad = float3(0.f);
    uint radCutNee = 0u;
#endif

    // Continuation bounces: byte-identical logic to the fused loop, minus the
    // bounce==0 primary-transmittance case, which cannot occur here (phase A
    // owns the primary escape term).
    while (traceLoop)
    {
#if USE_RAD_CACHE
        // Lever-1 warp-aware residual roulette (see gRadWarpRRLanes), REDESIGNED
        // per HANDOFF_9 5 after the first version failed the gate at -3.0%. Only
        // RESIDUAL paths are eligible: their remaining contribution is (S - C),
        // mean zero, so killing them costs the least variance in the kernel and
        // the CV already priced the survival algebra. When warp occupancy drops
        // below the knob, the warp is mostly waiting on stragglers and each one
        // delays 32 slots - killing them retires the warp. Three fixes:
        //  (1) COUNT ALL LIVE LANES. aliveWarp is read at the loop HEAD inside a
        //      UNIFORM guard (gRadWarpRRLanes is a cbuffer constant, so every
        //      lane still in the loop enters together) - so it is the true warp
        //      occupancy [SHADEOCC] measured, NOT the residual-only count the old
        //      code got by putting the wave op inside `if (radIsResidual)`. That
        //      bug made a warp with 2 residual paths read aliveWarp=2 -> p=0.25
        //      EVERY bounce, compounding to 1024x fireflies that read as bias.
        //  (2) BOUND THE COMPENSATION. p is floored at 0.5 (each event <=2x) and
        //      radRRComp caps the cumulative 1/p at 4x - past that the path is no
        //      longer rouletted, so its throughput cannot spiral. Both are RR-
        //      unbiased (any p in (0,1]; not-rouletting is exact).
        //  (3) COUNTER-BASED COIN. frame+pixel+bounce+ray hash, sg NOT advanced -
        //      generator-independent, like trRR bit4.
        // Wave op is uniform-guarded and separated from the atomic below, so it
        // is not the divergent-flow-around-atomic shape that miscompiled in
        // runs 127/128.
        if (gRadWarpRRLanes != 0u)
        {
            const uint aliveWarp = WaveActiveCountBits(true);
            if (radIsResidual && radRRComp < 4.f && aliveWarp < gRadWarpRRLanes)
            {
                const float p = max(float(aliveWarp) / float(gRadWarpRRLanes), 0.5f);
                uint hc = (gFrameCount * 0x9e3779b9u) ^ (pixel.x * 9781u) ^ (pixel.y * 6271u);
                hc = jenkinsHash(hc ^ (bounce * 0x85ebca6bu));
                hc = jenkinsHash(hc ^ asuint(ray.dir.x));
                hc = jenkinsHash(hc ^ asuint(ray.dir.y));
                const float coin = (hc >> 8) * 0x1p-24f;
                if (coin >= p)
                {
#if USE_WORK_STATS
                    InterlockedAdd(gRisStats[72], 1u); // Per-lane, wave-free (runs 127/128 lesson).
#endif
                    break;
                }
                throughput /= p;
                radRRComp /= p;
            }
        }

        // Unconditional post-cut residual roulette (Lever-2). The residual is
        // mean-zero (S - C), so per-bounce RR here is unbiased and the cheapest
        // variance in the kernel. Unlike warp-RR this is NOT gated on warp
        // occupancy - it caps EVERY residual straggler, which [PATHLEN] fingered
        // as the shade critical path (mean ~9 bounces, 12% past 16). This is the
        // point where Nubis stops computing the deep in-scatter entirely; we
        // keep it unbiased by compensating survivors. Counter-based coin (frame+
        // pixel+bounce+ray, sg untouched); distinct salt from warp-RR so the two
        // coins are independent if both are ever enabled.
        if (radIsResidual && gRadResidualTailQ > 0.f && radRRComp < 4.f)
        {
            const float p = 1.f - gRadResidualTailQ; // survival probability
            uint hc = (gFrameCount * 0x9e3779b9u) ^ (pixel.x * 9781u) ^ (pixel.y * 6271u);
            hc = jenkinsHash(hc ^ (bounce * 0xc2b2ae35u));
            hc = jenkinsHash(hc ^ asuint(ray.dir.x));
            hc = jenkinsHash(hc ^ asuint(ray.dir.z));
            const float coin = (hc >> 8) * 0x1p-24f;
            if (coin >= p)
                break;
            // Shared cumulative-compensation cap with warp-RR (radRRComp): per-
            // bounce RR compounds (1/p each survival), so bound the survivor's
            // throughput at 4x - past that, stop rouletting rather than mint a
            // firefly. Both are still RR-unbiased.
            throughput /= p;
            radRRComp /= p;
        }
#endif
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

#if USE_RAD_CACHE
        // The cut vertex. Both roles trigger BEFORE the vertex is processed,
        // so the suffix S is defined identically for deposit and consumption:
        // everything from this vertex onward, including the vertex itself.
        if (gRadCutBounce != 0u && bounce == gRadCutBounce)
        {
            if (radTraining)
            {
                radHasCut = true;
                radCutPos = pos;
                radCutThp = throughput;
                radCutRad = radiance;
                radCutNee = neeReservoir.count;
            }
            else
            {
                const float4 C = gRadTex.SampleLevel(gRadSmp, (pos - gRadOrigin) * gRadInvExtent, 0.f);
                if (C.a > 0.5f) // Confidence gate: an empty cache = plain estimator.
                {
                    // L = C + (S - C): bank the cache's mean...
                    radiance += throughput * C.rgb;
                    // ...then keep the path alive only to estimate the residual.
                    // This kill rate on RAW paths failed the matrix gate (it
                    // deleted real energy); on residuals the deleted term has
                    // mean ~0, which is the entire trick.
                    if (sampleNext1D(sg) >= gRadResidualSurvival)
                        break;
                    throughput /= gRadResidualSurvival;
                    radiance -= throughput * C.rgb;
                    radIsResidual = true; // Eligible for the warp-aware roulette above from the next bounce on.
                }
            }
        }
#endif

        const float g = gScene.getGridVolume(ds.instance).data.anisotropy;
        if (!scatterAtVertex(pos, wiB, g, bounce, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
            break;

        bounce++;
    }

    // Deferred single NEE for the whole path, exactly as the fused kernel.
    // Kept separable so the training deposit can attribute the correct SHARE
    // of it to the suffix.
    float3 neeContrib = float3(0.f);
    if (kSingleNeePerPath && kUseNEE && neeReservoir.count > 0)
    {
        const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(neeReservoir.g);
        neeContrib = neeReservoir.thp * evalNEE(neeReservoir.pos, neeReservoir.wi, phase, volumeSampler, sg) * float(neeReservoir.count);
        radiance += neeContrib;
    }

#if USE_RAD_CACHE
    if (radHasCut)
    {
        // Suffix = everything added after the snapshot, plus the share of the
        // deferred NEE belonging to post-cut vertices. The reservoir picks its
        // vertex uniformly, so the post-cut share IS the count ratio - correct
        // in expectation, which is all a training target needs.
        const float fPost = neeReservoir.count > 0u
            ? float(neeReservoir.count - min(radCutNee, neeReservoir.count)) / float(neeReservoir.count)
            : 0.f;
        const float3 suffix = (radiance - radCutRad) - neeContrib * (1.f - fPost);
        const float3 Ls = float3(
            radCutThp.x > 1e-6f ? suffix.x / radCutThp.x : 0.f,
            radCutThp.y > 1e-6f ? suffix.y / radCutThp.y : 0.f,
            radCutThp.z > 1e-6f ? suffix.z / radCutThp.z : 0.f
        );
        radSplat(radCutPos, max(Ls, float3(0.f)));
    }
#endif

    // Phase A wrote the background/escape terms; this adds the scatter terms.
    // One queue entry per pixel => single writer, plain read-modify-write.
    gOutputColor[pixel] += float4(radiance, 0.f);
    if (is_valid(gOutputLin))
        gOutputLin[pixel] += float4(radiance, 0.f);

#if USE_TEMPORAL_REUSE
    // Feed the scatter-sort predictor: this path's MEASURED bounce count,
    // read next frame by scatterClassMain. bounce >= 1 for every shaded path,
    // so 0 stays the "no path here last frame" sentinel. Field-only store
    // into the reservoir phase A wrote earlier this frame (one entry per
    // pixel => single writer; Falcor's UAV barrier orders the dispatches).
    // Run 138 lesson: the T hint (_pad0) is RR-quantized since trRR went
    // live - killed walks store exactly 0, survivors >= ~theta - so it
    // cannot rank work inside the thick mass. Last frame's actual path
    // length is the label itself, one frame stale.
    if (gTemporalEnable != 0)
        gReservoirCurr[pixel.y * gFrameDim.x + pixel.x]._pad1 = float(bounce);
#endif

#if USE_WORK_STATS
    // Shade-phase work lands in the same [WORK]/[COST] slots as the fused
    // kernel's shading bucket, so the log stays comparable across modes.
    {
        // [PATHLEN] path-length by radcache role (slots 85..93). One bin per
        // shaded path, HERE at the loop exit where bounce/radTraining/
        // radIsResidual are final, and past shadeMain's out-of-range return so
        // every lane holds a real queued path. Per-lane atomics (HANDOFF_9 6),
        // no wave op. 3 roles x { count, sum-of-bounces, stragglers(>=16) };
        // mean = sum/count. Tells us whether the long paths that set the warp
        // critical path are raw-uncut (0), cut->residual (1), or training (2).
        uint plCat = 0u; // raw: never cut (terminated early, or cache not confident at the cut)
#if USE_RAD_CACHE
        if (radTraining)        plCat = 2u; // training path: deposits at death, runs raw-length
        else if (radIsResidual) plCat = 1u; // cut at radCutBounce, continued only as residual
#endif
        const uint plBase = 85u + plCat * 3u;
        InterlockedAdd(gRisStats[plBase + 0u], 1u);
        InterlockedAdd(gRisStats[plBase + 1u], bounce);
        if (bounce >= 16u) InterlockedAdd(gRisStats[plBase + 2u], 1u);

        const uint a = WaveActiveSum(volumeSampler.statAabbTests);
        const uint a2 = WaveActiveSum(volumeSampler.statBrickCandidates);
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
        const uint h11 = WaveActiveSum(gvsTapSkipsNow());
        const uint h10 = WaveActiveSum(gvsBrickMissesNow());
        const uint h12 = WaveActiveSum(gvsBrickPromotesNow());

        // ---- shadeMain DIVERGENCE PROBE (slots 42-45) ----
        // Every lane here holds a REAL path (dense queue dispatch), so the
        // idle-warp clustering that made main's divergence benign cannot help:
        // any waste is path-LENGTH variance inside the warp, the thing
        // per-bounce requeueing would fix (the queue infra already exists).
        // Same identity as [LOOPOCC]: for a monotone-exit loop,
        // sum(iters) / (max(iters) * 32) IS instantaneous lane occupancy.
        // bounces = loop trips; work = total marching (cells + taps) so the
        // two disagreeing would show bounce COUNT vs bounce COST divergence.
        const uint sbSum = WaveActiveSum(bounce);
        const uint sbMax = WaveActiveMax(bounce);
        const uint swWork = gvsCellsNow() + gvsTapsNow();
        const uint swSum = WaveActiveSum(swWork);
        const uint swMax = WaveActiveMax(swWork);

        if (WaveIsFirstLane())
        {
            if (sbSum > 0) InterlockedAdd(gRisStats[42], sbSum);
            if (sbMax > 0) InterlockedAdd(gRisStats[43], sbMax);
            if (swSum > 0) InterlockedAdd(gRisStats[44], swSum);
            if (swMax > 0) InterlockedAdd(gRisStats[45], swMax);

            if (a > 0) InterlockedAdd(gRisStats[9], a);
            if (a2 > 0) InterlockedAdd(gRisStats[29], a2);
            if (b > 0) InterlockedAdd(gRisStats[10], b);
            if (g2 > 0) InterlockedAdd(gRisStats[15], g2);
            if (h4 > 0) InterlockedAdd(gRisStats[20], h4);
            if (h5 > 0) InterlockedAdd(gRisStats[21], h5);
            if (h6 > 0) InterlockedAdd(gRisStats[22], h6);
            if (h7 > 0) InterlockedAdd(gRisStats[23], h7);
            if (h8 > 0) InterlockedAdd(gRisStats[24], h8);
            if (h9 > 0) InterlockedAdd(gRisStats[27], h9);
            if (h10 > 0) InterlockedAdd(gRisStats[28], h10);
            if (h11 > 0) InterlockedAdd(gRisStats[30], h11);
            if (h12 > 0) InterlockedAdd(gRisStats[71], h12);
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

#if USE_COMPACTION
// ---------------------------------------------------------------------------
// Lever-1b: work-class counting sort of the scatter queue (MegaLights
// classify -> dense-dispatch pattern, MegaLightsTileClassification.ush).
//
// WHY: shadeMain's queue is wave-append order == rough screen order, so a
// warp mixes paths that die at bounce 1 with paths that diffuse 20+ bounces
// ([SHADEOCC] bounce occupancy 0.229) and every warp waits on its longest
// path. Sorting the queue by a work predictor puts similar path lengths in
// the same warp. The predictor is LAST frame's transmittance at the pixel
// (reservoir _pad0 - the same MegaLights-style hint adaptive M already
// trusts): low T = thick = long diffusion.
//
// IMAGE-IDENTICAL BY CONSTRUCTION: shadeMain seeds its generator by PIXEL
// precisely because queue order was already nondeterministic (see the seed
// comment there), so each path computes bit-identical values from any slot.
// Reordering changes only which threads share a warp. No gate needed - this
// is a pure scheduling change, same class as compaction itself.
//
// Three tiny dispatches between argsMain and shadeMain: classify (count-sized,
// per-entry class + histogram), offsets (1 thread, 4-wide exclusive prefix),
// scatter (count-sized, stable within a class up to atomic order - which is
// exactly as nondeterministic as the append order was).
// ---------------------------------------------------------------------------
RWStructuredBuffer<ScatterEntry> gScatterQueueSorted;
RWStructuredBuffer<uint> gScatterClass;      ///< Class per queue entry.
RWStructuredBuffer<uint> gScatterClassCount; ///< [0..3] histogram; [4..7] scatter cursors (start at class offsets).

[numthreads(64, 1, 1)]
void scatterClassMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= gScatterCount[0])
        return;
    uint c = 0;
#if USE_TEMPORAL_REUSE
    const uint2 pixel = uint2(gScatterQueue[idx].pixel & 0xffffu, gScatterQueue[idx].pixel >> 16);
    // Predictor v3 = LAST frame's MEASURED bounce count for this pixel's
    // path (written by shadeMain into _pad1). The label itself, one frame
    // stale - no proxy modelling. History of dead proxies, kept so nobody
    // resurrects them: v1 screen-scale T bins (run 137: all queued pixels
    // are low-T, everything class 0); v2 queue-scale T bins (run 138: the
    // T hint is RR-QUANTIZED since trRR went live - killed walks store
    // exactly 0, survivors >= ~theta = 0.05, so class 1 read EXACTLY 0 and
    // 89% piled into class 0; a T hint cannot rank work inside the thick
    // mass at all).
    // Edges bracket the measured distribution (avg path 8, warp-max 29):
    // 0 = no path here last frame (fresh pixel - bin with the short class,
    // most such pixels are silhouette flicker), 1..3 short, 4..9 the mass,
    // 10..19 long, 20+ the warp-killers.
    const float bHint = gReservoirPrev[pixel.y * gFrameDim.x + pixel.x]._pad1;
    if (bHint < 4.f) c = 0;
    else if (bHint < 10.f) c = 1;
    else if (bHint < 20.f) c = 2;
    else c = 3;
#endif
    // Temporal off: every entry lands in class 0 and the sort is a stable
    // no-op pass-through - correct, just useless, matching the adaptive-M
    // fallback behaviour for the same missing hint.
    gScatterClass[idx] = c;
    InterlockedAdd(gScatterClassCount[c], 1u);
#if USE_WORK_STATS
    // Class histogram to slots 73..76: a sort is only working if this is
    // spread across classes - run 137's degenerate binning would have been
    // visible in one log line instead of costing a run.
    InterlockedAdd(gRisStats[73 + c], 1u);
#endif
}

[numthreads(1, 1, 1)]
void scatterOffsetMain()
{
    uint off = 0;
    [unroll]
    for (uint c = 0; c < 4; ++c)
    {
        gScatterClassCount[4 + c] = off;
        off += gScatterClassCount[c];
    }
}

[numthreads(64, 1, 1)]
void scatterSortMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint idx = dispatchThreadId.x;
    if (idx >= gScatterCount[0])
        return;
    const uint c = gScatterClass[idx];
    uint dst;
    InterlockedAdd(gScatterClassCount[4 + c], 1u, dst);
    gScatterQueueSorted[dst] = gScatterQueue[idx];
}
#endif // USE_COMPACTION

// ---------------------------------------------------------------------------
// Wavefront backend for phase B: per-bounce requeueing.
//
// WHY: [SHADEOCC] measured the fused bounce loop at 0.271 lane occupancy -
// the average path dies at 8 bounces but the average warp's LONGEST path runs
// 29, and 27 lanes idle behind it. Roulette cannot fix this shape: the matrix
// gate FAILS every floor past 0.125 (+0.33% at 0.15) and every startBounce
// below 3, so the tail is load-bearing energy, not waste. The only unbiased
// lever left is scheduling: advance every live path ONE bounce per dispatch,
// requeue survivors, so warps stay dense (ceiling 1/0.383 ~ 2.6x on
// shadeMain's marching).
//
// BYTE-IDENTICAL BY CONSTRUCTION, and this is the part to keep true when
// editing: a path's per-bounce code is the fused loop body VERBATIM (same
// call order: sampleDistance -> scatterAtVertex -> deferred NEE at death),
// and the SampleGenerator STATE rides in PathState between dispatches - never
// re-seeded - so every path consumes the exact sample stream the fused loop
// would have. One path owns one pixel, so output writes stay single-writer.
// The fused shadeMain stays compiled and selectable (useWavefront off) as the
// same-session A/B reference; converged images from the two backends must be
// bit-identical, which is a stronger check than the +-0.15% gate.
// ---------------------------------------------------------------------------

/** Everything one in-flight path needs between bounces (~124 B).
    SampleGenerator and NeeVertexReservoir are embedded as POD - reflection
    sizes the buffer, so there is no hand-maintained stride to drift.
*/
struct PathState
{
    uint pixel;             ///< Packed x | y<<16 (one path per pixel: single writer).
    uint bounce;            ///< Bounce index of the NEXT scatter vertex.
    float3 org;             ///< Continuation ray origin (scatterAtVertex always
    float3 dir;             ///< writes Ray(pos, wo, 0, FLT_MAX), so tMin/tMax
                            ///< reconstruct exactly).
    float3 thp;             ///< Path throughput.
    float3 rad;             ///< Radiance this path has accumulated.
    NeeVertexReservoir nee; ///< Deferred single-NEE reservoir (resolved at death).
    SampleGenerator sg;     ///< RNG state - carried, NEVER re-seeded.
};
RWStructuredBuffer<PathState> gPathState; ///< Indexed by scatter-queue slot; a path's slot never changes.
RWStructuredBuffer<uint> gPathQueueIn;    ///< Slots of paths alive this bounce.
RWStructuredBuffer<uint> gPathCountIn;    ///< [0] = size of gPathQueueIn.
RWStructuredBuffer<uint> gPathQueueOut;   ///< Survivors, consumed next bounce.
RWStructuredBuffer<uint> gPathCountOut;   ///< [0] = size of gPathQueueOut.
RWByteAddressBuffer gBounceArgs;          ///< Indirect args for the next bounceMain.

/** Path death: resolve the deferred single NEE and write the pixel. Byte-for-
    byte the fused shadeMain tail, factored so init and bounce share it.
*/
void finishPath(
    const uint2 pixel,
    float3 radiance,
    const NeeVertexReservoir neeReservoir,
    inout VolumeInstanceSampler volumeSampler,
    inout SampleGenerator sg
)
{
    if (kSingleNeePerPath && kUseNEE && neeReservoir.count > 0)
    {
        const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(neeReservoir.g);
        radiance += neeReservoir.thp * evalNEE(neeReservoir.pos, neeReservoir.wi, phase, volumeSampler, sg) * float(neeReservoir.count);
    }
    gOutputColor[pixel] += float4(radiance, 0.f);
    if (is_valid(gOutputLin))
        gOutputLin[pixel] += float4(radiance, 0.f);
}

/** Shade-phase work counters, shared by both wavefront kernels so [WORK] and
    [COST] stay comparable with the fused backend (same slots, 20/21 = shading
    cells/taps). Wave ops with all live lanes, then lane 0 commits.
*/
void emitShadeStats(inout VolumeInstanceSampler volumeSampler)
{
#if USE_WORK_STATS
    const uint a = WaveActiveSum(volumeSampler.statAabbTests);
    const uint a2 = WaveActiveSum(volumeSampler.statBrickCandidates);
    const uint b = WaveActiveSum(volumeSampler.statSegments);
    const uint g2 = WaveActiveSum(volumeSampler.statRays);
    const uint h4 = WaveActiveSum(gvsCellsNow());
    const uint h5 = WaveActiveSum(gvsTapsNow());
    const uint h6 = WaveActiveSum(volumeSampler.statOverlapSteps);
    const uint h7 = WaveActiveSum(volumeSampler.statOverlapLookups);
    const uint h8 = WaveActiveSum(volumeSampler.statTailRays);
    const uint h9 = WaveActiveSum(gvsBrickHitsNow());
    const uint h11 = WaveActiveSum(gvsTapSkipsNow());
    const uint h10 = WaveActiveSum(gvsBrickMissesNow());
    const uint h12 = WaveActiveSum(gvsBrickPromotesNow());
    if (WaveIsFirstLane())
    {
        if (a > 0) InterlockedAdd(gRisStats[9], a);
        if (a2 > 0) InterlockedAdd(gRisStats[29], a2);
        if (b > 0) InterlockedAdd(gRisStats[10], b);
        if (g2 > 0) InterlockedAdd(gRisStats[15], g2);
        if (h4 > 0) InterlockedAdd(gRisStats[20], h4);
        if (h5 > 0) InterlockedAdd(gRisStats[21], h5);
        if (h6 > 0) InterlockedAdd(gRisStats[22], h6);
        if (h7 > 0) InterlockedAdd(gRisStats[23], h7);
        if (h8 > 0) InterlockedAdd(gRisStats[24], h8);
        if (h9 > 0) InterlockedAdd(gRisStats[27], h9);
        if (h10 > 0) InterlockedAdd(gRisStats[28], h10);
        if (h11 > 0) InterlockedAdd(gRisStats[30], h11);
        if (h12 > 0) InterlockedAdd(gRisStats[71], h12);
    }
#endif
}

/** Wavefront phase B entry: shade the RIS vertex (bounce 0) exactly as the
    fused shadeMain head does, then park the survivor in gPathState instead of
    looping.
*/
[numthreads(64, 1, 1)]
void shadeInitMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gvsSetOccSkipEnable(gOccSkipEnable);
    const uint queueIdx = dispatchThreadId.x;
    if (queueIdx >= gScatterCount[0])
        return;

    const ScatterEntry entry = gScatterQueue[queueIdx];
    const uint2 pixel = uint2(entry.pixel & 0xffffu, entry.pixel >> 16);

    // Seed by PIXEL with the same constant as the fused path: the wavefront
    // stream must be the same stream.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount + 0x9e3779b9u);
    VolumeInstanceSampler volumeSampler = {};

    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim);
    ray.tMax = FLT_MAX;
    const float3 wi = -ray.dir;

    float3 radiance = float3(0.f);
    const GridVolume gv = gScene.getGridVolume(entry.inst);
    float3 throughput = gv.data.albedo * entry.W;
    NeeVertexReservoir neeReservoir = {};

    const float3 scatterPos = ray.origin + entry.t * ray.dir;
    if (scatterAtVertex(scatterPos, wi, gv.data.anisotropy, 0, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
    {
        PathState ps;
        ps.pixel = entry.pixel;
        ps.bounce = 1;
        ps.org = ray.origin;
        ps.dir = ray.dir;
        ps.thp = throughput;
        ps.rad = radiance;
        ps.nee = neeReservoir;
        ps.sg = sg;
        gPathState[queueIdx] = ps;
        uint slot;
        InterlockedAdd(gPathCountOut[0], 1u, slot);
        gPathQueueOut[slot] = queueIdx;
    }
    else
    {
        finishPath(pixel, radiance, neeReservoir, volumeSampler, sg);
    }

    emitShadeStats(volumeSampler);
}

/** One bounce for every live path: the fused while-loop body, verbatim, with
    "break" replaced by path death and "loop again" by requeue.
*/
[numthreads(64, 1, 1)]
void bounceMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gvsSetOccSkipEnable(gOccSkipEnable);
    const uint i = dispatchThreadId.x;
    if (i >= gPathCountIn[0])
        return;
    const uint slot = gPathQueueIn[i];
    PathState ps = gPathState[slot];

    const uint2 pixel = uint2(ps.pixel & 0xffffu, ps.pixel >> 16);
    VolumeInstanceSampler volumeSampler = {};
    SampleGenerator sg = ps.sg;
    Ray ray = Ray(ps.org, ps.dir, 0.f, FLT_MAX);
    float3 throughput = ps.thp;
    float3 radiance = ps.rad;
    NeeVertexReservoir neeReservoir = ps.nee;
    uint bounce = ps.bounce;

    // Fused loop body, one trip. Divergence here is bounded by ONE bounce's
    // internal variance, not the whole path's - that is the entire point.
    bool alive = false;
    VolumeInstanceSampler::DistanceSample ds;
    if (!volumeSampler.sampleDistance(ray, sg, ds))
    {
        radiance += throughput * evalEnvironment(ray.dir);
    }
    else
    {
        throughput *= ds.thp;
        const float3 pos = ray.origin + ds.t * ray.dir;
        const float3 wiB = -ray.dir;
        if (bounce < kMaxBounces)
        {
            const float g = gScene.getGridVolume(ds.instance).data.anisotropy;
            if (scatterAtVertex(pos, wiB, g, bounce, ray, throughput, radiance, neeReservoir, volumeSampler, sg))
            {
                bounce++;
                alive = true;
            }
        }
    }

    if (alive)
    {
        ps.bounce = bounce;
        ps.org = ray.origin;
        ps.dir = ray.dir;
        ps.thp = throughput;
        ps.rad = radiance;
        ps.nee = neeReservoir;
        ps.sg = sg;
        gPathState[slot] = ps;
        uint o;
        InterlockedAdd(gPathCountOut[0], 1u, o);
        gPathQueueOut[o] = slot;
    }
    else
    {
        finishPath(pixel, radiance, neeReservoir, volumeSampler, sg);
    }

    emitShadeStats(volumeSampler);
}

/** Next bounce's indirect args from the live-path count. */
[numthreads(1, 1, 1)]
void bounceArgsMain()
{
    const uint n = gPathCountIn[0];
    gBounceArgs.Store3(0, uint3((n + 63u) / 64u, 1u, 1u));
}

/** Tail finisher: loop the remaining live paths to completion, fused-style.
    WHY: v1 ran all 64 requeue rounds and was MUCH slower - each round is
    args + clear + dispatch with serializing barriers (~192 sync points per
    frame), and past round ~10 only a few thousand paths remain, so ~50 rounds
    paid full pipeline-drain overhead to advance almost nothing. Requeueing
    only pays while populations keep warps dense; once they are thin, the
    fused loop's divergence costs less than the per-round overhead. So: K
    dense rounds (gWavefrontRounds, CPU-side), then this kernel drains the
    rest. Same loop body, same RNG stream => still byte-identical.
*/
[numthreads(64, 1, 1)]
void bounceTailMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gvsSetOccSkipEnable(gOccSkipEnable);
    const uint i = dispatchThreadId.x;
    if (i >= gPathCountIn[0])
        return;
    const uint slot = gPathQueueIn[i];
    PathState ps = gPathState[slot];

    const uint2 pixel = uint2(ps.pixel & 0xffffu, ps.pixel >> 16);
    VolumeInstanceSampler volumeSampler = {};
    SampleGenerator sg = ps.sg;
    Ray ray = Ray(ps.org, ps.dir, 0.f, FLT_MAX);
    float3 throughput = ps.thp;
    float3 radiance = ps.rad;
    NeeVertexReservoir neeReservoir = ps.nee;
    uint bounce = ps.bounce;

    // The fused shadeMain continuation loop, verbatim.
    while (true)
    {
        VolumeInstanceSampler::DistanceSample ds;
        if (!volumeSampler.sampleDistance(ray, sg, ds))
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

    finishPath(pixel, radiance, neeReservoir, volumeSampler, sg);
    emitShadeStats(volumeSampler);
}
#endif // USE_COMPACTION

// ---------------------------------------------------------------------------
// Sun-tau cache build (see the TauCB note for what this is and why it is
// unbiased). Two dispatches, both trivial next to one frame of marching:
// tauDirMain picks the direction, tauBuildMain walks one coarseOpticalDepth
// per cache voxel - the SAME function the per-pixel path calls when the cache
// is off, so a cache texel is byte-identical in semantics to the walk it
// replaces. 64^3 voxels x ~19 cells is ~5M DDA cells per rebuild, vs the
// 39.5M PER FRAME the live walks cost ([WORK] 2026-07-20).
// ---------------------------------------------------------------------------

/** Dominant light direction: luminance-weighted mean of 64 env importance
    samples. One thread on purpose - no reduction plumbing for a 64-iteration
    loop that runs once per rebuild. Fixed seed: the direction must not jitter
    between rebuilds or the cache would shimmer for no variance benefit.
*/
[numthreads(1, 1, 1)]
void tauDirMain()
{
    float3 acc = float3(0.f);
    if (kUseEnvLight)
    {
        SampleGenerator sg = SampleGenerator(uint2(0, 0), 4211);
        for (uint i = 0; i < 64; ++i)
        {
            EnvMapSample s = {};
            if (gParams.envMapSampler.sample(sampleNext2D(sg), s) && s.pdf > 0.f)
                acc += s.dir * (dot(s.Le, float3(0.2126f, 0.7152f, 0.0722f)) / s.pdf);
        }
    }
    // Degenerate env (or none): straight up. The cache is target-only, so a
    // wrong direction costs candidate-selection variance, never correctness.
    const float3 dir = dot(acc, acc) > 1e-12f ? normalize(acc) : float3(0.f, 1.f, 0.f);
    gTauDirBuf[0] = float4(dir, 1.f);
}

/** One coarseOpticalDepth walk per cache voxel, from voxel center to volume
    exit toward the dominant light. Occupancy skip is left at its static
    default here (no CB dependency); it only changes bake speed, not tau.
*/
[numthreads(4, 4, 4)]
void tauBuildMain(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid >= gTauDim)) return;
    const float3 pos = gTauOrigin + (float3(tid) + 0.5f) * gTauCellSize;
    VolumeInstanceSampler vs = {};
    const float tau = vs.coarseOpticalDepth(Ray(pos, gTauDirBuf[0].xyz, 0.f, FLT_MAX));
    // Clamp for R16F range; exp(-64) is 40 orders below the 1e-3 visibility
    // floor the target applies, so nothing is lost.
    gTauCacheUav[tid] = min(tau, 64.f);
}

/** Fold this frame's training deposits into the resolved radiance cache with
    an EMA, then clear the accumulators. Confidence (.a) rises toward 1 as a
    cell keeps receiving samples; consumption gates on it, so unpopulated
    regions never contribute a fake zero - they just run the plain estimator.
*/
[numthreads(4, 4, 4)]
void radResolveMain(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid >= gRadDim))
        return;
    const uint idx = ((tid.z * gRadDim.y + tid.y) * gRadDim.x + tid.x) * 4u;
    const uint4 a = uint4(gRadAccum[idx + 0], gRadAccum[idx + 1], gRadAccum[idx + 2], gRadAccum[idx + 3]);
    if (a.w > 0u)
    {
        const float3 mean = float3(a.xyz) / (kRadFxScale * float(a.w));
        float4 v = gRadResolveUav[tid];
        v.rgb = lerp(v.rgb, mean, gRadEma);
        v.a = lerp(v.a, 1.f, gRadEma);
        gRadResolveUav[tid] = v;
    }
    gRadAccum[idx + 0] = 0u;
    gRadAccum[idx + 1] = 0u;
    gRadAccum[idx + 2] = 0u;
    gRadAccum[idx + 3] = 0u;
}
