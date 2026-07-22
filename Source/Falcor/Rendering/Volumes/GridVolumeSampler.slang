/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Utils.PixelStats;
import GridVolumeSamplerParams;

#ifndef GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR
// GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR not defined!"
#endif

#ifndef GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER
// GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER not defined!"
#endif

#ifndef GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
// GRID_VOLUME_SAMPLER_USE_BRICKEDGRID must be defined in order to use this module.
#error "GRID_VOLUME_SAMPLER_USE_BRICKEDGRID not defined!"
#endif

// Mip level of the mean/range pyramids used by residual ratio tracking.
// Optional (defaulted, not #error'd) so existing users of this module that
// predate the residual estimator keep compiling unchanged.
#ifndef GRID_VOLUME_SAMPLER_RESIDUAL_MIP
#define GRID_VOLUME_SAMPLER_RESIDUAL_MIP 0
#endif

// Work counters for the ACTUAL marching this module does. Optional and off by
// default so every existing user compiles byte-identical. When enabled, these
// per-thread globals accumulate across every estimator call in the invocation;
// the host pass snapshots them around call sites to attribute cost per phase.
// Motivation (measured): the VolumePathTracer [WORK] log attributed ~0 of a
// ~470 ms frame because all marching happens inside this module, uncounted.
#ifndef GRID_VOLUME_SAMPLER_STATS
#define GRID_VOLUME_SAMPLER_STATS 0
#endif
#if GRID_VOLUME_SAMPLER_STATS
static uint gGvsStatCells = 0; ///< Majorant DDA cells stepped (range-tex fetch + stepDDA). The non-DDA estimators only tap, so their cost shows up entirely in taps.
static uint gGvsStatTaps = 0;  ///< Real density taps (stochastic atlas fetches) - the memory-bound operation.
static uint gGvsHomogBins[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }; ///< [HOMOG] histogram of sigma_mean/sigma_majorant over mip-0 majorant-DDA cells - the homogenization gauge (right-loaded = uniform interior, collapsible to an analytic segment).
#endif
void gvsStatCell()
{
#if GRID_VOLUME_SAMPLER_STATS
    gGvsStatCells++;
#endif
}
void gvsStatTap()
{
#if GRID_VOLUME_SAMPLER_STATS
    gGvsStatTaps++;
#endif
}

// ---------------------------------------------------------------------------
// Brick-coherent density-tap cache.
//
// WHY: Grid::lookupIndexTex is the hottest line in the volume kernel (~9.8% of
// all SM samples; top stall LONG SCOREBOARD at 86%). It is NOT bandwidth -
// L1TEX and VRAM both sit near 10% - it is LATENCY: atlasTex cannot issue until
// indirectionTex returns, so every tap pays a serialized two-hop dependent
// chase. Holding (brick, ptr, range) across taps removes both the rangeMean and
// indirection loads on a hit AND collapses the dependent hop, because the atlas
// address becomes computable from registers immediately.
//
// This is UE's Nanite Foliage idea (NaniteRasterizer.usf:1278 stages brick data
// in groupshared so the per-voxel loop never re-chases indirection), kept in
// registers instead of LDS because VNA's shared memory is already ~82 KB/SM
// against a ~100 KB ceiling.
//
// CORRECTNESS - READ BEFORE MOVING THESE:
// the cached ptr/range are only valid FOR ONE Grid. Two volume instances can
// produce the same brick coordinate with completely different atlas pointers, so
// a cache that outlives a single grid returns silently wrong densities - wrong
// image, no crash, no counter. These are reset at the entry of each walk, where
// the grid is fixed for the duration. Do not hoist the reset outward, and do not
// reuse this across a loop that switches instances.
// ---------------------------------------------------------------------------
static int3 gGvsBrickKey = int3(-1, -1, -1); ///< Cached brick coord; sentinel no valid index produces.
static uint3 gGvsBrickPtr = uint3(0, 0, 0);  ///< Cached indirection result for gGvsBrickKey.
static float2 gGvsBrickRange = float2(0.f, 0.f); ///< Cached (majorant, minorant) for gGvsBrickKey.
static float gGvsBrickMean = 0.f;            ///< Cached mean (rangeMeanTex .z) - the residual walk's control.
static uint gGvsBrickOcc = 0u;               ///< Cached occupancy mask for gGvsBrickKey (see Grid.slang).

// ---------------------------------------------------------------------------
// PREFETCH ENTRY (lever 2, 2026-07-20). The DDA loops are serial chains of
// single-entry misses: at mip 0 every DDA cell IS a brick, so each iteration's
// range/indirection/occupancy loads cannot issue until the previous
// iteration's majorant decided the loop - Nsight run 20_50 profiles the walk
// at LONG SCOREBOARD 30% with the SM at 50% and VRAM at 12% (latency-bound,
// not bandwidth). But the NEXT cell's address needs only ALU (stepDDA is pure
// arithmetic), so the walk can issue the next brick's loads WHILE the current
// brick's value is still in flight - software pipelining. This second entry
// holds those speculative loads; gvsBrickEnsure promotes it with register
// moves when the walk arrives, and a wasted prefetch (collision ended the
// walk) costs only the dead loads, never correctness.
// Same single-grid validity rule as the primary entry: reset clears BOTH.
//
// COMPILE-TIME, not a runtime uniform (run 137 lesson): the entry costs ~9
// registers whether or not a runtime flag reads it, and run 137 regressed
// gpuMs 26 -> 38 with bit-identical work counters - the register-pressure
// signature. A runtime toggle cannot A/B register pressure; only compiling
// the entry out can. The UI checkbox rebuilds the program.
// ---------------------------------------------------------------------------
#ifndef GVS_BRICK_PREFETCH
#define GVS_BRICK_PREFETCH 0
#endif
#if GVS_BRICK_PREFETCH
static int3 gGvsBrickKeyN = int3(-1, -1, -1);
static uint3 gGvsBrickPtrN = uint3(0, 0, 0);
static float2 gGvsBrickRangeN = float2(0.f, 0.f);
static float gGvsBrickMeanN = 0.f;
static uint gGvsBrickOccN = 0u;
#endif

/** Runtime A/B switch for the occupancy skip (HANDOFF_6 6.3).

    A UNIFORM, not a define, on purpose: the only trustworthy way to price this
    change is to flip it inside ONE process and compare [COST] lines from the
    same session. Cross-session gpuMs on this machine has twice shown ~25% drift
    with bit-identical work counters, which is larger than the effect being
    measured. A define would force a reload and reintroduce exactly that drift.

    Cost of the switch itself: the mask is loaded either way (it rides along with
    the range/ptr fetch already on the cache-miss path), so disabling this
    measures the skip, not the load. That asymmetry is deliberate - it isolates
    the branch + saved-load from the added fetch. Set 0 and the sampler is
    behaviourally identical to the pre-6.3 build.

    A per-thread static set from the caller's CB, NOT a module-scope `uniform`.
    Falcor core declares no module-scope uniforms anywhere, and introducing one
    here produced an E_INVALIDARG on the very first dispatchCompute - the loose
    global never got a valid binding. Statics are how this file already carries
    per-thread state (see gGvsBrickKey), so the flag rides in the same way and
    the pass owns the actual uniform.
*/
static uint gGvsOccSkipEnable = 1u;

/** Set by the entry point once per thread, from the pass's CB. */
void gvsSetOccSkipEnable(const uint enable)
{
    gGvsOccSkipEnable = enable;
}

/** Weighted delta-tracking majorant softening for the DISTANCE sampler.
    1 = true majorant = the exact analog estimator, byte-identical. Below 1 the
    free flight uses lerp(cellMean, cellMajorant, k), which is not a bound, and
    sampleDistanceDeltaTrackingLocalDDA pays for it with a per-collision weight.
    Only that one estimator reads this; every other walk passes weightK
    explicitly, so nothing can inherit it by accident.
*/
static float gGvsDistWeightK = 1.f;
void gvsSetDistWeightK(const float k)
{
    gGvsDistWeightK = k;
}
#if GRID_VOLUME_SAMPLER_STATS
static uint gGvsStatBrickHits = 0;   ///< Taps served from the cached brick.
static uint gGvsStatBrickMisses = 0; ///< Taps that had to re-chase the indirection.
static uint gGvsStatTapSkips = 0;    ///< Taps answered from the occupancy mask, atlasTex never read.
static uint gGvsStatBrickPromotes = 0; ///< Misses answered by the prefetch entry - register moves, loads already in flight.
/// stepToNextCollision's while (t < tFar) trip count for THIS lane. Declared
/// unconditionally (not under GVS_WAVE_UNIFORM_MIP) so [STEPOCC] reads the same
/// quantity in the baseline build and the lever build - an occupancy number is
/// only worth having if the A and the B are the same measurement.
static uint gGvsStatStepIters = 0;
static uint gGvsStatMipLifts = 0;    ///< Cells whose mip the warp actually raised above this lane's own level.
static uint gGvsStatMipLiftSum = 0;  ///< Sum of (lifted - own) over those cells - the average lift, not just its frequency.
#endif

// ---------------------------------------------------------------------------
// LEVER LEDGER, 2026-07-22. Five levers measured in one session, all null on
// frame time. Recorded here so the next person does not re-derive them from the
// same profiler numbers that suggested them the first time.
//
//   lever                        mechanism confirmed?      frame effect
//   ---------------------------  ------------------------  ----------------
//   occupancy skip (atlas reads) yes, 19.7M taps/frame     0.02 ms first
//                                skipped                   measured, ~0.5-1.4
//                                                          ms on the fixed
//                                                          instrument, always
//                                                          inside the floor
//   dead-path compile-out        yes, coarseOpticalDepth   none
//   (TAU_CACHE_APPLY,            and a 2nd evalTransmit-
//    USE_TRR_PROBE)              tance walk left main
//   per-instance walk setup      yes, brickCache hit       none (-0.62/+0.15/
//   (VolumeInstanceSampler)      +8 to +9 pp, identical    +0.50 ms over three
//                                lookups and cells         runs, floors 2.6-4.7)
//   register narrowing           NO - 125 -> 123 regs,     none
//   (this file)                  compiler had already
//                                stripped the fields
//
// THE PATTERN, which is worth more than any single row: every one of these was
// chosen because a profiler ranked it expensive - 5.6% of self-samples, 54% of
// taps, the register high-water mark - and removing it changed nothing. On this
// kernel "where the cycles are attributed" and "what would make the frame
// shorter" have come apart every single time they were tested.
//
// MEASUREMENT TRAPS THAT PRODUCED FALSE ANSWERS BEFORE THEY WERE FIXED:
//   - This machine drifts ~22% in gpuMs over ~3000 frames at BYTE-IDENTICAL
//     work, and recovers ~25% instantly after an idle pause. Any A-then-B
//     comparison charges B for the heat A produced. Interleave, and take the
//     spread between rounds of the SAME config as the floor.
//   - The retrain ramp after updatePass is ~600 frames. Shorter warm-ups
//     measure the ramp.
//   - The renderer is NOT bit-reproducible run to run (~1 ULP of half float on
//     volume pixels), so a bit-identity gate cannot pass. Use a same-config
//     control arm as the noise floor instead.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// WAVE-UNIFORM MAJORANT MIP (lever: Active Threads Per Warp).
//
// WHY: analysis7.yaml, read PER RANGE, puts "Active Threads Per Warp" at 24.27%
// with a 4.12x range-speedup ceiling and 73% frame gain - three times the
// register-count rule (1.36x / 25.6%) and fifty times long scoreboard. The SM
// issues at 52.6% of peak while long-scoreboard stalls sit at 1.5% and L2 hits
// 92%, so the frame is NOT latency-starved: it issues plenty of instructions,
// and 75.7% of the lane slots inside them are masked off. That is also the
// answer to the question five levers failed to answer - they all removed TAPS,
// work on the tex/lsu pipes, which do not even reach the top-5 SOL list, from
// lanes that were already idle. Op counts were never what the warp was paying.
//
// Both scheduling routes are already closed and measured: per-bounce requeueing
// (mUseWavefront) and the work-class queue sort (mUseScatterSort) each fixed
// path-LENGTH divergence and each came back slower. The wavefront note names
// the residue exactly - "single-bounce cost variance (3 vs 40 bricks marched)
// is most of the 0.383 work-occupancy gap". That residue is THIS loop. The
// while (t < tFar) DDA below costs its warp max(trips), not sum(trips), and
// reordering which lanes share a warp cannot change any single lane's trips.
//
// WHAT THIS DOES: the mip sets the DDA step size, and it is currently a purely
// per-lane quantity (the accessor climbs kDDAMipUpSpeed per empty cell, drops
// kDDAMipDownSpeed on a collision). So one lane grinding at mip 0 while its 31
// neighbours coast at mip 3 makes the entire warp pay the mip-0 trip count.
// Taking the COARSEST level any active lane wants makes the step size
// wave-uniform, so trip counts converge by construction - no queues, no
// barriers, no per-path state round-trip, which is what sank the wavefront
// backend. It trades serialized loop iterations on the ALU pipe (top SOL at
// 43%) for null collisions on tex/lsu (idle).
//
// UNBIASED - keep this true when editing. Delta tracking is unbiased for ANY
// majorant that bounds sigma over the cell it steps. A mip-m majorant is a max
// over a superset of the mip-0 cell, so coarsening only ever makes the bound
// looser, never invalid; that is precisely why the existing adaptive climb is
// already allowed to move the level at will. Lifting a lane's mip therefore
// introduces no bias regardless of what the other lanes are doing.
//
// The subtler half: the mip a lane uses now depends on OTHER lanes' RNG (their
// accessor levels, and whether they are still active in the loop). Conditional
// on every other lane's sample stream, this lane's majorant is still a fixed
// function of its own position along its own ray, so its estimate is unbiased;
// averaging over the other lanes' streams preserves that by the tower property.
// The argument rests on lanes having INDEPENDENT sample streams - which they
// do, since each is seeded by pixel - so do not introduce cross-lane sample
// sharing above this without revisiting it.
//
// NOT byte-identical, unlike compaction or the queue sort. Changing the mip
// changes how many cells the walk steps, which changes how many sampleNext1D
// draws it consumes, so the sample stream diverges from the reference. Same
// class as distWeightK: gate on a converged IMAGE comparison, never on
// bit-identity, and never on gpuMs alone.
//
// A LIFT CLAMP rather than a bare WaveActiveMax, because the trade has a sign
// that cannot be predicted from here. Over-coarsening a lane that sits inside
// dense cloud makes tauToGo run out almost immediately, which pushes iterations
// out of this inner loop and into the caller's while(1) retry loop - divergence
// moved rather than removed. gGvsWaveMipLift sweeps 0 (current behaviour)
// through 3 (full wave max) live, so the balance point gets measured instead of
// assumed. [STEPOCC] is the metric that decides it; gpuMs alone cannot, because
// this lever is supposed to trade one pipe's work for another's.
//
// COMPILE-TIME gate on top of the runtime lift: at lift 0 the wave reduction
// still costs an instruction in the hottest loop in the frame, and this file's
// own run-137 lesson is that a runtime flag cannot A/B something that costs
// whether or not it is read. GVS_WAVE_UNIFORM_MIP 0 compiles all of it out, so
// every other user of this module - and every ship build - stays unchanged.
// ---------------------------------------------------------------------------
#ifndef GVS_WAVE_UNIFORM_MIP
#define GVS_WAVE_UNIFORM_MIP 0
#endif
#if GVS_WAVE_UNIFORM_MIP
/** Maximum number of levels a lane may be lifted above its own accessor level
    by its warp. 0 = per-lane mip = the pre-lever behaviour (and the estimator
    is byte-identical there, since the clamp collapses before any wave op runs);
    3 = full WaveActiveMax. A per-thread static set from the pass CB, matching
    gGvsOccSkipEnable / gGvsDistWeightK - module-scope uniforms do not get a
    valid binding in this codebase (see the note on gGvsOccSkipEnable).
*/
static uint gGvsWaveMipLift = 0u;
void gvsSetWaveMipLift(const uint lift)
{
    gGvsWaveMipLift = lift;
}

/** Wave-uniform lift of one DDA cell's mip.

    Returns >= ownMip ALWAYS, and that is what makes this safe under any
    reconvergence behaviour the compiler chooses: HLSL wave ops see only the
    currently-active lanes, but however those lanes happen to be packed, THIS
    lane is one of them, so the max can never come back below its own level and
    can never turn a bounding majorant into a non-bounding one. A smaller-than-
    expected active mask costs opportunity, never correctness.
*/
int gvsWaveLiftMip(const int ownMip)
{
    if (gGvsWaveMipLift == 0u) return ownMip;
    const int lifted = min(WaveActiveMax(ownMip), ownMip + int(gGvsWaveMipLift));
#if GRID_VOLUME_SAMPLER_STATS
    // Did the mechanism actually fire? Without this a null result is
    // uninterpretable: a warp whose lanes already agree on the mip has nothing
    // to lift, and "no effect" would be indistinguishable from "no
    // opportunity". Registers only, no memory traffic, and stats-gated, so the
    // [SUBHOMOG] failure (a settled diagnostic still spending bandwidth in the
    // hot loop) cannot repeat here.
    if (lifted > ownMip)
    {
        gGvsStatMipLifts++;
        gGvsStatMipLiftSum += uint(lifted - ownMip);
    }
#endif
    return lifted;
}
#endif // GVS_WAVE_UNIFORM_MIP

/** Invalidate the brick cache. MUST be called at the start of every walk, before
    any tap, because the cache is only valid for a single Grid (see above).
    Clears BOTH entries - a prefetched brick from another grid's walk promoted
    later would be the exact silent-wrong-density bug the primary reset exists
    to prevent.
*/
void gvsBrickCacheReset()
{
    gGvsBrickKey = int3(-1, -1, -1);
#if GVS_BRICK_PREFETCH
    gGvsBrickKeyN = int3(-1, -1, -1);
#endif
}

/** Make the primary entry hold `brick`: hit, promote-from-prefetch, or full
    miss load. All callers below go through this, so hit/promote/miss
    accounting and the promote register-move path live in exactly one place.
*/
void gvsBrickEnsure(const Grid densityGrid, const int3 brick)
{
    if (!any(brick != gGvsBrickKey))
    {
#if GRID_VOLUME_SAMPLER_STATS
        gGvsStatBrickHits++;
#endif
        return;
    }
#if GVS_BRICK_PREFETCH
    if (!any(brick != gGvsBrickKeyN))
    {
        // The loads were issued cells ago (gvsBrickPrefetch); by now they have
        // landed or are about to. Promotion is register moves only.
        gGvsBrickKey = gGvsBrickKeyN;
        gGvsBrickRange = gGvsBrickRangeN;
        gGvsBrickMean = gGvsBrickMeanN;
        gGvsBrickPtr = gGvsBrickPtrN;
        gGvsBrickOcc = gGvsBrickOccN;
        gGvsBrickKeyN = int3(-1, -1, -1);
#if GRID_VOLUME_SAMPLER_STATS
        // A promote still counts as a miss: without the prefetch it WOULD have
        // been one, so the [COST] hit-rate stays comparable across sessions
        // and the promote share reads against the same denominator.
        gGvsStatBrickMisses++;
        gGvsStatBrickPromotes++;
#endif
        return;
    }
#endif // GVS_BRICK_PREFETCH
    gGvsBrickKey = brick;
    const float3 rm = densityGrid.rangeMeanTex[brick].xyz;
    gGvsBrickRange = rm.xy;
    gGvsBrickMean = rm.z;
    gGvsBrickPtr = densityGrid.indirectionTex[brick].xyz;
    gGvsBrickOcc = densityGrid.occupancyTex[brick];
#if GRID_VOLUME_SAMPLER_STATS
    gGvsStatBrickMisses++;
#endif
}

/** Issue the next brick's loads into the prefetch entry WITHOUT waiting for
    them. Call with the brick the walk will enter next (pure ALU to compute)
    while the current brick's work is still outstanding - that overlap is the
    entire point. Skips bricks already resident in either entry.
*/
void gvsBrickPrefetch(const Grid densityGrid, const int3 brick)
{
#if GVS_BRICK_PREFETCH
    if (!any(brick != gGvsBrickKey) || !any(brick != gGvsBrickKeyN)) return;
    gGvsBrickKeyN = brick;
    const float3 rm = densityGrid.rangeMeanTex[brick].xyz;
    gGvsBrickRangeN = rm.xy;
    gGvsBrickMeanN = rm.z;
    gGvsBrickPtrN = densityGrid.indirectionTex[brick].xyz;
    gGvsBrickOccN = densityGrid.occupancyTex[brick];
#endif
}

/** Brick-coherent replacement for Grid::lookupStochasticIndexTex.

    Bitwise identical to the uncached path: on a hit it reuses ptr/range fetched
    from the exact same addresses and the arithmetic is unchanged. The float3 ->
    int3 conversion reproduces the implicit TRUNCATION in lookupStochasticIndexTex
    (which does NOT floor, unlike lookupStochasticIndex).
*/
float gvsLookupMajorantCached(const Grid densityGrid, const int3 index, const int mip)
{
    // At mip 0 - the shipping config (residualMip 0) - lookupIndexLocalMajorantTex
    // loads rangeMeanTex[index >> 3].x, the SAME TEXEL the brick cache already
    // holds as gGvsBrickRange.x. On a hit the majorant costs nothing.
    //
    // The bigger win is ordering. stepToNextCollision fetches the majorant ONCE
    // per DDA cell, BEFORE any density tap in that cell. On a miss this populates
    // the FULL entry - range AND the indirection pointer - so taps that follow in
    // the same brick already have ptr resident and skip the dependent hop into
    // atlasTex. The indirection load is not added, it is MOVED EARLIER: off the
    // tap's critical path into a slot where its latency overlaps the DDA
    // arithmetic. Both fetches profile at LONG SCOREBOARD (majorant 76%, tap
    // 83-86%), so taking one off the dependent chain is the whole point.
    //
    // Above mip 0 the majorant lives in a different mip than the cache's mip-0
    // entry, so it falls through untouched - correctness does not depend on the
    // mip config.
    if (mip != 0) return densityGrid.lookupIndexLocalMajorantTex(index, mip);

    gvsBrickEnsure(densityGrid, index >> 3);
    return gGvsBrickRange.x;
}

/** Cache-served (majorant, minorant, mean) at mip 0 - the residual walk's
    per-node fetch. Same texel the cache already loads (rangeMeanTex[brick]
    == lookupIndexLocalRangeMeanTex at mip 0, see Grid.slang:327), so serving
    it from the entry is bitwise identical - and it fills ptr/occ EARLY, so
    the residual taps that follow inside the node hit instead of paying the
    dependent indirection chase on their first tap.
*/
float3 gvsLookupRangeMeanCached(const Grid densityGrid, const int3 index)
{
    gvsBrickEnsure(densityGrid, index >> 3);
    return float3(gGvsBrickRange, gGvsBrickMean);
}

float gvsLookupStochasticCached(const Grid densityGrid, const float3 index, const float3 u)
{
    const int3 vox = int3(index + (u - 0.5f));
    gvsBrickEnsure(densityGrid, vox >> 3);
    // Occupancy early-out (UE Nanite Foliage ProcessBrickPixel lesson): the mask
    // is already in a register, so a tap into an all-zero BC4 tile is answered
    // by one shift and one AND instead of a dependent atlasTex load. That load
    // profiles at LONG SCOREBOARD 83-86% and is the single most-stalled
    // instruction in the frame, so not issuing it is worth more than making it
    // faster.
    //
    // This is EXACT, not an approximation. A clear bit means every voxel in the
    // tile decodes to exactly 0, and 0 * (majorant - minorant) + minorant is
    // exactly the minorant, which is the value we return. See GridConverter.h
    // for why the granularity is a BC4 tile.
    if (gGvsOccSkipEnable != 0u && (gGvsBrickOcc & (1u << gridBrickTileBit(vox))) == 0u)
    {
#if GRID_VOLUME_SAMPLER_STATS
        gGvsStatTapSkips++;
#endif
        return gGvsBrickRange.y;
    }

    return densityGrid.atlasTex[(gGvsBrickPtr.xyz << 3) + (vox & 7)].x * (gGvsBrickRange.x - gGvsBrickRange.y) + gGvsBrickRange.y;
}

/** Helper class for sampling grid volumes in the scene.
    Note: For simplicity, this sampler only uses the first volume in the scene.
*/
struct GridVolumeSampler
{
    static const TransmittanceEstimator kTransmittanceEstimator = TransmittanceEstimator(GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR);
    static const DistanceSampler kDistanceSampler = DistanceSampler(GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER);
    static const int kResidualMip = GRID_VOLUME_SAMPLER_RESIDUAL_MIP;

    uint4 _dummy;

    // Multi-level DDA settings.
    static const float kDDAMipStart = 2.f;
    static const float kDDAMipUpSpeed = 0.33f;
    static const float kDDAMipDownSpeed = 2.f;


    /** Transform a ray from world space to grid index space.
        \param[in] gridVolume Grid volume object that stores the transform matrix we need.
        \param[in] densityGrid Grid object that stores the index space bounding box with which we need to offset coordinates, when using BrickedGrid.
        \param[in] rayOriginWorld Ray origin in world space.
        \param[in] rayDirWorld Ray direction in world space.
        \param[out] rayOriginIndex Ray origin in index space.
        \param[out] rayDirIndex Ray direction in index space (non-normalized!).
    */
    void transformRayToIndexSpace(const GridVolume gridVolume, const Grid densityGrid, const float3 rayOriginWorld, const float3 rayDirWorld, out float3 rayOriginIndex, out float3 rayDirIndex) {
        rayOriginIndex = mul(gridVolume.data.invTransform, float4(rayOriginWorld, 1.f)).xyz;
        rayDirIndex = mul(gridVolume.data.invTransform, float4(rayDirWorld, 0.f)).xyz;
#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
        // NanoVDB allows for negative indices, while textures require positive indices.
        // Thus, we simply translate the index-space box, matching a shift we do at grid conversion time on the host.
        rayOriginIndex -= densityGrid.getMinIndex();
#endif
    }

#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
    typedef float DensityAccessor; // dummy type to make interface to lookupDensityTex the same for both NanoVDB and Density Grid
    DensityAccessor createAccessor(const Grid densityGrid) { return kDDAMipStart; }
    float getAccessorLevel(const DensityAccessor a) { return a; }
    void increaseAccessorLevel(inout DensityAccessor a) { a = min(3.f, a + kDDAMipUpSpeed); }
    void decreaseAccessorLevel(inout DensityAccessor a) { a = max(0.f, a - kDDAMipDownSpeed); }
#else
    typedef Grid::Accessor DensityAccessor;
    DensityAccessor createAccessor(const Grid densityGrid) { return densityGrid.createAccessor(); }
    float getAccessorLevel(const DensityAccessor a) { return 0.f; }
    void increaseAccessorLevel(inout DensityAccessor a) { }
    void decreaseAccessorLevel(inout DensityAccessor a) { }
#endif

    /** Per-(grid, ray) walk state, hoisted out of the individual walks.

        WHY: every DDA walk in this file opened with the same six lines - fetch
        the Grid, reset the brick cache, transform the ray to index space (two
        4x4 mat-vecs), reciprocate the direction. That is fine when a walk is
        long. It is not fine here, because it isn't.

        MEASURED (log 185, 2560x1351, shipping config): the candidate-generation
        bucket runs 5.8 segments/pixel and 19.0 cells/pixel, and
        evalEscapeAndCandidatesRQ does 1 transmittance walk + nProc = 2 distance
        walks per segment. That is 17.4 preambles per pixel to service 19.0 loop
        iterations - the SETUP runs almost as often as the body it sets up, and
        all three walks on a segment share the same gridVolume and the same ray,
        so two thirds of it is recomputation of identical values.

        This is the direction this kernel wants. The note in coarseOpticalDepth
        records the opposite trade being tried - 81 floats of live state removed
        by recomputing them - and it cost 25% frame time. On an SM-bound kernel
        ALU is the scarce resource and registers are the cheap one, so paying a
        few registers to hold ipos/idir/ri across three walks is the trade that
        experiment says wins.

        NOT an estimator change. Nothing here touches the sample stream: the
        walks consume the same random numbers in the same order, and the values
        carried are bit-identical to the ones each walk used to compute for
        itself. Converged images must be BIT-identical, which is a far stronger
        gate than the statistical one the wave-mip lever needed.
    */
    struct WalkSetup
    {
        Grid densityGrid;
        float3 ipos;   ///< Ray origin in index space.
        float3 idir;   ///< Ray direction in index space (non-normalized).
        float3 ri;     ///< 1/idir, the DDA's per-axis slab reciprocal.
    };

    /** Open a walk on one (gridVolume, ray) pair.

        CACHE RESET LIVES HERE, and that placement is the correctness contract:
        the brick cache holds an atlas pointer valid for ONE Grid, so it must be
        invalidated whenever the grid changes and may NOT be reused across
        instances. Resetting per SETUP rather than per WALK is therefore still
        correct - a setup is exactly the point where the grid is chosen - and it
        is what lets the cache survive across the transmittance walk and the
        nProc distance walks that share a segment, instead of being thrown away
        twice per segment. Callers must build a fresh setup before switching
        grids; every caller that reuses one is reusing it within a single
        segment, where the grid is fixed by construction.
    */
    WalkSetup beginWalk(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir)
    {
        WalkSetup w;
        gScene.getGrid(gridVolume.getDensityGrid(), w.densityGrid);
        gvsBrickCacheReset();
        transformRayToIndexSpace(gridVolume, w.densityGrid, rayOrigin, rayDir, w.ipos, w.idir);
        w.ri = 1.f / w.idir;
        return w;
    }

    /** Rebuild a setup from index-space state a caller already holds.

        WHY THIS EXISTS AND WHY CALLERS MAY NOT JUST CACHE A WalkSetup:
        WalkSetup contains a Grid, and Grid holds five RESOURCE handles (four
        Texture3D plus a StructuredBuffer). Carrying one in a mutable local that
        is conditionally reassigned inside a loop asks the compiler for a phi
        node over opaque handles, which DXIL cannot represent - Slang has to
        legalize it and instead crashed outright (slang.dll, 0xc0000005, log
        198, 2026-07-22). Every WalkSetup must therefore stay `const`,
        block-scoped and initialised at its declaration, exactly as beginWalk's
        callers already use it.

        So a caller that wants to amortise the setup across loop iterations
        caches only the PLAIN state - ipos/idir/ri, nine floats, no handles -
        and calls this once per iteration to re-attach the handle. That is a
        buffer load, which the pre-change code did per iteration anyway; what it
        avoids is the two 4x4 mat-vecs, the reciprocal, and the brick-cache
        reset, which is the whole cost the amortisation targets.

        NO CACHE RESET HERE, deliberately: resuming means the grid has not
        changed, which is precisely the condition under which the brick cache's
        single-grid rule (see beginWalk) permits keeping it. Callers must call
        beginWalk, not this, whenever the grid changes.
    */
    WalkSetup resumeWalk(const GridVolume gridVolume, const float3 ipos, const float3 idir, const float3 ri)
    {
        WalkSetup w;
        gScene.getGrid(gridVolume.getDensityGrid(), w.densityGrid);
        w.ipos = ipos;
        w.idir = idir;
        w.ri = ri;
        return w;
    }

    struct DistanceSample
    {
        float t;    ///< Sampled distance.
        float3 thp; ///< Throughput.
    };

    /** Checks if a ray potentially intersects a volume.
        \param[in] ray Ray.
        \return Returns true if a volume potentially intersects the ray.
    */
    bool intersectsVolumes(const Ray ray)
    {
        if (gScene.getGridVolumeCount() == 0) return false;
        float2 nearFar;
        return intersectVolume(gScene.getGridVolume(0), ray, nearFar);
    }

    /** Perform a DDA step on a given mipmap level of (dilated) bricks.
        \param[in] rayOrigin Ray origin.
        \param[in] invRayDir Inverse ray direction.
        \param[in] pos Current position along ray.
        \param[in] mip Range texture mipmap level to traverse on (8^mip + 2 voxels).
        \return Distance from the ray origin to the next intersection with (dilated) brick borders.
    */
    float stepDDA(const float3 rayOrigin, const float3 invRayDir, const float3 pos, const int mip)
    {
        const float dim = 8 << mip;
        const float3 tofs = invRayDir * (select(invRayDir >= 0.f, dim + 0.5f, -0.5f) - rayOrigin);
        const float3 tmax = floor(pos * (1.f / dim)) * dim * invRayDir + tofs;
        return min(tmax.x, min(tmax.y, tmax.z));
    }

    /** Evaluate transmittance along a ray through volumes.
        \param[in] ray Ray.
        \param[in,out] sg Sample generator.
        \return Returns the transmittance.
    */
    float evalTransmittance<S : ISampleGenerator>(const Ray ray, inout S sg)
    {
        if (gScene.getGridVolumeCount() == 0) return 1.f;

        GridVolume gridVolume = gScene.getGridVolume(0);
        if (!gridVolume.hasDensityGrid()) return 1.f;

        float2 nearFar;
        if (!intersectVolume(gridVolume, ray, nearFar)) return 1.f;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));

        switch (kTransmittanceEstimator)
        {
        case TransmittanceEstimator::DeltaTracking:
            return evalTransmittanceDeltaTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::RatioTracking:
            return evalTransmittanceRatioTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::RatioTrackingLocalMajorant:
            return evalTransmittanceRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        case TransmittanceEstimator::ResidualRatioTrackingLocalMajorant:
            return evalTransmittanceResidualRatioTrackingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg);
        default:
            return 1.f;
        }
    }

    /** Sample a scattering distance along a ray through volumes.
        \param[in] ray Ray.
        \param[in,out] sg Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a valid scattering distance was sampled.
    */
    bool sampleDistance<S : ISampleGenerator>(const Ray ray, inout S sg, out DistanceSample ds)
    {
        ds = {};

        if (gScene.getGridVolumeCount() == 0) return false;

        GridVolume gridVolume = gScene.getGridVolume(0);
        if (!gridVolume.hasDensityGrid()) return false;

        float2 nearFar;
        if (!intersectVolume(gridVolume, ray, nearFar)) return false;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator(jenkinsHash(sg.next()));

        switch (kDistanceSampler)
        {
        case DistanceSampler::DeltaTracking:
            return sampleDistanceDeltaTracking(gridVolume, ray.origin, ray.dir, nearFar, tsg, ds);
        case DistanceSampler::DeltaTrackingLocalMajorant:
            return sampleDistanceDeltaTrackingLocalDDA(gridVolume, ray.origin, ray.dir, nearFar, tsg, ds);
        default:
            return false;
        }
    }

    float lookupDensityIndex(const float densityScale, const float3 ipos, const float3 u, const Grid densityGrid, inout DensityAccessor densityAccessor)
    {
        logVolumeLookup();
        gvsStatTap();
#if GRID_VOLUME_SAMPLER_USE_BRICKEDGRID
        // Brick-coherent path: see gvsLookupStochasticCached. Valid only because
        // every walk resets the cache before its first tap, so densityGrid is
        // fixed for the cache's lifetime.
        return densityScale * gvsLookupStochasticCached(densityGrid, ipos, u);
#else
        return densityScale * densityGrid.lookupStochasticIndex(ipos, u, densityAccessor);
#endif
    }

    bool intersectVolume(const GridVolume gridVolume, const Ray ray, out float2 nearFar)
    {
        // Intersect with volume bounds and get intersection interval along the view ray.
        AABB bounds = gridVolume.getBounds();
        intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, nearFar);
        nearFar.x = max(nearFar.x, ray.tMin);
        nearFar.y = min(nearFar.y, ray.tMax);
        return nearFar.x < nearFar.y;
    }

    float evalTransmittanceDeltaTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        const float densityScale = gridVolume.data.densityScale;
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        // The brick cache holds an atlas pointer valid for THIS grid only. Reset
        // at every walk entry so a brick cached during the previous instance's
        // walk can never be reused against a different grid's atlas - that would
        // be a silently wrong density, not a crash. See gvsBrickCacheReset.
        gvsBrickCacheReset();
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // Transform to index-space.
        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            if (t >= nearFar.y)
            {
                return 1.f;
            }
            const float d = lookupDensityIndex(densityScale, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // Russian roulette.
            if (sampleNext1D(sg) < d * invMajorant) return 0.f;
        }
        return 1.f;
    }

    float evalTransmittanceRatioTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        const float densityScale = gridVolume.data.densityScale;
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        // The brick cache holds an atlas pointer valid for THIS grid only. Reset
        // at every walk entry so a brick cached during the previous instance's
        // walk can never be reused against a different grid's atlas - that would
        // be a silently wrong density, not a crash. See gvsBrickCacheReset.
        gvsBrickCacheReset();
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);

        float Tr = 1.f;
        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Ratio tracking.
        float t = nearFar.x;
        while (true)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            if (t >= nearFar.y)
            {
                return Tr;
            }
            const float d = lookupDensityIndex(densityScale, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // The 'max' guards against non-bounding majorants which can happen due to density interpolation in NanoVDB.
            // They are rare but can lead to fireflies, and the max prevents them at the cost of (imperceptible) bias.
            Tr *= max(0.f, 1.f - d * invMajorant);
            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return Tr;
    }

    /** \param[in] weightK Free-flight majorant softening for WEIGHTED delta
        tracking. 1 = use the true cell majorant, i.e. every existing behaviour
        byte-identical. Below 1 the free flight is driven by
        sigmaBar_w = lerp(mean, majorant, weightK), which is NOT a bound - the
        caller MUST compensate (see sampleDistanceDeltaTrackingLocalDDA). Passed
        explicitly rather than read from a global so a caller that has not been
        taught to weight cannot silently inherit a non-bounding majorant and
        turn into a biased estimator.
    */
    /** Takes densityScale, not the GridVolume, because that is the only field
        it reads. This was a REGISTER experiment and it did not work: live
        registers here stayed at exactly 108 either way (see the refutation
        block in sampleDistanceDeltaTrackingLocalDDA). Kept because passing one
        float instead of a ~48-scalar struct is clearer, not because it is
        faster - it is not, measurably.
    */
    bool stepToNextCollision(const float densityScale, const Grid densityGrid, const float3 ipos, const float3 idir, const float3 ri,
                             float tauToGo, inout float t, const float tFar, bool adaptive, out float majorant, inout DensityAccessor accessor,
                             const float weightK)
    {
        // Warning: Changing the mip level based on 'tauToGo' below could potentially introduce bias, although it would not seem to be visible in practice.
        while (t < tFar)
        {
            gvsStatCell();
#if GRID_VOLUME_SAMPLER_STATS
            // [STEPOCC] denominator. gGvsStatCells cannot serve here: gvsStatCell
            // is also called by the tail DDA in VolumeInstanceSampler, so its
            // total mixes two different loops and the sum/max occupancy identity
            // (which holds per loop) would not apply to it.
            gGvsStatStepIters++;
#endif
            // Core DDA step. The level the accessor asks for is per-lane; the
            // level actually stepped may be lifted to keep the warp's step size
            // uniform (see the WAVE-UNIFORM MAJORANT MIP note above). Coarser is
            // always a valid bound, so this moves cost only.
            //
            // MEASUREMENT HAZARD, since three things below are gated on mip == 0:
            // the weighted-tracking mean, the [HOMOG] histogram, and the brick
            // prefetch. Lifting the mip makes all three fire LESS OFTEN, so under
            // a nonzero lift [HOMOG] thins out and the brick-cache hit rate moves
            // for reasons that have nothing to do with those features. Compare
            // them across lift settings only against a matched cell count, never
            // as raw totals.
            const int ownMip = int(round(getAccessorLevel(accessor)));
#if GVS_WAVE_UNIFORM_MIP
            const int mip = gvsWaveLiftMip(ownMip);
#else
            const int mip = ownMip;
#endif
            const float3 pos = ipos + idir * t;
            const float tExit = stepDDA(ipos, ri, pos, mip);
            logVolumeLookup();
            // Shares the brick cache and prefetches the indirection pointer for
            // the taps that follow in this cell. Same float3 -> int3 implicit
            // truncation as the direct call it replaces.
            majorant = densityScale * gvsLookupMajorantCached(densityGrid, pos, mip);
            // Weighted delta tracking: soften the free-flight majorant toward
            // the cell MEAN. [HOMOG] measured mean/majorant < 0.125 in 67% of
            // traversed cells, so the true majorant makes the mean free flight
            // ~8x shorter than the medium warrants and ~7 of every 8 proposed
            // collisions are null. Sampling against a smaller sigmaBar lengthens
            // the flight and removes those proposals outright - it attacks the
            // tap count, which is the measured cost, rather than the cell count.
            //
            // The mean is FREE here: gvsLookupMajorantCached just ran
            // gvsBrickEnsure for this brick, so gGvsBrickMean is the same texel
            // already resident. Only mip 0 caches it, so coarser mips fall back
            // to the true majorant - still exact, just unweighted.
            //
            // This is deliberately NOT a bound any more. Correctness now rests
            // entirely on the caller's weight; see the header note.
            if (weightK < 1.f && mip == 0)
                majorant = max(lerp(densityScale * gGvsBrickMean, majorant, weightK), 1e-8f);
#if GRID_VOLUME_SAMPLER_STATS
            // [HOMOG] gauge: bin sigma_mean/sigma_majorant for THIS traversed
            // cell. At mip 0 the fetch above just populated gGvsBrickRange.x
            // (majorant) and gGvsBrickMean (mean) from ONE texel, so both are
            // current and free - no added memory traffic. Above mip 0 the cache
            // holds a different mip's mean, so skip; adaptive-mip climbs happen
            // only over empty space (majorant 0), which the >0 guard drops
            // anyway. densityScale cancels in the ratio, so raw texel values
            // suffice; mean <= majorant, saturate guards boundary slop.
            if (mip == 0 && gGvsBrickRange.x > 0.f)
            {
                const float hratio = saturate(gGvsBrickMean / gGvsBrickRange.x);
                gGvsHomogBins[min(uint(hratio * 8.f), 7u)]++;

                // [SUBHOMOG] REMOVED (2026-07-22). It answered its question -
                // majRatio 0.607 at 4^3, against a break-even of 4.0 on the
                // measured cell/tap exchange rate, so finer granularity loses -
                // and then became a cost in its own right: an uncached
                // subRangeTex.Load() on EVERY mip-0 DDA cell, ~80M per frame at
                // 2560x1351, into a 15 MB texture that does not stay in L2.
                // Harmless at 960x540 where the frame was latency-bound; not
                // harmless once VRAM throughput hit 88% of peak. A diagnostic
                // whose question is settled does not get to keep spending
                // bandwidth. See the handoff for the numbers it produced.
            }
#endif
            // Speculative next-brick prefetch (see gvsBrickPrefetch). The next
            // cell's coordinate is pure ALU from tExit - the SAME expression
            // the next iteration truncates - so its loads can be in flight
            // while THIS cell's majorant is still being waited on. Wasted only
            // when this cell collides or exits, and a wasted prefetch is dead
            // loads, not wrong data. Mip-0 only: that is where the cache (and
            // the thick-pixel cost) lives; higher mips fall through unchanged.
            if (mip == 0 && tExit < tFar)
                gvsBrickPrefetch(densityGrid, int3(ipos + idir * tExit) >> 3);
            const float dt = tExit - t;
            t = tExit;
            // Test for (tentative) collisions.
            const float dtau = dt * majorant;
            tauToGo -= dtau;
            if (adaptive) increaseAccessorLevel(accessor);
            if (tauToGo > 0.f) continue;
            // Revert to collision.
            t += dt * tauToGo / dtau;
            if (adaptive) decreaseAccessorLevel(accessor);
            return (t < tFar);
        }
        return false;
    }

    float evalTransmittanceRatioTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        const float densityScale = gridVolume.data.densityScale;
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        // The brick cache holds an atlas pointer valid for THIS grid only. Reset
        // at every walk entry so a brick cached during the previous instance's
        // walk can never be reused against a different grid's atlas - that would
        // be a silently wrong density, not a crash. See gvsBrickCacheReset.
        gvsBrickCacheReset();
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);
        const float3 ri = 1.f / idir;
        // Ratio tracking.
        float Tr = 1.f;
        float t = nearFar.x;
        while (1)
        {
            float tauToGo = -log(1.f - sampleNext1D(sg)), majorant;
            // weightK 1: this ratio tracker's weight is (1 - d/majorant) and it
            // has no compensation for a non-bounding majorant, so it must keep
            // the true bound.
            if (!stepToNextCollision(densityScale, densityGrid, ipos, idir, ri, tauToGo, t, nearFar.y, true, majorant, densityAccessor, 1.f)) return Tr;
            const float d = lookupDensityIndex(densityScale, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);
            Tr *= max(0.f, 1.f - d / majorant);
            // Russian roulette.
            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return Tr;
    }


    /** Residual ratio tracking over the mean pyramid (Novak et al. 2014;
        VNA spec section 2 / P2 "prefilter == control variate").

        Per DDA node at mip kResidualMip the density is split
            sigma = sigma_c + sigma_r,   sigma_c = node MEAN (rangeMeanTex.z)
        so transmittance factorises EXACTLY: T = T_c * T_r, where
            T_c = exp(-integral of sigma_c)   -> ANALYTIC (zero variance), and
            T_r                               -> ratio-tracked with the residual
        majorant sigmaRbar = max(majorant - mean, mean - minorant), which bounds
        |sigma_r| for every lookup in the node: the converter folds the 1-voxel
        halo into [minorant, majorant], so even stochastic-trilinear taps that
        cross the brick border stay inside the bound.

        Properties worth knowing:
         - Unbiased at EVERY mip: the mip only changes which sigma_c is used,
           i.e. variance, never the expectation. That is the whole P2 claim,
           and it is directly testable by sweeping the mip against FLIP.
         - Uniform and empty nodes have sigmaRbar == 0 and cost zero stochastic
           work: their transmittance is purely analytic.
         - The residual collision rate sigmaRbar <= majorant, so this never
           takes more density taps than plain ratio tracking on the same node;
           where the mean tracks the field well it takes far fewer.
        The mip can now be chosen PER CALL (runtime parameter, overload below):
        unbiasedness at every mip is exactly what makes a per-ray footprint-
        driven choice legal - the choice is deterministic geometry, independent
        of the random numbers, so it only moves variance/cost, never the mean.
        The compile-time-kResidualMip signature forwards here unchanged.
    */
    float evalTransmittanceResidualRatioTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        return evalTransmittanceResidualRatioTrackingLocalDDA(gridVolume, rayOrigin, rayDir, nearFar, sg, kResidualMip);
    }

    float evalTransmittanceResidualRatioTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, const int residualMip)
    {
        return evalTransmittanceResidualRatioTrackingLocalDDA(beginWalk(gridVolume, rayOrigin, rayDir), gridVolume, nearFar, sg, residualMip);
    }

    /** As above, on a WalkSetup the caller already built (see WalkSetup). Use
        this when several walks share one (grid, ray) segment; the wrapper above
        is the single-walk case and stays byte-identical to what it replaced.
    */
    float evalTransmittanceResidualRatioTrackingLocalDDA<S : ISampleGenerator>(const WalkSetup w, const GridVolume gridVolume, const float2 nearFar, inout S sg, const int residualMip)
    {
        const float densityScale = gridVolume.data.densityScale;
        // Aliases, so the loop below is textually unchanged from the version
        // that computed these itself. They are SSA copies, not extra state.
        const Grid densityGrid = w.densityGrid;
        const float3 ipos = w.ipos;
        const float3 idir = w.idir;
        const float3 ri = w.ri;
        DensityAccessor densityAccessor = createAccessor(densityGrid);
        const float scale = densityScale;

        float tauC = 0.f; ///< Accumulated analytic control optical depth: T_c = exp(-tauC).
        float Tr = 1.f;   ///< Stochastic residual transmittance.

#if GVS_BRICK_PREFETCH
        // Value-carry software pipeline for the per-node texel (run 137
        // lesson: v1 routed this fetch through the FULL cache entry, which
        // added ptr/occ loads to every no-tap node - hit rate 61.6% -> 46.4%,
        // gpuMs 26 -> 38. This version adds ZERO loads: still exactly one
        // rangeMean texel per node, just issued one node EARLY, so its
        // latency hides under the current node's residual tracking. The
        // carried value needs no key: it was loaded for ipos + idir*tExit,
        // which IS the next iteration's pos, bitwise.)
        float3 rmNext = float3(0.f);
        bool rmNextValid = false;
#endif
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            gvsStatCell();
            // One DDA node at the chosen mip. stepDDA's half-voxel dilation
            // guarantees tExit > t, so the walk always makes progress (same
            // invariant the stock local-majorant loop relies on).
            const float3 pos = ipos + idir * t;
            const float tExit = min(stepDDA(ipos, ri, pos, residualMip), nearFar.y);
            const float dt = tExit - t;

            logVolumeLookup();
            // ONE fetch for (majorant, minorant, mean) - these live in a single
            // RGBA16Float texel. Reading range and mean separately cost two
            // fetches per cell at the identical address, and cells are the
            // dominant term in this pass's memory traffic.
#if GVS_BRICK_PREFETCH
            const float3 majMinMean = rmNextValid ? rmNext : densityGrid.lookupIndexLocalRangeMeanTex(int3(pos), residualMip);
            rmNextValid = tExit < nearFar.y;
            if (rmNextValid)
                rmNext = densityGrid.lookupIndexLocalRangeMeanTex(int3(ipos + idir * tExit), residualMip);
#else
            const float3 majMinMean = densityGrid.lookupIndexLocalRangeMeanTex(int3(pos), residualMip);
#endif
            const float2 majMin = majMinMean.xy;
            const float mean = majMinMean.z;
            const float sigmaC = scale * mean;
            const float sigmaRbar = scale * max(majMin.x - mean, mean - majMin.y);

            // Control transmittance over this node: analytic, zero variance.
            tauC += sigmaC * dt;

            if (sigmaRbar > 0.f)
            {
                // Ratio-track the residual inside [t, tExit].
                const float invSigmaRbar = 1.f / sigmaRbar;
                float tt = t;
                while (true)
                {
                    tt -= log(1.f - sampleNext1D(sg)) * invSigmaRbar;
                    if (tt >= tExit) break;
                    const float d = lookupDensityIndex(densityScale, ipos + tt * idir, sampleNext3D(sg), densityGrid, densityAccessor);
                    // (d - sigmaC) / sigmaRbar is in [-1, 1] by the bound above,
                    // so the factor is in [0, 2]: Tr may legitimately exceed 1
                    // where the control overshoots (negative residual). That is
                    // the "prefilter is allowed to be loose" part of P2.
                    Tr *= 1.f - (d - sigmaC) * invSigmaRbar;
                }
            }
            t = tExit;

            // Russian roulette on the residual factor, same guard as the stock
            // ratio tracker. Only fires on small Tr; unbiased because the
            // survivor is reweighted by 1/Tr and the analytic factor is
            // deterministic.
            if (Tr < 0.1f)
            {
                if (sampleNext1D(sg) >= Tr) return 0.f;
                Tr = 1.f;
            }
        }
        return exp(-tauC) * Tr;
    }

    bool sampleDistanceDeltaTracking<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        const float densityScale = gridVolume.data.densityScale;
        ds = {};

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(gridVolume.getDensityGrid(), densityGrid);
        // The brick cache holds an atlas pointer valid for THIS grid only. Reset
        // at every walk entry so a brick cached during the previous instance's
        // walk can never be reused against a different grid's atlas - that would
        // be a silently wrong density, not a crash. See gvsBrickCacheReset.
        gvsBrickCacheReset();
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        float3 ipos, idir;
        transformRayToIndexSpace(gridVolume, densityGrid, rayOrigin, rayDir, ipos, idir);
        const float invMajorant = 1.f / (gridVolume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            t -= log(1.f - sampleNext1D(sg)) * invMajorant;
            const float d = lookupDensityIndex(densityScale, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);

            // Scatter on real collision.
            if (sampleNext1D(sg) < d * invMajorant)
            {
                ds.t = t;
                ds.thp = gridVolume.data.albedo;
                return true;
            }
        }

        return false;
    }

    bool sampleDistanceDeltaTrackingLocalDDA<S : ISampleGenerator>(const GridVolume gridVolume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        return sampleDistanceDeltaTrackingLocalDDA(beginWalk(gridVolume, rayOrigin, rayDir), gridVolume, nearFar, sg, ds);
    }

    /** As above, on a WalkSetup the caller already built (see WalkSetup).

        This is the overload that matters: evalEscapeAndCandidatesRQ calls it
        nProc times per brick with the same grid and ray, and every one of those
        calls used to redo the getGrid, the two mat-vecs, the reciprocal, and a
        brick-cache flush that guaranteed the second process missed on its first
        tap.
    */
    bool sampleDistanceDeltaTrackingLocalDDA<S : ISampleGenerator>(const WalkSetup w, const GridVolume gridVolume, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        ds = {};
        ds.thp = 1.f;

        // Aliases, so the loop below is textually unchanged from the version
        // that computed these itself. They are SSA copies, not extra state.
        const Grid densityGrid = w.densityGrid;
        const float3 ipos = w.ipos;
        const float3 idir = w.idir;
        const float3 ri = w.ri;
        DensityAccessor densityAccessor = createAccessor(densityGrid);

        // The only two fields this walk reads, extracted before the loop so the
        // GridVolume dies here rather than staying live across it. Harmless and
        // arguably clearer, but see below: it did NOT do what it was written
        // for, and the direction it belongs to is closed.
        //
        // ================= REGISTER / OCCUPANCY DIRECTION: REFUTED ==========
        // 2026-07-22. The theory was that this function's register footprint
        // caps occupancy, so narrowing it would raise resident warps and hide
        // the latency the SM idles on. Every link in that chain was measured.
        //
        // THE SETUP (Nsight, Bottom-Up "Live Registers"): this function was the
        // register high-water mark of the whole shader at 125 live registers.
        // 65536 registers/SM / 125 = 16.4 warps; measured CS warp occupancy was
        // 14.8 of 48 slots, warp launch register-limited 23.7% of cycles, SM
        // throughput 47% with 57.7% of warp slots unallocated. Textbook
        // occupancy-limited signature.
        //
        // THE CHANGE: stepToNextCollision and lookupDensityIndex took the whole
        // GridVolume by value to read ONE float (densityScale); albedo was read
        // after the loop, pinning the struct live across it. GridVolumeData
        // carries two float4x4s - 32 floats of matrices provably unreachable
        // from an index-space walk - plus bounds and flags, ~48 scalars. Both
        // callees now take a float; both fields are hoisted here.
        //
        // THE RESULT: nothing.
        //     sampleDistanceDeltaTrackingLocalDDA   125 -> 123 registers
        //     stepToNextCollision                   108 -> 108
        //     CS warp occupancy                     14.8 -> 15.3 warps
        //     wall clock                            +0.50 ms (log 203),
        //                                           floor 4.69 ms
        // Slang had ALREADY eliminated the unused struct fields. The 125 is the
        // DDA's genuine working set - ipos/idir/ri, the Grid handles, the brick
        // cache globals, accessor, RNG state, t/majorant - not passenger data.
        // main 95 -> 67 and evalEscapeAndCandidatesRQ 111 -> 89 did drop, but
        // occupancy is set by the WORST function, so those buy nothing.
        //
        // DO NOT re-run this experiment by narrowing more signatures. The
        // source-level argument ("48 scalars passed to read one float") was
        // about the source text, not about what the compiler emits, and the
        // compiler was already right. If occupancy is attacked again it has to
        // be by shrinking the real working set - which means restructuring the
        // walk, not tidying parameter lists.
        // ===================================================================
        const float densityScale = gridVolume.data.densityScale;
        const float3 albedo = gridVolume.data.albedo;

        float t = nearFar.x;
        while (1)
        {
            float tauToGo = -log(1.f - sampleNext1D(sg)), majorant;
            if (!stepToNextCollision(densityScale, densityGrid, ipos, idir, ri, tauToGo, t, nearFar.y, true, majorant, densityAccessor, gGvsDistWeightK)) return false;

            // WEIGHTED delta tracking (Galtier et al. 2013; Novak et al. 2018
            // STAR 5). The general form draws a real collision with an arbitrary
            // probability Pc and carries
            //     real: w *= d / (sigmaBar * Pc)
            //     null: w *= (sigmaBar - d) / (sigmaBar * (1 - Pc))
            // which is unbiased for ANY Pc in (0,1] and ANY sigmaBar > 0 - the
            // majorant does not have to bound the medium.
            //
            // Choosing Pc = min(d / sigmaBar, 1) collapses this to something
            // almost free:
            //   d <= sigmaBar : Pc = d/sigmaBar, and BOTH weights are exactly 1
            //                   - identical to the analog tracker, including the
            //                   comparison below, which is the same test.
            //   d >  sigmaBar : Pc = 1, so the collision is always real and the
            //                   weight is d/sigmaBar > 1. The null branch has
            //                   probability zero and never runs.
            // So there is no running weight to carry: null collisions are always
            // weight 1, and a single factor max(1, d/sigmaBar) lands on the real
            // collision. With gGvsDistWeightK == 1, sigmaBar is the true majorant,
            // d <= sigmaBar always, the factor is 1, and this is byte-identical
            // to the estimator it replaces.
            //
            // COST OF BEING WRONG-SIDED: that factor compounds across bounces,
            // bounded per event by majorant/sigmaBar_w (~8x at weightK 0). This
            // trades tap count for variance, so it must be gated on an image
            // comparison, not on the frame timer.
            const float d = lookupDensityIndex(densityScale, ipos + t * idir, sampleNext3D(sg), densityGrid, densityAccessor);
            if (sampleNext1D(sg) * majorant < d)
            {
                // Real collision, terminate.
                ds.t = t;
                ds.thp = albedo * max(1.f, d / majorant);
                return true;
            }
            // Null collision, continue with new sample.
        }
        return false;
    }
};
