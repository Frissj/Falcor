# Test wrapper: frame-gated cache amortization, 1 correction frame in 8. Runs the
# real WdasSkyVNA.py config unchanged except for the two overrides below.
#
#   _RENDER_RES = (960, 540) is inherited (the daily Nubis-comparable path) - this
#     lever's whole point is frame time at ship resolution, and the gate that
#     priced it (log 164) ran there, so keep them comparable.
#   _AMORT_PERIOD = 8 -> frames 0, 8, 16... track the residual and train; the
#     other 7 in 8 consume the cache and die at the cut.
#
# WHAT TO LOOK AT, IN ORDER:
#  1. [PATHLEN] must alternate between [CORRECTION] and [consume]. The driver
#     now switches risStatsInterval to 63 whenever _AMORT_PERIOD is nonzero,
#     because 64 is a multiple of every sane period and would log only
#     correction frames - making a working lever look inert.
#  2. A consume frame's [PATHLEN] must be ~97% raw-uncut at mean ~= radCutBounce
#     with 0% stragglers, and cut/resid ~0. That is the mechanism; the gate
#     measured it at 63,200 of 64,661 paths.
#  3. THEN the frame time. Expect ~3.4 ms vs ship's ~3.9 at 960x540.
#
# AND THE PART THE TIMING CANNOT TELL YOU: this mode is CONSISTENT, not unbiased.
# Consume frames render the cache's mean, so any cache error shows up as image
# error. Compare against a full-res unbiased reference before believing it, and
# watch specifically for a periodic brightness ripple at the amortization period
# - correction and consume frames render different estimators, and until the
# cache converges the temporal accumulator is blending two different answers.
_AMORT_PERIOD = 8
exec(compile(open(r'C:\Users\Friss\Documents\Clouds\Falcor\scripts\WdasSkyVNA.py', encoding='utf-8').read(), 'WdasSkyVNA.py', 'exec'))
