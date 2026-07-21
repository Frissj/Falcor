# Test wrapper: warp-aware residual roulette at 16 lanes (HANDOFF_9 5 redesign,
# GATE PENDING). Runs the real WdasSkyVNA.py config unchanged except for the two
# overrides below.
#   _RENDER_RES = None -> FULL window res (NOT the 960x540 Nubis path). The gate
#     must run at full res: the 0.228 shade occupancy the redesign targets and the
#     1080p references both live there, and 960x540 changes the warp divergence.
#   _WARP_RR_LANES = 16 -> moderate threshold; rouletttes residual paths when a
#     warp is <16/32 (half) occupied.
# Score the result with VNA_RadCacheScore.py against the baseline (warp-RR off =
# WdasSkyVNA.py at full res) BEFORE trusting any speedup.
_RENDER_RES = None
_WARP_RR_LANES = 16
exec(compile(open(r'C:\Users\Friss\Documents\Clouds\Falcor\scripts\WdasSkyVNA.py', encoding='utf-8').read(), 'WdasSkyVNA.py', 'exec'))
