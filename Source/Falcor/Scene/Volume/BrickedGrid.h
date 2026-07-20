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
#pragma once
#include "Core/API/Texture.h"
#include "Utils/Math/Vector.h"
#include <cstdint>
#include <vector>

namespace Falcor
{
    struct BrickedGrid
    {
        // Per-brick (majorant, minorant, MEAN, unused) as RGBA16Float, 4 mips.
        // The majorant/minorant pair bounds the brick; the mean is the
        // control-variate field for residual ratio tracking (VNA spec section 2
        // / P2). These were two textures (RG16Float range + R16Float mean) with
        // identical dimensions and mip layouts, which cost the residual walk TWO
        // texture fetches per DDA cell at the SAME address. Packed into one
        // texel they cost one: measured ~230M -> ~165M fetches/frame at 1080p.
        // NOTE: majorant/minorant and mean are different quantities - the .xy
        // mips are a max/min pyramid, the .z mip is a mean pyramid. Do not
        // conflate them.
        ref<Texture> rangeMean;
        ref<Texture> indirection;
        ref<Texture> atlas;

        // Per-brick occupancy bitmask, R32Uint, mip-0 brick dims. Bit t is set
        // if BC4 tile t (4x4x1 voxels, t = (vx>>2) | ((vy>>2)<<1) | (vz<<2))
        // can decode to anything other than exactly 0. A clear bit lets the
        // sampler return the brick minorant WITHOUT touching the atlas - the
        // dependent load that dominates the frame. See GridConverter.h for why
        // the granularity is a BC4 tile and not a 2^3 cell (finer granularity
        // would be biased under BC4).
        ref<Texture> occupancy;

        // CPU-side copies of the range/mean pyramids, kept after conversion.
        // Consumers (UE-lesson ports, see VNA_UE_SOURCE_LESSONS.md):
        //  - brick-AABB extraction for the HW-BVH: occupied bricks at a chosen
        //    mip become procedural primitives in a per-(grid,mip) BLAS, which
        //    is how LoD becomes a property of the acceleration structure;
        //  - the merged-coarse-tail bake, which needs conservative per-cell
        //    bounds of the SUM of instanced grids.
        // Layout matches the GPU texture exactly: cumulative 4-mip brick grid,
        // one uint64 per brick = f16 majorant | f16 minorant << 16 |
        // f16 mean << 32 | 0 << 48, i.e. the RGBA16Float texel bit-for-bit.
        // Coordinates are in SHIFTED index space (NanoVDB index minus
        // getMinIndex()), i.e. the space the sampler marches in.
        std::vector<uint64_t> rangeMeanData;
        int3 leafDim[4] = {};       ///< Brick-grid dimensions per mip (mip m cells are 8<<m voxels wide).
        uint32_t leafOffset[4] = {}; ///< Start of each mip's cells within rangeMeanData.
    };
}
