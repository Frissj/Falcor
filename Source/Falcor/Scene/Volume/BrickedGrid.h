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
        ref<Texture> range;
        ref<Texture> indirection;
        ref<Texture> atlas;
        // Per-brick MEAN density, same brick grid and 4-mip layout as 'range'.
        // The control-variate field for residual ratio tracking (VNA spec
        // section 2 / P2). NOTE: 'range' stores majorant/minorant (max/min) -
        // a different quantity; the mips of 'range' are NOT a mean pyramid.
        ref<Texture> mean;

        // CPU-side copies of the range/mean pyramids, kept after conversion.
        // Consumers (UE-lesson ports, see VNA_UE_SOURCE_LESSONS.md):
        //  - brick-AABB extraction for the HW-BVH: occupied bricks at a chosen
        //    mip become procedural primitives in a per-(grid,mip) BLAS, which
        //    is how LoD becomes a property of the acceleration structure;
        //  - the merged-coarse-tail bake, which needs conservative per-cell
        //    bounds of the SUM of instanced grids.
        // Layout matches the GPU textures exactly: cumulative 4-mip brick grid,
        // rangeData = f16 majorant | f16 minorant << 16, meanData = f16 mean.
        // Coordinates are in SHIFTED index space (NanoVDB index minus
        // getMinIndex()), i.e. the space the sampler marches in.
        std::vector<uint32_t> rangeData;
        std::vector<uint16_t> meanData;
        int3 leafDim[4] = {};       ///< Brick-grid dimensions per mip (mip m cells are 8<<m voxels wide).
        uint32_t leafOffset[4] = {}; ///< Start of each mip's cells within rangeData/meanData.
    };
}
