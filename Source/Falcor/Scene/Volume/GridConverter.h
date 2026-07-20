/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "BrickedGrid.h"
#include "BC4Encode.h"
#include "Core/API/Device.h"
#include "Core/API/Formats.h"
#include "Utils/Logger.h"
#include "Utils/HostDeviceShared.slangh"
#include "Utils/NumericRange.h"
#include "Utils/Math/Vector.h"
#include "Utils/Timing/CpuTimer.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif
#include <nanovdb/NanoVDB.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <execution>
#include <vector>

namespace Falcor
{
    template <typename TexelType, unsigned int kBitsPerTexel> struct NanoVDBToBricksConverter;
    using NanoVDBConverterBC4 = NanoVDBToBricksConverter<uint64_t, 4>;
    using NanoVDBConverterUNORM8 = NanoVDBToBricksConverter<uint8_t, 8>;
    using NanoVDBConverterUNORM16 = NanoVDBToBricksConverter<uint16_t, 16>;

    template <typename TexelType, unsigned int kBitsPerTexel>
    struct NanoVDBToBricksConverter
    {
    public:
        NanoVDBToBricksConverter(const nanovdb::FloatGrid* grid);
        NanoVDBToBricksConverter(const NanoVDBToBricksConverter& rhs) = delete;

        BrickedGrid convert(ref<Device> pDevice);

    private:
        const static uint32_t kBrickSize = 8; // Must be 8, to match both NanoVDB leaf size.
        const static int32_t kBC4Compress = kBitsPerTexel == 4;

        void convertSlice(int z);
        void computeMip(int mip);

        inline uint3 getAtlasSizeBricks() const { return mAtlasSizeBricks; }
        inline uint3 getAtlasSizePixels() const { return mAtlasSizeBricks * kBrickSize; }
        inline uint32_t getAtlasMaxBrick() const { return mAtlasSizeBricks.x * mAtlasSizeBricks.y * mAtlasSizeBricks.z; }

        inline ResourceFormat getAtlasFormat() {
            switch (kBitsPerTexel) {
            case 4: return ResourceFormat::BC4Unorm;
            case 8: return ResourceFormat::R8Unorm;
            case 16: return ResourceFormat::R16Unorm;
            default: FALCOR_THROW("Unsupported bitdepth in NanoVDBToBricksConverter");
            }
        }

        inline float2 combineMajMin(float2 a, float2 b)
        {
            return float2(std::max(a.x, b.x), std::min(a.y, b.y));
        }

        inline float2 unpackMajMin(const uint64_t* data)
        {
            const uint16_t* data16 = (const uint16_t*)data;
            return float2(f16tof32(data16[0]), f16tof32(data16[1]));
        }

        inline void expandMinorantMajorant(float value, float& min_inout, float& maj_inout)
        {
            if (value < min_inout) min_inout = value;
            if (value > maj_inout) maj_inout = value;
        }

        const nanovdb::FloatGrid* mpFloatGrid;
        uint3 mAtlasSizeBricks;
        int3 mLeafDim[4];
        int3 mBBMin, mBBMax, mPixDim;
        uint32_t mLeafCount[4];
        // Per-brick (majorant, minorant, mean, unused) packed as 4x f16 = the
        // RGBA16Float texel bit-for-bit. The mean is the control-variate field
        // sigma_c for residual ratio tracking (VNA spec section 2 / P2); it is a
        // MEAN pyramid, while .xy's mips are a max/min pyramid - a different
        // quantity that cannot substitute for it. These were two arrays feeding
        // two textures; the residual walk read both at the SAME address every
        // DDA cell, so packing them halves that fetch.
        std::vector<uint64_t> mRangeMeanData;
        std::vector<uint32_t> mPtrData;
        std::vector<TexelType> mAtlasData;

        // Per-brick occupancy bitmask, one uint32 per mip-0 brick (UE Nanite
        // Foliage lesson, HANDOFF_6 6.3 - their FBrick carries a 64-bit mask and
        // steps DDA in registers with zero memory traffic).
        //
        // A bit is 1 if that group of voxels can decode to anything other than
        // exactly 0. Granularity is deliberately ONE BC4 TILE (4x4x1), giving
        // 2*2*8 = 32 tiles per 8^3 brick, so the whole mask is a single uint32
        // that lives in a register at runtime.
        //
        // WHY TILE GRANULARITY AND NOT 2^3 CELLS: the atlas is BC4-compressed
        // (Grid.cpp uses NanoVDBConverterBC4). BC4 fits endpoints across all 16
        // texels of a 4x4 tile, so a sub-cell that straddles tiles can decode to
        // a nonzero value even when its own source voxels were empty - masking
        // at that granularity would skip taps that should have returned more
        // than the minorant, i.e. a systematic BIAS. At tile granularity the
        // test is exact by construction: all 16 quantized texels 0 => both BC4
        // endpoints 0 => every texel decodes to exactly 0 => the sampler's
        // 0 * (majorant - minorant) + minorant is exactly the minorant, which
        // the brick cache already holds. Skipping is bit-identical, not an
        // approximation. Do not "improve" this to a finer granularity without
        // decoding the compressed block first.
        std::vector<uint32_t> mOccupancyData;
        std::atomic_uint32_t mNonEmptyCount;

        /** Pack one brick's (majorant, minorant, mean) into an RGBA16Float texel. */
        static inline uint64_t packRangeMean(float majorant, float minorant, float mean)
        {
            return (uint64_t)f32tof16(majorant)
                | ((uint64_t)f32tof16(minorant) << 16)
                | ((uint64_t)f32tof16(mean) << 32);
        }
        static inline float unpackMajorant(uint64_t v) { return f16tof32((uint32_t)(v & 0xffff)); }
        static inline float unpackMinorant(uint64_t v) { return f16tof32((uint32_t)((v >> 16) & 0xffff)); }
        static inline float unpackMean(uint64_t v)     { return f16tof32((uint32_t)((v >> 32) & 0xffff)); }
    };

    template <typename TexelType, unsigned int kBitsPerTexel>
    NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::NanoVDBToBricksConverter(const nanovdb::FloatGrid* grid)
    {
        mNonEmptyCount.store(0);
        mpFloatGrid = grid;
        auto& voxelbox = mpFloatGrid->indexBBox();
        mBBMin = (int3(voxelbox.min().x(), voxelbox.min().y(), voxelbox.min().z())) & (~7);
        mBBMax = (int3(voxelbox.max().x(), voxelbox.max().y(), voxelbox.max().z()) + 7) & (~7);
        mPixDim = mBBMax - mBBMin;
        mPixDim = (mPixDim + 63) & ~63; // Since we compute 4 mipmaps, the coarsest mip corresponds to 8*2^3 = 64 leaf voxels.
        for (uint i = 0; i < 4; ++i)
        {
            mLeafDim[i] = mPixDim / (8 << i);
            mLeafCount[i] = (mLeafDim[i].x * mLeafDim[i].y * mLeafDim[i].z) + (i ? mLeafCount[i - 1] : 0); // Cumulative leaf count up the mips.
        }
        uint leafCount = grid->tree().nodeCount(0);
        uint approxdim = 1u << uint(log2f((float)leafCount + 1.f) / 3.f); // Choose the first 2 dimensions to be powers of 2.
        uint lastdim = (leafCount + approxdim * approxdim - 1) / (approxdim * approxdim);
        mAtlasSizeBricks = uint3(approxdim, approxdim, lastdim);
        uint3 atlasSizePixels = getAtlasSizePixels();
        uint leafTexelCount = atlasSizePixels.x * atlasSizePixels.y * atlasSizePixels.z;
        mRangeMeanData.resize(mLeafCount[3]);
        mPtrData.resize(mLeafCount[0]);
        mOccupancyData.resize(mLeafCount[0]); // Zero = fully empty; filled per non-empty brick below.
        mAtlasData.resize(kBC4Compress ? (leafTexelCount / 16) : leafTexelCount);
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    void NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::convertSlice(int z)
    {
        uint3 atlasSizePixels = getAtlasSizePixels();
        uint brickMax = getAtlasMaxBrick();
        uint bricksPerSlice = mAtlasSizeBricks.x * mAtlasSizeBricks.y;
        uint pixelsPerSlice = atlasSizePixels.x * atlasSizePixels.y;

        size_t offset = z * mLeafDim[0].x * mLeafDim[0].y;
        uint64_t* rangemeandst = mRangeMeanData.data() + offset;
        uint32_t* ptrdst = mPtrData.data() + offset;
        uint32_t* occdst = mOccupancyData.data() + offset;
        auto a = mpFloatGrid->getAccessor();
        for (int y = 0; y < mLeafDim[0].y; ++y)
        {
            for (int x = 0; x < mLeafDim[0].x; ++x)
            {
                nanovdb::Coord ijk = { x * 8 + mBBMin.x, y * 8 + mBBMin.y, z * 8 + mBBMin.z };
                auto val = a.getValue(ijk);
                auto leaf = a.probeLeaf(ijk);
                float minorant = val, majorant = val;
                double valueSum = 0.0; // Mean of the brick's 8x8x8 voxels (control variate, VNA spec section 2).
                uint myleaf = 0;
                if (leaf)
                {
                    // Nanovdb only stores minorant/majorant for active voxels, but we need all of them... Grab the central 8x8x8 first the quick way.
                    const float* data = leaf->data()->mValues;
                    for (int i = 0; i < kBrickSize * kBrickSize * kBrickSize; ++i)
                    {
                        expandMinorantMajorant(data[i], minorant, majorant);
                        valueSum += data[i];
                    }
                    // We also need the 1-halo from neighbouring bricks. Fetch them in an order that maximises nanovdb's internal cache reuse.
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, j, kBrickSize)), minorant, majorant);
                    for (int j = 0; j < kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, -1, j)), minorant, majorant);
                    for (int j = 0; j < kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(i, kBrickSize, j)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, i)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) for (int i = 0; i < kBrickSize; ++i) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, i)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, -1)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(-1, j, kBrickSize)), minorant, majorant);
                    for (int j = -1; j <= kBrickSize; ++j) expandMinorantMajorant(a.getValue(ijk + nanovdb::Coord(kBrickSize, j, kBrickSize)), minorant, majorant);

                    if (minorant != majorant) myleaf = mNonEmptyCount.fetch_add(1);
                }
                if (majorant == minorant || myleaf >= brickMax || leaf == nullptr)
                {
                    // Force identical major and minor. With a collapsed range the
                    // sampler returns exactly 'majorant' everywhere in this brick,
                    // so that IS the runtime mean. Writing it makes the residual
                    // identically zero here: transmittance through uniform/empty
                    // bricks becomes purely analytic.
                    *rangemeandst++ = packRangeMean(majorant, majorant, majorant);
                    *ptrdst++ = 0;
                    // Collapsed range: the sampler returns exactly 'majorant'
                    // everywhere here regardless of what the atlas holds, so an
                    // all-zero mask makes every tap in this brick skip the atlas
                    // read and return the cached minorant (== majorant). Exact.
                    *occdst++ = 0;
                }
                else
                {
                    // True mean of the brick's voxels. Any value in [minorant,majorant]
                    // keeps residual tracking unbiased (control variate); the mean
                    // minimises the residual and thus the variance. Computed from the
                    // ORIGINAL voxel sum, before the majorant/minorant f16 nudge
                    // below - keep this order.
                    const float brickMean = float(valueSum * (1.0 / (kBrickSize * kBrickSize * kBrickSize)));
                    const float* data = leaf->data()->mValues;
                    majorant = f16tof32(f32tof16(majorant) + 1);
                    minorant = f16tof32(f32tof16(minorant));
                    *rangemeandst++ = packRangeMean(majorant, minorant, brickMean);
                    uint32_t atlasx = myleaf % mAtlasSizeBricks.x;
                    uint32_t atlasy = (myleaf / mAtlasSizeBricks.x) % mAtlasSizeBricks.y;
                    uint32_t atlasz = myleaf / bricksPerSlice;
                    *ptrdst++ = (atlasx + (atlasy << 8) + (atlasz << 16));

                    // Occupancy bits for this brick, indexed the same way the
                    // sampler will: tile = (vx>>2) | ((vy>>2)<<1) | (vz<<2),
                    // where (vx,vy,vz) = vox & 7.
                    uint32_t occ = 0;

                    if (!kBC4Compress) {
                        float invRange = ((1 << kBitsPerTexel) - 1.f) / (majorant - minorant);
                        TexelType* atlasdst = (TexelType*)mAtlasData.data() + atlasx * kBrickSize + atlasy * (atlasSizePixels.x * kBrickSize) + atlasz * (pixelsPerSlice * kBrickSize);
                        for (int pixz = 0; pixz < kBrickSize; ++pixz)
                        {
                            for (int pixy = 0; pixy < kBrickSize; ++pixy)
                            {
                                for (int pixx = 0; pixx < kBrickSize; ++pixx)
                                {
                                    float f = data[pixx * kBrickSize * kBrickSize + pixy * kBrickSize + pixz];
                                    const TexelType texel = TexelType((f - minorant) * invRange);
                                    // Same tile granularity as the BC4 path so
                                    // the runtime test is identical either way.
                                    if (texel != TexelType(0))
                                        occ |= 1u << ((pixx >> 2) | ((pixy >> 2) << 1) | (pixz << 2));
                                    *atlasdst++ = texel;
                                }
                                atlasdst += (atlasSizePixels.x - kBrickSize); // next scanline
                            }
                            atlasdst += (pixelsPerSlice - (atlasSizePixels.x * kBrickSize)); // next slice
                        }
                    }
                    else {
                        // BC4 compression:
                        float invRange = (255.f) / (majorant - minorant);
                        uint64_t* atlasdst = ((uint64_t*)mAtlasData.data() + atlasx * (kBrickSize / 4) + atlasy * ((atlasSizePixels.x / 4) * kBrickSize / 4) + atlasz * (pixelsPerSlice / 16 * kBrickSize));
                        for (int pixz = 0; pixz < kBrickSize; ++pixz)
                        {
                            for (int tiley = 0; tiley < kBrickSize; tiley += 4)
                            {
                                for (int tilex = 0; tilex < kBrickSize; tilex += 4) {
                                    uint8_t tilevals[4][4];
                                    uint8_t tileminorant = 255, tilemajorant = 0;
                                    for (int pixy = 0; pixy < 4; ++pixy)
                                    {
                                        for (int pixx = 0; pixx < 4; ++pixx)
                                        {
                                            float f = data[(pixx + tilex) * (kBrickSize * kBrickSize) + (pixy + tiley) * kBrickSize + pixz];
                                            uint8_t voxel = uint8_t((f - minorant) * invRange);
                                            tileminorant = std::min(tileminorant, voxel);
                                            tilemajorant = std::max(tilemajorant, voxel);
                                            tilevals[pixy][pixx] = voxel;
                                        }
                                    }
                                    // tilemajorant == 0 means all 16 quantized
                                    // texels are 0, so BC4 emits both endpoints
                                    // as 0 and every texel decodes to exactly 0.
                                    // Anything else must keep its bit set.
                                    if (tilemajorant != 0)
                                        occ |= 1u << ((tilex >> 2) | ((tiley >> 2) << 1) | (pixz << 2));

                                    CompressAlphaDxt5((uint8_t*)&tilevals[0][0], atlasdst);
                                    atlasdst++;
                                }
                                atlasdst += (atlasSizePixels.x / 4 - kBrickSize / 4); // next scanline
                            }
                            atlasdst += (pixelsPerSlice / 16 - (atlasSizePixels.x / 4 * kBrickSize / 4)); // next slice
                        } // z slice loop
                    } // bc4 compress?

                    *occdst++ = occ;
                } // non empty brick?
            } // x brick loop
        } // y brick loop
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    void NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::computeMip(int mip)
    {
        // One interleaved pyramid now: .xy reduces as max/min, .z as a mean.
        // Children are equal-size regions, so the average of the 8 child means
        // IS the exact mean of the parent region - the mean pyramid is exact at
        // every level, not an estimate.
        uint64_t* rangemeandst = mRangeMeanData.data() + mLeafCount[mip - 1];
        uint64_t* rangemeansrc = mRangeMeanData.data() + ((mip > 1) ? mLeafCount[mip - 2] : 0);
        int3 leafdim_src = mLeafDim[mip - 1];
        uint32_t rowstride_src = leafdim_src.x;
        uint32_t slicestride_src = leafdim_src.y * rowstride_src;

        int3 leafdim_tgt = mLeafDim[mip];
        uint32_t rowstride_tgt = leafdim_tgt.x;
        uint32_t slicestride_tgt = leafdim_tgt.y * rowstride_tgt;

        for (int z = 0; z < leafdim_tgt.z; ++z, rangemeansrc += slicestride_src)
        {
            for (int y = 0; y < leafdim_tgt.y; ++y, rangemeansrc += rowstride_src)
            {
                for (int x = 0; x < leafdim_tgt.x; ++x, rangemeansrc += 2)
                {
                    float2 majmin_dst = combineMajMin(
                        combineMajMin(
                            combineMajMin(unpackMajMin(rangemeansrc), unpackMajMin(rangemeansrc + 1)),
                            combineMajMin(unpackMajMin(rangemeansrc + rowstride_src), unpackMajMin(rangemeansrc + 1 + rowstride_src))
                        ),
                        combineMajMin(
                            combineMajMin(unpackMajMin(rangemeansrc + slicestride_src), unpackMajMin(rangemeansrc + slicestride_src + 1)),
                            combineMajMin(unpackMajMin(rangemeansrc + slicestride_src + rowstride_src), unpackMajMin(rangemeansrc + slicestride_src + 1 + rowstride_src))
                        )
                    );

                    const float meanSum =
                        unpackMean(rangemeansrc[0]) + unpackMean(rangemeansrc[1]) +
                        unpackMean(rangemeansrc[rowstride_src]) + unpackMean(rangemeansrc[1 + rowstride_src]) +
                        unpackMean(rangemeansrc[slicestride_src]) + unpackMean(rangemeansrc[slicestride_src + 1]) +
                        unpackMean(rangemeansrc[slicestride_src + rowstride_src]) + unpackMean(rangemeansrc[slicestride_src + 1 + rowstride_src]);

                    *rangemeandst++ = packRangeMean(majmin_dst.x, majmin_dst.y, meanSum * 0.125f);
                } // x
            } // y
        } // z
    }

    template <typename TexelType, unsigned int kBitsPerTexel>
    BrickedGrid NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::convert(ref<Device> pDevice)
    {
        auto t0 = CpuTimer::getCurrentTimePoint();
        auto range = NumericRange<int>(0, mLeafDim[0].z);
        std::for_each(std::execution::par, range.begin(), range.end(), [&](int z) { convertSlice(z); });
        for (int mip = 1; mip < 4; ++mip) computeMip(mip);

        BrickedGrid bricks;
        bricks.rangeMean = pDevice->createTexture3D(mLeafDim[0].x, mLeafDim[0].y, mLeafDim[0].z, ResourceFormat::RGBA16Float, 4, mRangeMeanData.data(), ResourceBindFlags::ShaderResource);
        bricks.indirection = pDevice->createTexture3D(mLeafDim[0].x, mLeafDim[0].y, mLeafDim[0].z, ResourceFormat::RGBA8Uint, 1, mPtrData.data(), ResourceBindFlags::ShaderResource);
        bricks.occupancy = pDevice->createTexture3D(mLeafDim[0].x, mLeafDim[0].y, mLeafDim[0].z, ResourceFormat::R32Uint, 1, mOccupancyData.data(), ResourceBindFlags::ShaderResource);
        bricks.atlas = pDevice->createTexture3D(getAtlasSizePixels().x, getAtlasSizePixels().y, getAtlasSizePixels().z, getAtlasFormat(), 1, mAtlasData.data(), ResourceBindFlags::ShaderResource);

        // Hand the CPU pyramids to the BrickedGrid (moved, not copied - the
        // converter is done with them). Needed by the HW-BVH brick-AABB
        // extraction and the merged-tail bake; see BrickedGrid.h.
        for (uint32_t i = 0; i < 4; ++i)
        {
            bricks.leafDim[i] = mLeafDim[i];
            bricks.leafOffset[i] = i ? mLeafCount[i - 1] : 0;
        }
        bricks.rangeMeanData = std::move(mRangeMeanData);

        double dt = CpuTimer::calcDuration(t0, CpuTimer::getCurrentTimePoint());
        logDebug("Converted '{}' in {:.4}ms: mNonEmptyCount {} vs max {}", mpFloatGrid->gridName(), dt, mNonEmptyCount.load(), getAtlasMaxBrick());
        return bricks;
    }
}
