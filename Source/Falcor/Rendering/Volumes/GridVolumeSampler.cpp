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
#include "GridVolumeSampler.h"
#include "Core/Error.h"

namespace Falcor
{
    GridVolumeSampler::GridVolumeSampler(RenderContext* pRenderContext, ref<IScene> pScene, const Options& options)
        : mpScene(pScene)
        , mOptions(options)
    {
        FALCOR_ASSERT(pScene);
    }

    DefineList GridVolumeSampler::getDefines() const
    {
        DefineList defines;
        defines.add("GRID_VOLUME_SAMPLER_USE_BRICKEDGRID", std::to_string((uint32_t)mOptions.useBrickedGrid));
        defines.add("GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR", std::to_string((uint32_t)mOptions.transmittanceEstimator));
        defines.add("GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER", std::to_string((uint32_t)mOptions.distanceSampler));
        defines.add("GRID_VOLUME_SAMPLER_RESIDUAL_MIP", std::to_string(mOptions.residualMip));
        return defines;
    }

    void GridVolumeSampler::bindShaderData(const ShaderVar& var) const
    {
        FALCOR_ASSERT(var.isValid());
    }

    bool GridVolumeSampler::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        if (widget.checkbox("Use BrickedGrid", mOptions.useBrickedGrid))
        {
            if (!mOptions.useBrickedGrid) {
                // Switch back to modes not requiring bricked grid.
                if (requiresBrickedGrid(mOptions.transmittanceEstimator)) mOptions.transmittanceEstimator = TransmittanceEstimator::RatioTracking;
                if (requiresBrickedGrid(mOptions.distanceSampler)) mOptions.distanceSampler = DistanceSampler::DeltaTracking;
            }
            dirty = true;
        }
        if (widget.dropdown("Transmittance Estimator", mOptions.transmittanceEstimator))
        {
            // Enable bricked grid if the chosen mode requires it.
            if (requiresBrickedGrid(mOptions.transmittanceEstimator)) mOptions.useBrickedGrid = true;
            dirty = true;
        }
        if (widget.dropdown("Distance Sampler", mOptions.distanceSampler))
        {
            // Enable bricked grid if the chosen mode requires it.
            if (requiresBrickedGrid(mOptions.distanceSampler)) mOptions.useBrickedGrid = true;
            dirty = true;
        }
        if (mOptions.transmittanceEstimator == TransmittanceEstimator::ResidualRatioTrackingLocalMajorant)
        {
            // Mip of the control-variate (mean) field. Unbiased at every level;
            // sweeping it should change noise/speed but NOT the converged image.
            dirty |= widget.var("Residual mip", mOptions.residualMip, 0u, 3u);
            widget.tooltip("Mean-pyramid level for residual ratio tracking.\n0 = per-brick (8^3) mean, 3 = coarsest (64^3).\nThe converged image must be identical at every level.", true);
        }

        return dirty;
    }
}
