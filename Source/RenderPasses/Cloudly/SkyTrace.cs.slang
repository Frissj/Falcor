/***************************************************************************
 # Cloud raymarch (counterpart of Cloudly's SkyTraceCS).
 # Runs at 1/traceDownscale resolution; each frame traces a different full-res
 # pixel inside every block and TemporalUpscale fills in the rest.
 #
 # Output: color    = cloud radiance incl. aerial perspective (rgb) + view transmittance (a)
 #         distance = transmittance-weighted cloud distance, world units
 **************************************************************************/
import RenderPasses.Cloudly.CloudlyCommon;

Texture2D<float> gDepth; ///< Hardware depth (optional, 1 = background).
RWTexture2D<float4> gOutColor;
RWTexture2D<float> gOutDistance;

cbuffer TraceCB
{
    uint2 gFrameDim; ///< Full resolution.
    uint2 gSubPixel; ///< Full-res pixel traced inside each block this frame.
    uint gHasDepth;
}

struct MarchState
{
    float3 Lsun;  ///< Accumulated sun scattering (before sunIntensity).
    float Lamb;   ///< Accumulated ambient scattering weight.
    float T;      ///< View transmittance.
    float distSum;
    float distWeight;
    uint steps;
    bool done;
};

/// Smallest step length at a traced distance (TraceMinStepSmallest / _Far, MinStepSmallestFar_Alpha, near ramp).
float minStepLength(float tracedDistance)
{
    float farRatio = exp2Decay(tracedDistance, gParams.minStepSmallestFarAlpha);
    float minStep = lerp(gParams.traceMinStepSmallest, gParams.traceMinStepSmallestFar, farRatio);
    float nearRatio = linearstep(0.f, 1000.f, tracedDistance);
    return lerp(lerp(8.f, 18.4f, nearRatio), minStep, nearRatio);
}

void marchSegment(inout MarchState st, float3 ro, float3 rd, float2 seg, float3 sunDir, float cosTheta, float jitter, bool cirrusSegment)
{
    if (st.done || seg.y <= seg.x)
        return;

    float bailOD = gParams.cloudBailOutTransmittanceLogViewRay;
    float segLen = seg.y - seg.x;
    float t = seg.x;
    float step = minStepLength(t);
    t += step * jitter;
    float golden = jitter;

    while (t < seg.y && st.steps < gParams.traceMaxSteps)
    {
        st.steps++;
        golden = frac(golden + kGolden);
        float3 p = ro + rd * t;
        float fade = cloudDistanceFade(t);
        CloudEval e = evalCloud(p, t, fade, false);

        float minStep = minStepLength(t);
        float d = allSdf(e);
        // SDF-driven stepping (SDF_Factor 0.45) clamped to the smallest step; jittered to hide banding.
        float stepLen = max(minStep, abs(d) * 0.45f);
        if (e.density > 0.f)
            stepLen = min(stepLen, 4.f * minStep);
        else if (gParams.useGameVolumes == 0)
            stepLen = min(stepLen, max(8.f * minStep, 0.02f * t)); // The procedural pseudo-SDF is not a distance bound.
        if (cirrusSegment)
            stepLen = min(stepLen, max(minStep, segLen * 0.25f));
        stepLen *= 0.75f + 0.5f * golden;
        float dt = min(stepLen, seg.y - t);

        if (e.density > 0.f)
        {
            float sigma = e.density * gParams.sigmaScatCloud;
            float h = altitude(p);
            float od = sunRayOpticalDepth(p, sunDir, e, t, golden);
            float3 sunColor = sunTransmittance(p, sunDir);
            float Ssun = sunScattering(e, od, cosTheta, h);
            float Samb = sigma * ambientHeightAO(h);

            float stepT = exp(-sigma * dt);
            float integ = (1.f - stepT) / sigma; // Energy-conserving integration over the step.
            st.Lsun += st.T * Ssun * sunColor * integ;
            st.Lamb += st.T * Samb * integ;

            float prevT = st.T;
            st.T *= stepT;
            st.distSum += t * (prevT - st.T);
            st.distWeight += prevT - st.T;

            if (-log(max(st.T, 1e-12f)) > bailOD)
            {
                st.T = 0.f;
                st.done = true;
                return;
            }
        }
        t += dt;
    }
}

[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.traceDim))
        return;

    uint2 fullPixel = min(id.xy * gParams.traceDownscale + gSubPixel, gFrameDim - 1);
    float2 pixelPos = float2(fullPixel) + 0.5f;
    float3 rd = viewDirForPixel(pixelPos, float2(gFrameDim));

    float sceneDist = kInfinity;
    if (gHasDepth != 0)
        sceneDist = sceneDistanceFromDepth(gDepth[fullPixel], pixelPos, float2(gFrameDim));
    float sceneDistM = sceneDist >= kInfinity ? kInfinity : worldToMeters(sceneDist);

    float3 ro = cameraPlanetPos();
    float3 sunDir = normalize(gParams.lightSourceDirection);
    float cosTheta = dot(rd, sunDir);
    float jitter = temporalJitter(fullPixel, gParams.frameIndex);

    float baseAltoTop = gParams.useGameVolumes != 0
        ? gParams.cloudMaxHeightForBaseAltoTrace
        : min(gParams.cloudMaxHeightForBaseAltoTrace,
              max(gParams.cloudLayersStartHeight.x + gParams.layerThickness.x, gParams.cloudLayersStartHeight.y + gParams.layerThickness.y));
    float2 segBA = rayShell(ro, rd, rBot() + gParams.cloudLayersStartHeight.x, rBot() + baseAltoTop, min(gParams.cloudMaxTraceDistance, sceneDistM));
    float2 segCi = rayShell(ro, rd, rBot() + gParams.cloudLayersStartHeight.z, rBot() + gParams.cloudLayersStartHeight.z + gParams.cirrusThickness,
                            min(gParams.cloudMaxTraceDistanceCirrus, sceneDistM));

    MarchState st;
    st.Lsun = float3(0.f);
    st.Lamb = 0.f;
    st.T = 1.f;
    st.distSum = 0.f;
    st.distWeight = 0.f;
    st.steps = 0;
    st.done = false;

    // March the segments front to back.
    if (segBA.x <= segCi.x || segCi.y <= segCi.x)
    {
        marchSegment(st, ro, rd, segBA, sunDir, cosTheta, jitter, false);
        marchSegment(st, ro, rd, segCi, sunDir, cosTheta, jitter, true);
    }
    else
    {
        marchSegment(st, ro, rd, segCi, sunDir, cosTheta, jitter, true);
        marchSegment(st, ro, rd, segBA, sunDir, cosTheta, jitter, false);
    }

    float3 L = st.Lsun * gParams.sunIntensity + st.Lamb * gParams.cloudSkyAmbientLightIntensity * cloudAmbientLight() * (1.f / (4.f * kPI));

    float cloudDist;
    if (st.distWeight > 1e-5f)
        cloudDist = st.distSum / st.distWeight;
    else if (segBA.y > segBA.x)
        cloudDist = segBA.x;
    else
        cloudDist = min(gParams.cloudMaxTraceDistance, sceneDistM);

    // Aerial perspective between the camera and the clouds (starts at AerialPerspectiveStartDistance).
    if (st.T < 1.f)
    {
        float tAP = max(0.f, cloudDist - gParams.aerialPerspectiveStartDistance);
        float3 roAP = ro + rd * min(gParams.aerialPerspectiveStartDistance, cloudDist);
        SkyResult ap = integrateSky(roAP, rd, sunDir, tAP, 16, true, false, jitter);
        L = L * ap.transmittance + ap.L * (1.f - st.T);
    }

    gOutColor[id.xy] = float4(L, st.T);
    gOutDistance[id.xy] = metersToWorld(min(cloudDist, 1e7f));
}
