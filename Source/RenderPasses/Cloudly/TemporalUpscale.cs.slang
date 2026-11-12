/***************************************************************************
 # Temporal upscale (counterpart of Cloudly's TemporalUpscale.usf /
 # CheckerBoardResolveCS). Reconstructs full-resolution clouds from the
 # low-resolution trace plus reprojected history.
 **************************************************************************/
import RenderPasses.Cloudly.CloudlyCommon;

Texture2D<float4> gTraceColor;
Texture2D<float> gTraceDistance;
Texture2D<float4> gHistoryColor;
Texture2D<float> gHistoryDistance;
RWTexture2D<float4> gOutColor;
RWTexture2D<float> gOutDistance;

cbuffer UpscaleCB
{
    uint2 gFrameDim;
    uint2 gSubPixel;
    uint gHistoryValid;
    float gHistoryWeight; ///< Weight of history for pixels not traced this frame.
}

[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint2 pixel = id.xy;
    if (any(pixel >= gFrameDim))
        return;

    uint ds = gParams.traceDownscale;
    int2 lowPixel = int2(pixel / ds);
    int2 lowMax = int2(gParams.traceDim) - 1;
    bool tracedThisFrame = all((pixel % ds) == gSubPixel);

    // Current estimate: exact sample if this pixel was traced, else bilinear from the low-res trace.
    // Low-res texel i covers full-res pixel i * ds + gSubPixel this frame.
    float2 uv = (float2(pixel) - float2(gSubPixel) + 0.5f * ds) / float2(gParams.traceDim * ds);
    float4 cur = tracedThisFrame ? gTraceColor[lowPixel] : gTraceColor.SampleLevel(gLinearClamp, uv, 0);
    float curDist = tracedThisFrame ? gTraceDistance[lowPixel] : gTraceDistance.SampleLevel(gLinearClamp, uv, 0);

    // Neighbourhood bounds from the 3x3 low-res texels (clamps stale history).
    float4 nMin = float4(1e30f), nMax = float4(-1e30f);
    [unroll]
    for (int y = -1; y <= 1; y++)
    [unroll]
    for (int x = -1; x <= 1; x++)
    {
        float4 c = gTraceColor[clamp(lowPixel + int2(x, y), 0, lowMax)];
        nMin = min(nMin, c);
        nMax = max(nMax, c);
    }

    float4 result = cur;
    float resultDist = curDist;

    if (gHistoryValid != 0)
    {
        // Reproject the cloud's representative point into the previous frame.
        float3 dirW = viewDirForPixel(float2(pixel) + 0.5f, float2(gFrameDim));
        float3 posW = gCam.posW + dirW * curDist;
        float4 prevH = mul(gCam.prevViewProj, float4(posW, 1.f));
        float2 prevNdc = prevH.xy / prevH.w;
        float2 prevUv = float2(0.5f, -0.5f) * prevNdc + 0.5f;

        if (prevH.w > 0.f && all(prevUv >= 0.f) && all(prevUv <= 1.f))
        {
            float4 hist = gHistoryColor.SampleLevel(gLinearClamp, prevUv, 0);
            float histDist = gHistoryDistance.SampleLevel(gLinearClamp, prevUv, 0);
            hist = clamp(hist, nMin, nMax);
            float w = tracedThisFrame ? gHistoryWeight * 0.5f : gHistoryWeight;
            result = lerp(cur, hist, w);
            resultDist = lerp(curDist, histDist, w);
        }
    }

    gOutColor[pixel] = result;
    gOutDistance[pixel] = resultDist;
}
