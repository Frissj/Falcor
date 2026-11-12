/***************************************************************************
 # Composite (counterpart of Cloudly's CompositeAtmosphereCS):
 #   background pixels -> procedural sky + sun disk,
 #   geometry          -> aerial perspective,
 #   then the resolved clouds on top using their transmittance.
 **************************************************************************/
import RenderPasses.Cloudly.CloudlyCommon;

Texture2D<float4> gSceneColor;
Texture2D<float> gDepth;
Texture2D<float4> gClouds;
RWTexture2D<float4> gOutput;

cbuffer CompositeCB
{
    uint2 gFrameDim;
    uint gHasDepth;
    uint gHasSceneColor;
    uint gSceneAerialPerspective;
}

[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint2 pixel = id.xy;
    if (any(pixel >= gFrameDim))
        return;

    float4 scene = gHasSceneColor != 0 ? gSceneColor[pixel] : float4(0.f, 0.f, 0.f, 1.f);
    float2 pixelPos = float2(pixel) + 0.5f;
    float3 rd = viewDirForPixel(pixelPos, float2(gFrameDim));
    float3 ro = cameraPlanetPos();
    float3 sunDir = normalize(gParams.lightSourceDirection);
    float jitter = temporalJitter(pixel, gParams.frameIndex);

    float depth = gHasDepth != 0 ? gDepth[pixel] : 1.f;
    if (depth >= 1.f)
    {
        if (gParams.drawSky != 0)
        {
            SkyResult sky = integrateSky(ro, rd, sunDir, kInfinity, 32, true, true, jitter);
            float3 bg = sky.L;
            if (gParams.drawSunDisk != 0 && rayGround(ro, rd) < 0.f)
                bg += sunDiskLuminance(rd, sunDir) * sky.transmittance;
            scene.rgb = bg;
        }
    }
    else if (gSceneAerialPerspective != 0)
    {
        float dist = worldToMeters(sceneDistanceFromDepth(depth, pixelPos, float2(gFrameDim)));
        float tAP = max(0.f, dist - gParams.aerialPerspectiveStartDistance);
        if (tAP > 0.f)
        {
            float3 roAP = ro + rd * gParams.aerialPerspectiveStartDistance;
            SkyResult ap = integrateSky(roAP, rd, sunDir, tAP, 16, true, false, jitter);
            scene.rgb = scene.rgb * ap.transmittance + ap.L;
        }
    }

    float4 clouds = gClouds[pixel];
    gOutput[pixel] = float4(scene.rgb * clouds.a + clouds.rgb, scene.a);
}
