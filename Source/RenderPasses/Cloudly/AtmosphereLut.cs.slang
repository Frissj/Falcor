/***************************************************************************
 # Atmosphere lookup tables (counterparts of Cloudly's OpticalDepthTex /
 # MultiScatteringLut / DistantSkyLightLut passes), after Hillaire 2020.
 **************************************************************************/
import RenderPasses.Cloudly.CloudlyCommon;

RWTexture2D<float4> gOutput;

/// Transmittance from (r, mu) to the top of the atmosphere. 256x64.
[numthreads(8, 8, 1)]
void transmittanceLut(uint3 id: SV_DispatchThreadID)
{
    uint2 dim;
    gOutput.GetDimensions(dim.x, dim.y);
    if (any(id.xy >= dim))
        return;
    float2 uv = (float2(id.xy) + 0.5f) / float2(dim);
    float r, mu;
    transmittanceLutRMu(uv, r, mu);

    float3 ro = float3(0.f, r, 0.f);
    float3 rd = float3(sqrt(max(0.f, 1.f - mu * mu)), mu, 0.f);
    float tEnd = raySphere(ro, rd, rTop()).y;
    const uint N = 40;
    float dt = tEnd / N;
    float3 od = float3(0.f);
    for (uint i = 0; i < N; i++)
    {
        float3 s = ro + rd * ((i + 0.5f) * dt);
        od += sampleAtmosphere(altitude(s)).extinction * dt;
    }
    gOutput[id.xy] = float4(exp(-od), 1.f);
}

/// Uniformly distributed direction on the sphere from a 2D sample.
float3 sphereDir(float2 u)
{
    float z = 1.f - 2.f * u.x;
    float r = sqrt(max(0.f, 1.f - z * z));
    float phi = 2.f * kPI * u.y;
    return float3(r * cos(phi), z, r * sin(phi));
}

/// Multiple scattering transfer (Hillaire 2020, eq. 5-10). 32x32, (cos sun zenith, altitude).
[numthreads(8, 8, 1)]
void multiScatteringLut(uint3 id: SV_DispatchThreadID)
{
    uint2 dim;
    gOutput.GetDimensions(dim.x, dim.y);
    if (any(id.xy >= dim))
        return;
    float2 uv = (float2(id.xy) + 0.5f) / float2(dim);
    float cosSunZenith = uv.x * 2.f - 1.f;
    float h = uv.y * (rTop() - rBot());
    float3 ro = float3(0.f, rBot() + max(h, 1.f), 0.f);
    float3 sunDir = float3(sqrt(saturate(1.f - cosSunZenith * cosSunZenith)), cosSunZenith, 0.f);

    const uint kDirs = 8;
    const uint kSteps = 20;
    float3 L2 = float3(0.f);
    float3 fms = float3(0.f);
    for (uint j = 0; j < kDirs; j++)
    for (uint i = 0; i < kDirs; i++)
    {
        float3 rd = sphereDir((float2(i, j) + 0.5f) / kDirs);
        float2 top = raySphere(ro, rd, rTop());
        float tMax = top.y;
        float ground = rayGround(ro, rd);
        if (ground > 0.f)
            tMax = ground;
        float dt = tMax / kSteps;
        float3 T = float3(1.f);
        float3 L = float3(0.f);
        float3 f = float3(0.f);
        for (uint s = 0; s < kSteps; s++)
        {
            float3 p = ro + rd * ((s + 0.5f) * dt);
            AtmosphereSample a = sampleAtmosphere(altitude(p));
            float3 scat = a.rayleighScat + a.mieScat;
            float3 stepT = exp(-a.extinction * dt);
            float3 integ = (1.f - stepT) / max(a.extinction, 1e-12f);
            float3 Tsun = sunTransmittance(p, sunDir);
            L += T * scat * Tsun * (1.f / (4.f * kPI)) * integ;
            f += T * scat * integ;
            T *= stepT;
        }
        if (ground > 0.f)
        {
            float3 pg = ro + rd * ground;
            float3 n = normalize(pg);
            L += T * sunTransmittance(pg + n, sunDir) * saturate(dot(n, sunDir)) * gParams.groundAlbedo / kPI;
        }
        L2 += L;
        fms += f;
    }
    float inv = 1.f / float(kDirs * kDirs); // Isotropic phase over the sphere: (1/4pi) * (4pi / N).
    L2 *= inv;
    fms *= inv;
    gOutput[id.xy] = float4(L2 / (1.f - fms), 1.f);
}

/// Mean single-scattered sky radiance over the sphere at 16 altitudes from 1 km to 20 km (DistantSkyLightLutCS).
[numthreads(16, 1, 1)]
void distantSkyLightLut(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= 16)
        return;
    float3 sunDir = normalize(gParams.lightSourceDirection);
    float3 ro = float3(0.f, rBot() + lerp(1000.f, 20000.f, float(id.x) / 15.f), 0.f);
    const uint kDirs = 8;
    float3 sum = float3(0.f);
    for (uint j = 0; j < kDirs; j++)
    for (uint i = 0; i < kDirs; i++)
    {
        float3 rd = sphereDir((float2(i, j) + 0.5f) / kDirs);
        sum += integrateSky(ro, rd, sunDir, kInfinity, 10, false, false, 0.3f).L;
    }
    // Illuminance = sum * (4pi / N); times the uniform phase 1/(4pi) gives the mean radiance.
    gOutput[uint2(id.x, 0)] = float4(sum / float(kDirs * kDirs), 1.f);
}
