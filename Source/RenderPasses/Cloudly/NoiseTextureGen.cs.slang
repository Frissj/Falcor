/***************************************************************************
 # Tileable 3D cloud noise (counterpart of Cloudly's NoiseTextureGen.usf:
 # PerlinWeight / WorleyWeight, NumOctaves, Persistence).
 #   Base   (gOutput, kDetail=0): R = Perlin-Worley, G/B/A = Worley fBm (freq 4, 8, 16)
 #   Detail (gOutput, kDetail=1): R/G/B = Worley fBm (freq 2, 4, 8)
 **************************************************************************/

cbuffer NoiseCB
{
    uint gSize;
    uint gDetail;
}

RWTexture3D<float4> gOutput;

float3 hash33(int3 p)
{
    uint3 q = uint3(p) * uint3(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return float3(q) * (1.f / 4294967295.f);
}

int3 wrap(int3 c, int period) { return ((c % period) + period) % period; }

/// Inverted tileable Worley (F1) noise in [0,1]; 1 at feature points.
float worley(float3 p, int period)
{
    float3 cell = floor(p);
    float3 f = p - cell;
    float minD = 1e9f;
    [unroll]
    for (int z = -1; z <= 1; z++)
    [unroll]
    for (int y = -1; y <= 1; y++)
    [unroll]
    for (int x = -1; x <= 1; x++)
    {
        int3 o = int3(x, y, z);
        float3 fp = float3(o) + hash33(wrap(int3(cell) + o, period));
        float3 d = fp - f;
        minD = min(minD, dot(d, d));
    }
    return 1.f - saturate(sqrt(minD));
}

float3 gradient(int3 c, int period)
{
    return normalize(hash33(wrap(c, period)) * 2.f - 1.f + 1e-4f);
}

/// Tileable gradient (Perlin) noise in [-1,1].
float perlin(float3 p, int period)
{
    int3 c = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.f - 15.f) + 10.f);
    float n[8];
    [unroll]
    for (int i = 0; i < 8; i++)
    {
        int3 o = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        n[i] = dot(gradient(c + o, period), f - float3(o));
    }
    float x0 = lerp(lerp(n[0], n[1], u.x), lerp(n[2], n[3], u.x), u.y);
    float x1 = lerp(lerp(n[4], n[5], u.x), lerp(n[6], n[7], u.x), u.y);
    return lerp(x0, x1, u.z);
}

float worleyFbm(float3 uvw, int freq)
{
    return worley(uvw * freq, freq) * 0.625f
         + worley(uvw * freq * 2, freq * 2) * 0.25f
         + worley(uvw * freq * 4, freq * 4) * 0.125f;
}

float perlinFbm(float3 uvw, int freq, int octaves)
{
    float sum = 0.f, amp = 1.f, norm = 0.f;
    for (int o = 0; o < octaves; o++)
    {
        int f = freq << o;
        sum += perlin(uvw * f, f) * amp;
        norm += amp;
        amp *= 0.5f;
    }
    return sum / norm * 0.5f + 0.5f;
}

[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id >= gSize))
        return;
    float3 uvw = (float3(id) + 0.5f) / gSize;

    if (gDetail == 0)
    {
        float w4 = worleyFbm(uvw, 4);
        float w8 = worleyFbm(uvw, 8);
        float w16 = worleyFbm(uvw, 16);
        float p = perlinFbm(uvw, 4, 6);
        // Perlin-Worley: Perlin dilated by Worley (billowy, connected shapes).
        float pw = saturate((p - (w4 - 1.f)) / (1.f - (w4 - 1.f)));
        pw = saturate(pw * 1.2f - 0.1f);
        gOutput[id] = float4(pw, w4, w8, w16);
    }
    else
    {
        gOutput[id] = float4(worleyFbm(uvw, 2), worleyFbm(uvw, 4), worleyFbm(uvw, 8), 1.f);
    }
}
