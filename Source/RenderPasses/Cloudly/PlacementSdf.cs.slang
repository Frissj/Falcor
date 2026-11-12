// Base/Alto placement from a scalar density texture. Seed and distance rules
// follow the shipped InitializeTexturesCS, JumpFloodPassCS and distance decode.
cbuffer PlacementCB
{
    uint3 gDim;
    uint gJump;
    float gSurface;
    float gSurfaceFixupStrength;
}

Texture3D<float> gDensity;
Texture3D<uint> gSeeds;
RWTexture3D<uint> gNextSeeds;
RWTexture3D<float> gDistance;
SamplerState gDensitySampler;

static const uint kInvalidSeed = 0xffffffffu;

uint packSeed(int3 p)
{
    return uint(p.x) | (uint(p.y) << 12) | (uint(p.z) << 24);
}

int3 unpackSeed(uint seed)
{
    return int3(seed & 0xfffu, (seed >> 12) & 0xfffu, seed >> 24);
}

int3 addressTexel(int3 p)
{
    // Placement volumes have power-of-two horizontal dimensions.
    return int3(p.xy & (int2(gDim.xy) - 1), clamp(p.z, 0, int(gDim.z) - 1));
}

bool isSolid(int3 p)
{
    return gDensity[addressTexel(p)] > gSurface;
}

float periodicDistanceSquared(int3 p, int3 q)
{
    int3 delta = abs(p - q);
    delta.xy = min(delta.xy, int2(gDim.xy) - delta.xy);
    return dot(float3(delta), float3(delta));
}

[numthreads(8, 8, 1)]
void initialize(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gDim))
        return;
    int3 p = int3(id);
    bool solid = isSolid(p);
    bool boundary = solid != isSolid(p + int3(1, 0, 0)) || solid != isSolid(p - int3(1, 0, 0)) ||
                    solid != isSolid(p + int3(0, 1, 0)) || solid != isSolid(p - int3(0, 1, 0)) ||
                    solid != isSolid(p + int3(0, 0, 1)) || solid != isSolid(p - int3(0, 0, 1));
    gNextSeeds[id] = boundary ? packSeed(p) : kInvalidSeed;
}

[numthreads(8, 8, 1)]
void jumpFlood(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gDim))
        return;
    int3 p = int3(id);
    uint nearest = gSeeds[id];
    float3 initialDelta = float3(unpackSeed(nearest) - p);
    float best = dot(initialDelta, initialDelta);
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        int3 neighbor = addressTexel(p + int3(x, y, z) * int(gJump));
        if (all(neighbor == p))
            continue;
        uint candidate = gSeeds[neighbor];
        if (candidate == kInvalidSeed)
            continue;
        float distance = periodicDistanceSquared(p, unpackSeed(candidate));
        if (distance < best)
        {
            best = distance;
            nearest = candidate;
        }
    }
    gNextSeeds[id] = nearest;
}

float sampleDensity(float3 p)
{
    return gDensity.SampleLevel(gDensitySampler, (p + 0.5f) / float3(gDim), 0);
}

float3 densityGradient(float3 p)
{
    const float e = 0.8f;
    const float3 x = float3(e, 0, 0), y = float3(0, e, 0), z = float3(0, 0, e);
    return float3(sampleDensity(p + x) - sampleDensity(p - x),
                  sampleDensity(p + y) - sampleDensity(p - y),
                  sampleDensity(p + z) - sampleDensity(p - z)) / (2.f * e);
}

float refineSurface(int3 voxel)
{
    // The original searches from a half-texel offset and accepts only points
    // that converge to within 0.02 of the configured density level.
    float3 start = float3(voxel) + 0.5f;
    float3 position = start;
    for (uint i = 0; i < 16; ++i)
    {
        float3 gradient = densityGradient(position);
        float gradientSquared = dot(gradient, gradient);
        if (gradientSquared == 0.f)
            break;
        float stepWeight = float(i) / 15.f;
        if (gSurfaceFixupStrength < 1.f)
        {
            float3 displacement = position - start;
            stepWeight *= exp(dot(displacement, displacement) / log(1.f - gSurfaceFixupStrength));
        }
        position += gradient * ((gSurface - sampleDensity(position)) * stepWeight / gradientSquared);
    }
    return abs(gSurface - sampleDensity(position)) <= 0.02f ? length(position - start) : 0.f;
}

[numthreads(8, 8, 1)]
void decodeDistance(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gDim))
        return;
    int3 p = int3(id);
    uint seed = gSeeds[id];
    float distance = seed == kInvalidSeed ? 1e15f : sqrt(periodicDistanceSquared(p, unpackSeed(seed)));
    if (distance == 0.f && gSurfaceFixupStrength > 0.f)
        distance = refineSurface(p);
    gDistance[id] = isSolid(p) ? -distance : distance;
}
