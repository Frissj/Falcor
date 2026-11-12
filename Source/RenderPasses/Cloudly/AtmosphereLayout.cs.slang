// Placement fields adapted from Atmosphere (src/shaders/sky.ts and src/noise.ts).
// Copyright (c) 2026 Harada Moses Kiyohide (@takustaqu)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

cbuffer LayoutCB
{
    uint3 gDim;
    uint gLayer;
    float2 gOffset;
    float2 gRotation; // Cosine and sine of the placement yaw.
    float2 gScale;
    float3 gMapExtent;
    float3 gMapOffset;
    float gStartHeight;
    float gThickness;
    float gAmount;
}

RWTexture3D<float> gLayoutDensity;

float lattice(float2 cell)
{
    // Match Atmosphere's repeating, quantized 8-bit noise lattice.
    cell = frac(cell / 256.f) * 256.f;
    precise float3 p = frac(cell.xyx * 0.1031f);
    precise float d = (p.x * (p.y + 33.33f) + p.y * (p.z + 33.33f)) + p.z * (p.x + 33.33f);
    p += d;
    precise float value = (p.x + p.y) * p.z;
    return round(frac(value) * 255.f) / 255.f;
}

float valueNoise(float2 p)
{
    float2 cell = floor(p);
    float2 f = smoothstep(0.f, 1.f, frac(p));
    return lerp(
        lerp(lattice(cell), lattice(cell + float2(1.f, 0.f)), f.x),
        lerp(lattice(cell + float2(0.f, 1.f)), lattice(cell + 1.f), f.x),
        f.y
    );
}

float2 nextOctave(float2 p) { return p * 2.03f + float2(17.3f, 9.1f); }

float cumulus(float2 plane, float height, float amount)
{
    // Atmosphere's locality field, downwind crown lean, stretch and lump placement.
    float2 patchUV = plane * 0.09f + 61.f;
    float patch = 0.667f * valueNoise(patchUV) + 0.333f * valueNoise(nextOctave(patchUV)) - 0.5f;
    float edge = 0.88f - 0.52f * amount + 0.34f * patch * (1.f - smoothstep(0.55f, 0.9f, amount));
    float2 uv = (plane + float2(0.35f * 1.1875f * height, 0.f)) / float2(1.15f, 1.f);
    // At placement-map resolution the two largest octaves carry the field.
    float field = 0.5f * valueNoise(uv) + 0.25f * valueNoise(nextOctave(uv)) + 0.21875f * 0.5f;
    float lump = 0.28f + 0.85f * smoothstep(0.35f, 0.8f, valueNoise(uv * 0.9f + 31.7f));
    field = (field * 0.64f + 0.47f * 0.36f - 0.5f) * 2.2f + 0.5f;
    return lerp(field, lump, 0.45f) - edge - 0.06f * height * height;
}

float altocumulus(float2 plane, float height, float amount)
{
    // The flock envelope survives the placement grid. Cloudly supplies the finer cloudlets.
    float2 uv = plane * 0.4f + 21.f;
    float n0 = valueNoise(uv);
    uv = nextOctave(uv);
    float n1 = valueNoise(uv);
    float n2 = valueNoise(nextOctave(uv));
    float coverage = smoothstep(0.f, 0.75f, amount);
    float patch = 0.5f * n0 + 0.25f * n1 + 0.1875f * n2;
    float edge = lerp(0.58f, 0.37f, coverage);
    return patch - edge - 0.16f * (height - 0.5f) * (height - 0.5f);
}

[numthreads(8, 8, 1)]
void rasterize(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gDim))
        return;
    float3 uv = (float3(id) + 0.5f) / float3(gDim);
    float altitude = uv.z * gMapExtent.z - gMapOffset.z;
    float height = (altitude - gStartHeight) / gThickness;
    if (gAmount <= 0.f || altitude < gStartHeight - gMapExtent.z / gDim.z || height >= 1.f)
    {
        gLayoutDensity[id] = 0.f;
        return;
    }

    // Inverse translation and yaw in the flat map's X/Z plane.
    float2 map = frac(uv.xy - gOffset / gMapExtent.xy) * gMapExtent.xy - gMapOffset.xy;
    float2 local = float2(dot(map, gRotation), dot(map, float2(-gRotation.y, gRotation.x))) / gScale;
    float depth = saturate(height);
    float field = gLayer == 0
        ? cumulus(local / 1500.f, depth, saturate(gAmount))
        : altocumulus(local / 1000.f, depth, saturate(gAmount));
    // Continuous density gradients for Cloudly's subvoxel surface correction.
    float top = smoothstep(0.65f, 1.f, depth);
    gLayoutDensity[id] = saturate(0.5f + field - 0.7f * top);
}
