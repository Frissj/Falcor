/***************************************************************************
 # Cloudly-style volumetric clouds: shared density / lighting / atmosphere code.
 #
 # Everything is evaluated in metres in "planet space": the planet centre is the
 # origin, +y is up at the world origin. The lighting model follows the one used
 # by the Cloudly plugin (see README.md); cloud shapes are procedural and exposed
 # through a pseudo signed distance so that the plugin's SDF-driven density,
 # edge fade and dark-edge terms apply unchanged.
 **************************************************************************/
import RenderPasses.Cloudly.CloudlyParams;

static const float kPI = 3.14159265359f;
static const float kInfinity = 1e30f;
static const float kGolden = 0.618033989f;

cbuffer CloudlyCB
{
    CloudlyParams gParams;
    CloudlyCamera gCam;
    CloudlyGameData gGame;
}

// Game cloud volumes (CloudTex_*), used when gParams.useGameVolumes != 0.
Texture3D<float4> gGameBaseSdf;    ///< RG8: base / alto placement SDF (encoded).
Texture3D<float4> gGameDisp1;      ///< BC5: displacement 1.
Texture3D<float4> gGameDisp2;      ///< BC4: displacement 2.
Texture3D<float4> gGameCirrusSdf;  ///< RGBA8: near / far SDF (encoded), near / far density.
Texture3D<float4> gGameDispCirrus; ///< R8: cirrus displacement.
SamplerState gSdfSampler;          ///< Linear, wrap in x/y, clamp in z.
SamplerState gDispSampler;         ///< Linear, wrap in x/y/z (CloudTex_Sampler).

Texture3D<float4> gBaseNoise;   ///< R: Perlin-Worley, GBA: Worley fBm at increasing frequencies.
Texture3D<float4> gDetailNoise; ///< RGB: Worley fBm at increasing frequencies.
Texture2D<float4> gTransmittanceLut;  ///< 256x64, (mu, r) parameterisation.
Texture2D<float4> gMultiScatteringLut; ///< 32x32, (cos sun zenith, altitude).
Texture2D<float4> gDistantSkyLightLut; ///< 16x1, mean sky radiance at altitudes 1..20 km.
SamplerState gLinearWrap;
SamplerState gLinearClamp;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float sq(float x) { return x * x; }
float remap(float v, float l0, float h0, float l1, float h1) { return l1 + (v - l0) * (h1 - l1) / (h0 - l0); }
float linearstep(float a, float b, float x) { return saturate((x - a) / (b - a)); }
float exp2Decay(float x, float alpha) { return 1.f - exp2(-alpha * x); }
float luminance(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }

/// Interleaved gradient noise, animated with the golden ratio.
float temporalJitter(uint2 pixel, uint frame)
{
    float ign = frac(52.9829189f * frac(dot(float2(pixel), float2(0.06711056f, 0.00583715f))));
    return frac(ign + float(frame % 64) * kGolden);
}

/// Ray through a full-resolution pixel position (Falcor pinhole camera, no jitter).
float3 viewDirForPixel(float2 pixelPos, float2 frameDim)
{
    float2 p = pixelPos / frameDim;
    float2 ndc = float2(2.f, -2.f) * p + float2(-1.f, 1.f);
    return normalize(ndc.x * gCam.cameraU + ndc.y * gCam.cameraV + gCam.cameraW);
}

/// Distance (world units) from the camera to the surface stored in a hardware depth buffer, kInfinity for background.
float sceneDistanceFromDepth(float depth, float2 pixelPos, float2 frameDim)
{
    if (depth >= 1.f)
        return kInfinity;
    float2 p = pixelPos / frameDim;
    float2 ndc = float2(2.f, -2.f) * p + float2(-1.f, 1.f);
    float4 posH = mul(gCam.invViewProj, float4(ndc, depth, 1.f));
    return length(posH.xyz / posH.w - gCam.posW);
}

float worldToMeters(float d) { return d * gParams.metersPerUnit; }
float metersToWorld(float d) { return d / gParams.metersPerUnit; }

float rBot() { return gParams.earthRadius; }
float rTop() { return gParams.earthRadius + gParams.atmMaxAltitude; }

/// Camera altitude in metres (world y plus the origin altitude).
float cameraAltitudeM() { return gCam.posW.y * gParams.metersPerUnit + gParams.cameraAltitude; }

/// Planet space keeps the planet centre straight below the camera, like the plugin's flat game map
/// (GetEarthCenter() follows WorldCameraOrigin.xy).
float3 cameraPlanetPos() { return float3(0.f, gParams.earthRadius + cameraAltitudeM(), 0.f); }

/// World position (world units) to planet space (metres).
float3 planetPos(float3 posW) { return (posW - gCam.posW) * gParams.metersPerUnit + cameraPlanetPos(); }

/// Planet space to the plugin's flat-map coordinates (x, y horizontal, z flat height), metres.
float3 gameFlatPos(float3 p)
{
    float2 camXY = float2(gCam.posW.x, gCam.posW.z) * gParams.metersPerUnit + gGame.gameOffsetXY;
    return float3(p.x + camXY.x, p.z + camXY.y, p.y - gParams.earthRadius);
}

float altitude(float3 p) { return length(p) - gParams.earthRadius; }

/// Ray / origin-centred sphere. Returns (tNear, tFar); tFar < 0 means miss.
float2 raySphere(float3 ro, float3 rd, float radius)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.f)
        return float2(-1.f, -1.f);
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

/// Distance to the ground along the ray, or -1 if it does not hit.
float rayGround(float3 ro, float3 rd)
{
    float2 g = raySphere(ro, rd, rBot());
    return g.x > 0.f ? g.x : -1.f;
}

/// Segment of the ray inside the spherical shell [rInner, rOuter], clipped to [0, tMax] and to the ground.
float2 rayShell(float3 ro, float3 rd, float rInner, float rOuter, float tMax)
{
    float2 outer = raySphere(ro, rd, rOuter);
    if (outer.y < 0.f)
        return float2(0.f, 0.f);
    float2 inner = raySphere(ro, rd, rInner);
    float r = length(ro);
    float t0, t1;
    if (r < rInner)
    {
        t0 = inner.y;
        t1 = outer.y;
    }
    else if (r < rOuter)
    {
        t0 = 0.f;
        t1 = (inner.x > 0.f) ? inner.x : outer.y;
    }
    else
    {
        t0 = max(outer.x, 0.f);
        t1 = (inner.x > 0.f) ? inner.x : outer.y;
    }
    float ground = rayGround(ro, rd);
    if (ground > 0.f)
        t1 = min(t1, ground);
    return float2(max(t0, 0.f), min(t1, tMax));
}

// ---------------------------------------------------------------------------
// Atmosphere (SkyConstantsUniform: Rayleigh / Mie / ozone, Hillaire 2020 LUTs)
// ---------------------------------------------------------------------------

struct AtmosphereSample
{
    float3 rayleighScat;
    float3 mieScat;
    float3 extinction;
};

AtmosphereSample sampleAtmosphere(float h)
{
    h = max(h, 0.f);
    float rhoR = exp(-h / gParams.h0Rayleigh);
    float rhoM = exp(-h / gParams.h0Mie);
    float rhoO = max(0.f, 1.f - abs(h - gParams.ozoneCenterHeight) / gParams.ozoneHalfThickness);
    AtmosphereSample s;
    s.rayleighScat = gParams.sigmaScatRayleigh * rhoR;
    s.mieScat = gParams.sigmaScatMie * rhoM;
    s.extinction = s.rayleighScat + gParams.sigmaExtMie * rhoM + gParams.sigmaExtOzone * rhoO;
    return s;
}

float phaseRayleigh(float c) { return 3.f / (16.f * kPI) * (1.f + c * c); }

float phaseHG(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.f + g2 - 2.f * g * cosTheta;
    return (1.f - g2) / (4.f * kPI * denom * sqrt(denom));
}

/// Cornette-Shanks (CornetteSingleScatteringPhaseFunction).
float phaseCornetteShanks(float cosTheta, float g)
{
    float g2 = g * g;
    float k = 3.f / (8.f * kPI) * (1.f - g2) / (2.f + g2);
    float denom = 1.f + g2 - 2.f * g * cosTheta;
    return k * (1.f + cosTheta * cosTheta) / (denom * sqrt(denom));
}

float2 transmittanceLutUv(float r, float mu)
{
    float H = sqrt(max(0.f, rTop() * rTop() - rBot() * rBot()));
    float rho = sqrt(max(0.f, r * r - rBot() * rBot()));
    float disc = r * r * (mu * mu - 1.f) + rTop() * rTop();
    float d = max(0.f, -r * mu + sqrt(max(disc, 0.f)));
    float dMin = rTop() - r;
    float dMax = rho + H;
    return float2((d - dMin) / max(dMax - dMin, 1e-3f), rho / H);
}

void transmittanceLutRMu(float2 uv, out float r, out float mu)
{
    float H = sqrt(rTop() * rTop() - rBot() * rBot());
    float rho = H * uv.y;
    r = sqrt(rho * rho + rBot() * rBot());
    float dMin = rTop() - r;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    mu = d == 0.f ? 1.f : (H * H - rho * rho - d * d) / (2.f * r * d);
    mu = clamp(mu, -1.f, 1.f);
}

/// Atmospheric transmittance from p towards direction dir up to the top of the atmosphere (ignores the planet).
float3 atmosphereTransmittance(float3 p, float3 dir)
{
    float r = length(p);
    float mu = dot(p, dir) / r;
    return gTransmittanceLut.SampleLevel(gLinearClamp, transmittanceLutUv(r, mu), 0).rgb;
}

/// Transmittance towards the sun including the planet shadow.
float3 sunTransmittance(float3 p, float3 sunDir)
{
    if (rayGround(p, sunDir) > 0.f)
        return float3(0.f);
    return atmosphereTransmittance(p, sunDir);
}

float3 multipleScattering(float3 p, float3 sunDir)
{
    float r = length(p);
    float cosSunZenith = dot(p / r, sunDir);
    float2 uv = float2(cosSunZenith * 0.5f + 0.5f, saturate((r - rBot()) / (rTop() - rBot())));
    // Half-texel remap for the 32x32 LUT.
    uv = (uv * 31.f + 0.5f) / 32.f;
    return gMultiScatteringLut.SampleLevel(gLinearClamp, uv, 0).rgb;
}

struct SkyResult
{
    float3 L;           ///< In-scattered radiance (already multiplied by sunIntensity).
    float3 transmittance;
};

/// Single scattering + LUT multiple scattering along a view ray, up to tMax (metres).
SkyResult integrateSky(float3 ro, float3 rd, float3 sunDir, float tMax, uint steps, bool useMs, bool addGround, float jitter)
{
    SkyResult res;
    res.L = float3(0.f);
    res.transmittance = float3(1.f);

    float2 top = raySphere(ro, rd, rTop());
    if (top.y < 0.f)
        return res;
    float t0 = max(top.x, 0.f);
    float t1 = top.y;
    float ground = rayGround(ro, rd);
    bool hitGround = ground > 0.f && ground < tMax;
    if (ground > 0.f)
        t1 = min(t1, ground);
    t1 = min(t1, tMax);
    if (t1 <= t0)
        return res;

    float cosTheta = dot(rd, sunDir);
    float pR = phaseRayleigh(cosTheta);
    float pM = phaseCornetteShanks(cosTheta, gParams.miePhaseG);

    float dt = (t1 - t0) / steps;
    for (uint i = 0; i < steps; i++)
    {
        float t = t0 + (i + jitter) * dt;
        float3 s = ro + rd * t;
        AtmosphereSample a = sampleAtmosphere(altitude(s));
        float3 Tsun = sunTransmittance(s, sunDir);
        float3 inScat = (a.rayleighScat * pR + a.mieScat * pM) * Tsun;
        if (useMs)
            inScat += (a.rayleighScat + a.mieScat) * multipleScattering(s, sunDir) * gParams.multipleScatteringScale;
        float3 stepT = exp(-a.extinction * dt);
        res.L += res.transmittance * (inScat - inScat * stepT) / max(a.extinction, 1e-12f);
        res.transmittance *= stepT;
    }

    if (addGround && hitGround)
    {
        float3 pg = ro + rd * ground;
        float3 n = normalize(pg);
        float3 Tsun = sunTransmittance(pg + n * 1.f, sunDir);
        res.L += res.transmittance * Tsun * gParams.groundAlbedo * (saturate(dot(n, sunDir)) / kPI);
    }

    res.L *= gParams.sunIntensity;
    return res;
}

/// Sun disk luminance (GetSunDiskLuminance): illuminance spread over the disk with a soft edge.
float3 sunDiskLuminance(float3 rd, float3 sunDir)
{
    float cosTheta = dot(rd, sunDir);
    float sunCos = gParams.sunDiskCosine;
    if (cosTheta <= sunCos)
        return float3(0.f);
    float solidAngle = 2.f * kPI * (1.f - sunCos);
    float softEdge = saturate(2.f * (cosTheta - sunCos) / (1.f - sunCos));
    return float3(softEdge * gParams.sunIntensity / solidAngle);
}

// ---------------------------------------------------------------------------
// Cloud shapes: procedural layers exposed as pseudo signed distances (metres,
// negative inside), then the plugin's density model (GetCloudDensity).
// ---------------------------------------------------------------------------

struct CloudEval
{
    float density;  ///< rho_w in [0,1]
    float dBase;    ///< (Pseudo) SDF of the base layer, metres, negative inside.
    float dAlto;
    float dCirrus;
    float2 layerTrans; ///< Bottom-to-top transition of base / alto (packing_layer_trans3).
    bool inCirrus;
};

// ---------------------------------------------------------------------------
// Game volumes: GetCloudSDF / GetCloudSDF_Cirrus / GetCloudDensity
// ---------------------------------------------------------------------------

float opU(float a, float b) { return min(a, b); }
float opI(float a, float b) { return max(a, b); }
float her2(float x, float r) { return 0.5f * ((1.f / r) * x * x + r); }

/// Smooth intersection (opI_weld_quadric).
float opIWeldQuadric(float a, float b, float r)
{
    float c = (a + b) * 0.5f;
    float x = abs(a - c);
    return c + (x > r ? x : her2(x, r));
}

/// Double-sided smooth intersection (opI_weld_quadric_double_sided), .x only.
float opIWeldQuadricDoubleSided(float a, float b, float r)
{
    float c = (a + b) * 0.5f;
    float x = abs(a - c);
    return c + (x > r ? x : her2(x, r));
}

/// Decode an 8-bit logistic-encoded SDF sample to texels (Unorm8ToPixelDistance, curve 0 = base/alto, 1 = cirrus).
float decodeSdf(float x, float k)
{
    x = clamp(x, 1e-6f, 1.f - 1e-6f);
    x = saturate(0.5f + (-log(1.f / x - 1.f)) / k);
    x = x * 2.f - (1.f + 1.f / 255.f);
    return x * 32.f;
}

/// Sine of the half-angle of a trace pixel's cone (GetRoundConeApertureAngle).
float coneSinAlpha()
{
    float tanHalfFovY = length(gCam.cameraV) / length(gCam.cameraW);
    float traceHeight = float(gParams.traceDim.y * gParams.traceDownscale) * gGame.mipTraceScale;
    return (2.f / traceHeight) * 0.70710678f * tanHalfFovY;
}

/// CalculateMipLevel: mip whose texel matches the pixel cone radius at viewDist.
float gameMip(float viewDist, float texelWorldSize)
{
    float coneRadius = viewDist * coneSinAlpha();
    return clamp(log2(max(1.f, coneRadius / texelWorldSize)), 0.f, gGame.maxMip);
}

/// Placement SDFs (CloudTex_SamplerBASE / CIRRUS): wrap x/y, clamp z.
float4 sampleGame(Texture3D<float4> tex, float3 pp, float3 uvOffset, float3 uvScale, float mip)
{
    return tex.SampleLevel(gSdfSampler, (pp + uvOffset) * uvScale, mip);
}

/// Displacement / density volumes (CloudTex_Sampler): wrap on every axis.
float4 sampleGameDisp(Texture3D<float4> tex, float3 pp, float3 uvOffset, float3 uvScale, float mip)
{
    return tex.SampleLevel(gDispSampler, (pp + uvOffset) * uvScale, mip);
}

/// Fade of the displacement strength with distance to the base surface.
float fadeBandLinear(float d, float modulate, float s, float e) { return 1.f - modulate * linearstep(s, e, d); }
float fadeBandSmooth(float d, float modulate, float s, float e) { return 1.f - modulate * smoothstep(s, e, d); }

float rotateDeg(inout float2 p, float deg)
{
    float a = radians(deg);
    float c = cos(a), s = sin(a);
    p = float2(c * p.x - s * p.y, s * p.x + c * p.y);
    return 0.f;
}

/// Layered animated gyroid (HotsureEffect detail noise).
float layeredGyroid(float3 p)
{
    const float advection = 1.4f;
    float3 trans = float3(gGame.hotsureAnimOffsetPreScaled, 0.f);
    p += trans;
    p.z *= gGame.hotsureZScale;
    float d = 0.f, amplitude = 1.f, trk = 1.f;
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        p += sin(p.zyx * 0.75f * trk + trans) * gGame.hotsureWhirl;
        trk *= advection;
        d += abs(dot(cos(p), sin(p)) * amplitude);
        float2 xy = p.xy; rotateDeg(xy, 33.f); p.xy = xy;
        float2 yz = p.yz; rotateDeg(yz, 66.f); p.yz = yz;
        p *= gGame.hotsureFrequency;
        amplitude *= gGame.hotsurePersistence;
    }
    return d * (1.f / 3.f);
}

float hotsureDensity(float3 pp, float h, float dBase, float dAlto, float baseDensity, float viewDist)
{
    float2 mask = saturate(-float2(dBase, dAlto));
    mask *= linearstep(0.003f, 0.004f, baseDensity);
    float2 r0 = gGame.hotsureLayerRemap0 * mask;
    float2 r1 = gGame.hotsureLayerRemap1 * mask;
    if (gGame.enableHotsure == 0 || max(max(r0.x, r0.y), max(r1.x, r1.y)) == 0.f)
        return 0.f;
    float m = (1.f - exp2(-baseDensity * gGame.hotsureLowBaseDensityDecay)) * exp2(-baseDensity * gGame.hotsureHighBaseDensityDecay) *
              linearstep(gGame.hotsureFadeDistanceEnd, gGame.hotsureFadeDistanceStart, viewDist) *
              linearstep(-50.f, 0.f, h - gParams.cloudLayersStartHeight.x);
    if (m == 0.f)
        return 0.f;
    float detail = 1.f - layeredGyroid(pp * gGame.hotsurePositionScaleRcp);
    float2 v = lerp(r0, r1, detail);
    detail = saturate(max(v.x, v.y));
    detail = detail * detail * detail;
    detail *= detail;
    return detail * m;
}

CloudEval evalCloudGame(float3 p, float viewDist, float distanceFade, bool cheap)
{
    CloudEval e;
    e.density = 0.f;
    e.inCirrus = false;

    float h = altitude(p);
    float3 pp = gameFlatPos(p);
    // CalculateCurveCorrectedPosition: the earth centre sits under the camera, so the horizontal
    // offset from it is p.xz in planet space.
    pp.z -= sqrt(max(0.f, sq(gParams.earthRadius) - dot(p.xz, p.xz))) - gParams.earthRadius;
    pp += float3(gParams.windOffset.x, gParams.windOffset.z, 0.f);

    float2 dLayersMin = h - gParams.cloudLayersStartHeight.xy;
    float2 minClamp = gParams.cloudLayersStartHeight.xy;
    if (gParams.cloudLayersDensityBottom.x < 0.001f)
        minClamp.x += max(0.f, gParams.cloudLayersTransitionStartDensity.x);
    if (gParams.cloudLayersDensityBottom.y < 0.001f)
        minClamp.y += max(0.f, gParams.cloudLayersTransitionStartDensity.y);
    float2 dLayersMinClamp = h - minClamp;

    // Placement SDF (base in x, alto in y).
    float2 sdf = sampleGame(gGameBaseSdf, pp, gGame.baseUVOffset, gGame.baseUVScale, gameMip(viewDist, gGame.baseTexelWorldSize)).xy;
    sdf = float2(decodeSdf(sdf.x, 15.f), decodeSdf(sdf.y, 15.f)) * gGame.baseTexelWorldSize - float2(gGame.sdfOffsetBase, gGame.sdfOffsetAlto);
    float dBase = sdf.x + gGame.disp1Base.z;
    float dAlto = sdf.y + gGame.disp1Alto.z;

    // Displacement 1.
    float4 disp1 = float4(0.f);
    if (abs(dBase) < gGame.disp1Base.y || abs(dAlto) < gGame.disp1Alto.y)
    {
        float2 t = sampleGameDisp(gGameDisp1, pp, gGame.disp1UVOffset, gGame.disp1UVScale, gameMip(viewDist, gGame.texelWorldSize.x)).xy;
        disp1 = float4(t.x, t.y, t.x, t.x); // Alto reuses the base top channel (no ALTO_DISP1, flags 8/9 off).
    }
    float2 layerTrans = smoothstep(gGame.cloudLayersTransitionStart, gParams.cloudLayersTransitionEnd, dLayersMin);
    float2 layerTrans2 = lerp(gGame.cloudLayersBottomBleedDisp1, float2(1.f), layerTrans);
    float disp2Modulate = exp2(-saturate(disp1.x) * gGame.disp2Base.w);
    {
        float d1 = saturate(disp1.y * (1.f - layerTrans.x) + disp1.x * layerTrans2.x);
        d1 *= fadeBandLinear(dBase, smoothstep(gGame.disp1BaseFade.z, 1.f, layerTrans.x), gGame.disp1BaseFade.x, gGame.disp1BaseFade.y);
        dBase -= d1 * gGame.disp1Base.x;
    }
    {
        float d1 = saturate(disp1.w * (1.f - layerTrans.y) + disp1.z * layerTrans2.y);
        d1 *= fadeBandLinear(dAlto, smoothstep(gGame.disp1AltoFade.z, 1.f, layerTrans.y), gGame.disp1AltoFade.x, gGame.disp1AltoFade.y);
        dAlto -= d1 * gGame.disp1Alto.x;
    }

    // Displacement 2.
    dBase += gGame.disp2Base.z;
    dAlto += gGame.disp2Alto.z;
    if (abs(dBase) < gGame.disp2Base.y || abs(dAlto) < gGame.disp2Alto.y)
    {
        float t = sampleGameDisp(gGameDisp2, pp, gGame.disp2UVOffset, gGame.disp2UVScale, gameMip(viewDist, gGame.texelWorldSize.y)).x;
        float b2 = fadeBandSmooth(dBase, layerTrans.x, gGame.disp2BaseFade.x, gGame.disp2BaseFade.y) *
                   lerp(gGame.cloudLayersBottomBleedDisp2.x, 1.f, layerTrans.x) * disp2Modulate * t;
        dBase -= b2 * gGame.disp2Base.x;
        float a2 = fadeBandSmooth(dAlto, layerTrans.y, gGame.disp2AltoFade.x, gGame.disp2AltoFade.y) *
                   lerp(gGame.cloudLayersBottomBleedDisp2.y, 1.f, layerTrans.y) * t * disp2Modulate;
        dAlto -= a2 * gGame.disp2Alto.x;
    }

    // Clip against the layer floors with a soft weld.
    dBase = opIWeldQuadricDoubleSided(dBase, -dLayersMinClamp.x, gGame.cloudLayersSoftRadius.x);
    dAlto = opIWeldQuadricDoubleSided(dAlto, -dLayersMinClamp.y, gGame.cloudLayersSoftRadius.y);

    // Cirrus.
    float dCirrus = 1e12f, dCirrus2d = 1e12f, cirrusDisp = 1.f;
    float2 cirrusDens = float2(0.f);
    {
        float dLow = h - gParams.cloudLayersStartHeight.z;
        float dHigh = h - (gParams.cloudLayersStartHeight.z + gParams.cirrusThickness);
        float dBand = max(-dLow, dHigh);
        float dPlacement = 1e12f;
        if (dBand < 5.f)
        {
            float4 s = sampleGame(gGameCirrusSdf, pp, gGame.cirrusUVOffset, gGame.cirrusUVScale, min(gameMip(viewDist, gGame.cirrusTexelWorldSize), 1.f));
            float sub1 = decodeSdf(s.x, 8.f) * gGame.cirrusTexelWorldSize - gGame.sdfOffsetCirrus;
            float sub2 = decodeSdf(s.y, 8.f) * gGame.cirrusTexelWorldSize - gGame.sdfOffsetCirrusFar;
            cirrusDens = s.zw;
            dPlacement = opU(sub1, opI(sub2, -viewDist + gGame.cirrusFarTransitionStart));
        }
        dCirrus = dPlacement;
        if (dCirrus < gGame.disp1Cirrus.y)
        {
            cirrusDisp = sampleGameDisp(gGameDispCirrus, pp, gGame.dispCirrusUVOffset, gGame.dispCirrusUVScale, gameMip(viewDist, gGame.texelWorldSize.z)).x;
            dCirrus += gGame.disp1Cirrus.z - cirrusDisp * gGame.disp1Cirrus.x;
        }
        dCirrus2d = dCirrus;
        dCirrus = opIWeldQuadric(dBand, dCirrus, gGame.cloudLayersSoftRadius.z);
    }

    e.dBase = dBase;
    e.dAlto = dAlto;
    e.dCirrus = dCirrus;
    e.layerTrans = smoothstep(gParams.cloudLayersTransitionStartDensity, gParams.cloudLayersTransitionEnd, dLayersMin);
    e.inCirrus = h >= gParams.cloudLayersStartHeight.z && h <= gParams.cloudLayersStartHeight.z + gParams.cirrusThickness;

    // GetCloudDensity.
    if (min(min(dBase, dAlto), dCirrus) >= 0.f)
        return e;
    float2 layerDensity = lerp(gParams.cloudLayersDensityBottom, gParams.cloudLayersDensity.xy, e.layerTrans);
    float2 layerEdgeFade = lerp(gParams.cloudLayersEdgeFadeDistanceBottom, gParams.cloudLayersEdgeFadeDistance.xy, e.layerTrans);
    float cloudDensity = 0.f;
    float edgeFadeDistance = 100e3f;
    if (dBase < 0.f)
    {
        cloudDensity += layerDensity.x; // RainDensityCurve = 1 (rain disabled), DensityBoostAdd = 0.
        edgeFadeDistance = min(edgeFadeDistance, layerEdgeFade.x);
    }
    if (dAlto < 0.f)
    {
        cloudDensity = max(cloudDensity, layerDensity.y);
        edgeFadeDistance = min(edgeFadeDistance, layerEdgeFade.y);
    }
    cloudDensity *= distanceFade * distanceFade * step(0.f, h); // CloudMinHeight = 0
    float edgeFade = smoothstep(0.f, edgeFadeDistance, -opU(dBase, dAlto));
    float edgeFadeForHotsure = edgeFade;

    if (dCirrus < 0.f)
    {
        float den = lerp(cirrusDens.x, cirrusDens.y, smoothstep(gGame.cirrusFarTransitionStart, gGame.cirrusFarTransitionEnd, viewDist));
        float cirrusDensity = den * gParams.cloudLayersDensity.z * lerp(gGame.cirrusZeroDisplacementFade, 1.f, cirrusDisp);
        cloudDensity += cirrusDensity * max(gParams.cirrusFarFadeMinOpacity, sq(1.f - saturate((viewDist - 5000.f) / (gParams.cirrusFarFadeDistance - 5000.f))));
    }
    edgeFade = saturate(edgeFade + smoothstep(0.f, gParams.cloudLayersEdgeFadeDistance.z, -dCirrus) * linearstep(0.f, gParams.cloudLayersEdgeFadeDistance.w, -dCirrus2d));

    float density = edgeFade;
    if (!cheap)
        density += hotsureDensity(pp, h, dBase, dAlto, edgeFadeForHotsure, viewDist);
    e.density = saturate(density * cloudDensity);
    return e;
}

float coverageMap(float3 pw)
{
    float2 uv = pw.xz / gParams.weatherScale;
    float a = gBaseNoise.SampleLevel(gLinearWrap, float3(uv, 0.25f), 0).r;
    float b = gBaseNoise.SampleLevel(gLinearWrap, float3(uv * 2.7f + 0.37f, 0.75f), 0).g;
    return saturate(a * 0.75f + b * 0.35f - 0.05f);
}

/// Unclamped shape value of a cumulus-like layer: > 0 inside.
float layerShape(float3 pw, float hf, float coverage, float weather, uint layer, bool cheap)
{
    float cov = saturate(remap(weather, 1.f - coverage, 1.f, 0.f, 1.f));
    float3 off = layer == 0 ? float3(0.f) : float3(0.31f, 0.53f, 0.17f);
    float4 n = gBaseNoise.SampleLevel(gLinearWrap, pw / gParams.baseNoiseScale + off, 0);
    float fbm = n.g * 0.625f + n.b * 0.25f + n.a * 0.125f;
    float shape = remap(n.r, fbm - 1.f, 1.f, 0.f, 1.f);

    float profile;
    if (layer == 0)
    {
        // Flat base, rounded tops; towers grow with the local coverage.
        float topFrac = lerp(0.25f, 1.f, cov);
        profile = linearstep(0.f, 0.06f, hf) * linearstep(topFrac, topFrac * 0.55f, hf);
    }
    else
    {
        profile = linearstep(0.f, 0.15f, hf) * linearstep(1.f, 0.55f, hf);
    }

    float v = remap(shape * profile, 1.f - cov, 1.f, 0.f, 1.f) * cov;
    if (v > 0.f && !cheap)
    {
        float3 dn = gDetailNoise.SampleLevel(gLinearWrap, pw / gParams.detailNoiseScale, 0).rgb;
        float detail = dn.r * 0.625f + dn.g * 0.25f + dn.b * 0.125f;
        float mod = lerp(detail, 1.f - detail, saturate(hf * 6.f)); // Wispy bottoms, billowy tops.
        v = remap(v, mod * gParams.detailErosion, 1.f, 0.f, 1.f);
    }
    return v;
}

CloudEval evalCloudProcedural(float3 p, float viewDist, float distanceFade, bool cheap)
{
    CloudEval e;
    e.density = 0.f;
    e.dBase = 1e12f;
    e.dAlto = 1e12f;
    e.dCirrus = 1e12f;
    e.layerTrans = float2(1.f);
    e.inCirrus = false;

    float h = altitude(p);
    float3 pw = p + gParams.windOffset;
    float weather = coverageMap(pw);

    // Base and alto.
    float2 z = h - gParams.cloudLayersStartHeight.xy;
    float2 hf = z / gParams.layerThickness;
    if (hf.x >= 0.f && hf.x <= 1.f && gParams.layerCoverage.x > 0.f)
        e.dBase = -layerShape(pw, hf.x, gParams.layerCoverage.x, weather, 0, cheap) * gParams.sdfScale;
    if (hf.y >= 0.f && hf.y <= 1.f && gParams.layerCoverage.y > 0.f)
        e.dAlto = -layerShape(pw, hf.y, gParams.layerCoverage.y, 1.f - weather * 0.5f, 1, cheap) * gParams.sdfScale;

    e.layerTrans = smoothstep(gParams.cloudLayersTransitionStartDensity, gParams.cloudLayersTransitionEnd, z);
    float2 layerDensity = lerp(gParams.cloudLayersDensityBottom, gParams.cloudLayersDensity.xy, e.layerTrans);
    float2 layerEdgeFade = lerp(gParams.cloudLayersEdgeFadeDistanceBottom, gParams.cloudLayersEdgeFadeDistance.xy, e.layerTrans);

    float cloudDensity = 0.f;
    float edgeFadeDistance = 100e3f;
    if (e.dBase < 0.f)
    {
        cloudDensity += layerDensity.x;
        edgeFadeDistance = min(edgeFadeDistance, layerEdgeFade.x);
    }
    if (e.dAlto < 0.f)
    {
        cloudDensity = max(cloudDensity, layerDensity.y);
        edgeFadeDistance = min(edgeFadeDistance, layerEdgeFade.y);
    }
    cloudDensity *= distanceFade * distanceFade;
    float edgeFade = smoothstep(0.f, edgeFadeDistance, -min(e.dBase, e.dAlto));

    // Cirrus: thin slab, streaky density, 2D coverage.
    float zc = h - gParams.cloudLayersStartHeight.z;
    if (zc >= 0.f && zc <= gParams.cirrusThickness && gParams.layerCoverage.z > 0.f)
    {
        e.inCirrus = true;
        float3 q = pw / (gParams.baseNoiseScale * float3(gParams.cirrusStretch, 1.f, 1.f)) + float3(0.71f, 0.13f, 0.29f);
        float4 n = gBaseNoise.SampleLevel(gLinearWrap, float3(q.x, zc / gParams.cirrusThickness * 0.05f, q.z), 0);
        float streak = saturate(remap(n.r * 0.6f + n.g * 0.4f, 1.f - gParams.layerCoverage.z, 1.f, 0.f, 1.f));
        float cov2d = remap(1.f - weather, 1.f - gParams.layerCoverage.z, 1.f, 0.f, 1.f);
        float d2d = -cov2d * gParams.cloudLayersEdgeFadeDistance.w;
        float slab = min(zc, gParams.cirrusThickness - zc);
        e.dCirrus = streak > 0.f ? -min(slab, streak * gParams.sdfScale) : 1e12f;
        float cirrusDensity = streak * gParams.cloudLayersDensity.z;
        float farFade = max(gParams.cirrusFarFadeMinOpacity, sq(1.f - saturate((viewDist - 5000.f) / (gParams.cirrusFarFadeDistance - 5000.f))));
        cloudDensity += cirrusDensity * farFade;
        edgeFade = saturate(edgeFade + smoothstep(0.f, gParams.cloudLayersEdgeFadeDistance.z, -e.dCirrus) * linearstep(0.f, gParams.cloudLayersEdgeFadeDistance.w, -d2d));
    }

    e.density = saturate(edgeFade * cloudDensity);
    return e;
}

CloudEval evalCloud(float3 p, float viewDist, float distanceFade, bool cheap)
{
    if (gParams.useGameVolumes != 0)
        return evalCloudGame(p, viewDist, distanceFade, cheap);
    return evalCloudProcedural(p, viewDist, distanceFade, cheap);
}

float allSdf(CloudEval e) { return min(min(e.dBase, e.dAlto), e.dCirrus); }

/// Height fraction over the whole cloud band (CloudMinHeight = 0 .. CloudMaxHeightForBaseAltoTrace).
float bandHeightFraction(float h) { return saturate(h / gParams.cloudMaxHeightForBaseAltoTrace); }

float cloudDistanceFade(float t) { return 1.f - smoothstep(gParams.cloudFadeBeginDistance, gParams.cloudMaxTraceDistance, t); }

// ---------------------------------------------------------------------------
// Cloud lighting (calc_Fr_c, TraceSunRayCloudOpticalDepth, multiple scattering)
// ---------------------------------------------------------------------------

/// calc_Fr_c(cos_theta, g, EccentricityAttenuation, bCirrus, shadow)
float cloudPhase(float cosTheta, float g, float eccentricity, bool cirrus, float trSun)
{
    g *= eccentricity;
    float basePhase = lerp(gParams.backscatteringScaleFactor, 1.f, smoothstep(-0.15f, 0.f, g)) * phaseHG(cosTheta, g);
    float phaseFinal;
    if (cirrus)
        phaseFinal = max(basePhase, gParams.phaseScaleHGCirrus * phaseHG(cosTheta, 0.99f - gParams.hgG2Cirrus));
    else
        phaseFinal = max(basePhase, gParams.phaseScaleHG * phaseHG(cosTheta, 0.99f - gParams.hgG2));
    // SLShadowOcclusion: the forward lobe fades out in shadowed regions.
    phaseFinal = lerp(basePhase, phaseFinal, saturate(lerp(1.f, trSun * trSun, gParams.slShadowOcclusionScale)));
    return gParams.phaseScale * phaseFinal * eccentricity;
}

/// Cloud optical depth towards the sun in density x metres (multiply by sigmaScatCloud).
/// Six geometrically growing steps, trapezoidal integration, golden-ratio jitter, early bail-out.
float sunRayOpticalDepth(float3 p, float3 sunDir, CloudEval start, float viewDist, float jitterSeed)
{
    float stepFirstLen = gParams.sunRayStepFirstLen;
    if (gParams.sunRayStepFirstLenAsSdfFraction != 0.f)
        stepFirstLen = clamp(abs(allSdf(start)) * gParams.sunRayStepFirstLenAsSdfFraction, gParams.sunRayStepFirstLenAsSdfMinMax.x, gParams.sunRayStepFirstLenAsSdfMinMax.y);

    float2 band = rayShell(p, sunDir, rBot() + gParams.cloudLayersStartHeight.x,
                           rBot() + max(gParams.cloudLayersStartHeight.z + gParams.cirrusThickness, gParams.cloudMaxHeightForBaseAltoTrace), kInfinity);
    float tEnd = band.y;

    float t = 0.f;
    float tNonJittered = 0.f;
    float nextStop = stepFirstLen * gParams.sunRayStepBase;
    float od = 0.f;
    float prev = start.density;
    float halfFactor = 0.f;
    for (uint i = 0; i < 6 && t < tEnd; i++, halfFactor = 0.5f)
    {
        float jitter = frac(jitterSeed + float(i) * kGolden);
        float nextStopJittered = lerp(nextStop, tNonJittered, gParams.sunRayStepJitterVariance * jitter);
        tNonJittered = nextStop;
        float factor = halfFactor * (nextStopJittered - t);
        t = nextStopJittered;
        float3 s = p + sunDir * t;
        float cur = evalCloud(s, viewDist, 1.f, gParams.useGameVolumes != 0 || i >= 3).density; // Sun samples skip Hotsure (mips_flags != 0).
        od += factor * (prev + cur);
        prev = cur;
        if (od * gParams.sigmaScatCloud > gParams.cloudBailOutTransmittanceLogSunRay)
            break;
        nextStop *= gParams.sunRayStepBase;
    }
    return od;
}

/// Distance into the cloud used by the dark-edge term (bUseDarkEdge path).
float darkEdgeDistance(CloudEval e)
{
    float2 edgeFade = lerp(gParams.cloudLayersEdgeFadeDistanceBottom, gParams.cloudLayersEdgeFadeDistance.xy, e.layerTrans);
    float dBase = e.dBase + edgeFade.x * gParams.darkEdgeSdfBias.x;
    float dAlto = e.dAlto + edgeFade.y * gParams.darkEdgeSdfBias.y;
    float d = min(min(dBase, dAlto), e.dCirrus);
    return min(0.f, d);
}

/// Directional (sun) scattering coefficient at a sample, without sun intensity / atmosphere colour.
/// Sum over CloudScatteringTimes octaves of rho * sigma * c^i * phase_i * exp(-od * a^i * sigma).
float sunScattering(CloudEval e, float od, float cosTheta, float h)
{
    float sigma = gParams.sigmaScatCloud;
    float hf = bandHeightFraction(h);
    float dist = abs(darkEdgeDistance(e));
    bool inFadeArea = -dist < gParams.darkEdgeRange;

    float s = 0.f;
    float a = 1.f, b = 1.f, c = 1.f;
    uint n = max(gParams.cloudScatteringTimes, 1u);
    for (uint i = 0; i < n; i++)
    {
        float odSun = od * a;
        float trSun = exp(-odSun * sigma);
        float g = gParams.hgG1;
        if (gParams.useDarkEdge != 0)
        {
            float gEdge = lerp(gParams.minG, gParams.maxG,
                               exp(-odSun * gParams.sunRayTrScale * sigma) * exp(-e.density * sigma * gParams.backscatteringIntensity * dist * b));
            gEdge = clamp(gEdge, gParams.minG, gParams.maxG);
            gEdge = inFadeArea ? gEdge : min(gEdge, 0.2f);
            g = lerp(gEdge, gParams.hgG1, hf * hf);
        }
        s += trSun * b * cloudPhase(cosTheta, g, c, e.inCirrus, trSun);
        a *= gParams.msAttenuation;
        b *= gParams.msContribution;
        c *= gParams.msEccentricityAttenuationForG;
    }
    if (n == 1)
        s = exp(-od * sigma) * cloudPhase(cosTheta, gParams.hgG1, 1.f, e.inCirrus, 1.f);
    return e.density * sigma * s;
}

/// Height ambient occlusion (CloudSkyAmbientLightEnableHeightAO).
float ambientHeightAO(float h)
{
    float hf = bandHeightFraction(h);
    float ao = smoothstep(0.f, max(0.f, gParams.cloudBottomAOStartHeight), hf);
    float occ = gParams.cloudSkyAmbientLightBottomOcclusion;
    return pow(saturate(ao + saturate(1.f - occ)), max(1.f, (occ - 1.f) * 10.f));
}

/// Sky ambient light used to light clouds (DistantSkyLightLut, fade to grey, overlay colour).
float3 cloudAmbientLight()
{
    // The plugin samples the last texel (20 km) regardless of the sample altitude.
    float3 A = gDistantSkyLightLut.SampleLevel(gLinearClamp, float2(1.f, 0.5f), 0).rgb;
    float lum = luminance(A);
    A = lerp(A, float3(lum), 0.3f); // CloudSkyAmbientLightFadeToGray
    if (lum > 0.f)
    {
        float len = length(A);
        float3 col = A / len;
        A = len * normalize(col + gParams.cloudSkyAmbientLightOverlayColor);
    }
    return A;
}
