add_plugin(Cloudly)

target_sources(Cloudly PRIVATE
    Cloudly.cpp
    Cloudly.h
    CloudlyParams.slang
    CloudlyCommon.slang
    AtmosphereLut.cs.slang
    NoiseTextureGen.cs.slang
    SkyTrace.cs.slang
    TemporalUpscale.cs.slang
    CompositeWithScreen.cs.slang
)

target_copy_shaders(Cloudly RenderPasses/Cloudly)

target_source_group(Cloudly "RenderPasses")
