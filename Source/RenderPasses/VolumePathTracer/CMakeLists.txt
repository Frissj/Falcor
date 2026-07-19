add_plugin(VolumePathTracer)

target_sources(VolumePathTracer PRIVATE
    VolumePathTracer.cpp
    VolumePathTracer.h
    VolumePathTracer.cs.slang
    VolumeInstanceSampler.slang
)

target_copy_shaders(VolumePathTracer RenderPasses/VolumePathTracer)

target_source_group(VolumePathTracer "RenderPasses")
