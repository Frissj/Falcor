# VNA Step 2, part 2 - turn the captured pair into numbers.
# Loads the newest vna_1spp / vna_ref EXRs from vna_measurements\ and builds:
#   ImageLoader(1spp) --> FLIPPass.testImage        FLIP error map + pooled mean
#   ImageLoader(ref)  --> FLIPPass.referenceImage   (check "Compute pooled FLIP
#                                                    values" box shows the mean
#                                                    in the FLIPPass UI)
#   ImageLoader(1spp) --> ErrorMeasurePass.Source   MSE, appended per frame to
#   ImageLoader(ref)  --> ErrorMeasurePass.Reference  vna_measurements\vna_mse.csv
#
# Run with:
#   Mogwai.exe --script scripts/VNA_Step2_Flip.py
#
# READING THE RESULT
#  - The FLIP mean (FLIPPass UI, "Compute pooled FLIP values") is the headline
#    number: perceptual distance of today's 1-spp frame from ground truth.
#    That is the gap section-5 ReSTIR has to close.
#  - vna_mse.csv records MSE per frame (all rows identical here - static
#    inputs; the pass appends one row per rendered frame).
#  - The errorMapDisplay output shows WHERE the error lives (expect: worst in
#    thin wisps and dark underbelly).

import glob
import os

from falcor import *

OUT_DIR = r"C:\Users\Friss\Documents\Clouds\Falcor\vna_measurements"

def newest(pattern):
    files = sorted(glob.glob(os.path.join(OUT_DIR, pattern)), key=os.path.getmtime)
    if not files:
        raise RuntimeError(
            "VNA: no files match {} in {} - run scripts/VNA_Step2_Measure.py first.".format(pattern, OUT_DIR))
    return files[-1]

# The linear HDR captures (AccumulatePass.output), not the tonemapped PNGs.
test_path = newest("vna_1spp.VolumePathTracer.AccumulatePass.output.*.exr")
ref_path = newest("vna_ref.VolumePathTracer.AccumulatePass.output.*.exr")
print("VNA: test      = {}".format(test_path))
print("VNA: reference = {}".format(ref_path))

def render_graph_Flip():
    g = RenderGraph("VNAFlip")

    Test = createPass("ImageLoader", {'filename': test_path, 'mips': False, 'srgb': False})
    g.addPass(Test, "Test")
    Ref = createPass("ImageLoader", {'filename': ref_path, 'mips': False, 'srgb': False})
    g.addPass(Ref, "Ref")

    # HDR-FLIP on linear inputs; pooled values give the mean/min/max in the UI.
    FLIP = createPass("FLIPPass", {'isHDR': True, 'computePooledFLIPValues': True, 'useMagma': True})
    g.addPass(FLIP, "FLIP")
    g.addEdge("Test.dst", "FLIP.testImage")
    g.addEdge("Ref.dst", "FLIP.referenceImage")
    g.markOutput("FLIP.errorMapDisplay")

    # MSE, logged to CSV so there is a recorded artifact, not a UI readout.
    Err = createPass("ErrorMeasurePass", {
        'MeasurementsFilePath': os.path.join(OUT_DIR, "vna_mse.csv"),
        'ComputeSquaredDifference': True,
        'IgnoreBackground': False,
        'UseLoadedReference': False,
        'ReportRunningError': False,
    })
    g.addPass(Err, "Err")
    g.addEdge("Test.dst", "Err.Source")
    g.addEdge("Ref.dst", "Err.Reference")
    g.markOutput("Err.Output")

    return g

m.addGraph(render_graph_Flip())
print("VNA: FLIP mean is in the FLIPPass UI group; MSE rows land in vna_mse.csv.")
