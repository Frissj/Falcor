@echo off
setlocal
REM ===========================================================================
REM  VNA Shader Profiler - GPU Trace + the SM sampling profiler.
REM
REM  WHY THIS EXISTS SEPARATELY FROM VNA_NsightTrace.cmd:
REM  a plain GPU Trace answers "which unit is saturated" (answer: SM, at ~50-60%
REM  with half the warp slots empty). It does NOT say WHICH INSTRUCTIONS are
REM  burning the SM, because it collects no samples - the Shader Pipelines panel
REM  reports "Total Samples: 0" and every stall column is blank.
REM
REM  --real-time-shader-profiler turns on a high-speed sampling profiler in the
REM  SM, which populates per-instruction sample counts and warp stall reasons.
REM  That is the readout that serves the actual strategy: REDUCE INSTRUCTIONS
REM  EXECUTED. Register-pressure work has now returned <=3% on three of three
REM  attempts, so instruction count is the lever worth aiming at.
REM
REM  WHAT THIS CANNOT TELL YOU: register allocation per source line. That is a
REM  whole-kernel property the compiler decides globally; no Nsight view breaks
REM  it down. Do not re-run this expecting to find the register high-water mark.
REM
REM  --architecture Ada IS REQUIRED. real-time-shader-profiler is an ARCH-SCOPED
REM  option: without --architecture naming the GPU it applies to, ngfx accepts
REM  the flag silently and configures nothing. First attempt at this script
REM  omitted it and produced a trace with "Total Samples: 0". Ada is correct for
REM  the RTX 4080 Laptop; metric-set-id 2 is Ada's "Top-Level Triage", the same
REM  set the GUI defaults to.
REM
REM  TIMING: ngfx documents the SM sampling profiler as incurring no shader
REM  execution overhead (it consumes PCIe TX bandwidth, not SM cycles), so this
REM  is NOT necessarily slower than a plain trace. Still do not compare gpuMs
REM  across runs - the cross-run clock drift in HANDOFF_5 section 2.1 applies
REM  regardless and swamps most real wins.
REM
REM  SOURCE CORRELATION NEEDS --debug-shaders. Without it Nsight reports
REM  "Failed to find any debug symbols" and silently disables Shader Source
REM  browsing, Source-level Hot Spots and Function Call trees - you still get
REM  samples and the Instruction Mix, but nothing maps back to Slang.
REM  --debug-shaders is a Mogwai flag (Mogwai.cpp, "debug-shaders") wired to
REM  ProgramManager::setGenerateDebugInfoEnabled, so this needs NO rebuild.
REM
REM  CAVEAT: debug info can perturb codegen slightly. Treat sample ATTRIBUTION
REM  from this run as authoritative, but re-read absolute register counts from a
REM  plain VNA_NsightTrace.cmd run without --debug-shaders.
REM
REM  Usage:  VNA_NsightShaderProf.cmd [path-to-mogwai-script]
REM          defaults to scripts\WdasSkyVNA.py
REM ===========================================================================

set "NGFX=C:\Program Files\NVIDIA Corporation\Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\ngfx.exe"
set "FALCOR=C:\Users\Friss\Documents\Clouds\Falcor"
set "BIN=%FALCOR%\build\windows-vs2022\bin\Release"
set "OUTDIR=%FALCOR%\vna_measurements\nsight"

set "SCRIPT=%~1"
if "%SCRIPT%"=="" set "SCRIPT=%FALCOR%\scripts\WdasSkyVNA.py"

if not exist "%NGFX%" (
    echo [VNA] ngfx.exe not found at:
    echo       %NGFX%
    echo       Nsight Graphics may have updated - check the version folder under
    echo       C:\Program Files\NVIDIA Corporation\
    pause
    exit /b 1
)
if not exist "%BIN%\Mogwai.exe" (
    echo [VNA] Mogwai.exe not found at %BIN%
    pause
    exit /b 1
)
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo [VNA] Nsight GPU Trace + Real-Time Shader Profiler
echo [VNA]   script : %SCRIPT%
echo [VNA]   output : %OUTDIR%
echo [VNA]   waiting 200 frames for the sustained clock state before tracing
echo [VNA]   arch   : Ada, metric-set 2 (Top-Level Triage), SM sampler ON
echo.

REM --start-after-frames 200   : skip shader compile + boost-clock frames.
REM                              MEASURED: 800 was tried (on the theory that the
REM                              gpuMs drift past frame 128 meant the clock had not
REM                              converged) and made no meaningful difference to
REM                              the sampled frames, so it went back to 200. The
REM                              residual gpuMs variance is not a warmup effect and
REM                              waiting longer does not remove it - which is why
REM                              attribution comes from samples, not from timings.
REM --architecture Ada         : scopes the two options below to this GPU.
REM --metric-set-id 2          : Ada's "Top-Level Triage" (the GUI default).
REM --real-time-shader-profiler: the whole point of this script (see header).
REM --multi-pass-metrics       : REQUIRED for the Trace Analysis view. Without
REM                              it that view banners "Analysis view works best
REM                              with the multi-pass metrics enabled - analysis
REM                              results will be limited" and collapses almost
REM                              every category: the 2026-07-20 capture produced
REM                              exactly ONE populated entry under "Warp Launch
REM                              Stalled by Reasons" (CS Warp Launch Stalled
REM                              Register Allocation, 25.4%). That single entry
REM                              was enough to prove the kernel is register-
REM                              limited, but SM Warp Issue Stalls, SM Warp
REM                              Occupancy and Unit Throughputs all stayed empty.
REM                              Costs extra passes over MULTIPLE frames, so the
REM                              capture takes longer - worth it for analysis
REM                              runs, drop it for a quick sample-only pass.
REM --limit-to-frames 5        : a handful of steady-state frames is plenty; 5
REM                              frames already yields ~8M SM samples, ample for
REM                              the entries that matter.
REM --max-duration-ms 4000     : safety bound on the CAPTURE, not the wait - the
REM                              200-frame wait already exceeded 4000ms of
REM                              wall-clock and traced fine, and 800 frames still
REM                              does. NOTE: ngfx HARD CAPS this below 10000;
REM                              passing 16000 is rejected outright with
REM                              "Invalid activity options".
"%NGFX%" ^
  --activity "GPU Trace Profiler" ^
  --platform Windows ^
  --exe "%BIN%\Mogwai.exe" ^
  --dir "%BIN%" ^
  --args "--script %SCRIPT% --debug-shaders" ^
  --output-dir "%OUTDIR%" ^
  --architecture Ada ^
  --metric-set-id 2 ^
  --real-time-shader-profiler ^
  --multi-pass-metrics ^
  --start-after-frames 200 ^
  --limit-to-frames 5 ^
  --max-duration-ms 4000 ^
  --no-timeout

echo.
echo [VNA] done - trace written to %OUTDIR%
echo [VNA] Open the .gputrace, select the ExecuteIndirect (shadeMain), then use
echo [VNA] the Shader Source tab. "Total Samples" in Shader Pipelines must now
echo [VNA] be non-zero; if it is still 0 the sampler did not arm.
pause
