@echo off
setlocal
REM ===========================================================================
REM  VNA GPU Trace - launches Mogwai under Nsight Graphics' GPU Trace Profiler.
REM
REM  WHY: every VNA perf number so far comes from work counters (cells/taps/
REM  aabbTests) plus gpuMs. Those cannot rank the three remaining cost centres
REM  against each other:
REM      1. TLAS traversal   - 30M AABB tests/frame, counted by NOTHING
REM      2. the tap's dependent chase (indirectionTex -> atlasTex)
REM      3. cell fetches     - now 1/cell after the rangeMean pack
REM  GPU Trace gives SM occupancy, warp stall reasons and memory throughput,
REM  which is what actually separates them.
REM
REM  START FRAME MATTERS. This machine's GPU drops to a sustained clock around
REM  frame ~128 and stays there (14.8ms cold -> ~27ms steady, with IDENTICAL
REM  work counters). Tracing frame 0 profiles a boost-clock frame that does not
REM  represent the shipping state, so we wait 200 frames first.
REM
REM  Usage:  VNA_NsightTrace.cmd [path-to-mogwai-script]
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

echo [VNA] Nsight GPU Trace
echo [VNA]   script : %SCRIPT%
echo [VNA]   output : %OUTDIR%
echo [VNA]   waiting 200 frames for the sustained clock state before tracing
echo.

REM --start-after-frames 200 : skip shader compile + boost-clock frames.
REM --limit-to-frames 5      : a handful of steady-state frames is plenty.
REM --max-duration-ms 4000   : safety bound; 5 frames at ~27ms needs far less,
REM                            but the wait-for-frame-200 happens inside it.
"%NGFX%" ^
  --activity "GPU Trace Profiler" ^
  --platform Windows ^
  --exe "%BIN%\Mogwai.exe" ^
  --dir "%BIN%" ^
  --args "--script %SCRIPT%" ^
  --output-dir "%OUTDIR%" ^
  --start-after-frames 200 ^
  --limit-to-frames 5 ^
  --max-duration-ms 4000 ^
  --no-timeout

echo.
echo [VNA] done - trace written to %OUTDIR%
echo [VNA] Open the .gputrace in the Nsight Graphics UI to read it.
echo [VNA] GPU perf counters are already open to all users on this machine
echo [VNA] (RmProfilingAdminOnly=0), so this should not need elevation.
pause
