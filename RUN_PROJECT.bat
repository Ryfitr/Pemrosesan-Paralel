@echo off
echo.
echo ====================================================================
echo == PROYEK ELIMINASI GAUSS-JORDAN PARALEL (CUDA/CPU) - FINAL == 
echo ====================================================================

echo [1/4] Kompilasi Program...
g++ -std=c++17 matrix_generator.cpp -o generator_exec.exe
g++ -std=c++17 CPU_Sequential.cpp -o cpu_exec.exe
nvcc GPU_Parallel.cu -o gpu_exec.exe

REM 5 ukuran matriks
set SIZES=256 512 1024 2048 4096

echo.
echo [2/4] Generate 30 Matriks (5 ukuran x 6 matriks per ukuran)...
for %%N in (%SIZES%) do (
    for /L %%I in (1,1,6) do (
        echo Generate matrix N=%%N ID=%%I
        generator_exec.exe %%N %%I
    )
)

echo.
echo [3/4] TEST CPU (1x per matriks) -> cpu_results.csv
echo =================================================================
for %%N in (%SIZES%) do (
    for /L %%I in (1,1,6) do (
        echo CPU: N=%%N ID=%%I
        cpu_exec.exe %%N %%I
    )
)

echo.
echo [4/4] TEST GPU (1x per matriks) -> gpu_results.csv
echo =================================================================
for %%N in (%SIZES%) do (
    for /L %%I in (1,1,6) do (
        echo GPU: N=%%N ID=%%I
        gpu_exec.exe %%N %%I
    )
)

echo.
echo ==========================================
echo == SELESAI! 30 SPL CPU ^& GPU Selesai   ==
echo == File: cpu_results.csv ^& gpu_results.csv ==
echo ==========================================
pause
