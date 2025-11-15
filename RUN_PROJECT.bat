@echo off
setlocal

echo.
echo ====================================================================
echo == PROYEK ELIMINASI GAUSS-JORDAN PARALEL (CUDA/CPU) ==
echo ====================================================================

:: 1. Kompilasi Program
echo.
echo [1/3] Kompilasi Program CPU Sequential...
g++ -std=c++17 CPU_Sequential.cpp -o cpu_exec
if errorlevel 1 goto error_cpu

echo.
echo [2/3] Kompilasi Program GPU Parallel (CUDA)...
nvcc GPU_Parallel.cu -o gpu_exec -allow-unsupported-compiler
if errorlevel 1 goto error_gpu

:: 2. Eksekusi Multi-Skala: Perintah Langsung Berurutan
echo.
echo [3/3] Eksekusi Multi-Skala dan Analisis Kinerja...

:: Header CSV (Comma Separated Values)
echo N,Waktu_CPU_ms,Residual_CPU,Waktu_GPU_ms,Residual_GPU > RESULT_FINAL.csv

:: --- N=256 ---
echo.
echo Menjalankan untuk N=256... (Sangat cepat)
:: Eksekusi dan tambahkan output N,Waktu,Residual ke file CSV
cpu_exec.exe 256 >> RESULT_FINAL.csv
gpu_exec.exe 256 >> RESULT_FINAL.csv

:: --- N=512 ---
echo.
echo Menjalankan untuk N=512... (Cepat)
cpu_exec.exe 512 >> RESULT_FINAL.csv
gpu_exec.exe 512 >> RESULT_FINAL.csv

:: --- N=1024 ---
echo.
echo Menjalankan untuk N=1024... (Sedang)
cpu_exec.exe 1024 >> RESULT_FINAL.csv
gpu_exec.exe 1024 >> RESULT_FINAL.csv

:: --- N=2048 ---
echo.
echo Menjalankan untuk N=2048... (Membutuhkan beberapa detik)
cpu_exec.exe 2048 >> RESULT_FINAL.csv
gpu_exec.exe 2048 >> RESULT_FINAL.csv

:: --- N=4096 ---
echo.
echo Menjalankan untuk N=4096... (Membutuhkan waktu lama, menit)
cpu_exec.exe 4096 >> RESULT_FINAL.csv
gpu_exec.exe 4096 >> RESULT_FINAL.csv

echo.
echo ====================================================================
echo == PROYEK SELESAI DIJALANKAN. ==
echo == DATA LENGKAP TERSIMPAN DALAM FILE: RESULT_FINAL.csv ==
echo ====================================================================
goto end

:error_cpu
echo.
echo ERROR: Kompilasi CPU gagal.
goto end

:error_gpu
echo.
echo ERROR: Kompilasi GPU gagal.
goto end

:end
pause
endlocal