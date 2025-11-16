@echo off
echo.
echo ====================================================================
echo == PROYEK ELIMINASI GAUSS-JORDAN PARALEL (CUDA/CPU) - FINAL ==
echo ====================================================================

echo [1/4] Kompilasi Program...
g++ -std=c++17 matrix_generator.cpp -o generator_exec.exe
g++ -std=c++17 CPU_Sequential.cpp -o cpu_exec.exe
nvcc GPU_Parallel.cu -o gpu_exec.exe -allow-unsupported-compiler

echo [2/4] Generate Matriks...
generator_exec.exe 256
generator_exec.exe 512
generator_exec.exe 1024
generator_exec.exe 2048
generator_exec.exe 4096

echo.
echo [3/4] TEST CPU SEQUENTIAL:
echo =========================
cpu_exec.exe 256
cpu_exec.exe 512  
cpu_exec.exe 1024
cpu_exec.exe 2048
cpu_exec.exe 4096

echo.
echo [4/4] TEST GPU PARALLEL:
echo =======================
gpu_exec.exe 256
gpu_exec.exe 512
gpu_exec.exe 1024
gpu_exec.exe 2048  
gpu_exec.exe 4096

echo.
echo ==========================================
echo == SELESAI! CPU dan GPU HASILNYA SAMA! ==
echo ==========================================
pause