# IMPLEMENTASI PARALEL ELIMINASI GAUSS-JORDAN BERBASIS CUDA

## 🎯 Abstraksi Proyek

Proyek ini bertujuan untuk menganalisis **efisiensi dan skalabilitas** algoritma **Eliminasi Gauss-Jordan (GJE)** yang diimplementasikan secara sekuensial (CPU) dan paralel (GPU melalui **CUDA**).

Analisis berfokus pada:
1.  Perbandingan Waktu Eksekusi dan perhitungan **Speedup**.
2.  Evaluasi **Stabilitas Numerik** (Residual) untuk SPL skala besar.

---

## 🛠️ Lingkungan Pengembangan

### 1. Kebutuhan Perangkat Keras (Perangkat Uji)
* [cite_start]**CPU:** Intel(R) Core(TM) i5-12500H [cite: 59, 157]
* [cite_start]**GPU:** NVIDIA GeForce RTX 3050 Notebook [cite: 59, 157]
* [cite_start]**Sistem Operasi:** Windows 11 [cite: 59, 158]

### 2. Kebutuhan Perangkat Lunak

| No. | Perangkat Lunak | Tujuan Utama |
| :---: | :--- | :--- |
| 1 | **NVIDIA CUDA Toolkit (v13.0)** | [cite_start]Kompilasi program paralel (`nvcc`). [cite: 60, 159] |
| 2 | **Microsoft Visual Studio (atau Build Tools)** | Menyediakan *toolchain* dan *linker* Windows. |
| 3 | **C++ Compiler (g++)** | [cite_start]Kompilasi program sekuensial (`g++`). [cite: 159] |
| 4 | **Visual Studio Code (VS Code)** | Lingkungan pengembangan utama. |

### 3. Struktur Direktori

Pastikan file-file berikut berada di dalam folder utama proyek (`Pemrosesan Paralel/`):

* `CPU_Sequential.cpp` (Implementasi Sekuensial)
* `GPU_Parallel.cu` (Implementasi Paralel CUDA)
* `RUN_PROJECT.bat` (Skrip Eksekusi)

---

## 🚀 Prosedur Eksekusi

### 1. Persiapan Terminal

Eksekusi harus dilakukan dari terminal yang telah memuat variabel lingkungan yang benar (Visual Studio build environment).

1.  **Buka** **Visual Studio Developer Command Prompt** atau **Terminal VS Code**.
2.  **Arahkan Direktori** ke folder proyek Anda (Ganti `/path/to/project/` dengan lokasi folder proyek Anda yang sebenarnya):
    ```bash
    cd /path/to/project/Pemrosesan Paralel
    ```

### 2. Pelaksanaan Uji

Jalankan skrip yang akan mengompilasi dan mengeksekusi pengujian multi-skala:

```bash
RUN_PROJECT.bat
