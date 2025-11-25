# ⚙️ Eliminasi Gauss-Jordan dengan Partial Pivoting (CPU vs. GPU)

Proyek ini membandingkan kinerja algoritma **Eliminasi Gauss-Jordan** dengan **Partial Pivoting** untuk menyelesaikan sistem persamaan linear $A \times X = b$ pada dua implementasi komputasi yang berbeda: **CPU Sequential** (tradisional) dan **GPU Parallel** (menggunakan CUDA).

---

## 📋 Deskripsi Proyek

Tujuan utama proyek ini adalah menganalisis **efisiensi dan _speedup_** yang ditawarkan oleh komputasi paralel GPU (khususnya untuk operasi intensif seperti eliminasi matriks) dibandingkan dengan implementasi _sequential_ CPU. Kami berfokus pada matriks berukuran besar.

### 🎯 Tentang Partial Pivoting

Partial Pivoting adalah teknik untuk **meningkatkan stabilitas numerik** selama eliminasi matriks. Ini dilakukan dengan menukar baris matriks agar elemen _pivot_ $A[k][k]$ selalu memiliki **nilai absolut terbesar** di kolom yang sedang diproses.



| Keuntungan Partial Pivoting |
| :--- |
| * Stabilitas numerik lebih baik |
| * Mengurangi potensi _error_ pembulatan |
| * Mencegah kegagalan saat elemen _pivot_ bernilai nol |

---

## 🏗️ Struktur Proyek

| File | Deskripsi |
| :--- | :--- |
| `CPU_Sequential.cpp` | Implementasi Eliminasi Gauss-Jordan versi **CPU Sequential** (C++). |
| `GPU_Parallel.cu` | Implementasi Eliminasi Gauss-Jordan versi **GPU Parallel** (CUDA). |
| `matrix_generator.cpp` | Program _helper_ untuk menghasilkan matriks uji $A$ dan vektor $b$ dengan berbagai ukuran. |
| `RUN_PROJECT.sh` | Skrip **Bash** untuk mengotomatisasi kompilasi dan pengujian. |
| `.gitignore` | File konfigurasi Git. |

---

## 🔧 Teknologi yang Digunakan

* **C++17:** Bahasa pemrograman utama.
* **CUDA (Compute Unified Device Architecture):** Platform komputasi paralel GPU.
* **NVIDIA CUDA Toolkit:** Diperlukan untuk kompilasi kode `.cu`.
* **GCC/G++:** _Compiler_ standar C++ untuk Linux/Bash.

---

## 🚀 Cara Menjalankan (Menggunakan Bash)

Pastikan Anda memiliki **NVIDIA CUDA Toolkit** dan **GCC** terpasang di sistem Anda.

### 1. Kompilasi Program

```bash
# Kompilasi Matrix Generator
g++ -std=c++17 matrix_generator.cpp -o generator_exec

# Kompilasi Versi CPU Sequential
g++ -std=c++17 CPU_Sequential.cpp -o cpu_exec

# Kompilasi Versi GPU Parallel (membutuhkan nvcc)
nvcc GPU_Parallel.cu -o gpu_exec -allow-unsupported-compiler
```

### 2. Jalankan Test

```bash
./RUN_PROJECT.bat
```

## 📊 Hasil Performance (Contoh Benchmark)

Data ini adalah contoh hasil benchmark yang menunjukkan waktu eksekusi dalam milidetik (ms) dan faktor speedup GPU terhadap CPU.

| Ukuran Matriks | CPU (ms) | Residual (≈ Akurasi) | GPU (ms)| Residual (≈ Akurasi) |
| :--- | :--- | :--- | :--- | :--- |
| 256×256 | ~35 | xxxxx | ~218 | xxxxx |
| 512×512 | ~276 | xxxxx | ~500 | xxxxx |
| 1024×1024 | xxxx | xxxxx | xxxx | xxxxx |
| 2048×2048 | xxxxx | xxxxx | xxxx | xxxxx |
| 4096×4096 | xxxxx | xxxxx | xxxx | xxxxx |
