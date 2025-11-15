# IMPLEMENTASI PARALEL ELIMINASI GAUSS-JORDAN BERBASIS CUDA

## 🎯 Ringkasan Proyek

Proyek ini mengimplementasikan dan membandingkan kinerja algoritma **Eliminasi Gauss-Jordan (GJE)** secara sekuensial (pada **CPU**) dan paralel (pada **GPU** menggunakan **CUDA**).

Tujuan utama proyek adalah:
1.  Mengukur tingkat percepatan (*speedup*) dan skalabilitas ketika diterapkan pada **Sistem Persamaan Linear (SPL) skala besar**.
2.  Menganalisis **stabilitas numerik** algoritma GJE tanpa *pivoting* pada skala matriks hingga $4096 \times 4096$.

---

## 🛠️ Lingkungan Pengembangan dan Dependensi

Untuk mengompilasi dan mereplikasi hasil eksperimen ini, lingkungan berikut harus disiapkan:

### 1. Kebutuhan Perangkat Keras (Perangkat Uji)
* **CPU:** Intel(R) Core(TM) i5-12500H
* **GPU:** NVIDIA GeForce RTX 3050 Notebook (P = 2048 CUDA Cores)
* **Sistem Operasi:** Windows 11

### 2. Prasyarat Instalasi Perangkat Lunak

| No. | Perangkat Lunak | Tujuan |
| :---: | :--- | :--- |
| 1 | **NVIDIA CUDA Toolkit (v12.x/13.x)** | Menyediakan *compiler* `nvcc` dan *library* CUDA untuk kompilasi dan eksekusi GPU. |
| 2 | **Microsoft Visual Studio (atau Build Tools)** | Menyediakan *toolchain* dan *linker* Windows yang wajib diintegrasikan dengan CUDA Toolkit. |
| 3 | **C++ Compiler (g++)** | Digunakan untuk mengompilasi program CPU sekuensial. |
| 4 | **Git** | Sistem Kontrol Versi (VCS) untuk pengelolaan kode dan *branch*. |
| 5 | **Visual Studio Code (VS Code)** | *Code Editor* utama yang digunakan untuk pengembangan. |

### 3. Konfigurasi Lingkungan (PATH)

* **Aplikasi yang Digunakan:** Eksekusi proyek harus dilakukan dari terminal yang telah memuat variabel lingkungan yang benar.
* **Cara Pelaksanaan:** Gunakan **Visual Studio Developer Command Prompt** atau terminal VS Code yang dimuat dengan *profil Visual Studio build tools*. Terminal ini secara otomatis mengatur **PATH** agar `g++` dan `nvcc` dapat dikenali.

---

## 🚀 Cara Menjalankan Proyek

### 1. Struktur File Proyek