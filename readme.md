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

### 2. Prasyarat Instalasi Perangkat Lunak & Aplikasi

| No. | Perangkat Lunak | Tujuan |
| :---: | :--- | :--- |
| 1 | **NVIDIA CUDA Toolkit (v12.x/13.x)** | Menyediakan *compiler* `nvcc` dan *library* CUDA. |
| 2 | **Microsoft Visual Studio (atau Build Tools)** | Menyediakan *toolchain* dan *linker* Windows yang wajib diintegrasikan dengan CUDA Toolkit. |
| 3 | **C++ Compiler (g++)** | Digunakan untuk mengompilasi program CPU sekuensial. |
| 4 | **Git** | Sistem Kontrol Versi (VCS) untuk pengelolaan kode. |
| 5 | **Visual Studio Code (VS Code)** | **Aplikasi Editor** utama dan terminal terintegrasi. |

### 3. Konfigurasi Lingkungan (PATH)

* **Cara Pelaksanaan:** Gunakan **Visual Studio Developer Command Prompt** atau terminal VS Code yang dimuat dengan *profil Visual Studio build tools*. Terminal ini secara otomatis mengatur **PATH** agar `g++` dan `nvcc` dapat dikenali.

---

## 🚀 Cara Menjalankan Proyek

### 1. Struktur File Proyek

Pastikan direktori Anda memiliki struktur file berikut:
Pemrosesan Paralel/ ├── CPU_Sequential.cpp ├── GPU_Parallel.cu └── RUN_PROJECT.bat

### 2. Eksekusi

1.  **Buka** **Visual Studio Developer Command Prompt** atau **Terminal VS Code**.
2.  **Pindah Direktori** ke folder proyek:
    ```bash
    cd "C:\Users\ariyo\Documents\Vs Code\Pemrosesan Paralel"
    ```
3.  **Jalankan *Batch Script*:**
    ```bash
    RUN_PROJECT.bat
    ```

> **RUN\_PROJECT.bat** akan menjalankan kompilasi (`g++` dan `nvcc`) diikuti oleh eksekusi multi-skala $N=256$ hingga $4096$. Setiap pengujian diulang 5 kali untuk mendapatkan waktu rata-rata.

---

## 📊 Temuan Eksperimental (Hasil Baseline Tanpa Pivoting)

Hasil di *branch* **`baseline-no-pivoting`** menunjukkan kinerja yang kuat tetapi kerentanan numerik yang parah.

<details>
<summary>Klik untuk melihat Data Kinerja dan Residual Lengkap</summary>

| Ukuran Matriks ($N$) | $T_{CPU}$ (ms) | $R_{CPU}$ (Residual) | $T_{GPU}$ (ms) | $R_{GPU}$ (Residual) | Speedup ($S$) |
| :------------------: | :------------: | :------------------: | :-----------: | :------------------: | :-----------: |
| 256 | 83.6482 | 0.4097 | 39.7831 | nan | 2.10 |
| 512 | 644.3658 | 14.4033 | 103.4166 | nan | 6.23 |
| 1024 | 5200.2214 | 33.5738 | 684.0984 | nan | 7.60 |
| 2048 | 42506.4422 | 447.0670 | 3539.0935 | nan | 12.01 |
| **4096** | **309941.8208** | **69148.6719** | **21670.7520** | nan | **14.30** |

</details>

### 1. Poin Kinerja

* **Percepatan Tinggi:** *Speedup* mencapai **14.30** pada matriks $4096 \times 4096$, membuktikan efisiensi komputasi paralel GPU untuk masalah $O(N^3)$.

### 2. Poin Stabilitas Numerik

* **Kegagalan Total:** Nilai Residual yang sangat besar ($69148.6719$) dan kemunculan **`nan`** membuktikan bahwa algoritma Gauss-Jordan **tanpa *Partial Pivoting*** tidak stabil dan tidak dapat menghasilkan solusi yang akurat pada SPL skala besar.

---

## ➡️ Status Pengembangan (*Branches*)

| Branch | Status | Tujuan |
| :--- | :--- | :--- |
| `main` | Stable | Versi stabil utama proyek. |
| **`baseline-no-pivoting`** | **Complete** | **BASELINE Data:** Mengunci kode yang menghasilkan data eksperimental awal (termasuk *speedup* tinggi dan kegagalan *residual*). |
| `implementasi-pivoting` | In Progress | Pengembangan fitur untuk mengimplementasikan **Partial Pivoting** guna mengatasi masalah ketidakstabilan numerik. |