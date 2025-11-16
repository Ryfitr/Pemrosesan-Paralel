⚙️ Pemrosesan Paralel – Gauss-Jordan dengan Pivoting
🧭 Deskripsi

Proyek ini membandingkan performa algoritma Gauss-Jordan Elimination pada dua implementasi berbeda:

💻 CPU (Sequential)

⚡ GPU (CUDA Parallel)

Keduanya digunakan untuk menyelesaikan sistem persamaan linear A × x = b dengan ukuran matriks besar (256×256 hingga 4096×4096).
Versi ini sudah menggunakan Partial Pivoting agar hasil perhitungan lebih stabil dan akurat.

🧩 Struktur Proyek
📁 CPU_Sequential.cpp     → Implementasi versi CPU (Sequential)
📁 GPU_Parallel.cu        → Implementasi versi GPU (CUDA Parallel)
📁 matrix_generator.cpp   → Pembuat dataset matriks & vektor x_true
📁 RUN_PROJECT.bat        → Skrip otomatis untuk uji CPU vs GPU
📄 .gitignore             → Mengabaikan file hasil build (.exe, .bin)
📘 README.md              → Dokumentasi proyek

🚀 Cara Menjalankan
1️⃣ Kompilasi generator matriks
g++ matrix_generator.cpp -O2 -o gen.exe

2️⃣ Buat dataset uji
gen.exe 256
gen.exe 512
gen.exe 1024
gen.exe 2048
gen.exe 4096


📦 Hasil: matrix_*.bin dan xtrue_*.bin

3️⃣ Kompilasi solver CPU & GPU
g++ CPU_Sequential.cpp -O3 -o cpu_exec.exe
nvcc GPU_Parallel.cu -O3 -o gpu_exec.exe -allow-unsupported-compiler

4️⃣ Jalankan pengujian
cpu_exec.exe 1024
gpu_exec.exe 1024


Atau jalankan semua ukuran otomatis:

RUN_PROJECT.bat

📊 Contoh Hasil (rata-rata 5x pengujian)
Ukuran Matriks	CPU (ms)	GPU (ms)	Residual
256×256	35	5	< 1e-4
1024×1024	700	70	< 1e-4
4096×4096	>10 000	500	< 1e-3

⏱️ Waktu aktual dapat berbeda tergantung spesifikasi perangkat keras.

🧠 Apa Itu Pivoting?

Pivoting adalah proses menukar baris matriks selama eliminasi Gauss-Jordan agar elemen pivot (A[k][k]) selalu memiliki nilai absolut terbesar di kolom tersebut.

🎯 Tujuan Pivoting

🔹 Meningkatkan stabilitas numerik — menghindari pembagian dengan nilai yang sangat kecil.

🔹 Mengurangi propagasi error akibat pembulatan floating-point.

🔹 Mencegah kegagalan eliminasi ketika elemen diagonal utama bernilai nol.

📘 Contoh Sederhana

Tanpa pivoting:

[ 0  2 | 4 ]
[ 1  3 | 5 ]


Pivot pertama bernilai 0 → algoritma gagal.
Dengan pivoting → baris ditukar sehingga pivot ≠ 0 dan proses berjalan normal.

💡 Kenapa Digunakan di Proyek Ini?

Versi awal (tanpa pivot) memang lebih cepat, tetapi sering menghasilkan error besar pada matriks acak atau besar.
Dengan partial pivoting, performa sedikit menurun, namun hasil jauh lebih stabil dan akurat — membuat perbandingan CPU vs GPU lebih valid secara ilmiah.
