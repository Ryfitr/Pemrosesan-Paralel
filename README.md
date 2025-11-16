Pemrosesan Paralel – Gauss-Jordan dengan Pivoting
Deskripsi

Proyek ini membandingkan performa algoritma eliminasi Gauss-Jordan pada dua implementasi berbeda:

CPU (sequential)

GPU (CUDA parallel)

Keduanya digunakan untuk menyelesaikan sistem persamaan linear A x = b dengan ukuran matriks besar (256×256 hingga 4096×4096).
Versi ini sudah menggunakan Partial Pivoting agar hasil perhitungan lebih stabil dan akurat.

Struktur Proyek
CPU_Sequential.cpp     → Implementasi versi CPU
GPU_Parallel.cu        → Implementasi versi GPU (CUDA)
matrix_generator.cpp   → Pembuat dataset matriks & vektor x_true
RUN_PROJECT.bat        → Skrip otomatis untuk uji CPU vs GPU
.gitignore             → Mengabaikan file hasil build (.exe, .bin)
README.md              → Dokumentasi proyek

Cara Menjalankan

Kompilasi generator matriks

g++ matrix_generator.cpp -O2 -o gen.exe


Buat dataset uji

gen.exe 256
gen.exe 512
gen.exe 1024
gen.exe 2048
gen.exe 4096


File yang dihasilkan: matrix_*.bin dan xtrue_*.bin

Kompilasi solver CPU dan GPU

g++ CPU_Sequential.cpp -O3 -o cpu_exec.exe
nvcc GPU_Parallel.cu -O3 -o gpu_exec.exe -allow-unsupported-compiler


Jalankan pengujian

cpu_exec.exe 1024
gpu_exec.exe 1024


Atau jalankan semua ukuran otomatis:

RUN_PROJECT.bat

Contoh Hasil (rata-rata 5 kali pengujian)
Ukuran	CPU (ms)	GPU (ms)	Residual
256×256	35 ms	5 ms	< 1e-4
1024×1024	700 ms	70 ms	< 1e-4
4096×4096	>10 s	500 ms	< 1e-3

Hasil bervariasi tergantung spesifikasi CPU dan GPU.

Penjelasan Pivoting

Pivoting adalah proses menukar baris matriks selama eliminasi agar elemen pivot (A[k][k]) memiliki nilai absolut terbesar di kolom yang sedang diproses.

Tujuan Pivoting

Menghindari pembagian dengan nilai sangat kecil yang bisa menyebabkan error numerik.

Meningkatkan stabilitas hasil perhitungan.

Mencegah kegagalan eliminasi jika elemen diagonal utama bernilai nol.

Contoh Sederhana

Tanpa pivoting:

[ 0  2 | 4 ]
[ 1  3 | 5 ]


Pivot pertama bernilai 0 → algoritma gagal.
Dengan pivoting, baris ditukar sehingga pivot ≠ 0 dan proses berjalan normal.

Kenapa digunakan di proyek ini?

Versi awal tanpa pivot bekerja tetapi sering menghasilkan kesalahan pada matriks besar atau acak.
Dengan partial pivoting, performa GPU sedikit menurun tetapi hasil menjadi lebih stabil dan akurat, sehingga perbandingan CPU vs GPU menjadi valid secara ilmiah.
