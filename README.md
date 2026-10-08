# Dokumen Implementasi CUDA pada Simple CUDA Demo

## Ringkasan
Notebook `simple_cuda_demo.ipynb` menunjukkan penggunaan CUDA dasar dengan PyTorch untuk memfilter sinyal sintetis ECG menggunakan moving average filter (1D convolution) yang dijalankan pada GPU. Untuk memperlihatkan percepatan GPU, ukuran sinyal ditingkatkan menjadi 30 detik dengan sampling 500 Hz dan 12 lead (seperti ECG standar). Implementasi meliputi:
- Pemeriksaan ketersediaan CUDA.
- Pembuatan sinyal sintetis multi-lead.
- Pemindahan data ke device (GPU atau CPU).
- Implementasi fungsi moving average filter menggunakan 1D convolution dengan padding refleks.
- Pengukuran waktu eksekusi pada GPU dan CPU untuk perbandingan.
- Visualisasi hasil filtering (jika matplotlib tersedia).

## Detail Implementasi
1. **Penentuan Device**: Menggunakan `torch.cuda.is_available()` untuk memilih antara GPU dan CPU.
2. **Pembuatan Sinyal**: Sinyal ECG sintetis dibuat dengan menjumlahkan komponen sinusoidal dan noise untuk menghasilkan 12 lead dengan durasi 30 detik.
3. **Filtering**: Fungsi `moving_average_filter` menerapkan konvolusi 1D dengan kernel rata-rata berukuran yang dapatディスアジャスト (default 11). Setiap lead diproses secara terpisah.
4. **Pengukuran Waktu**: Waktu eksekusi diukur menggunakan `time.time()` dengan `torch.cuda.synchronize()` untuk GPU agar waktu benar-benar tercatat.
5. **Perbandingan**: Sama sekali filtering dijalankan pada CPU untuk memperlihatkan speedup.
6. **Hasil**: Variance sinyal sebelum dan sesudah filtering dihitung untuk memastikan smoothing terjadi, dan plot ditampilkan jika matplotlib tersedia.

## Cara Menggunakan
- Jalankan notebook di環境 yang mendukung GPU (seperti Kaggle atau Google Colab dengan akselerator GPU diaktifkan).
- Pastikan PyTorch terinstal dengan dukungan CUDA.
- Notebook akan secara otomatis menggunakan GPU jika tersedia, otherwise fallback ke CPU.

## Catatan
- Untuk sinyal yang sangat kecil, overhead peluncuran GPU dapat membuat CPU lebih cepat; keuntungan GPU terlihat pada sinyal yang lebih besar seperti yang digunakan dalam demo ini.
- Implementasi ini adalah demonstrasi konsep dan tidak menggantikan filter yang lebih kompleks yang digunakan dalam pra-pemrosesan ECG sejati.