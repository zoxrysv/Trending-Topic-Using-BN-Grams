# 📊 Trending Topic Detection System Using Temporal BN-Grams & Weight Topic Length

Sistem berbasis web untuk mendeteksi lonjakan kata kunci (*n-grams*) dan menganalisis dinamika perubahan tren secara temporal terkait program **Makan Bergizi Gratis (MBG)** pada media sosial Twitter/X. Aplikasi ini dibangun menggunakan Python dan kerangka kerja **Streamlit**.

---

## 🚀 Fitur Utama

Sistem ini dirancang secara modular mengikuti alur pemrosesan *text mining* tingkat lanjut:

*   **Dataset Management:** Unggah dan kelola dataset Twitter beserta *Ground Truth* (Google Trends & Portal Berita Nasional) untuk tiap rentang waktu (`P1` - `P4`).
*   **Preliminary Analysis:** Validasi awal dataset dengan fitur *inline editing* untuk pelabelan relevansi, kategori media, dan akses URL *tweet* secara langsung.
*   **Sequential Preprocessing:** Pembersihan teks bertahap (*Case Folding, Noise Removal, Remove Non-Alpha, Stopword Removal*, dan *Stemming*).
*   **Temporal BN-Grams Execution:** Ekstraksi topik dalam 3 tahap: Pembobotan kata, Klasterisasi *n-grams* (1-Gram, 2-Gram, 3-Gram), dan *Topic Ranking* berdasarkan skor *DF-IDF Max*.
*   **Weight Topic Length Analysis:** Metode pembanding temporal untuk mendeteksi stabilitas tren antar *timeslot* dengan indikator status **TETAP** atau **BERUBAH**.
*   **Performance Evaluation:** Pengujian performa sistem menggunakan metrik *Topic Recall* (TR), *Keyword Precision* (KP), dan *Keyword Recall* (KR).

---

## 🛠️ Arsitektur Teknologi & Metode

Aplikasi ini mengintegrasikan beberapa pendekatan ilmiah:
1.  **Text Mining & NLP:** Menggunakan teknik pemrosesan bahasa alami untuk bahasa Indonesia.
2.  **Temporal BN-Grams:** Modifikasi pembobotan *DF-IDFt* (Document Frequency - Inverse Document Frequency Temporal) untuk menangkap sensitivitas tren berdasarkan urutan waktu.
3.  **Weight Topic Length ($s_t$):** Digunakan sebagai parameter pembanding matematis untuk melihat pergeseran atau kejenuhan suatu klaster informasi.

---

## 💻 Cara Instalasi & Menjalankan Aplikasi

Pastikan Anda sudah menginstal **Python (version >= 3.8)** di perangkat Anda.

### 1. Kloning Repositori
git clone [https://github.com/USERNAME_LU/Trending-Topic-Using-BN-Grams.git](https://github.com/USERNAME_LU/Trending-Topic-Using-BN-Grams.git)
cd Trending-Topic-Using-BN-Grams


## 2. install depedensi
pip install -r requirements.txt

## 3. Menjalankan Aplikasi
streamlit run app.py

##  Struktur Folder
├── app.py                  # File utama aplikasi Streamlit
├── components/             # Komponen visual dan fungsi pembantu UI
├── data/                   # Folder penyimpanan dataset mentah & ground truth
├── src/                    # Source code core algoritma
│   ├── preprocessing.py    # Pipeline pembersihan teks
│   ├── bngrams.py          # Logika ekstraksi & pembobotan DF-IDFt
│   └── weight_length.py    # Perhitungan komparasi temporal (st)
├── requirements.txt        # Daftar dependency library Python
└── README.md               # Dokumentasi proyek


