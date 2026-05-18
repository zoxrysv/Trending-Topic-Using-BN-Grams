import streamlit as st
import pandas as pd
from modules.preliminary import (
    relevant_percentage_from_dataset,
    media_percentage_from_dataset,
    calculate_entropy
)

st.set_page_config(layout="wide")
st.title("📊 Preliminary Dataset Analysis")
st.write("Analisis awal dataset untuk mengevaluasi kelayakan pendeteksian trending topic.")

if "datasets" not in st.session_state:
    st.warning("⚠️ Upload dataset dulu di halaman Dataset.")
    st.stop()

# Looping untuk 4 dataset
for i in [1, 2, 3, 4]:
    df = st.session_state["datasets"].get(i)

    if df is not None:
        with st.container():
            st.subheader(f"📂 Dataset {i}")
            
            # --- 1. FITUR EDIT DATASET ---
            with st.expander(f"📝 Lihat & Edit Dataset {i}", expanded=False):
                st.info("Tips: Kamu bisa mengubah nilai pada kolom 'Relevan' atau 'Media' langsung di tabel bawah ini.")
                
                # Menggunakan data_editor agar bisa diedit user
                edited_df = st.data_editor(
                    df, 
                    use_container_width=True, 
                    key=f"editor_{i}",
                    num_rows="dynamic" # Bisa tambah/hapus baris jika perlu
                )
                
                col_save, _ = st.columns([1, 4])
                if col_save.button(f"💾 Simpan Perubahan Dataset {i}", key=f"save_{i}"):
                    st.session_state["datasets"][i] = edited_df
                    st.success(f"✅ Perubahan pada Dataset {i} berhasil disimpan ke memori!")
                    st.rerun()

            # --- 2. TOMBOL EKSEKUSI PRELIMINARY ---
            if st.button(f"🚀 Jalankan Analisis Preliminary Dataset {i}", key=f"exec_{i}"):
                st.divider()
                
                # Progress bar pemanis
                with st.spinner(f"Menghitung metrik untuk Dataset {i}..."):
                    # Menampilkan metrik dalam kolom
                    c1, c2, c3, c4 = st.columns(4)

                    # Perhitungan Relevansi
                    rel = relevant_percentage_from_dataset(df)
                    c1.metric("Relevant Tweets", f"{rel}%")

                    # Perhitungan Media vs Individual
                    media_stat = media_percentage_from_dataset(df)
                    c2.metric("Media Account", f"{media_stat['media']}%")
                    c3.metric("Individual Account", f"{media_stat['individual']}%")

                    # Perhitungan Entropy
                    entropy = calculate_entropy(df, text_col="full_text")
                    c4.metric("Entropy Value", entropy)

                    # Penjelasan Singkat untuk Sidang
                    
                   
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()