import streamlit as st
import pandas as pd
import os
from modules.preprocessing import (
    case_folding,
    remove_noise,
    remove_non_alpha,
    stopword_removal,
    stemming
)

st.title("🧹 Multi-Preprocessing Teks")

# Konfigurasi
INDEX_LIST = [1, 2, 3, 4]
TEXT_COL = "full_text"

# ===============================
# INIT SESSION STATE
# ===============================
if "df_cleans" not in st.session_state:
    st.session_state.df_cleans = {i: None for i in INDEX_LIST}

# ===============================
# FUNGSI PROSES
# ===============================
def run_preprocessing(df):
    """Fungsi pembantu untuk menjalankan semua tahapan preprocessing"""
    df = df.copy()
    # Pastikan kolom teks ada
    if TEXT_COL not in df.columns:
        st.error(f"Kolom '{TEXT_COL}' tidak ditemukan!")
        return None
    
    with st.spinner("Sedang memproses..."):
        df["case_folding"] = df[TEXT_COL].astype(str).apply(case_folding)
        df["noise_removed"] = df["case_folding"].apply(remove_noise)
        df["non_alpha_removed"] = df["noise_removed"].apply(remove_non_alpha)
        df["stopword_removed"] = df["non_alpha_removed"].apply(stopword_removal)
        df["clean_text"] = df["stopword_removed"].apply(stemming)
    return df

# ===============================
# UI TABS UNTUK 4 DATASET
# ===============================
tabs = st.tabs([f"Dataset {i}" for i in INDEX_LIST])

for i, tab in enumerate(tabs, 1):
    with tab:
        RAW_PATH = f"data/dataset_{i}.csv"
        CLEAN_PATH = f"data/cleaned_dataset_{i}.csv"

        # 1. Cek keberadaan file Raw
        if not os.path.exists(RAW_PATH):
            st.warning(f"File {RAW_PATH} belum di-upload di menu Management.")
            continue

        df_raw = pd.read_csv(RAW_PATH, sep=";", encoding="utf-8-sig") # Sesuaikan sep=";" jika perlu
        
        # 2. Auto-load Clean Data jika sudah ada di folder
        if os.path.exists(CLEAN_PATH) and st.session_state.df_cleans[i] is None:
            st.session_state.df_cleans[i] = pd.read_csv(CLEAN_PATH)

        # 3. Tombol Aksi
        col1, col2 = st.columns(2)
        
        # --- TOMBOL PROSES ---
        if st.session_state.df_cleans[i] is None:
            with col1:
                if st.button(f"🚀 Proses Dataset {i}", key=f"btn_run_{i}"):
                    processed_df = run_preprocessing(df_raw)
                    if processed_df is not None:
                        processed_df.to_csv(CLEAN_PATH, index=False)
                        st.session_state.df_cleans[i] = processed_df
                        st.success(f"Dataset {i} Berhasil diproses!")
                        st.rerun()
        
        # --- TOMBOL RESET ---
        else:
            with col2:
                if st.button(f"🔁 Reset Dataset {i}", key=f"btn_reset_{i}"):
                    if os.path.exists(CLEAN_PATH):
                        os.remove(CLEAN_PATH)
                    st.session_state.df_cleans[i] = None
                    st.warning(f"Data {i} direset.")
                    st.rerun()

        # 4. Tampilkan Hasil jika sudah ada
        if st.session_state.df_cleans[i] is not None:
            st.subheader(f"🔎 Preview Hasil {i}")
            st.dataframe(st.session_state.df_cleans[i][[
                TEXT_COL, "case_folding", "noise_removed", 
                "non_alpha_removed", "stopword_removed", "clean_text"
            ]].head(10))
            st.download_button(
                label=f"📥 Download Cleaned Dataset {i}",
                data=st.session_state.df_cleans[i].to_csv(index=False),
                file_name=f"cleaned_dataset_{i}.csv",
                mime="text/csv"
            )