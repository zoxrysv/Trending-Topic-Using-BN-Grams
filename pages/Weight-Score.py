import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from modules.weight_utils import calculate_comparison_metrics

# --- 1. SETUP HALAMAN ---
st.set_page_config(page_title="Weight Score Analysis", layout="wide")

st.title("⚖️ Weighting Result Analysis")
st.caption("Perbandingan Hasil Ranking: Baseline (DF-IDF) vs Proposed Method (Weighted Topic-Length)")

# --- 2. CONFIG & STATE ---
INDEX_LIST = [1, 2, 3, 4]

if "weight_results" not in st.session_state:
    st.session_state.weight_results = {i: None for i in INDEX_LIST}

# --- 3. SIDEBAR PARAMETER ---
with st.sidebar:
    st.header("⚙️ Parameter Jurnal")
    alpha = st.slider("Alpha (α)", 0.0, 1.0, 0.7, 0.05)
    threshold = st.slider("Threshold Clustering", 0.1, 1.5, 0.5, 0.1) 
    linkage_method = st.radio("Linkage Method", ["average", "single"]) 
    
    st.divider()
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.weight_results = {i: None for i in INDEX_LIST}
        st.rerun()

# --- 4. TABS PER DATASET ---
tabs = st.tabs([f"Dataset {i}" for i in INDEX_LIST])

for i, tab in enumerate(tabs, 1):
    with tab:
        PATH_CLEANED = f"data/cleaned_dataset_{i}.csv"
        PATH_TRENDING = f"data/top_trending_topics_{i}.csv"
        PATH_SAVE = f"data/weight_score_{i}.csv"

        if not os.path.exists(PATH_CLEANED) or not os.path.exists(PATH_TRENDING):
            st.warning(f"⚠️ Data untuk Dataset {i} belum lengkap.")
            continue

        # --- LOAD DATA ---
        df_raw = pd.read_csv(PATH_CLEANED)
        df_ngrams = pd.read_csv(PATH_TRENDING)

        # Standarisasi kolom waktu
        if 'time_slot' not in df_raw.columns:
            for col in ['created_at', 'date', 'Timestamp']:
                if col in df_raw.columns:
                    df_raw = df_raw.rename(columns={col: 'time_slot'})
                    break

        # --- TOMBOL EKSEKUSI ---
        if st.button(f"🚀 Hitung Bobot (Dataset {i})", key=f"btn_calc_{i}"):
            with st.spinner("Sedang menghitung s_t score..."):
                result = calculate_comparison_metrics(
                    df_ngrams, df_raw, 
                    alpha=alpha, cluster_method=linkage_method, t_threshold=threshold
                )
                st.session_state.weight_results[i] = result
                result.to_csv(PATH_SAVE, index=False)
                st.rerun()

        # --- TAMPILKAN HASIL ---
        weight_df = st.session_state.weight_results[i]
        if weight_df is None and os.path.exists(PATH_SAVE):
            weight_df = pd.read_csv(PATH_SAVE)
            st.session_state.weight_results[i] = weight_df

        if weight_df is not None:
            method_name = "Average Linkage" if linkage_method == "average" else "Single Linkage"
            st.success(f"Analisis Menggunakan: **{method_name}**")

            # --- LOGIKA PERBANDINGAN ---
            dfidf_col = 'max_dfidf' if 'max_dfidf' in weight_df.columns else 'trend_score'
            
            baseline = weight_df.sort_values(['time_slot', dfidf_col], ascending=[True, False]).groupby('time_slot').head(1)
            proposed = weight_df.sort_values(['time_slot', 'topic_score_st'], ascending=[True, False]).groupby('time_slot').head(1)
            
            comparison = pd.merge(
                baseline[['time_slot', 'topic_core', dfidf_col]].rename(columns={'topic_core': 'Topik Baseline (DF-IDF)', dfidf_col: 'Skor DF-IDF'}),
                proposed[['time_slot', 'topic_core', 'topic_score_st', 'Lt', 'Nt']].rename(columns={'topic_core': f'Topik Proposed ({linkage_method})', 'topic_score_st': 'Skor St', 'Lt': 'Panjang (Lt)', 'Nt': 'Volume (Nt)'}),
                on='time_slot'
            )
            comparison['Status Perubahan'] = np.where(comparison['Topik Baseline (DF-IDF)'] != comparison[f'Topik Proposed ({linkage_method})'], '🔥 BERUBAH', '✅ TETAP')

            # --- METRIK GLOBAL ---
            col1, col2, col3 = st.columns(3)
            ns_real = len(df_raw) 
            col1.metric("Total Sampel (Ns)", int(ns_real))
            col2.metric("Avg Lt (Panjang Klaster)", f"{weight_df['Lt'].mean():.2f}")
            col3.metric("Avg Nt (Volume Klaster)", f"{weight_df['Nt'].mean():.1f}")

            st.divider()

            # --- VISUALISASI ---
            t1, t2 = st.tabs(["📋 Tabel Perbandingan", "📈 Analisis Korelasi"])

            with t1:
                st.subheader(f"🏆 Side-by-Side Comparison ({method_name})")
                
                # Membuat dua kolom menyamping
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### 1️⃣ Baseline: Max DF-IDF")
                    st.caption("Ranking berdasarkan skor keterpopuleran")
                    baseline_display = comparison[['time_slot', 'Topik Baseline (DF-IDF)', 'Skor DF-IDF']].copy()
                    st.dataframe(baseline_display, use_container_width=True, hide_index=True)

                with col_right:
                    st.markdown(f"#### 2️⃣ Proposed: Weighted ($S_t$)")
                    st.caption("Ranking berdasarkan struktur klaster (Lt & Nt)")
                    proposed_display = comparison[['time_slot', f'Topik Proposed ({linkage_method})', 'Skor St', 'Status Perubahan']].copy()
                    
                    # Style khusus untuk kolom kanan saja agar perubahan terlihat kontras
                    def style_proposed(row):
                        color = '#422121' if row['Status Perubahan'] == '🔥 BERUBAH' else '' 
                        return [f'background-color: {color}' for _ in row]

                    st.dataframe(
                        proposed_display.style.apply(style_proposed, axis=1), 
                        use_container_width=True,
                        hide_index=True
                    )
                
                st.divider()
                # Info tambahan di bawah tabel menyamping
                total_slots = len(comparison)
                num_changes = len(comparison[comparison['Status Perubahan'] == '🔥 BERUBAH'])
                percent_changes = (num_changes / total_slots) * 100 if total_slots > 0 else 0

                st.info(
                    f"""
                    **Total Topic Change:**  
                    Dari **{total_slots} time slot**, terdapat **{num_changes} slot** yang mengalami perubahan topik utama  
                    (**{percent_changes:.2f}%** dari seluruh slot).
                    """
                )

            with t2:
                fig = px.scatter(
                    weight_df, x="Lt", y="Nt", size="topic_score_st", color="topic_score_st",
                    hover_name="topic_core", color_continuous_scale="Viridis",
                    title="Distribusi Bobot Topik (Lt vs Nt)"
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Lihat Data Mentah"):
                st.dataframe(weight_df)