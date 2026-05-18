import streamlit as st
import pandas as pd
from modules.bngrams import compute_temporal_dfidf, calculate_distance_matrix
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(layout="wide")
st.title("📊 BN-Grams - Multi Dataset Analysis")

# ===============================
# CONFIG & INDEX
# ===============================
INDEX_LIST = [1, 2, 3, 4]

# ===============================
# SIDEBAR KONTROL
# ===============================
st.sidebar.header("⚙️ Parameter Sistem")
use_ner = st.sidebar.checkbox("✅ Gunakan NER Weighting", value=True)
freq = st.sidebar.selectbox("Time Slot", {"Per Jam": "H", "Harian": "D"}.items(), format_func=lambda x: x[0])[1]
min_df = st.sidebar.slider("Minimal DF per Slot", 1, 10, 3)
top_k_slot = st.sidebar.slider("Top-K Trending per Slot", 3, 20, 5)
cluster_method = st.sidebar.selectbox("Metode Linkage", ["average", "complete", "single"], index=0)

# ===============================
# INIT SESSION STATE
# ===============================
if "trend_results" not in st.session_state:
    st.session_state.trend_results = {i: None for i in INDEX_LIST}

# ===============================
# UI TABS
# ===============================
tabs = st.tabs([f"Analisis Dataset {i}" for i in INDEX_LIST])

for i, tab in enumerate(tabs, 1):
    with tab:
        CLEAN_DATA_PATH = f"data/cleaned_dataset_{i}.csv"
        GT_PATH = f"data/ground_truth_{i}.csv"
        SAVE_TREND_PATH = f"data/top_trending_topics_{i}.csv"

        if not os.path.exists(CLEAN_DATA_PATH):
            st.warning(f"⚠️ Dataset {i} tidak ditemukan.")
            continue

        df_clean = pd.read_csv(CLEAN_DATA_PATH)

        if st.button(f"🚀 Jalankan Analisis BN-Grams Dataset {i}", key=f"btn_trend_{i}"):
            with st.spinner(f"Memproses Dataset {i}..."):
                current_gt = GT_PATH if os.path.exists(GT_PATH) else None
                # Eksekusi Tahap 1: DF-IDF Temporal
                df_trend = compute_temporal_dfidf(
                    df_clean, "clean_text", "created_at", (1,3), 
                    freq, use_ner, min_df, top_k_slot, SAVE_TREND_PATH, current_gt
                )
                st.session_state.trend_results[i] = df_trend
                st.success(f"✅ Analisis Dataset {i} Selesai!")

        # --- TAMPILKAN HASIL ---
        if st.session_state.trend_results[i] is not None:
            res = st.session_state.trend_results[i]

            # 1. TAHAP PEMBOBOTAN BN-GRAMS
            st.subheader(f"1️⃣ Hasil Pembobotan N-Gram Trending (Temporal DF-IDF)")
            st.dataframe(res.sort_values("trend_score", ascending=False), use_container_width=True)

            # 2. TAHAP KLASTERISASI HIERARKI
            st.divider()
            st.subheader(f"2️⃣ Hasil Klasterisasi Topik (Hierarchical Clustering)")
            
            # Hitung Matriks Jarak
            list_ngrams_unique = res["ngram"].unique().tolist()
            dist_matrix = calculate_distance_matrix(list_ngrams_unique, df_clean["clean_text"])

            # Eksekusi Clustering sesuai Jurnal (Distance Threshold 0.5)
            # cluster_k (n_clusters) di-set None agar jarak yang menentukan jumlah cluster
            model = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=0.5, 
                linkage=cluster_method, 
                metric='precomputed'
            )
            cluster_labels = model.fit_predict(dist_matrix)
            
            cluster_mapping = dict(zip(list_ngrams_unique, cluster_labels))
            res["cluster_id"] = res["ngram"].map(cluster_mapping)

            # --- TABEL RINGKASAN TOPIK ---
            st.write("📋 **Daftar Topik Terdeteksi (Grup N-Gram)**")
            
            # Grouping anggota klaster
            df_summary = res.groupby("cluster_id")["ngram"].unique().apply(lambda x: ", ".join(x)).reset_index()
            df_summary.columns = ["ID Topik", "N-Gram Terkait"]
            
            # Hitung jumlah anggota
            df_summary["Jumlah Kata"] = res.groupby("cluster_id")["ngram"].nunique().values
            
            # Tampilkan Tabel Final yang Bersih
            st.table(df_summary.sort_values("Jumlah Kata", ascending=False))

            # 3. VISUALISASI MATRIKS (Opsional)
            with st.expander("📊 Lihat Detail Matriks Jarak antar N-Gram"):
                df_dist_matrix = pd.DataFrame(dist_matrix, index=list_ngrams_unique, columns=list_ngrams_unique)
                st.dataframe(df_dist_matrix, use_container_width=True)
                st.caption("Nilai 0.0 berarti kata selalu muncul bersama, 1.0 berarti tidak pernah muncul bersama.")

            # Tombol Download Hasil
            csv_data = res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Hasil Analisis Dataset {i}",
                data=csv_data,
                file_name=f"hasil_bngrams_dataset_{i}.csv",
                mime='text/csv',
            )
            
            # 3️⃣ TAHAP RANKING TOPIK
            st.divider()
            st.subheader("3️⃣ Ranking Trending Topics")
            st.warning("⚠️ Syarat: Minimal 2 n-gram per klaster untuk hasil yang lebih representatif.")

            # Grouping dan Filtering: Minimal 2 kata per klaster
            ranked_topics = []
            cluster_counts = res.groupby("cluster_id")["ngram"].nunique()
            valid_cluster_ids = cluster_counts[cluster_counts >= 2].index.tolist()

            for c_id in valid_cluster_ids:
                cluster_data = res[res["cluster_id"] == c_id]
                
                # Sesuai Rujukan: Ambil nilai bobot tertinggi (highest score)
                highest_score = cluster_data["trend_score"].max()
                members = cluster_data["ngram"].unique().tolist()
                
                ranked_topics.append({
                    "Cluster ID": c_id,
                    "Topic Score (Highest Bobot)": round(highest_score, 4),
                    "Total N-Grams": len(members),
                    "Top Keywords": ", ".join(members)
                })

            if ranked_topics:
                # Sorting berdasarkan bobot tertinggi anggota
                df_final = pd.DataFrame(ranked_topics).sort_values("Topic Score (Highest Bobot)", ascending=False).reset_index(drop=True)
                df_final.index += 1
                
                st.write("### 📋 Tabel Ranking Topik")
                st.table(df_final)

                # Visualisasi Ranking
                st.bar_chart(df_final, x="Top Keywords", y="Topic Score (Highest Bobot)")

                # Detail per Topik dalam Expander
                with st.expander("🔍 Lihat Detail Anggota tiap Topik"):
                    for row in df_final.itertuples():
                        st.write(f"**Topik {row.Index} (Highest Score: {row._2})**")
                        detail = res[res["cluster_id"] == row._1].sort_values("trend_score", ascending=False)
                        st.dataframe(detail[["ngram", "time_slot", "trend_score"]], use_container_width=True)
            else:
                st.info("Tidak ada topik yang memenuhi syarat minimal 2 n-gram.")