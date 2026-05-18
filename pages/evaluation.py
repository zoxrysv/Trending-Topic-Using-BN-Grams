import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.evaluation_utils import evaluate
import os

st.set_page_config(page_title="Evaluasi Multi-Dataset", layout="wide")

st.title("📊 Evaluasi Real Score (Multi-Dataset)")

# =========================
# SIDEBAR CONFIG
# =========================
INDEX_LIST = [1, 2, 3, 4]

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    top_k = st.slider("Top-K Keywords (Ranking)", 5, 100, 20)
    st.info("Evaluasi performa sistem terhadap Ground Truth pada Top-K data.")

# =========================
# METODE TABS
# =========================
method_tabs = st.tabs(["🔠 BN-Grams", "📏 Weight Topic Length"])

# ======================================================
# TAB 1 : BN-GRAMS
# ======================================================
with method_tabs[0]:
    st.subheader("🔠 Evaluasi Metode BN-Grams")

    tabs = st.tabs([f"Evaluasi Dataset {i}" for i in INDEX_LIST])

    for i, tab in enumerate(tabs, 1):
        with tab:
            TREND_PATH = f"data/top_trending_topics_{i}.csv"
            GT_PATH = f"data/ground_truth_{i}.csv"

            if not os.path.exists(TREND_PATH) or not os.path.exists(GT_PATH):
                st.warning(f"⚠️ Data untuk Dataset {i} belum lengkap.")
                continue

            if st.button(f"🚀 Jalankan Evaluasi BN-Grams Dataset {i}", key=f"bn_btn_{i}"):

                results = []
                with st.spinner(f"Menghitung metrik BN-Grams Dataset {i}..."):
                    for n in [1, 2, 3]:
                        score = evaluate(
                            method="bngrams",
                            ngram_n=n,
                            top_k=top_k,
                            trending_path=TREND_PATH,
                            gt_path=GT_PATH
                        )
                        score["Model"] = f"{n}-Gram"
                        results.append(score)

                df_eval = pd.DataFrame(results)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📋 Tabel Hasil")
                    st.dataframe(df_eval.style.format({
                        "Topic Recall": "{:.2f}",
                        "Keyword Precision": "{:.2f}",
                        "Keyword Recall": "{:.2f}"
                    }), use_container_width=True)

                with col2:
                    st.subheader("📈 Grafik Metrik")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = df_eval["Model"]
                    ax.plot(x, df_eval["Topic Recall"], marker='o', label="Topic Recall")
                    ax.plot(x, df_eval["Keyword Precision"], marker='s', label="Keyword Precision")
                    ax.plot(x, df_eval["Keyword Recall"], marker='^', label="Keyword Recall")
                    ax.set_ylim(0, 1.1)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                # ---------- ANALISIS TOP-K ----------
                st.divider()
                st.subheader(f"🔍 Analisis Variasi Top-K (BN-Grams Dataset {i})")

                all_ks = [5, 10, 20, 50]
                comparison_results = []

                for k_val in all_ks:
                    for n in [1, 2, 3]:
                        res = evaluate("bngrams", n, k_val, TREND_PATH, GT_PATH)
                        res["K"] = k_val
                        res["Model"] = f"{n}-Gram"
                        comparison_results.append(res)

                df_compare = pd.DataFrame(comparison_results)
                st.dataframe(df_compare[["K", "Model", "Keyword Precision", "Keyword Recall", "Topic Recall"]],
                             use_container_width=True)

# ======================================================
# TAB 2 : WEIGHT TOPIC LENGTH
# ======================================================
with method_tabs[1]:
    st.subheader("📏 Evaluasi Metode Weight Topic Length")

    tabs = st.tabs([f"Evaluasi Dataset {i}" for i in INDEX_LIST])

    for i, tab in enumerate(tabs, 1):
        with tab:
            WTL_PATH = f"data/weight_score_{i}.csv"
            GT_PATH = f"data/ground_truth_{i}.csv"

            if not os.path.exists(WTL_PATH) or not os.path.exists(GT_PATH):
                st.warning(f"⚠️ Data Dataset {i} belum lengkap (Weight Score / Ground Truth).")
                continue

            if st.button(f"🚀 Jalankan Evaluasi WTL Dataset {i}", key=f"wtl_btn_{i}"):

                with st.spinner(f"Menghitung metrik WTL Dataset {i}..."):
                    score = evaluate(
                        method="wtl",
                        ngram_n=None,
                        top_k=top_k,
                        trending_path=WTL_PATH,
                        gt_path=GT_PATH
                    )

                df_eval = pd.DataFrame([score])

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📋 Tabel Hasil")
                    st.dataframe(df_eval.style.format({
                        "Topic Recall": "{:.2f}",
                        "Keyword Precision": "{:.2f}",
                        "Keyword Recall": "{:.2f}"
                    }), use_container_width=True)

                with col2:
                    st.subheader("📈 Grafik Metrik")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(df_eval.columns, df_eval.iloc[0])
                    ax.set_ylim(0, 1.1)
                    ax.set_ylabel("Score")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

# ======================================================
# RINGKASAN AKHIR (BN-GRAMS BENCHMARK)
# ======================================================
st.divider()
if st.sidebar.button("📊 Ringkasan Semua Dataset (BN-Grams 2-Gram)"):
    st.subheader("🏁 Perbandingan Ringkas Semua Dataset (2-Gram)")

    summary_data = []
    for i in INDEX_LIST:
        T_PATH = f"data/top_trending_topics_{i}.csv"
        G_PATH = f"data/ground_truth_{i}.csv"

        if os.path.exists(T_PATH) and os.path.exists(G_PATH):
            res = evaluate("bngrams", 2, top_k, T_PATH, G_PATH)
            summary_data.append({
                "Dataset": f"Dataset {i}",
                "2-Gram Precision": res["Keyword Precision"]
            })

    if summary_data:
        st.bar_chart(pd.DataFrame(summary_data).set_index("Dataset"))
    else:
        st.error("Data tidak cukup untuk membuat ringkasan.")
