import pandas as pd
import numpy as np

# =========================================================
# MAIN DISPATCHER
# =========================================================
def evaluate(method, ngram_n=None, top_k=20, trending_path="", gt_path=""):
    if method == "bngrams":
        return evaluate_bngrams(ngram_n, top_k, trending_path, gt_path)

    elif method == "wtl":
        return evaluate_wtl(top_k, trending_path, gt_path)

    else:
        raise ValueError("Metode evaluasi tidak dikenali. Gunakan 'bngrams' atau 'wtl'.")


# =========================================================
# BN-GRAMS (PUNYA LU, DIPERTAHANKAN)
# =========================================================
def evaluate_bngrams(ngram_n, top_k, trending_path, gt_path):
    try:
        tweets = pd.read_csv(trending_path, sep=",", encoding="utf-8-sig")
        gt = pd.read_csv(gt_path, sep=",", encoding="utf-8-sig")
    except Exception as e:
        print(f"Error: {e}")
        return {"Topic Recall": 0, "Keyword Precision": 0, "Keyword Recall": 0}

    # 1. Preprocessing Output Sistem (KBT)
    tweets["ngram"] = tweets["ngram"].astype(str).str.lower().str.strip()
    tweets['n_temp'] = tweets['ngram'].apply(lambda x: len(x.split()))
    df_n = tweets[tweets['n_temp'] == ngram_n].head(top_k)

    if df_n.empty:
        return {"Topic Recall": 0, "Keyword Precision": 0, "Keyword Recall": 0}

    kbt_list = df_n['ngram'].tolist()
    kbt_set = set(kbt_list)

    # 2. Preprocessing Ground Truth (KGT)
    kgt_all_phrases = []
    for row in gt["keywords"].dropna().astype(str):
        parts = [p.strip().lower() for p in row.split(',') if p.strip()]
        kgt_all_phrases.extend(parts)

    kgt_set = set(kgt_all_phrases)

    # 3. Hitung Hits
    hits = 0
    for s_gram in kbt_set:
        sys_words = set(s_gram.split())
        is_match = False

        for gt_phrase in kgt_set:
            gt_words = set(gt_phrase.split())
            overlap = sys_words.intersection(gt_words)

            threshold = 2 if ngram_n == 3 else 1

            if len(overlap) >= threshold:
                is_match = True
                break

        if is_match:
            hits += 1

    # 4. Hitung metrik
    keyword_precision = hits / len(kbt_set) if len(kbt_set) > 0 else 0
    keyword_recall = hits / len(kgt_set) if len(kgt_set) > 0 else 0

    topics_found = 0
    for gt_topic in kgt_set:
        if any(gt_topic in s_gram or s_gram in gt_topic for s_gram in kbt_list):
            topics_found += 1

    topic_recall = topics_found / len(kgt_set) if len(kgt_set) > 0 else 0

    return {
        "Topic Recall": topic_recall,
        "Keyword Precision": keyword_precision,
        "Keyword Recall": keyword_recall
    }


# =========================================================
# WEIGHT TOPIC LENGTH (BARU)
# CSV: time_slot, topic_core, ngrams, topic_score_st, Lt, Nt, Ns, max_dfidf
# =========================================================
def evaluate_wtl(top_k, trending_path, gt_path):
    try:
        df = pd.read_csv(trending_path, sep=",", encoding="utf-8-sig")
        gt = pd.read_csv(gt_path, sep=",", encoding="utf-8-sig")
    except Exception as e:
        print(f"Error: {e}")
        return {"Topic Recall": 0, "Keyword Precision": 0, "Keyword Recall": 0}

    # Ambil Top-K berdasarkan ranking file (anggap sudah terurut)
    df = df.head(top_k)

    # ===== PREPROCESS SYSTEM OUTPUT =====
    system_topics = df["topic_core"].dropna().astype(str).str.lower().str.strip().tolist()

    system_keywords = []
    for row in df["ngrams"].dropna().astype(str):
        parts = [p.strip().lower() for p in row.split(',') if p.strip()]
        system_keywords.extend(parts)

    kbt_set = set(system_keywords)

    # ===== PREPROCESS GROUND TRUTH =====
    kgt_all_phrases = []
    for row in gt["keywords"].dropna().astype(str):
        parts = [p.strip().lower() for p in row.split(',') if p.strip()]
        kgt_all_phrases.extend(parts)

    kgt_set = set(kgt_all_phrases)

    if len(kbt_set) == 0 or len(kgt_set) == 0:
        return {"Topic Recall": 0, "Keyword Precision": 0, "Keyword Recall": 0}

    # ===== HIT KEYWORD =====
    hits = 0
    for s_gram in kbt_set:
        sys_words = set(s_gram.split())
        is_match = False

        for gt_phrase in kgt_set:
            gt_words = set(gt_phrase.split())
            overlap = sys_words.intersection(gt_words)

            if len(overlap) >= 1:   # WTL pakai threshold ringan
                is_match = True
                break

        if is_match:
            hits += 1

    keyword_precision = hits / len(kbt_set)
    keyword_recall = hits / len(kgt_set)

    # ===== TOPIC RECALL (pakai topic_core) =====
    topics_found = 0
    for gt_topic in kgt_set:
        if any(gt_topic in sys_topic or sys_topic in gt_topic for sys_topic in system_topics):
            topics_found += 1

    topic_recall = topics_found / len(kgt_set)

    return {
        "Topic Recall": topic_recall,
        "Keyword Precision": keyword_precision,
        "Keyword Recall": keyword_recall
    }
