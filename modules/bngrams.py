import pandas as pd
import numpy as np
from collections import defaultdict,Counter  
import re

def calculate_distance_matrix(ngrams_list, tweets_series):
    n = len(ngrams_list)
    dist_matrix = np.zeros((n, n))
    tweets = [str(t).lower() for t in tweets_series]
    
    # Pre-calculate kemunculan tiap n-gram di tiap tweet
    occurence_map = {}
    for gram in ngrams_list:
        occurence_map[gram] = np.array([1 if gram.lower() in t else 0 for t in tweets], dtype=int)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                dist_matrix[i, j] = 0
                continue
                
            occ1 = occurence_map[ngrams_list[i]]
            occ2 = occurence_map[ngrams_list[j]]
            
            # A: Jumlah tweet mengandung g1 DAN g2
            A = np.sum(occ1 & occ2)
            # B: Jumlah tweet mengandung g1
            B = np.sum(occ1)
            # C: Jumlah tweet mengandung g2
            C = np.sum(occ2)
            
            # Rumus (2.15): d = 1 - (A / min(B, C))
            min_bc = min(B, C)
            
            if min_bc > 0:
                dist = 1 - (A / min_bc)
            else:
                dist = 1.0
                
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix

def calculate_cluster_score(cluster_members, df_clean):
    """
    Implementasi Persamaan (4) dan (5) dari Jurnal Indra (2019)
    """
    delta = 0.5
    # Hitung total kemunculan semua term unik (n)
    all_text = " ".join(df_clean['clean_text'].astype(str))
    all_words = all_text.split()
    total_N_u = len(all_words) # Total occurrences
    n_types = len(set(all_words)) # Number of term types
    
    cluster_total_score = 0
    
    # Ambil tweet yang mengandung anggota cluster
    # (Docs_c dalam jurnal: dokumen yang mengandung setidaknya satu keyword cluster)
    relevant_docs = df_clean[df_clean['clean_text'].str.contains('|'.join(cluster_members))]
    num_docs_c = len(relevant_docs)

    for _, doc in relevant_docs.iterrows():
        doc_text = str(doc['clean_text']).lower()
        doc_words = doc_text.split()
        
        sum_exp_p = 0
        for word in cluster_members:
            # Persamaan (5): p(w|corpus)
            N_w = all_words.count(word.lower()) # Total appearance in corpus
            p_w = (N_w + delta) / (total_N_u + (delta * n_types))
            
            # w_ij: jumlah term j muncul di dokumen i
            w_ij = doc_words.count(word.lower())
            
            if w_ij > 0:
                sum_exp_p += np.exp(-p_w) * w_ij

        cluster_total_score += sum_exp_p
        
    # Persamaan (4): Score C = |Docs_c| * Sigma Sigma ...
    return num_docs_c * cluster_total_score


def compute_temporal_dfidf(
    df,
    text_col="clean_text",
    date_col="created_at",
    ngram_range=(1, 3), 
    freq="H",
    use_ner=False,
    min_df=3,
    top_k_per_slot=5,
    save_path=None,
    ground_truth_path=None
):
    # 1. Persiapan Data & Time Slots (Tetap sama)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, text_col])
    df["time_slot"] = df[date_col].dt.to_period(freq).astype(str)

    slots = sorted(df["time_slot"].unique())
    total_timeslots = len(slots)
    slot_term_freq = defaultdict(lambda: defaultdict(int))

    # 2. Ekstraksi N-Gram
    from modules.ngrams_utils import extract_ngrams
    for slot in slots:
        slot_df = df[df["time_slot"] == slot]
        for text in slot_df[text_col]:
            grams = extract_ngrams(str(text), ngram_range)
            for g in set(grams):
                slot_term_freq[slot][g] += 1

    # ======================================================
    # 3. NER GAZETTEER (Rujukan: Yusuf Syaifudin - indonesia-ner)
    # ======================================================
    # Daftar entitas ini dikategorikan agar kamu bisa melaporkannya di Bab 4
    org_list = {'bpkp', 'kppn', 'kpk', 'kemenkeu', 'dpr', 'bpk', 'pemerintah', 'polri', 'tni', 'kejaksaan'}
    per_list = {'jokowi', 'prabowo', 'gibran', 'anies', 'ganjar', 'mahfud', 'luhut', 'menteri', 'presiden'}
    loc_list = {'jakarta', 'indonesia', 'nusantara', 'jawa', 'sumatera', 'kalimantan', 'sulawesi', 'papua'}

    # Gabungkan semua untuk regex search
    all_entities = org_list | per_list | loc_list
    # Menggunakan escape untuk menangani karakter khusus jika ada
    entity_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, all_entities)) + r')\b', re.IGNORECASE)

    records = []
    for idx, slot in enumerate(slots):
        prev_slots = slots[:idx]

        for term, df_i in slot_term_freq[slot].items():
            if df_i < min_df:
                continue

            sum_df_prev = sum(slot_term_freq[s].get(term, 0) for s in prev_slots)

            # --- FORMULA BNgram ---
            pembagi_internal = (sum_df_prev / total_timeslots) + 1
            penyebut = np.log10(pembagi_internal) + 1
            df_idf_t_score = (df_i + 1) / penyebut

            # ======================================================
            # 4. NER WEIGHTING LOGIC
            # ======================================================
            weight = 1.0
            is_entity = False
            entity_type = "NON-ENT"

            if use_ner:
                term_lower = term.lower()
                # Cek apakah n-gram mengandung entitas rujukan
                match = entity_pattern.search(term_lower)
                if match:
                    weight = 1.5  # Sesuai Jurnal BNgram
                    is_entity = True
                    # Identifikasi tipe untuk kolom tambahan
                    matched_word = match.group(0).lower()
                    if matched_word in org_list: entity_type = "ORGANIZATION"
                    elif matched_word in per_list: entity_type = "PERSON"
                    elif matched_word in loc_list: entity_type = "LOCATION"

            final_score = df_idf_t_score * weight

            records.append({
                "ngram": term,
                "time_slot": slot,
                "df": df_i,
                "sum_df_prev": sum_df_prev,
                "trend_score": round(final_score, 6),
                "boost": weight,
                "is_entity": is_entity,
                "entity_type": entity_type
            })

    # 5. Sorting & Ranking (Tetap sama)
    df_result = pd.DataFrame(records)
    if df_result.empty: return df_result

    df_result = df_result.sort_values(["time_slot", "trend_score"], ascending=[True, False])
    df_top = df_result.groupby("time_slot").head(top_k_per_slot).reset_index(drop=True)

    if save_path:
        df_top.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df_top