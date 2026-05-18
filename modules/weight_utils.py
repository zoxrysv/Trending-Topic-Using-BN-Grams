import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from modules.bngrams import calculate_distance_matrix
def calculate_comparison_metrics(df_ngrams, df_raw, alpha=0.7, cluster_method='average', t_threshold=0.7):
    all_results = []
    
    # 1. Deteksi kolom waktu
    raw_time_col = next((col for col in ['time_slot', 'created_at', 'date'] if col in df_raw.columns), None)
    if not raw_time_col:
        raise KeyError("Kolom waktu tidak ditemukan di df_raw.")

    for slot, slot_data in df_ngrams.groupby('time_slot'):
        if len(slot_data) < 2:
            continue
            
        # Filter tweet pada slot waktu yang bersangkutan
        mask = df_raw[raw_time_col].astype(str).str.contains(str(slot), na=False)
        current_tweets = df_raw[mask]['clean_text']
        Ns = len(current_tweets)
        
        if Ns == 0: 
            Ns = int(slot_data['df'].sum()) 

        # --- Proses Clustering (Sesuai Jurnal BN-Grams) ---
        ngram_col = 'ngram' if 'ngram' in slot_data.columns else 'ngrams'
        list_ngrams = slot_data[ngram_col].tolist()
        
        # HITUNG JARAK BERDASARKAN KO-OKURENSI (Persamaan 3)
        dist_matrix = calculate_distance_matrix(list_ngrams, current_tweets)
        
        # Konversi matriks simetris ke bentuk condensed matrix untuk scipy linkage
        # Menggunakan rumus: dist_matrix[np.triu_indices(n, k=1)]
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(dist_matrix, checks=False)
        
        # Clustering dengan linkage sesuai pilihan sidebar (single/average)
        Z = linkage(condensed_dist, method=cluster_method)
        
        slot_data = slot_data.copy()
        slot_data['cluster_id'] = fcluster(Z, t=t_threshold, criterion='distance')
        
        # Agregasi Topik
        topic_groups = slot_data.groupby('cluster_id').agg({
            ngram_col: list,
            'df': 'sum', 
            'trend_score' if 'trend_score' in slot_data.columns else 'max_dfidf': 'max'
        })
        
        L_max = topic_groups[ngram_col].apply(len).max()

        for cid, row in topic_groups.iterrows():
            terms = row[ngram_col]
            Lt = len(terms)
            Nt = row['df']
            
            # Rumus St Score (Proposed Method)
            st_score = alpha * (Lt / L_max) + (1 - alpha) * (Nt / Ns)
            
            all_results.append({
                "time_slot": slot,
                "topic_core": terms[0], 
                "ngrams": ", ".join(terms[:10]),
                "topic_score_st": round(st_score, 4),
                "Lt": Lt,
                "Nt": Nt,
                "Ns": Ns,
                "max_dfidf": row['trend_score' if 'trend_score' in slot_data.columns else 'max_dfidf']
            })

    return pd.DataFrame(all_results)