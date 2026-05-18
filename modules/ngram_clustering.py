def cluster_ngrams(dfidf, tweets_df, top_k=30, min_df=3, alpha=0.7):
    from collections import defaultdict
    import pandas as pd

    clusters = defaultdict(list)

    # ===== 1. FILTER TRENDING CANDIDATE =====
    dfidf = dfidf[dfidf["df"] >= min_df]

    for slot in dfidf["time_slot"].unique():
        slot_df = dfidf[dfidf["time_slot"] == slot] \
                    .sort_values("df_idft", ascending=False) \
                    .head(top_k)

        for _, row in slot_df.iterrows():
            key = " ".join(row["ngram"].split()[:2])  # inti topik (lebih stabil)
            clusters[(key, slot)].append(row)

    if not clusters:
        return []

    # ===== 2. TOTAL TWEET PER SLOT =====
    tweets_df["slot"] = pd.to_datetime(tweets_df["created_at"]).dt.to_period("H").astype(str)
    total_tweets_per_slot = tweets_df.groupby("slot").size().to_dict()

    # ===== 3. Lmax =====
    L_max = max(len(v) for v in clusters.values())

    results = []
    for i, ((core, slot), items) in enumerate(clusters.items()):
        Lt = len(items)

        # estimasi tweet count berbasis df tertinggi (lebih realistis dari sum)
        Nt = max(x["df"] for x in items)
        Ns = total_tweets_per_slot.get(str(slot), 1)

        st = alpha * (Lt / L_max) + (1 - alpha) * (Nt / Ns)

        results.append({
            "time_slot": slot,
            "cluster_id": i,
            "topic_core": core,
            "ngrams": ", ".join(sorted({x["ngram"] for x in items}, key=len, reverse=True)),
            "max_dfidf": max(x["df_idft"] for x in items),
            "topic_score_st": round(st, 4),
            "Lt": Lt,
            "Nt": Nt,
            "Ns": Ns
        })

    return sorted(results, key=lambda x: x["topic_score_st"], reverse=True)
