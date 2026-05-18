import numpy as np
from collections import Counter

# ---------- 1. PERSENTASE RELEVAN ----------
def relevant_percentage_from_dataset(df):
    # pastikan kolom standar
    df.columns = df.columns.str.strip().str.lower()
    df['relevan'] = df['relevan'].astype(str).str.strip().str.lower()

    total = len(df)
    relevant = len(df[df['relevan'] == "relevan"])
    
    return round((relevant / total) * 100, 2)


# ---------- 2. MEDIA VS INDIVIDUAL ----------
def media_percentage_from_dataset(df):
    df.columns = df.columns.str.strip().str.lower()
    df['media'] = df['media'].astype(str).str.strip().str.lower()

    total = len(df)
    media = len(df[df['media'] == "media"])
    individual = len(df[df['media'] == "individual"])

    return {
        "media": round((media / total) * 100, 2),
        "individual": round((individual / total) * 100, 2)
    }


# ---------- 3. ENTROPY ----------
def calculate_entropy(df, text_col="full_text"):
    df.columns = df.columns.str.strip().str.lower()
    all_tokens = []

    for text in df[text_col].dropna():
        # langsung split spasi, tanpa hapus simbol / hashtag / URL
        all_tokens.extend(str(text).split())

    counter = Counter(all_tokens)
    N = sum(counter.values())

    # jika dataset kosong, hindari ZeroDivisionError
    if N == 0:
        return 0

    entropy = -sum((count / N) * np.log10(count / N) for count in counter.values())
    return round(entropy, 4)
def calculate_entropy_with_max(df, text_col="full_text"):
    all_tokens = []
    for text in df[text_col].dropna():
        all_tokens.extend(str(text).split())

    counter = Counter(all_tokens)
    N = sum(counter.values()) # Total Term
    V = len(counter)         # Kata Unik (Vocabulary)

    if N == 0: return 0, 0

    # Rumus 2.13 (Shannon dengan log10)
    entropy = -sum((count / N) * np.log10(count / N) for count in counter.values())
    
    # Nilai Maksimum yang mungkin (Jika semua kata muncul rata)
    h_max = np.log10(V) if V > 0 else 0
    
    return round(entropy, 4), round(h_max, 4)