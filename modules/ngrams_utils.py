from collections import Counter

def generate_ngrams(tokens, n):
    """Membuat n-gram dan menggabungkannya menjadi string."""
    if n == 1:
        return tokens
    # Zip digunakan untuk menggeser token sebanyak n kali untuk membuat kombinasi
    return [" ".join(gram) for gram in zip(*[tokens[i:] for i in range(n)])]

def extract_ngrams(text, ngram_range=(1, 3)):
    """
    Mengubah teks menjadi list n-gram gabungan.
    Contoh: ngram_range=(1, 2) akan menghasilkan list berisi unigram DAN bigram.
    """
    if not isinstance(text, str):
        return []
    
    tokens = text.split()
    all_grams = []
    
    # Memastikan ngram_range dalam bentuk tuple/list (start, end)
    if isinstance(ngram_range, int):
        # Jika user hanya input satu angka, anggap sebagai n-gram spesifik
        all_grams.extend(generate_ngrams(tokens, ngram_range))
    else:
        # Loop dari n terkecil ke n terbesar (misal 1 sampai 3)
        start, end = ngram_range
        for n in range(start, end + 1):
            all_grams.extend(generate_ngrams(tokens, n))
            
    return all_grams

def count_ngrams(texts, ngram_range=(1, 3)):
    """Menghitung frekuensi n-gram dari sekumpulan teks."""
    counter = Counter()
    for text in texts:
        counter.update(extract_ngrams(text, ngram_range))
    return counter