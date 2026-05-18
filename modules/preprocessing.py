import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords as nltk_stopwords

# =========================
# INIT TOOLS
# =========================
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

stop_factory = StopWordRemoverFactory()
stopwords_sastrawi = set(stop_factory.get_stop_words())
stopwords_nltk = set(nltk_stopwords.words("indonesian"))

extra_stopwords = {
    "utk","yg","yang","untuk","pada","dari","dengan","agar","karena",
    "hingga","sehingga","supaya","sampai","sebelum","sesudah",
    "atau","tapi","namun","kalau","jadi","saja","lagi","aja","pun","nya"
}

STOPWORDS = stopwords_sastrawi | stopwords_nltk | extra_stopwords

stemmer = StemmerFactory().create_stemmer()

# =========================
# STEP FUNCTIONS
# =========================
def case_folding(text):
    return text.lower()

def remove_noise(text):
    text = re.sub(r"http\S+|www\S+|\S+\.co/\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"rt[\s]+", "", text)
    return text

def remove_non_alpha(text):
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stopword_removal(text):
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

def stemming(text):
    return stemmer.stem(text)
