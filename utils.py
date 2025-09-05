# app/utils.py
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import nltk
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

PII_PATTERNS = [
    # emails
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '<EMAIL>'),
    # phone numbers (simple)
    (re.compile(r'\b(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b'), '<PHONE>'),
    # PAN-like (India generic alpha numeric pattern) - example
    (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'), '<PAN>'),
    # simple numeric ids
    (re.compile(r'\b\d{6,}\b'), '<NUMBER>')
]

def redact_pii(text: str) -> str:
    t = text
    for pat, repl in PII_PATTERNS:
        t = pat.sub(repl, t)
    return t

def normalize_text(text: str) -> str:
    # normalize whitespace, remove repeated punctuation, basic cleaning
    t = text.strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[\u200B-\u200D\uFEFF]', '', t)  # zero-width
    t = re.sub(r'([!?.,]){2,}', r'\1', t)
    return t

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return 'unknown'

def simple_tokenize(text: str):
    from nltk.tokenize import word_tokenize
    toks = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    toks = [t for t in toks if t not in STOP_WORDS]
    return toks
