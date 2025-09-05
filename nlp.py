# app/nlp.py
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from wordcloud import WordCloud
import io
from PIL import Image
import base64

# Lazy-load pipelines to avoid heavy startup until first call
_sentiment_pipe = None
_summarizer_pipe = None
_kw_model = None
_embedding_model = None

def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        # multilingual sentiment baseline; can be changed to a custom fine-tuned model
        _sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return _sentiment_pipe

def get_summarizer():
    global _summarizer_pipe
    if _summarizer_pipe is None:
        # English summarizer; replace with mT5 for multilingual summarization if needed
        _summarizer_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer_pipe

def get_kw_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    return _kw_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# Helpers
def predict_sentiment(text: str):
    pipe = get_sentiment_pipe()
    out = pipe(text[:512])  # protect from huge inputs
    # nlptown returns labels '1 star'..'5 star'. Map to Negative/Neutral/Positive
    label = out[0]['label']
    score = float(out[0]['score'])
    try:
        stars = int(label.split()[0])
    except:
        stars = 3
    if stars <= 2:
        cls = 'negative'
    elif stars == 3:
        cls = 'neutral'
    else:
        cls = 'positive'
    return {'label': cls, 'confidence': score, 'raw': out[0]}

def summarize_text(text: str, max_length=60):
    pipe = get_summarizer()
    # summarizer expects sufficiently long text; for short texts return same text truncated
    if len(text.split()) < 10:
        s = text if len(text) < 250 else text[:250]
        return s
    try:
        out = pipe(text, max_length=max_length, min_length=10, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        # fallback to return leading sentence
        return text.split('. ')[0]

def extract_keywords(text: str, top_n=8):
    kw = get_kw_model()
    try:
        keys = kw.extract_keywords(text, keyphrase_ngram_range=(1,2), top_n=top_n)
        return [{'term': k[0], 'score': float(k[1])} for k in keys]
    except Exception:
        toks = text.split()
        uniq = list(dict.fromkeys(toks))
        return [{'term': t, 'score': 0.0} for t in uniq[:top_n]]

def generate_wordcloud_image(texts, width=800, height=400):
    # texts: list of strings
    combined = " ".join(texts)
    wc = WordCloud(width=width, height=height, background_color='white', collocations=False).generate(combined)
    img = wc.to_image()
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return "data:image/png;base64," + b64

def semantic_similarity(a, b):
    emb = get_embedding_model()
    ea = emb.encode([a])
    eb = emb.encode([b])
    return float(cosine_similarity(ea, eb)[0,0])
