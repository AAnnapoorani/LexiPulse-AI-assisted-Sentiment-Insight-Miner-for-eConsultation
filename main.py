# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
from .utils import normalize_text, redact_pii, detect_language, simple_tokenize
from .nlp import predict_sentiment, summarize_text, extract_keywords, generate_wordcloud_image
from typing import List
import os

app = FastAPI(title="eConsultation Sentiment Prototype")

# serve static UI
if not os.path.exists('app/static'):
    os.makedirs('app/static', exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    html = open("app/static/index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

@app.post("/ingest_csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file with columns: id, draft_id, text, created_at (optional)
    Returns per-row processed results with sentiment, summary, keywords.
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        # try excel
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            return JSONResponse({"error": "Unable to read file. Provide CSV or Excel."}, status_code=400)
    if 'text' not in df.columns and 'comment' not in df.columns:
        return JSONResponse({"error": "CSV must contain 'text' or 'comment' column."}, status_code=400)
    text_col = 'text' if 'text' in df.columns else 'comment'
    results = []
    all_texts_for_wc = []
    for idx, row in df.iterrows():
        raw = str(row[text_col])
        norm = normalize_text(raw)
        pii = redact_pii(norm)
        lang = detect_language(pii)
        # You may decide to translate non-English here. This prototype keeps original.
        sentiment = predict_sentiment(pii)
        summary = summarize_text(pii)
        keywords = extract_keywords(pii)
        results.append({
            'id': str(row.get('id', idx)),
            'draft_id': row.get('draft_id', None),
            'lang': lang,
            'text': raw,
            'clean_text': pii,
            'sentiment': sentiment,
            'summary': summary,
            'keywords': keywords
        })
        all_texts_for_wc.append(pii)
    wc_b64 = generate_wordcloud_image(all_texts_for_wc)
    return JSONResponse({'results': results, 'wordcloud': wc_b64})

@app.get("/health")
async def health():
    return {"status": "ok"}
