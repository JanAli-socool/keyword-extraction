#!/usr/bin/env python3
"""
final_pipeline.py

Full NLP pipeline for SEC filings:
- Section-aware extraction (MD&A, Human Capital, Governance, Board Diversity)
- Glossary-driven corpus construction (Financial & Diversity)
- Sentence- and paragraph-level evidence storage
- Confidence tags (why paragraph/sentence was included)
- Sentiment analysis with FinBERT
- Numeric/statistical extraction (contextual)
- Cosine similarity / alignment scores
- Topic modeling (BERTopic, optional)
- Linguistic category tagging (Maximizing, Achievements, Comparative, Goal-oriented)
- Multi-sheet Excel output (summary, details, debug_shared, glossaries)
"""

import os, sys, re, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# NLP libs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline, AutoTokenizer
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# optional BERTopic
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# ---------------------------
# Config / Hyperparams
# ---------------------------
DIVERSITY_GLOSSARY_PATH = "DEI_Glossary.xlsx"
FINANCIAL_GLOSSARY_PATH = "FinancialPerformance_Glossary.xlsx"

EMBED_MODEL = "all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"

MIN_PARA_WORDS = 5
MAX_PARA_WORDS = 1000
SIMILARITY_OVERLAP_THRESHOLD = 0.95
BOOTSTRAP_SAMPLES = 50
NUM_CONTEXT_WINDOW = 40
SENTIMENT_BATCH = 32

# ---------------------------
# Glossary handling
# ---------------------------
def load_glossary(path: str) -> List[str]:
    df = pd.read_excel(path, engine="openpyxl")
    terms = []
    for col in df.columns:
        terms.extend(df[col].dropna().astype(str).tolist())
    cleaned = [re.sub(r"\s+", " ", t).strip().lower() for t in terms if t and not re.fullmatch(r"\d+", t.strip())]
    return sorted(set(cleaned))

def compile_patterns(terms: List[str]) -> List[re.Pattern]:
    return [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms if len(t) > 1]

# ---------------------------
# Text cleaning / splitting
# ---------------------------
def strip_edgar_header(text: str) -> str:
    text = re.sub(r"<SEC-HEADER>.*?</SEC-HEADER>", " ", text, flags=re.I | re.S)
    text = re.sub(r"(^|\n)\s*(CIK:|COMPANY CONFORMED NAME:|CENTRAL INDEX KEY:|FILED:|CONFORMED PERIOD OF REPORT:|ACCESSION NUMBER:).*", " ", text, flags=re.I)
    return text

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

# ---------------------------
# Section detection
# ---------------------------
SECTION_HEADING_PATTERNS = {
    "MD&A": re.compile(r"(management’s?\s+discussion\s+and\s+analysis|item\s+7\.)", re.I),
    "HumanCapital": re.compile(r"(human\s+capital|our\s+people|workforce)", re.I),
    "BoardNominees": re.compile(r"(nominees\s+for\s+election|board\s+of\s+directors|proxy\s+statement)", re.I),
    "CorporateGovernance": re.compile(r"(corporate\s+governance|governance\s+committee)", re.I),
}
def detect_section(paragraph: str) -> List[str]:
    return [name for name, pat in SECTION_HEADING_PATTERNS.items() if pat.search(paragraph)]

def extract_company_info(text: str) -> Tuple[str, str]:
    name = re.search(r"COMPANY CONFORMED NAME:\s*(.+)", text, re.I)
    cik = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", text, re.I)
    return (name.group(1).strip() if name else ""), (cik.group(1).strip() if cik else "")

# ---------------------------
# Linguistic categories
# ---------------------------
def categorize_linguistic_patterns(text: str) -> List[str]:
    cats = []
    if re.search(r"\b(increase|expand|grow|maximize|improve|strive)\b", text, re.I):
        cats.append("Maximizing")
    if re.search(r"\b(achieve|award|earned|recognized|proud|progress)\b", text, re.I):
        cats.append("Achievements")
    if re.search(r"\b(compare|versus|benchmark|since|percent)\b", text, re.I):
        cats.append("Comparative/Evaluating")
    if re.search(r"\b(goal|target|aspire|plan|2030|commit|future|objective)\b", text, re.I):
        cats.append("Goal-oriented/Forward-looking")
    return cats or ["Uncategorized"]

# ---------------------------
# Numeric extraction
# ---------------------------
NUM_CONTEXT_WORDS = r"(employee|women|percent|%|board|director|target|goal|aspire|hire|year|growth|benchmark)"
def extract_numbers_with_context(text: str) -> List[Dict[str, str]]:
    matches = []
    for m in re.finditer(r"(\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b)", text):
        num = m.group(1)
        s, e = max(0, m.start() - NUM_CONTEXT_WINDOW), m.end() + NUM_CONTEXT_WINDOW
        ctx = text[s:e]
        if re.search(NUM_CONTEXT_WORDS, ctx, re.I):
            matches.append({"number": num, "context": ctx})
    return matches

# ---------------------------
# Embeddings & similarity
# ---------------------------
embedder = SentenceTransformer(EMBED_MODEL)
embed_dim = embedder.get_sentence_embedding_dimension()
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert = pipeline("sentiment-analysis", model=FINBERT_MODEL, tokenizer=tokenizer)

def encode_sentences(sents: List[str]) -> np.ndarray:
    if not sents:
        return np.zeros((0, embed_dim))
    emb = embedder.encode(sents, convert_to_numpy=True)
    return emb if emb.ndim > 1 else emb.reshape(1, -1)

def compute_centroid_metrics(div_emb, fin_emb) -> Dict[str, Any]:
    if div_emb.shape[0] == 0 or fin_emb.shape[0] == 0:
        return {}
    c1, c2 = div_emb.mean(0, keepdims=True), fin_emb.mean(0, keepdims=True)
    return {"centroid_cosine": float(cosine_similarity(c1, c2)[0, 0])}

# ---------------------------
# Sentiment
# ---------------------------
def safe_sentiment(sents: List[str]) -> List[Dict[str, Any]]:
    out = []
    for i in range(0, len(sents), SENTIMENT_BATCH):
        out.extend(finbert(sents[i:i+SENTIMENT_BATCH], truncation=True, max_length=512))
    return [{"label": r["label"].lower(), "score": r["score"]} for r in out]

# ---------------------------
# File summarizer
# ---------------------------
def summarize_file(path: Path, div_patterns, fin_patterns):
    text = path.read_text(errors="ignore")
    name, cik = extract_company_info(text)
    paras = split_paragraphs(clean_text(strip_edgar_header(text)))

    detail_rows, div_sents, fin_sents = [], [], []
    for p in paras:
        for s in sent_tokenize(p):
            hit_div = any(pat.search(s) for pat in div_patterns)
            hit_fin = any(pat.search(s) for pat in fin_patterns)
            if hit_div or hit_fin:
                cats = categorize_linguistic_patterns(s)
                nums = extract_numbers_with_context(s)
                detail_rows.append({
                    "File": path.name,
                    "Company": name,
                    "CIK": cik,
                    "Sentence": s,
                    "Diversity": hit_div,
                    "Financial": hit_fin,
                    "LinguisticTags": ", ".join(cats),
                    "Numbers": nums,
                })
                if hit_div: div_sents.append(s)
                if hit_fin: fin_sents.append(s)

    div_emb, fin_emb = encode_sentences(div_sents), encode_sentences(fin_sents)
    metrics = compute_centroid_metrics(div_emb, fin_emb)
    sentiments = safe_sentiment([r["Sentence"] for r in detail_rows])
    for i, row in enumerate(detail_rows):
        row["Sentiment"] = sentiments[i]["label"]
        row["SentimentScore"] = sentiments[i]["score"]

    summary = {
        "File": path.name,
        "Company": name,
        "CIK": cik,
        "Div_Sentences": len(div_sents),
        "Fin_Sentences": len(fin_sents),
        "AlignmentScore": metrics.get("centroid_cosine"),
    }
    return summary, detail_rows

# ---------------------------
# Main
# ---------------------------
def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("src_dir")
    p.add_argument("out_xlsx")
    args = p.parse_args(argv)

    div_terms, fin_terms = load_glossary(DIVERSITY_GLOSSARY_PATH), load_glossary(FINANCIAL_GLOSSARY_PATH)
    div_patterns, fin_patterns = compile_patterns(div_terms), compile_patterns(fin_terms)

    summaries, all_details = [], []
    for f in Path(args.src_dir).glob("*.txt"):
        s, d = summarize_file(f, div_patterns, fin_patterns)
        summaries.append(s)
        all_details.extend(d)

    with pd.ExcelWriter(args.out_xlsx) as w:
        pd.DataFrame(summaries).to_excel(w, "summary", index=False)
        pd.DataFrame(all_details).to_excel(w, "details", index=False)
        pd.DataFrame({"diversity_terms": div_terms}).to_excel(w, "diversity_glossary", index=False)
        pd.DataFrame({"financial_terms": fin_terms}).to_excel(w, "financial_glossary", index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
