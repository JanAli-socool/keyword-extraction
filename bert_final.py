#!/usr/bin/env python3
"""
final_pipeline_bertopic.py

Same as final_pipeline_full.py but using BERTopic for topic modeling
 - Global topic modeling on all text (sentences / paragraphs)
 - Diversity: both sentence + paragraph topics
 - Financial: sentence topics only, plus combined topics
 - Multi-level metadata fallback
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import hdbscan
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from umap import UMAP
import nltk
# Download standard punkt
nltk.download("punkt", quiet=True)

# Fallback for punkt_tab if needed
try:
    nltk.data.find("tokenizers/punkt_tab/english.pickle")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

from bertopic import BERTopic

# ---------- CONFIG ----------
DIVERSITY_GLOSSARY_PATH = "DEI_Glossary.xlsx"

# ---------- IO helpers ----------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def clean_sec_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    for t in soup.find_all(["table", "script", "style", "noscript", "iframe"]):
        t.decompose()
    for tag in soup.find_all():
        if tag.name and ":" in tag.name:
            tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\|\|+", " ", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", " ", text)
    text = re.sub(r"\b[A-Z]{2,5}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_html_file(path: Path) -> str:
    raw = read_text_file(path)
    return clean_sec_text(raw)

# ---------- Metadata extraction ----------
def extract_from_raw_header(raw: str) -> Tuple[str, str, str]:
    company, cik, year = "", "", ""
    m_name = re.search(r"COMPANY CONFORMED NAME:\s*(.+?)(?:\r|\n)", raw, re.I)
    if m_name: company = m_name.group(1).strip()
    m_cik = re.search(r"CENTRAL INDEX KEY:\s*([0-9\-]{4,20})", raw, re.I)
    if m_cik: cik = m_cik.group(1).strip()
    m_year = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})", raw, re.I)
    if m_year: year = m_year.group(1).strip()
    return company, cik, year

def extract_from_html_title(raw: str) -> Tuple[str, str]:
    soup = BeautifulSoup(raw, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    company_guess = title.split("-")[0].strip() if title else ""
    return company_guess, title

def extract_from_filename(fname: str) -> Tuple[str, str, str]:
    stem = Path(fname).stem
    m_cik = re.search(r"(\d{7,10})", fname)
    cik = m_cik.group(1) if m_cik else ""
    m_year = re.search(r"(20\d{2}|19\d{2})", fname)
    year = m_year.group(1) if m_year else ""
    parts = re.split(r"[_\-\.]", stem)
    company = parts[0] if parts else ""
    return company, cik, year

def consolidate_metadata(filename: Path, raw: str) -> Tuple[str, str, str]:
    comp, cik, year = extract_from_raw_header(raw)
    if not comp or not cik or not year:
        title_comp, _ = extract_from_html_title(raw)
        if not comp and title_comp: comp = title_comp
    if not comp or not cik or not year:
        f_comp, f_cik, f_year = extract_from_filename(str(filename.name))
        if not comp and f_comp: comp = f_comp
        if not cik and f_cik: cik = f_cik
        if not year and f_year: year = f_year
    if (not comp or not cik) and filename.parent:
        parent = filename.parent.name
        p_comp, p_cik, p_year = extract_from_filename(parent)
        if not comp and p_comp: comp = p_comp
        if not cik and p_cik: cik = p_cik
        if not year and p_year: year = p_year
    cik = cik.strip()
    if cik.isdigit() and len(cik) < 10:
        cik = cik.zfill(10)
    return comp.strip(), cik, year

# ---------- Glossary ----------
def load_glossary(path: str) -> List[str]:
    df = pd.read_excel(path, engine="openpyxl")
    terms = []
    for col in df.columns:
        terms.extend(df[col].dropna().astype(str).tolist())
    cleaned = [re.sub(r"\s+", " ", t).strip().lower() for t in terms if t]
    return sorted(set(cleaned))

def compile_term_patterns(terms: List[str]):
    out = []
    for t in terms:
        t_flexible = re.escape(t).replace(r"\ ", r"\s+")
        pat = re.compile(r"\b" + t_flexible + r"\b", flags=re.I)
        out.append((t, pat))
    return out

# ---------- MD&A extraction ----------
def extract_mdna_text(raw_text: str) -> str:
    mdna_match = re.search(
        r"(item\s*7[^a-zA-Z]{0,3}.*?management.*?discussion.*?analysis.*?)(?=item\s*7a|item\s*8|item\s*9|signatures?)",
        raw_text,
        flags=re.I | re.S
    )
    return mdna_match.group(1) if mdna_match else ""

# ---------- BERTopic wrapper ----------
# Now handles small input sizes gracefully.. currently always returns dict for topic_labels..

def run_bertopic(sentences, n_neighbors=5, n_components=2, min_dist=0.1):
    if not sentences:
        return None
    
    n_points = len(sentences)

    # Case 1: Only one sentence → assign topic 0 directly
    if n_points == 1:
        return {
            "topic_ids": [0],
            "topic_labels": {0: "SingleSentence"}   # dictionary now
        }

    # Case 2: Only two sentences → assign simple topics directly
    if n_points == 2:
        return {
            "topic_ids": [0, 1],
            "topic_labels": {0: "Topic0", 1: "Topic1"}  # dictionary now
        }

    # Case 3: Normal case → Run BERTopic safely
    n_neighbors = max(2, min(n_neighbors, n_points - 1))  # Ensure >= 2
    min_cluster_size = min(2, n_points)
    min_samples = min(1, n_points)

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        random_state=42
    )
    
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True
    )
    
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )

    topics, probs = model.fit_transform(sentences)
    topic_info = model.get_topic_info()

    topic_map = {
        row["Topic"]: " ".join([w for w, _ in model.get_topic(row["Topic"])])
        for _, row in topic_info.iterrows() if row["Topic"] != -1
    }

    topic_labels = {tid: topic_map.get(tid, "Misc") for tid in set(topics)}

    return {
        "topic_ids": topics,
        "topic_labels": topic_labels  # always dictionary
    }


# ---------- Main Pipeline ----------
def run_pipeline(src_dir: str, out_xlsx: str, limit: int = 200):
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(src)

    div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
    div_term_patterns = compile_term_patterns(div_terms)

    files = sorted(list(src.glob("*.txt")) + list(src.glob("*.htm")) + list(src.glob("*.html")))
    files = files[:limit]

    dei_sentence_texts, dei_paragraph_texts, fin_sentence_texts = [], [], []
    dei_sent_meta, fin_sent_meta = [], []

    for path in tqdm(files, desc="Scanning files"):
        raw_for_meta = read_text_file(path)
        cleaned = clean_sec_text(raw_for_meta)
        company, cik, year = consolidate_metadata(path, raw_for_meta)
        filing_type = "DEF14" if "def" in path.stem.lower() else "10-K"

        # DEI sentences & paragraphs
        paragraphs = [p for p in re.split(r"(?:\r\n\r\n|\n\s*\n|\r\n|\n){1,}", cleaned) if p.strip()]
        for p in paragraphs:
            para_terms = [t for t, pat in div_term_patterns if pat.search(p)]
            if para_terms:
                dei_paragraph_texts.append(p)
                for s in sent_tokenize(p):
                    if any(pat.search(s) for _, pat in div_term_patterns):
                        matched = sorted(set([t for t, pat in div_term_patterns if pat.search(s)]))
                        dei_sentence_texts.append(s)
                        dei_sent_meta.append({
                            "Company": company,
                            "CIK": cik,
                            "Year": year,
                            "FilingType": filing_type,
                            "Sentence": s,
                            "Paragraph": p,
                            "DEI_Words": matched,
                            "SourceFile": path.name
                        })

        # MD&A for financials
        if filing_type == "10-K":
            mdna = extract_mdna_text(raw_for_meta)
            if mdna:
                mdna_clean = clean_sec_text(mdna)
                for s in sent_tokenize(mdna_clean):
                    if len(s.split()) > 3 and re.search(r"[A-Za-z]", s):
                        fin_sentence_texts.append(s)
                        fin_sent_meta.append({
                            "Company": company,
                            "CIK": cik,
                            "Year": year,
                            "FilingType": filing_type,
                            "Sentence": s,
                            "SourceFile": path.name
                        })

    # ---- Run BERTopic on global corpora ----
    div_sent_topics = run_bertopic(dei_sentence_texts)
    div_para_topics = run_bertopic(dei_paragraph_texts)
    fin_sent_topics = run_bertopic(fin_sentence_texts)
    fin_all_topics = run_bertopic([" ".join(fin_sentence_texts)])

    # ---- Map topics back ----
    diversity_rows = []
    for meta in dei_sent_meta:
        s_idx = dei_sentence_texts.index(meta["Sentence"])
        p_idx = dei_paragraph_texts.index(meta["Paragraph"]) if meta["Paragraph"] in dei_paragraph_texts else -1
        tid_sent = div_sent_topics["topic_ids"][s_idx] if div_sent_topics else None
        tid_para = div_para_topics["topic_ids"][p_idx] if div_para_topics and p_idx >= 0 else None
        label_sent = div_sent_topics["topic_labels"].get(tid_sent, "") if div_sent_topics else ""
        label_para = div_para_topics["topic_labels"].get(tid_para, "") if div_para_topics else ""
        for word in meta["DEI_Words"]:
            diversity_rows.append({
                **meta,
                "DEI_Word": word,
                "Topic_ID_Sentence": tid_sent,
                "Topic_Label_Sentence": label_sent,
                "Topic_ID_Paragraph": tid_para,
                "Topic_Label_Paragraph": label_para
            })

    financial_rows = []
    for meta in fin_sent_meta:
        s_idx = fin_sentence_texts.index(meta["Sentence"])
        tid = fin_sent_topics["topic_ids"][s_idx] if fin_sent_topics else None
        label = fin_sent_topics["topic_labels"].get(tid, "") if fin_sent_topics else ""
        financial_rows.append({**meta, "Topic_ID": tid, "Topic_Label": label})

    # ---- Write Excel ----
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        pd.DataFrame(diversity_rows).to_excel(writer, sheet_name="diversity_output", index=False)
        pd.DataFrame(financial_rows).to_excel(writer, sheet_name="financial_output", index=False)
        pd.DataFrame({"diversity_terms": div_terms}).to_excel(writer, sheet_name="diversity_glossary", index=False)

    print(f"✅ Wrote {out_xlsx}")
    print(f"DEI rows: {len(diversity_rows)}, Financial rows: {len(financial_rows)}")

# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run BERTopic pipeline")
    ap.add_argument("src_dir", help="Directory with filings (.txt/.htm/.html)")
    ap.add_argument("out_xlsx", help="Output Excel filename")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    run_pipeline(args.src_dir, args.out_xlsx, limit=args.limit)



