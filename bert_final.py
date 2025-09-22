#!/usr/bin/env python3
"""
final_pipeline_with_topics.py

Enhanced pipeline:
 - Section-aware extraction (MD&A, Human Capital, Governance, Board Diversity)
 - Glossary-driven corpus construction (Financial & Diversity)
 - Sentence- and paragraph-level evidence storage (full sentence)
 - MatchReason (which glossary terms matched) for auditability
 - Sentiment analysis with FinBERT
 - Numeric/statistical extraction (contextual)
 - Cosine similarity / alignment scores
 - Topic modeling (BERTopic optional, LDA fallback)
 - Linguistic category tagging (Maximizing, Achievements, Comparative, Goal-oriented)
 - Multi-sheet Excel output (summary, details, topics, debug_shared, glossaries)
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
from transformers import pipeline, AutoTokenizer
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# Topic modelling fallbacks
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# sklearn LDA fallback
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------
# Config / Hyperparams
# ---------------------------
DIVERSITY_GLOSSARY_PATH = "DEI_Glossary.xlsx"
FINANCIAL_GLOSSARY_PATH = "FinancialPerformance_Glossary.xlsx"

EMBED_MODEL = "all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"

MIN_PARA_WORDS = 5
MAX_PARA_WORDS = 1000
NUM_CONTEXT_WINDOW = 40
SENTIMENT_BATCH = 32
TOPIC_NUM = 8  # default number of topics for LDA fallback

# ---------------------------
# Glossary handling (flexible whitespace)
# ---------------------------
def load_glossary(path: str) -> List[str]:
    df = pd.read_excel(path, engine="openpyxl")
    terms = []
    for col in df.columns:
        terms.extend(df[col].dropna().astype(str).tolist())
    cleaned = [re.sub(r"\s+", " ", t).strip().lower() for t in terms if t and not re.fullmatch(r"\d+", t.strip())]
    return sorted(set(cleaned))

def compile_term_patterns(terms: List[str]) -> List[Tuple[str, re.Pattern]]:
    """
    For each glossary term produce (term, compiled_pattern) where internal spaces allow flexible whitespace/linebreaks.
    """
    out = []
    for t in terms:
        # escape, then allow arbitrary whitespace between words
        t_escaped = re.escape(t)
        t_flexible = t_escaped.replace(r"\ ", r"\s+")
        pattern = re.compile(r"\b" + t_flexible + r"\b", flags=re.I)
        out.append((t, pattern))
    return out

# ---------------------------
# Text reading / cleaning / splitting
# ---------------------------
def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def strip_edgar_header(text: str) -> str:
    text = re.sub(r"<SEC-HEADER>.*?</SEC-HEADER>", " ", text, flags=re.I | re.S)
    text = re.sub(r"(^|\n)\s*(CIK:|COMPANY CONFORMED NAME:|CENTRAL INDEX KEY:|FILED:|CONFORMED PERIOD OF REPORT:|ACCESSION NUMBER:).*", " ", text, flags=re.I)
    return text

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def split_paragraphs(text: str) -> List[str]:
    # split on double newline OR single newline to be robust across EDGAR formats
    paras = re.split(r"(?:\r\n\r\n|\n\s*\n|\r\n|\n)", text)
    return [p.strip() for p in paras if p.strip()]

# ---------------------------
# Section detection (use your patterns)
# ---------------------------
SECTION_HEADING_PATTERNS = {
    "MD&A": re.compile(r"(management’s?\s+discussion\s+and\s+analysis|item\s+7\.)", re.I),
    "HumanCapital": re.compile(r"(human\s+capital|our\s+people|workforce|human capital resources)", re.I),
    "BoardNominees": re.compile(r"(nominees\s+for\s+election|board\s+of\s+directors|nominee[s]?\b|proxy\s+statement)", re.I),
    "CorporateGovernance": re.compile(r"(corporate\s+governance|governance\s+committee|nominating\s+and\s+governance)", re.I),
}
def detect_section(paragraph: str) -> List[str]:
    tags = []
    for name, pat in SECTION_HEADING_PATTERNS.items():
        if pat.search(paragraph):
            tags.append(name)
    return tags

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
NUM_CONTEXT_WORDS = r"(employee|employees|women|men|percent|%|board|director|directors|target|goal|aspire|hire|hiring|headcount|FTE|year|age|salary|compensation|increase|decrease|growth|benchmark)"
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
# Embeddings & sentiment (FinBERT)
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
        return {"centroid_cosine": None, "n_div": int(div_emb.shape[0]), "n_fin": int(fin_emb.shape[0])}
    c1, c2 = div_emb.mean(0, keepdims=True), fin_emb.mean(0, keepdims=True)
    return {"centroid_cosine": float(cosine_similarity(c1, c2)[0, 0]), "n_div": int(div_emb.shape[0]), "n_fin": int(fin_emb.shape[0])}

def safe_sentiment(sents: List[str]) -> List[Dict[str, Any]]:
    out = []
    for i in range(0, len(sents), SENTIMENT_BATCH):
        out.extend(finbert(sents[i:i+SENTIMENT_BATCH], truncation=True, max_length=512))
    return [{"label": r["label"].lower(), "score": r["score"]} for r in out]

# ---------------------------
# Topic modeling wrappers
# ---------------------------
def run_bertopic(sentences: List[str]):
    if not BERTOPIC_AVAILABLE:
        return None
    try:
        model = BERTopic(verbose=False)
        topics, probs = model.fit_transform(sentences)
        return {"model": model, "topics": topics, "probs": probs}
    except Exception as e:
        print("BERTopic failed:", e)
        return None

def run_lda(sentences: List[str], n_topics: int = TOPIC_NUM):
    # simple LDA fallback on sentences
    if not sentences:
        return None
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(sentences)
    if X.shape[1] == 0:
        return None
    lda = LatentDirichletAllocation(n_components=min(n_topics, X.shape[1]), random_state=0)
    topics = lda.fit_transform(X)
    topic_ids = np.argmax(topics, axis=1).tolist()
    # derive topic labels from top words
    words = np.array(vectorizer.get_feature_names_out())
    topic_labels = {}
    for t in range(lda.n_components):
        top_indices = np.argsort(lda.components_[t])[-8:][::-1]
        topic_labels[t] = " ".join(words[top_indices])
    return {"model": lda, "topic_ids": topic_ids, "topic_labels": topic_labels}

# ---------------------------
# Core file summarizer (sentence-level evidence, section, matchreason)
# ---------------------------
def summarize_file(path: Path, div_term_patterns, fin_term_patterns, dataset_label: str):
    raw = read_file(path)
    cleaned = clean_text(strip_edgar_header(raw))
    name, cik = extract_company_info(raw)
    paras = split_paragraphs(cleaned)

    detail_rows = []
    div_sents = []
    fin_sents = []

    for p in paras:
        sections = detect_section(p)
        for s in sent_tokenize(p):
            matched_div = []
            matched_fin = []
            # find matching glossary terms (collect terms themselves)
            for term, pat in div_term_patterns:
                if pat.search(s):
                    matched_div.append(term)
            for term, pat in fin_term_patterns:
                if pat.search(s):
                    matched_fin.append(term)
            if matched_div or matched_fin:
                cats = categorize_linguistic_patterns(s)
                nums = extract_numbers_with_context(s)
                detail = {
                    "File": path.name,
                    "Company": name,
                    "CIK": cik,
                    "Dataset": dataset_label,   # 👈 NEW COLUMN
                    "Section": ";".join(sections) if sections else "",
                    "Sentence": s,
                    "Diversity": bool(matched_div),
                    "Financial": bool(matched_fin),
                    "MatchedTerms": ";".join(matched_div + matched_fin),
                    "LinguisticTags": ", ".join(cats),
                    "Numbers": nums
                }
                detail_rows.append(detail)
                if matched_div:
                    div_sents.append(s)
                if matched_fin:
                    fin_sents.append(s)

    # embeddings & alignment
    div_emb = encode_sentences(div_sents)
    fin_emb = encode_sentences(fin_sents)
    metrics = compute_centroid_metrics(div_emb, fin_emb)

    # sentiment for detail rows (batch)
    sentences_for_sent = [r["Sentence"] for r in detail_rows]
    sentiments = safe_sentiment(sentences_for_sent) if sentences_for_sent else []
    for i, r in enumerate(detail_rows):
        if i < len(sentiments):
            r["Sentiment"] = sentiments[i]["label"]
            r["SentimentScore"] = sentiments[i]["score"]
        else:
            r["Sentiment"] = None
            r["SentimentScore"] = None

    summary = {
        "File": path.name,
        "Company": name,
        "CIK": cik,
        "Dataset": dataset_label,   # 👈 NEW COLUMN
        "Total_Paragraphs": len(paras),
        "Div_Sentences": len(div_sents),
        "Fin_Sentences": len(fin_sents),
        "AlignmentScore": metrics.get("centroid_cosine"),
        "n_div": metrics.get("n_div"),
        "n_fin": metrics.get("n_fin")
    }

    return summary, detail_rows

# ---------------------------
# Main runner: collect, topic-model, write Excel
# ---------------------------
# def run_pipeline(src_dir: str, out_xlsx: str, limit: int = 10, use_bertopic: bool = False):
#     src = Path(src_dir)
#     if not src.exists():
#         raise FileNotFoundError(src)

#     # load glossaries
#     div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
#     fin_terms = load_glossary(FINANCIAL_GLOSSARY_PATH)
#     div_term_patterns = compile_term_patterns(div_terms)
#     fin_term_patterns = compile_term_patterns(fin_terms)

#     # files = sorted([p for p in src.glob("*.txt")])[:limit]
#     files = sorted(list(src.glob("*.txt")) + list(Path("data2").glob("*.txt")))[:limit]

#     summaries = []
#     all_details = []

#     for path in tqdm(files, desc="Processing files"):
#         try:
#             s, d = summarize_file(path, div_term_patterns, fin_term_patterns)
#             summaries.append(s)
#             all_details.extend(d)
#         except Exception as e:
# #             print(f"Error processing {path.name}: {e}")
# def run_pipeline(src_dir: str, out_xlsx: str, limit: int = 10, use_bertopic: bool = False):
#     src = Path(src_dir)
#     extra = Path("data2")  # DEF14 directory

#     if not src.exists():
#         raise FileNotFoundError(src)
#     if not extra.exists():
#         raise FileNotFoundError(extra)
# def run_pipeline(src_dir, out_xlsx, limit=5, use_bertopic=False):
#     if not os.path.exists(src_dir):
#         raise FileNotFoundError(f"Source directory not found: {src_dir}")

#     files = [f for f in os.listdir(src_dir) if f.endswith(".txt")]
#     if not files:
#         raise FileNotFoundError(f"No .txt files found in directory: {src_dir}")

#     # load glossaries
#     div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
#     fin_terms = load_glossary(FINANCIAL_GLOSSARY_PATH)
#     div_term_patterns = compile_term_patterns(div_terms)
#     fin_term_patterns = compile_term_patterns(fin_terms)

#     # collect files with dataset labels
#     files = [(p, "10-K") for p in sorted(src.glob("*.txt"))]
#     files += [(p, "DEF14") for p in sorted(extra.glob("*.txt"))]
#     files = files[:limit]   # still respects limit

#     summaries = []
#     all_details = []

#     for path, dataset_label in tqdm(files, desc="Processing files"):
#         try:
#             s, d = summarize_file(path, div_term_patterns, fin_term_patterns, dataset_label)
#             summaries.append(s)
#             all_details.extend(d)
#         except Exception as e:
#             print(f"Error processing {path.name}: {e}")


#     # write intermediate sheets
#     summary_df = pd.DataFrame(summaries)
#     details_df = pd.DataFrame(all_details)

#     # Topic modeling on sentences in details_df
#     topics_rows = []
#     if not details_df.empty:
#         sentences = details_df["Sentence"].astype(str).tolist()
#         if use_bertopic and BERTOPIC_AVAILABLE:
#             tm = run_bertopic(sentences)
#             if tm:
#                 topic_ids = tm["topics"]
#                 # create topic labels from model
#                 topic_info = tm["model"].get_topic_info()
#                 # extract representative keywords via get_topic
#                 topic_labels = {}
#                 for tid in set(topic_ids):
#                     top_words = tm["model"].get_topic(tid)
#                     if top_words:
#                         topic_labels[tid] = ", ".join([w for w, _ in top_words[:10]])
#                     else:
#                         topic_labels[tid] = ""
#                 for i, row in details_df.reset_index(drop=True).iterrows():
#                     topics_rows.append({
#                         "File": row["File"],
#                         "Company": row["Company"],
#                         "CIK": row["CIK"],
#                         "Sentence": row["Sentence"],
#                         "Topic_ID": int(topic_ids[i]) if i < len(topic_ids) else None,
#                         "Topic_Label": topic_labels.get(topic_ids[i], "") if i < len(topic_ids) else "",
#                         "Diversity": row.get("Diversity", False),
#                         "Financial": row.get("Financial", False),
#                         "MatchedTerms": row.get("MatchedTerms", "")
#                     })
#         else:
#             # LDA fallback
#             lda_res = run_lda(sentences, n_topics=TOPIC_NUM)
#             if lda_res:
#                 topic_ids = lda_res["topic_ids"]
#                 topic_labels = lda_res["topic_labels"]
#                 for i, row in details_df.reset_index(drop=True).iterrows():
#                     tid = topic_ids[i] if i < len(topic_ids) else None
#                     topics_rows.append({
#                         "File": row["File"],
#                         "Company": row["Company"],
#                         "CIK": row["CIK"],
#                         "Sentence": row["Sentence"],
#                         "Topic_ID": int(tid) if tid is not None else None,
#                         "Topic_Label": topic_labels.get(tid, "") if tid is not None else "",
#                         "Diversity": row.get("Diversity", False),
#                         "Financial": row.get("Financial", False),
#                         "MatchedTerms": row.get("MatchedTerms", "")
#                     })

#     topics_df = pd.DataFrame(topics_rows)

#     # Build debug_shared sheet placeholder (could extend to detect paragraph overlaps)
#     debug_shared_df = pd.DataFrame([])

#     # Write Excel with multi-sheets: summary, details, topics, debug_shared, glossaries
#     with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
#         summary_df.to_excel(writer, sheet_name="summary", index=False)
#         details_df.to_excel(writer, sheet_name="details", index=False)
#         topics_df.to_excel(writer, sheet_name="topics", index=False)
#         debug_shared_df.to_excel(writer, sheet_name="debug_shared", index=False)
#         pd.DataFrame({"diversity_terms": div_terms}).to_excel(writer, sheet_name="diversity_glossary", index=False)
#         pd.DataFrame({"financial_terms": fin_terms}).to_excel(writer, sheet_name="financial_glossary", index=False)

#     print(f"Wrote {out_xlsx} — summary rows: {len(summary_df)}, details rows: {len(details_df)}, topics rows: {len(topics_df)}")
def run_pipeline(src_dir, out_xlsx, limit=5, use_bertopic=False):
    src = Path(src_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    files = [(p, "Uploaded") for p in sorted(src.glob("*.txt"))[:limit]]
    if not files:
        raise FileNotFoundError(f"No .txt files found in directory: {src_dir}")

    # load glossaries
    div_terms = load_glossary(DIVERSITY_GLOSSARY_PATH)
    fin_terms = load_glossary(FINANCIAL_GLOSSARY_PATH)
    div_term_patterns = compile_term_patterns(div_terms)
    fin_term_patterns = compile_term_patterns(fin_terms)

    summaries = []
    all_details = []

    for path, dataset_label in tqdm(files, desc="Processing files"):
        try:
            s, d = summarize_file(path, div_term_patterns, fin_term_patterns, dataset_label)
            summaries.append(s)
            all_details.extend(d)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    # write intermediate sheets
    summary_df = pd.DataFrame(summaries)
    details_df = pd.DataFrame(all_details)

    # topic modeling...
    topics_rows = []
    if not details_df.empty:
        sentences = details_df["Sentence"].astype(str).tolist()
        if use_bertopic and BERTOPIC_AVAILABLE:
            tm = run_bertopic(sentences)
            if tm:
                topic_ids = tm["topics"]
                topic_labels = {}
                for tid in set(topic_ids):
                    top_words = tm["model"].get_topic(tid)
                    topic_labels[tid] = ", ".join([w for w, _ in top_words[:10]]) if top_words else ""
                for i, row in details_df.reset_index(drop=True).iterrows():
                    topics_rows.append({
                        "File": row["File"],
                        "Company": row["Company"],
                        "CIK": row["CIK"],
                        "Sentence": row["Sentence"],
                        "Topic_ID": int(topic_ids[i]) if i < len(topic_ids) else None,
                        "Topic_Label": topic_labels.get(topic_ids[i], "") if i < len(topic_ids) else "",
                        "Diversity": row.get("Diversity", False),
                        "Financial": row.get("Financial", False),
                        "MatchedTerms": row.get("MatchedTerms", "")
                    })
        else:
            lda_res = run_lda(sentences, n_topics=TOPIC_NUM)
            if lda_res:
                topic_ids = lda_res["topic_ids"]
                topic_labels = lda_res["topic_labels"]
                for i, row in details_df.reset_index(drop=True).iterrows():
                    tid = topic_ids[i] if i < len(topic_ids) else None
                    topics_rows.append({
                        "File": row["File"],
                        "Company": row["Company"],
                        "CIK": row["CIK"],
                        "Sentence": row["Sentence"],
                        "Topic_ID": int(tid) if tid is not None else None,
                        "Topic_Label": topic_labels.get(tid, "") if tid is not None else "",
                        "Diversity": row.get("Diversity", False),
                        "Financial": row.get("Financial", False),
                        "MatchedTerms": row.get("MatchedTerms", "")
                    })

    topics_df = pd.DataFrame(topics_rows)
    debug_shared_df = pd.DataFrame([])

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        details_df.to_excel(writer, sheet_name="details", index=False)
        topics_df.to_excel(writer, sheet_name="topics", index=False)
        debug_shared_df.to_excel(writer, sheet_name="debug_shared", index=False)
        pd.DataFrame({"diversity_terms": div_terms}).to_excel(writer, sheet_name="diversity_glossary", index=False)
        pd.DataFrame({"financial_terms": fin_terms}).to_excel(writer, sheet_name="financial_glossary", index=False)

    print(f"Wrote {out_xlsx} — summary rows: {len(summary_df)}, details rows: {len(details_df)}, topics rows: {len(topics_df)}")




# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run final pipeline with topics")
    ap.add_argument("src_dir", help="Directory with .txt filings")
    ap.add_argument("out_xlsx", help="Output Excel filename")
    ap.add_argument("--limit", type=int, default=10, help="Max number of files to process (default 10)")
    ap.add_argument("--bertopic", action="store_true", help="Use BERTopic (if installed). Otherwise uses LDA fallback.")
    args = ap.parse_args()

    run_pipeline(args.src_dir, args.out_xlsx, limit=args.limit, use_bertopic=args.bertopic)


