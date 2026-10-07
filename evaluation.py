

"""
evaluation.py
-------------
Runs every question in Evaluation_Results/test_questions.csv through the RAG
pipeline and writes REAL measured results to Evaluation_Results/evaluation_results.csv
plus a short summary (evaluation_summary.txt). Nothing is hard-coded or simulated.

test_questions.csv columns:
    question_id, question, expected_keywords (';'-separated), expected_pages (';'-separated),
    answerable (yes/no), category

Per-question metrics:
    retrieval_hit@K    : 1 if any retrieved chunk is on an expected page (answerable only)
    mrr                : 1/rank of the first retrieved chunk on an expected page (answerable only)
    citation_page_hit  : 1 if any cited page is an expected page (answerable only)
    keyword_recall     : fraction of expected keywords present in the answer (answerable only)
    refused            : 1 if the system said sufficient evidence was not found
    refusal_correct    : 1 if refused == (answerable == "no")
    latency_s          : end-to-end time per question

Usage (from Code/):
    python evaluation.py                              # uses model_config.txt settings
    python evaluation.py --provider none              # no-LLM baseline (retrieval + extractive)
    python evaluation.py --top_k 5 --chunk_size 600 --chunk_overlap 100
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ingestion import PROJECT_ROOT, load_config, setup_logging
from rag_pipeline import RAGPipeline

logger = logging.getLogger("hld_rag.evaluation")
EVAL_DIR = PROJECT_ROOT / "Evaluation_Results"
DEFAULT_PDF = "D:\\prachi\\code\\Input_Data\\Synthetic_AUTOSAR_HLD.pdf"

RESULT_FIELDS = [
    "question_id", "category", "question", "answerable", "expected_pages", "expected_keywords",
    "answer", "supported", "refused", "refusal_correct", "retrieved_pages", "top_score",
    "retrieval_hit_at_k", "mrr", "cited_pages", "cited_sections", "citation_page_hit",
    "keywords_found", "keyword_recall", "latency_s", "provider", "top_k", "chunk_size", "chunk_overlap", "note",
]


def _split(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def load_questions(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


def evaluate_one(rag: RAGPipeline, row: Dict, cfg: Dict) -> Dict:
    answerable = row["answerable"].strip().lower() in ("yes", "y", "true", "1")
    exp_pages = {int(p) for p in _split(row.get("expected_pages", ""))}
    exp_kw = _split(row.get("expected_keywords", ""))

    res = rag.answer(row["question"])
    retrieved_pages = [e.chunk.page for e in res.evidence]
    cited_pages = [c["page"] for c in res.citations]
    refused = not res.supported

    out = {
        "question_id": row["question_id"], "category": row.get("category", ""), "question": row["question"],
        "answerable": "yes" if answerable else "no", "expected_pages": row.get("expected_pages", ""),
        "expected_keywords": row.get("expected_keywords", ""), "answer": res.answer.replace("\n", " "),
        "supported": int(res.supported), "refused": int(refused), "refusal_correct": int(refused == (not answerable)),
        "retrieved_pages": ";".join(map(str, retrieved_pages)),
        "top_score": round(res.evidence[0].score, 4) if res.evidence else "",
        "cited_pages": ";".join(map(str, cited_pages)),
        "cited_sections": " | ".join(c["section"] for c in res.citations),
        "latency_s": round(res.latency_s, 3), "provider": res.provider, "top_k": cfg["TOP_K"],
        "chunk_size": cfg["CHUNK_SIZE"], "chunk_overlap": cfg["CHUNK_OVERLAP"], "note": res.note,
        "retrieval_hit_at_k": "", "mrr": "", "citation_page_hit": "", "keywords_found": "", "keyword_recall": "",
    }
    if answerable:
        ranks = [i + 1 for i, p in enumerate(retrieved_pages) if p in exp_pages]
        out["retrieval_hit_at_k"] = int(bool(ranks))
        out["mrr"] = round(1.0 / ranks[0], 4) if ranks else 0.0
        out["citation_page_hit"] = int(any(p in exp_pages for p in cited_pages))
        ans_low = res.answer.lower()
        found = [k for k in exp_kw if k.lower() in ans_low]
        out["keywords_found"] = ";".join(found)
        out["keyword_recall"] = round(len(found) / len(exp_kw), 4) if exp_kw else ""
    return out


def _mean(rows: List[Dict], key: str):
    vals = [float(r[key]) for r in rows if r[key] != ""]
    return round(statistics.mean(vals), 4) if vals else None


def summarize(results: List[Dict], cfg: Dict, pdf: Path) -> str:
    ans = [r for r in results if r["answerable"] == "yes"]
    una = [r for r in results if r["answerable"] == "no"]
    lines = [
        "AUTOSAR HLD RAG - Evaluation Summary",
        f"Run at            : {datetime.now().isoformat(timespec='seconds')}",
        f"Document          : {pdf.name}",
        f"LLM provider      : {cfg['LLM_PROVIDER']}",
        f"Embedding model   : {cfg['EMBEDDING_MODEL']}",
        f"Chunk size/overlap: {cfg['CHUNK_SIZE']}/{cfg['CHUNK_OVERLAP']}   Top-K: {cfg['TOP_K']}   Threshold: {cfg['SIMILARITY_THRESHOLD']}",
        f"Questions         : {len(results)} ({len(ans)} answerable, {len(una)} unanswerable)",
        "",
        f"Retrieval Hit@K (answerable)        : {_mean(ans, 'retrieval_hit_at_k')}",
        f"MRR (answerable)                    : {_mean(ans, 'mrr')}",
        f"Citation page accuracy (answerable) : {_mean(ans, 'citation_page_hit')}",
        f"Keyword recall (answerable)         : {_mean(ans, 'keyword_recall')}",
        f"False refusals (answerable refused) : {sum(int(r['refused']) for r in ans)} / {len(ans)}",
        f"Correct refusals (unanswerable)     : {sum(int(r['refused']) for r in una)} / {len(una)}",
        f"Overall refusal decision accuracy   : {_mean(results, 'refusal_correct')}",
        f"Mean latency (s)                    : {_mean(results, 'latency_s')}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the AUTOSAR HLD RAG pipeline")
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--questions", default=str(EVAL_DIR / "test_questions.csv"))
    ap.add_argument("--output", default=str(EVAL_DIR / "evaluation_results.csv"))
    ap.add_argument("--provider", help="gemini / groq / openai / ollama / none")
    ap.add_argument("--top_k", type=int)
    ap.add_argument("--chunk_size", type=int)
    ap.add_argument("--chunk_overlap", type=int)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between questions (API rate limits)")
    args = ap.parse_args()

    cfg = load_config()
    for arg, key in [("provider", "LLM_PROVIDER"), ("top_k", "TOP_K"), ("chunk_size", "CHUNK_SIZE"),
                     ("chunk_overlap", "CHUNK_OVERLAP"), ("threshold", "SIMILARITY_THRESHOLD")]:
        if getattr(args, arg) is not None:
            cfg[key] = getattr(args, arg)
    setup_logging(cfg, "evaluation.log")

    pdf = Path(args.pdf)
    questions = load_questions(Path(args.questions))
    logger.info("Evaluating %d questions on %s with provider=%s", len(questions), pdf.name, cfg["LLM_PROVIDER"])

    rag = RAGPipeline(cfg)
    rag.build([pdf])

    results = []
    for i, row in enumerate(questions, 1):
        try:
            r = evaluate_one(rag, row, cfg)
        except Exception as exc:
            logger.exception("Question %s failed", row.get("question_id"))
            r = {k: "" for k in RESULT_FIELDS}
            r.update(question_id=row.get("question_id"), question=row.get("question"), note=f"ERROR: {exc}",
                     answerable=row.get("answerable"), refused=0, refusal_correct=0, latency_s=0)
        results.append(r)
        print(f"[{i}/{len(questions)}] {r['question_id']}: refused={r['refused']} hit={r['retrieval_hit_at_k']} "
              f"kw={r['keyword_recall']} ({r['latency_s']}s)")
        if args.delay:
            time.sleep(args.delay)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)

    summary = summarize(results, cfg, pdf)
    (out.parent / "evaluation_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()