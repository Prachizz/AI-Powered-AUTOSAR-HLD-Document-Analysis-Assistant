

"""
rag_pipeline.py
---------------
Online half of the RAG pipeline:

    question -> query embedding -> FAISS top-K retrieval -> evidence gate
             -> prompt (retrieved context only) -> LLM -> grounded answer
             -> page/section citations

Supported LLM back-ends (set LLM_PROVIDER in model_config.txt):
    gemini : Google Gemini (OpenAI-compatible endpoint), key in env GEMINI_API_KEY
    groq   : Groq cloud (OpenAI-compatible), key in env GROQ_API_KEY
    openai : OpenAI, key in env OPENAI_API_KEY
    ollama : local model served by Ollama (no key)
    none   : no LLM - extractive answer made of the best evidence sentences

Quick CLI test:
    python rag_pipeline.py --pdf ../Input_Data/Synthetic_AUTOSAR_BCM_HLD.pdf -q "Which UDS services are supported?"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ingestion import (
    DEFAULT_PROMPTS_PATH,
    PROJECT_ROOT,
    Chunk,
    build_knowledge_base,
    embed_texts,
    load_config,
    load_vector_store,
    resolve_path,
    setup_logging,
)

try:  # optional: read API keys from Code/.env
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

logger = logging.getLogger("hld_rag.pipeline")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def load_prompts(path: Path | str = DEFAULT_PROMPTS_PATH) -> Dict[str, str]:
    """Parse prompts.txt into {BLOCK_NAME: text}."""
    blocks: Dict[str, List[str]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\[([A-Z_]+)\]\s*$", line.strip())
        if m:
            current = m.group(1)
            blocks[current] = []
        elif current and not line.lstrip().startswith("#"):
            blocks[current].append(line)
    prompts = {k: "\n".join(v).strip() for k, v in blocks.items()}
    for required in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "NO_EVIDENCE_MESSAGE"):
        if required not in prompts:
            raise ValueError(f"prompts.txt is missing the [{required}] block")
    prompts["SYSTEM_PROMPT"] = prompts["SYSTEM_PROMPT"].replace("{no_evidence}", prompts["NO_EVIDENCE_MESSAGE"])
    return prompts


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    rank: int
    score: float
    chunk: Chunk

    @property
    def tag(self) -> str:
        return f"S{self.rank}"


@dataclass
class RAGAnswer:
    question: str
    answer: str
    supported: bool
    citations: List[Dict] = field(default_factory=list)
    evidence: List[RetrievedChunk] = field(default_factory=list)
    provider: str = ""
    latency_s: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# LLM back-ends
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    pass


def call_llm(system_prompt: str, user_prompt: str, cfg: Dict) -> str:
    provider = str(cfg["LLM_PROVIDER"]).lower()
    temperature = float(cfg["TEMPERATURE"])
    max_tokens = int(cfg["MAX_TOKENS"])
    timeout = float(cfg["LLM_TIMEOUT_SECONDS"])

    if provider in ("gemini", "groq", "openai"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Package 'openai' is not installed: pip install openai") from exc
        if provider == "gemini":
            # Gemini's OpenAI-compatible endpoint
            key, model = os.getenv("GEMINI_API_KEY"), cfg["GEMINI_MODEL"]
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif provider == "groq":
            key, model, base_url = os.getenv("GROQ_API_KEY"), cfg["GROQ_MODEL"], "https://api.groq.com/openai/v1"
        else:
            key, model, base_url = os.getenv("OPENAI_API_KEY"), cfg["OPENAI_MODEL"], None
        if not key:
            raise LLMError(f"Environment variable {provider.upper()}_API_KEY is not set")
        client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "ollama":
        import requests

        url = str(cfg["OLLAMA_URL"]).rstrip("/") + "/api/chat"
        try:
            r = requests.post(
                url,
                json={
                    "model": cfg["OLLAMA_MODEL"],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed ({exc}). Is `ollama serve` running?") from exc
        return r.json().get("message", {}).get("content", "").strip()

    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, cfg: Optional[Dict] = None, prompts: Optional[Dict[str, str]] = None):
        self.cfg = cfg or load_config()
        self.prompts = prompts or load_prompts()
        self.index = None
        self.chunks: List[Chunk] = []
        self.meta: Dict = {}
        setup_logging(self.cfg)

    # ---------- knowledge base ----------
    def build(self, pdf_paths: Sequence[str | Path], doc_names: Optional[Sequence[str]] = None, force: bool = False) -> Dict:
        t0 = time.time()
        self.index, self.chunks, self.meta = build_knowledge_base(pdf_paths, self.cfg, doc_names, force=force)
        logger.info("Knowledge base ready in %.1fs", time.time() - t0)
        return self.meta

    def load(self, store_dir: Optional[Path] = None) -> Dict:
        self.index, self.chunks, self.meta = load_vector_store(store_dir or resolve_path(self.cfg["VECTOR_STORE_DIR"]))
        return self.meta

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)

    # ---------- retrieval ----------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if not self.ready:
            raise RuntimeError("Knowledge base is empty - upload/ingest a PDF first.")
        k = min(int(top_k or self.cfg["TOP_K"]), len(self.chunks))
        qvec = embed_texts([query], self.cfg["EMBEDDING_MODEL"])
        scores, ids = self.index.search(qvec, k)
        results = [
            RetrievedChunk(rank=r + 1, score=float(s), chunk=self.chunks[i])
            for r, (s, i) in enumerate(zip(scores[0], ids[0]))
            if i != -1
        ]
        logger.info("Retrieved %d chunks for query '%s' (best=%.3f)", len(results), query[:80], results[0].score if results else -1)
        return results

    # ---------- prompting ----------
    @staticmethod
    def format_context(evidence: List[RetrievedChunk]) -> str:
        parts = []
        for e in evidence:
            c = e.chunk
            parts.append(f"[{e.tag}] (Document: {c.doc_name} | Page: {c.page} | Section: {c.section})\n{c.text}")
        return "\n\n".join(parts)

    def _extractive_answer(self, question: str, evidence: List[RetrievedChunk]) -> str:
        """No-LLM fallback: pick the evidence sentences most similar to the question."""
        sentences, tags = [], []
        for e in evidence:
            for s in re.split(r"(?<=[.!?])\s+", e.chunk.text):
                if len(s) > 25:
                    sentences.append(s.strip())
                    tags.append(e.tag)
        if not sentences:
            return self.prompts["NO_EVIDENCE_MESSAGE"]
        vecs = embed_texts([question] + sentences, self.cfg["EMBEDDING_MODEL"])
        sims = vecs[1:] @ vecs[0]
        best = np.argsort(-sims)[:3]
        lines = [f"- {sentences[i]} [{tags[i]}]" for i in sorted(best)]
        return "Extractive answer (no LLM configured) - most relevant statements from the document:\n" + "\n".join(lines)

    # ---------- main entry point ----------
    def answer(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        t0 = time.time()
        question = question.strip()
        no_ev = self.prompts["NO_EVIDENCE_MESSAGE"]
        provider = str(self.cfg["LLM_PROVIDER"]).lower()
        threshold = float(self.cfg["SIMILARITY_THRESHOLD"])

        evidence = self.retrieve(question, top_k)
        relevant = [e for e in evidence if e.score >= threshold]

        # Gate 1: retrieval confidence. Nothing relevant -> refuse without calling the LLM.
        if not relevant:
            logger.info("Evidence gate: best score below threshold %.2f -> no answer", threshold)
            return RAGAnswer(question, no_ev, False, [], evidence, provider, time.time() - t0,
                             note=f"Best similarity {evidence[0].score:.2f} < threshold {threshold:.2f}" if evidence else "")

        note = ""
        if provider == "none":
            text = self._extractive_answer(question, relevant)
        else:
            user_prompt = self.prompts["USER_PROMPT_TEMPLATE"].format(context=self.format_context(relevant), question=question)
            try:
                text = call_llm(self.prompts["SYSTEM_PROMPT"], user_prompt, self.cfg)
            except Exception as exc:  # keep the app usable if the API fails
                logger.error("LLM call failed: %s", exc)
                text = self._extractive_answer(question, relevant)
                note = f"LLM unavailable ({exc}); showing extractive fallback."

        # Gate 2: the LLM itself said the context is insufficient
        supported = not self._is_refusal(text, no_ev)
        if not supported:
            text = no_ev
        citations = self._build_citations(text, relevant) if supported else []
        result = RAGAnswer(question, text, supported, citations, evidence, provider, time.time() - t0, note)
        logger.info("Answered in %.2fs | supported=%s | citations=%d", result.latency_s, supported, len(citations))
        return result

    @staticmethod
    def _is_refusal(text: str, no_ev: str) -> bool:
        t = text.lower()
        return (not t.strip()) or no_ev.lower()[:45] in t or "sufficient evidence was not found" in t

    @staticmethod
    def _build_citations(answer: str, relevant: List[RetrievedChunk]) -> List[Dict]:
        """Citations = sources the answer actually referenced ([S#]); if the model
        cited nothing, fall back to all chunks that passed the evidence gate."""
        used = set(re.findall(r"\[S(\d+)\]", answer))
        chosen = [e for e in relevant if str(e.rank) in used] or relevant
        seen, cites = set(), []
        for e in chosen:
            key = (e.chunk.doc_name, e.chunk.page, e.chunk.section)
            if key in seen:
                continue
            seen.add(key)
            cites.append({"source": e.tag, "document": e.chunk.doc_name, "page": e.chunk.page,
                          "section": e.chunk.section, "score": round(e.score, 3)})
        return cites


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_answer(res: RAGAnswer) -> None:
    print("\n" + "=" * 80)
    print("Q:", res.question)
    print("-" * 80)
    print(res.answer)
    if res.note:
        print(f"\n(note: {res.note})")
    print("-" * 80)
    print("Supported:", res.supported, "| Provider:", res.provider, f"| {res.latency_s:.2f}s")
    for c in res.citations:
        print(f"  [{c['source']}] {c['document']} | Page {c['page']} | {c['section']} (score {c['score']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about an AUTOSAR HLD PDF")
    parser.add_argument("--pdf", nargs="*", default=[str(PROJECT_ROOT / "Input_Data" / "Synthetic_AUTOSAR_BCM_HLD.pdf")])
    parser.add_argument("-q", "--question", help="Question (omit for interactive mode)")
    parser.add_argument("--provider", help="Override LLM_PROVIDER (gemini/groq/openai/ollama/none)")
    parser.add_argument("--top_k", type=int)
    args = parser.parse_args()

    cfg = load_config()
    if args.provider:
        cfg["LLM_PROVIDER"] = args.provider
    rag = RAGPipeline(cfg)
    rag.build(args.pdf)

    if args.question:
        _print_answer(rag.answer(args.question, args.top_k))
        return
    print("Interactive mode - type a question (empty line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        _print_answer(rag.answer(q, args.top_k))


if __name__ == "__main__":
    main()