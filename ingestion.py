

"""
ingestion.py
------------
Offline half of the RAG pipeline:

    PDF -> page-wise text extraction -> cleaning -> section detection
        -> chunking (configurable size/overlap) -> embeddings -> FAISS index -> disk

Also hosts the shared helpers used by every other module:
    load_config(), setup_logging(), PROJECT_ROOT

Run standalone:
    python ingestion.py --pdf ../Input_Data/Synthetic_AUTOSAR_BCM_HLD.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "Model_Prompts_Config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.txt"
DEFAULT_PROMPTS_PATH = CONFIG_DIR / "prompts.txt"

logger = logging.getLogger("hld_rag.ingestion")


def _auto_cast(value: str):
    """Convert config strings to int / float / bool where possible."""
    v = value.strip()
    if v.lower() in {"true", "yes"}:
        return True
    if v.lower() in {"false", "no"}:
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Dict:
    """Read KEY=VALUE config file. Environment variables override file values."""
    cfg: Dict = {}
    path = Path(path)
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            cfg[key.strip()] = _auto_cast(val.split(" #")[0])
    else:
        logger.warning("Config file %s not found - using defaults", path)

    defaults = {
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EMBEDDING_BATCH_SIZE": 32,
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 150,
        "MIN_CHUNK_CHARS": 40,
        "TOP_K": 4,
        "SIMILARITY_THRESHOLD": 0.30,
        "LLM_PROVIDER": "gemini",
        "GEMINI_MODEL": "gemini-2.5-flash",
        "GROQ_MODEL": "llama-3.1-8b-instant",
        "OPENAI_MODEL": "gpt-4o-mini",
        "OLLAMA_MODEL": "llama3.2:3b",
        "OLLAMA_URL": "http://localhost:11434",
        "TEMPERATURE": 0.0,
        "MAX_TOKENS": 600,
        "LLM_TIMEOUT_SECONDS": 60,
        "VECTOR_STORE_DIR": "Code/vector_store",
        "LOG_DIR": "Code/logs",
        "LOG_LEVEL": "INFO",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    # environment overrides (never used for secrets in the file itself)
    for k in list(cfg.keys()):
        if k in os.environ:
            cfg[k] = _auto_cast(os.environ[k])
    return cfg


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


_LOGGING_DONE = False


def setup_logging(cfg: Optional[Dict] = None, log_name: str = "rag_app.log") -> None:
    """Console + rotating-file logging. Safe to call multiple times."""
    global _LOGGING_DONE
    if _LOGGING_DONE:
        return
    from logging.handlers import RotatingFileHandler

    cfg = cfg or load_config()
    log_dir = resolve_path(cfg.get("LOG_DIR", "Code/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(cfg.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    root = logging.getLogger("hld_rag")
    root.setLevel(level)
    fh = RotatingFileHandler(log_dir / log_name, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    root.propagate = False
    _LOGGING_DONE = True


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PageLine:
    text: str
    is_heading: bool = False


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page: int          # 1-based page number
    section: str       # e.g. "6.2 Diagnostic Event Manager (DEM)" or "N/A"
    text: str

    def citation(self) -> str:
        return f"{self.doc_name} | Page {self.page} | Section: {self.section}"


# ---------------------------------------------------------------------------
# 1. Extraction (PyMuPDF) - keeps page numbers and font info for headings
# ---------------------------------------------------------------------------
# Numbered heading, e.g. "3", "3.2", "10.1.4" followed by a capitalised title
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][A-Za-z0-9 ,/()&\-–:'+]{2,90})$")


def extract_pages(pdf_path: str | Path) -> List[List[PageLine]]:
    """Return a list (one item per page) of lines, each flagged as heading or not.

    A line is a heading when it matches the numbered-heading pattern AND is
    rendered bigger/bolder than the page's body text. This avoids table cells
    such as "10 ms" being treated as section titles.
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        import fitz  # older PyMuPDF

    pdf_path = Path(pdf_path)
    pages: List[List[PageLine]] = []
    with fitz.open(pdf_path) as doc:
        logger.info("Opened '%s' (%d pages)", pdf_path.name, doc.page_count)
        for page in doc:
            data = page.get_text("dict")
            raw_lines = []
            sizes = []
            for block in data.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    size = max(s.get("size", 0) for s in spans)
                    bold = any((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                    raw_lines.append((text, size, bold))
                    sizes.extend(s.get("size", 0) for s in spans)

            body_size = statistics.median(sizes) if sizes else 10.0
            lines: List[PageLine] = []
            for text, size, bold in raw_lines:
                looks_numbered = bool(HEADING_RE.match(text))
                styled = size >= body_size + 0.8 or bold
                lines.append(PageLine(text=text, is_heading=looks_numbered and styled))
            pages.append(lines)
    return pages


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def _norm_for_repeat(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages: List[List[PageLine]]) -> List[List[PageLine]]:
    """Remove repeated headers/footers and page counters, fix whitespace & hyphenation."""
    n_pages = len(pages)
    counts: Counter = Counter()
    for lines in pages:
        for key in {_norm_for_repeat(l.text) for l in lines}:
            counts[key] += 1
    # a line (digits ignored) appearing on >= 60% of pages (min 3) is header/footer boilerplate
    repeated = {k for k, c in counts.items() if n_pages >= 3 and c >= max(3, int(0.6 * n_pages))}
    page_counter = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)

    cleaned: List[List[PageLine]] = []
    removed = 0
    for lines in pages:
        out: List[PageLine] = []
        for l in lines:
            t = l.text.replace(" ", " ").replace("­", "")
            t = re.sub(r"[ \t]+", " ", t).strip()
            if not t or page_counter.match(t) or (_norm_for_repeat(t) in repeated and not l.is_heading):
                removed += 1
                continue
            # join words broken with a hyphen at line end: "configu-" + "ration"
            if out and out[-1].text.endswith("-") and not out[-1].is_heading and t[:1].islower():
                out[-1].text = out[-1].text[:-1] + t
                continue
            out.append(PageLine(text=t, is_heading=l.is_heading))
        cleaned.append(out)
    logger.info("Cleaning removed %d boilerplate/empty lines", removed)
    return cleaned


# ---------------------------------------------------------------------------
# 3. Section-aware chunking
# ---------------------------------------------------------------------------
def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-based sliding window that never cuts inside a word and
    prefers to break at sentence ends."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            # prefer sentence boundary in the last 30% of the window, else a space
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut < int(chunk_size * 0.7):
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        # move forward to a word boundary so overlap doesn't start mid-word
        while 0 < next_start < n and text[next_start - 1] not in " \n":
            next_start += 1
        start = max(next_start, start + 1)
    return chunks


def chunk_document(
    pages: List[List[PageLine]],
    doc_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 40,
) -> List[Chunk]:
    """Build chunks that never span two pages (so the page citation is exact)
    and never mix two sections. The section heading is prefixed to each chunk
    to give the embedding model extra context."""
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks: List[Chunk] = []
    current_section = "N/A"
    for page_no, lines in enumerate(pages, start=1):
        segments: List[tuple] = []  # (section, [lines])
        buf: List[str] = []
        for l in lines:
            if l.is_heading:
                if buf:
                    segments.append((current_section, buf))
                    buf = []
                current_section = l.text
            else:
                buf.append(l.text)
        if buf:
            segments.append((current_section, buf))

        for section, seg_lines in segments:
            seg_text = " ".join(seg_lines)
            seg_text = re.sub(r"\s+", " ", seg_text).strip()
            for piece in _split_with_overlap(seg_text, chunk_size, chunk_overlap):
                if len(piece) < min_chunk_chars:
                    continue
                cid = f"{Path(doc_name).stem}_p{page_no}_c{len(chunks)}"
                chunks.append(Chunk(cid, doc_name, page_no, section, piece))
    logger.info("Created %d chunks from '%s' (size=%d, overlap=%d)", len(chunks), doc_name, chunk_size, chunk_overlap)
    return chunks


# ---------------------------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------------------------
_EMBEDDER_CACHE: Dict[str, object] = {}


def get_embedder(model_name: str):
    """Load (and cache) a SentenceTransformer model. Downloads once (~90 MB)."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' ...", model_name)
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _EMBEDDER_CACHE[model_name]


def embed_texts(texts: Sequence[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    model = get_embedder(model_name)
    vecs = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors -> inner product == cosine similarity
    )
    return vecs.astype("float32")


def chunk_embedding_text(c: Chunk) -> str:
    """Text actually embedded: section title + chunk body."""
    return f"{c.section}\n{c.text}" if c.section != "N/A" else c.text


# ---------------------------------------------------------------------------
# 5. FAISS vector store
# ---------------------------------------------------------------------------
def build_faiss_index(vectors: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search, fine for small corpora
    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, vectors.shape[1])
    return index


def save_vector_store(index, chunks: List[Chunk], meta: Dict, out_dir: Path) -> None:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Vector store saved to %s", out_dir)


def load_vector_store(store_dir: Path):
    import faiss

    store_dir = Path(store_dir)
    index = faiss.read_index(str(store_dir / "index.faiss"))
    chunks = [Chunk(**d) for d in json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))]
    meta = json.loads((store_dir / "meta.json").read_text(encoding="utf-8"))
    logger.info("Loaded vector store from %s (%d chunks)", store_dir, len(chunks))
    return index, chunks, meta


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------
def build_knowledge_base(
    pdf_paths: Sequence[str | Path],
    cfg: Dict,
    doc_names: Optional[Sequence[str]] = None,
    store_dir: Optional[Path] = None,
    force: bool = False,
):
    """Full ingestion for one or more PDFs. Re-uses the saved index when the same
    files were already indexed with the same settings (unless force=True).

    Returns (faiss_index, chunks, meta)."""
    pdf_paths = [Path(p) for p in pdf_paths]
    doc_names = list(doc_names) if doc_names else [p.name for p in pdf_paths]
    store_dir = Path(store_dir) if store_dir else resolve_path(cfg["VECTOR_STORE_DIR"])

    signature = {
        "files": sorted(f"{n}:{file_sha256(p)}" for n, p in zip(doc_names, pdf_paths)),
        "embedding_model": cfg["EMBEDDING_MODEL"],
        "chunk_size": int(cfg["CHUNK_SIZE"]),
        "chunk_overlap": int(cfg["CHUNK_OVERLAP"]),
    }
    if not force and (store_dir / "meta.json").exists():
        try:
            index, chunks, meta = load_vector_store(store_dir)
            if meta.get("signature") == signature:
                logger.info("Index already up to date - skipping re-embedding")
                return index, chunks, meta
        except Exception as exc:  # corrupt store -> rebuild
            logger.warning("Could not reuse vector store (%s); rebuilding", exc)

    all_chunks: List[Chunk] = []
    page_counts = {}
    for path, name in zip(pdf_paths, doc_names):
        pages = clean_pages(extract_pages(path))
        page_counts[name] = len(pages)
        if not any(pages):
            logger.warning("No extractable text in %s (scanned PDF?)", name)
        all_chunks.extend(
            chunk_document(pages, name, int(cfg["CHUNK_SIZE"]), int(cfg["CHUNK_OVERLAP"]), int(cfg["MIN_CHUNK_CHARS"]))
        )
    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded PDF(s).")

    vectors = embed_texts([chunk_embedding_text(c) for c in all_chunks], cfg["EMBEDDING_MODEL"], int(cfg["EMBEDDING_BATCH_SIZE"]))
    index = build_faiss_index(vectors)
    meta = {"signature": signature, "page_counts": page_counts, "num_chunks": len(all_chunks), "dim": int(vectors.shape[1])}
    save_vector_store(index, all_chunks, meta, store_dir)
    return index, all_chunks, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into a FAISS vector store")
    parser.add_argument("--pdf", nargs="+", required=True, help="Path(s) to PDF file(s)")
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--chunk_overlap", type=int)
    parser.add_argument("--force", action="store_true", help="Rebuild even if index is up to date")
    parser.add_argument("--show", type=int, default=3, help="Print the first N chunks")
    args = parser.parse_args()

    cfg = load_config()
    if args.chunk_size:
        cfg["CHUNK_SIZE"] = args.chunk_size
    if args.chunk_overlap is not None:
        cfg["CHUNK_OVERLAP"] = args.chunk_overlap
    setup_logging(cfg, "ingestion.log")

    _, chunks, meta = build_knowledge_base(args.pdf, cfg, force=args.force)
    print(f"\nIndexed {meta['num_chunks']} chunks. Pages: {meta['page_counts']}")
    sections = sorted({c.section for c in chunks}, key=lambda s: [int(x) if x.isdigit() else 0 for x in s.split()[0].split(".")] if s[0].isdigit() else [999])
    print(f"Detected {len(sections)} sections, e.g.: {sections[:8]}")
    for c in chunks[: args.show]:
        print("-" * 70)
        print(c.citation())
        print(c.text[:300], "...")


if __name__ == "__main__":
    main()