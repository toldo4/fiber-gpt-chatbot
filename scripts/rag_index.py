"""
Hybrid retrieval over the pre-built index-parts/ corpus.

Dense cosine similarity alone is unreliable on this corpus because the queries
and the documents are full of exact tokens — "AOAC 991.43", "2017.16", "FOS" —
that embeddings blur together. Retrieval here fuses a dense ranking with a BM25
ranking using reciprocal rank fusion, and the tokenizer keeps method codes
intact as single terms.

    from rag_index import HybridIndex, OpenAIEmbedder

    index = HybridIndex.load(Path("index-parts"))
    hits = index.search("resistant starch in legumes", OpenAIEmbedder(api_key), top_k=6)
    for h in hits:
        print(h.file, h.section, h.score, h.text)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# Chunks shorter than this are page furniture — running heads, figure captions,
# "Contents lists available at ScienceDirect" — and only ever displace real
# content from the top-k.
MIN_CHUNK_CHARS = 120

_BOILERPLATE = re.compile(
    r"contents lists available|journal homepage|sciencedirect\.com|"
    r"^\s*(supplementary data|acknowledge?ments?|references)\b|"
    r"all rights reserved|doi:\s*\S+\s*$",
    re.IGNORECASE,
)

# Keeps "991.43" and "2017.16" as single tokens instead of splitting on the dot.
_TOKEN = re.compile(r"[a-z]+[0-9]*|[0-9]+(?:\.[0-9]+)+|[0-9]+")

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    section: str | None
    text: str


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk, carrying enough provenance to cite it."""
    chunk: Chunk
    score: float
    dense_rank: int | None
    lexical_rank: int | None

    @property
    def file(self) -> str:
        return self.chunk.file

    @property
    def section(self) -> str:
        return self.chunk.section or "General"

    @property
    def text(self) -> str:
        return self.chunk.text

    def cite(self) -> str:
        return f"{self.file} — {self.section}"


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


# ── Embedding with an on-disk cache ───────────────────────────────────────────

class OpenAIEmbedder:
    """Embeds queries with the same model used to build the index.

    Node retrieval queries are fixed by the decision tree, so caching them on
    disk makes repeat runs of the same tree essentially free.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        cache_path: Path | None = None,
    ) -> None:
        import openai

        self.model = model
        self._client = openai.OpenAI(api_key=api_key)
        self._cache_path = cache_path or (Path(__file__).parent / ".embed_cache.json")
        self._cache: dict[str, list[float]] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def embed(self, text: str) -> list[float]:
        key = hashlib.sha1(f"{self.model}\x00{text}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        resp = self._client.embeddings.create(model=self.model, input=text)
        vec = list(resp.data[0].embedding)
        self._cache[key] = vec
        try:
            self._cache_path.write_text(json.dumps(self._cache))
        except OSError:
            pass
        return vec


# ── Index ─────────────────────────────────────────────────────────────────────

class HybridIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunk count does not match embedding matrix rows")
        self.chunks = chunks
        self.matrix = matrix          # L2-normalised, float32
        self._build_lexical()

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        parts_dir: Path,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        cache_path: Path | None = None,
    ) -> HybridIndex:
        part_files = sorted(parts_dir.glob("index-part-*.json"))
        if not part_files:
            raise FileNotFoundError(f"no index-part-*.json found in {parts_dir}")

        cache_path = cache_path or (Path(__file__).parent / ".rag_cache.npz")
        fingerprint = cls._fingerprint(part_files, min_chunk_chars)

        cached = cls._read_cache(cache_path, fingerprint)
        if cached is not None:
            return cls(*cached)

        chunks: list[Chunk] = []
        vectors: list[Sequence[float]] = []
        for part_file in part_files:
            part = json.loads(part_file.read_text())
            for raw in part["chunks"]:
                text = (raw.get("text") or "").strip()
                if len(text) < min_chunk_chars or _BOILERPLATE.search(text):
                    continue
                chunks.append(Chunk(
                    id=raw["id"],
                    file=raw["file"],
                    section=raw.get("section"),
                    text=text,
                ))
                vectors.append(raw["embedding"])

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-12)

        cls._write_cache(cache_path, fingerprint, chunks, matrix)
        return cls(chunks, matrix)

    @staticmethod
    def _fingerprint(part_files: list[Path], min_chunk_chars: int) -> str:
        h = hashlib.sha1(f"v2:{min_chunk_chars}".encode())
        for p in part_files:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()

    @staticmethod
    def _read_cache(path: Path, fingerprint: str) -> tuple[list[Chunk], np.ndarray] | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != fingerprint:
                    return None
                meta = json.loads(str(data["meta"]))
                matrix = data["matrix"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        chunks = [Chunk(id=m[0], file=m[1], section=m[2], text=m[3]) for m in meta]
        return chunks, matrix

    @staticmethod
    def _write_cache(
        path: Path, fingerprint: str, chunks: list[Chunk], matrix: np.ndarray
    ) -> None:
        meta = [[c.id, c.file, c.section, c.text] for c in chunks]
        try:
            np.savez_compressed(
                path,
                fingerprint=np.array(fingerprint),
                meta=np.array(json.dumps(meta)),
                matrix=matrix,
            )
        except OSError:
            pass

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _build_lexical(self) -> None:
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        lengths = np.zeros(len(self.chunks), dtype=np.float32)

        for doc_id, chunk in enumerate(self.chunks):
            terms = tokenize(chunk.text)
            lengths[doc_id] = len(terms)
            counts: dict[str, int] = defaultdict(int)
            for t in terms:
                counts[t] += 1
            for term, tf in counts.items():
                postings[term][doc_id] = tf

        n_docs = max(len(self.chunks), 1)
        self._avg_len = float(lengths.mean()) if len(lengths) else 0.0
        self._doc_lens = lengths
        self._postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, docs in postings.items():
            doc_ids = np.fromiter(docs.keys(), dtype=np.int32, count=len(docs))
            tfs = np.fromiter(docs.values(), dtype=np.float32, count=len(docs))
            df = len(docs)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            self._postings[term] = (doc_ids, tfs, idf)

    def _bm25_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._avg_len <= 0:
            return scores
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self._doc_lens / self._avg_len)
        for term in tokenize(query):
            entry = self._postings.get(term)
            if entry is None:
                continue
            doc_ids, tfs, idf = entry
            scores[doc_ids] += idf * (tfs * (BM25_K1 + 1)) / (tfs + norm[doc_ids])
        return scores

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 6,
        candidate_k: int = 40,
    ) -> list[Hit]:
        """Fuse a dense and a lexical ranking with reciprocal rank fusion.

        RRF is used rather than a weighted score sum because cosine similarity
        and BM25 are on incomparable scales and BM25's range shifts with the
        query length.
        """
        if not self.chunks:
            return []

        q = np.asarray(embedder.embed(query), dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        dense = self.matrix @ q

        lexical = self._bm25_scores(query)

        k = min(candidate_k, len(self.chunks))
        dense_order = np.argsort(-dense)[:k]
        lexical_order = np.argsort(-lexical)[:k]

        dense_rank = {int(d): i for i, d in enumerate(dense_order)}
        lexical_rank = {
            int(d): i for i, d in enumerate(lexical_order) if lexical[d] > 0
        }

        fused: dict[int, float] = defaultdict(float)
        for doc_id, rank in dense_rank.items():
            fused[doc_id] += 1.0 / (RRF_K + rank)
        for doc_id, rank in lexical_rank.items():
            fused[doc_id] += 1.0 / (RRF_K + rank)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Hit(
                chunk=self.chunks[doc_id],
                score=score,
                dense_rank=dense_rank.get(doc_id),
                lexical_rank=lexical_rank.get(doc_id),
            )
            for doc_id, score in ordered
        ]


class BoundRetriever:
    """An index plus an embedder, matching the engine's `Retriever` protocol."""

    def __init__(self, index: HybridIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        return self.index.search(query, self.embedder, top_k=top_k)


def build_context(hits: Sequence[Hit], max_chars: int = 9000) -> str:
    """Render hits as a numbered context block.

    Chunks are included whole. Truncating each chunk to a fixed width cuts
    mid-sentence and routinely discards the sentence that carries the answer;
    dropping a whole low-ranked chunk to stay inside the budget loses less.
    """
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, 1):
        block = f"[Source {i}] {hit.cite()}\n{hit.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
