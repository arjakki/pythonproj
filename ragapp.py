"""
ragapp.py — RAG (Retrieval-Augmented Generation) pipeline using the Anthropic SDK.

Dependencies: anthropic, numpy
Install:  pip install anthropic numpy

Usage:
    python ragapp.py                         # run built-in demo
    from ragapp import Pipeline              # use as a library
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import anthropic


# ── Chunking ─────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


# ── TF-IDF vector store (numpy + stdlib only) ────────────────────────────────────

class TFIDFStore:
    """
    Lightweight in-memory vector store.

    Vectorizes text with TF-IDF and retrieves via cosine similarity.
    No external ML dependencies — pure numpy + stdlib.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf_weights: dict[str, float] = {}
        self.chunks: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = np.zeros(len(self.vocab))
        for token, count in tf.items():
            if token in self.vocab:
                vec[self.vocab[token]] = (count / total) * self.idf_weights.get(token, 0.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def build(self, chunks: list[str]) -> None:
        """Fit vocabulary and IDF weights, then index all chunks."""
        self.chunks = list(chunks)
        N = len(chunks)
        if N == 0:
            return
        df: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            for token in set(self._tokenize(chunk)):
                df[token] += 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.idf_weights = {
            t: math.log((N + 1) / (df[t] + 1)) + 1
            for t in self.vocab
        }
        self._matrix = np.stack([self._vectorize(c) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (chunk, score) pairs sorted by cosine similarity."""
        if not self.chunks:
            return []
        qvec = self._vectorize(query)
        if np.linalg.norm(qvec) == 0:
            return []
        scores: np.ndarray = self._matrix @ qvec
        top_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0.0]


# ── System prompt (stable — will be prompt-cached) ───────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the user's question using only the provided context passages. "
    "If the context does not contain sufficient information, say so honestly. "
    "Cite the relevant passage number(s) when quoting or paraphrasing. "
    "Do not fabricate facts that are not present in the context."
)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────────

@dataclass
class Pipeline:
    """
    RAG pipeline: ingest documents → chunk → index → retrieve → generate.

    Methods:
        ingest(sources)          — load documents from a list of strings or a folder path
        query(question)          — retrieve relevant chunks and generate an answer
        query_with_usage(question) — like query(), but also returns token usage info
    """

    model: str = "claude-sonnet-4-6"
    chunk_size: int = 400       # words per chunk
    overlap: int = 80           # word overlap between consecutive chunks
    top_k: int = 5              # passages to retrieve per query
    max_tokens: int = 1024      # max generation tokens

    _store: TFIDFStore = field(default_factory=TFIDFStore, repr=False)
    _client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        ),
        repr=False,
    )

    # ── Ingestion ────────────────────────────────────────────────────────────────

    def ingest(self, sources: list[str] | str) -> int:
        """
        Load and index documents.

        Args:
            sources: Either a list of raw text strings, or a path to a directory
                     whose .txt files will be loaded recursively.
        Returns:
            Number of chunks indexed.
        """
        texts: list[str] = []

        if isinstance(sources, str):
            p = Path(sources)
            if not p.is_dir():
                raise ValueError(f"'{sources}' is not a directory")
            for txt_file in sorted(p.glob("**/*.txt")):
                texts.append(txt_file.read_text(encoding="utf-8"))
        else:
            texts = list(sources)

        all_chunks: list[str] = []
        for text in texts:
            all_chunks.extend(chunk_text(text.strip(), self.chunk_size, self.overlap))

        self._store.build(all_chunks)
        return len(all_chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        """Return top-k (chunk, score) pairs for a question."""
        return self._store.search(question, top_k=self.top_k)

    # ── Generation ───────────────────────────────────────────────────────────────

    def _build_request(self, question: str, context_text: str) -> dict:
        """
        Build the Claude API request dict with prompt caching.

        Cache placement strategy (prefix = tools → system → messages):
          1. System prompt  — stable across all queries; cache_control here
             caches the system instruction block.
          2. Context block  — changes per query but can be reused when the same
             top-k passages are retrieved for similar questions. Setting
             cache_control here lets repeated queries over the same document set
             hit the cache. For small demos the 2048-token minimum (Sonnet 4.6)
             may not be met; in production with larger docs this pays off well.
          3. Question text  — volatile; no cache_control (always after last breakpoint).
        """
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    # Stable across all queries — cache aggressively.
                    # Minimum cacheable prefix on Sonnet 4.6 is 2048 tokens;
                    # silently skipped if content is shorter (no error).
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Context passages:\n\n{context_text}",
                            # Cache context so that repeated questions over the
                            # same retrieved passages avoid re-processing.
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            # Volatile — placed after the last cache breakpoint.
                            "text": f"Question: {question}",
                        },
                    ],
                }
            ],
        )

    def query(self, question: str) -> str:
        """Retrieve relevant passages and return Claude's answer."""
        results = self._store.search(question, top_k=self.top_k)
        if not results:
            return "No relevant context found in the indexed documents."

        context_text = "\n\n".join(
            f"[Passage {i + 1}]\n{chunk}"
            for i, (chunk, _) in enumerate(results)
        )
        response = self._client.messages.create(**self._build_request(question, context_text))
        return next((b.text for b in response.content if b.type == "text"), "")

    def query_with_usage(self, question: str) -> tuple[str, dict]:
        """
        Like query(), but also returns a usage dict for observability.

        Usage dict keys:
            input_tokens, output_tokens,
            cache_creation_input_tokens, cache_read_input_tokens
        """
        results = self._store.search(question, top_k=self.top_k)
        if not results:
            return "No relevant context found in the indexed documents.", {}

        context_text = "\n\n".join(
            f"[Passage {i + 1}]\n{chunk}"
            for i, (chunk, _) in enumerate(results)
        )
        response = self._client.messages.create(**self._build_request(question, context_text))
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", 0
            ),
        }
        answer = next((b.text for b in response.content if b.type == "text"), "")
        return answer, usage


# ── Sample documents for the demo ────────────────────────────────────────────────

_SAMPLE_DOCS = [
    """
    Python is a high-level, general-purpose programming language created by Guido van Rossum
    and first released in 1991. It is known for its clear syntax, readability, and significant
    indentation style. Python supports multiple programming paradigms including structured,
    object-oriented, and functional programming. It is widely used in data science, machine
    learning, web development, scripting, and automation. The Python Package Index (PyPI)
    hosts hundreds of thousands of third-party packages, making it one of the largest
    software ecosystems in existence. Python's design philosophy emphasizes code readability
    and developer productivity over raw execution speed.
    """,
    """
    Machine learning is a branch of artificial intelligence that enables systems to learn and
    improve from experience without being explicitly programmed. It focuses on developing
    algorithms and statistical models that allow computers to perform tasks without explicit
    instructions, relying instead on patterns and inference. Machine learning categories
    include supervised learning (training on labeled data), unsupervised learning (finding
    patterns in unlabeled data), semi-supervised learning, and reinforcement learning (learning
    through rewards and penalties). Common algorithms include linear regression, decision trees,
    random forests, support vector machines, and neural networks. Deep learning, a subfield
    using multi-layer neural networks, has driven many recent breakthroughs in image recognition,
    natural language processing, and game playing.
    """,
    """
    Retrieval-Augmented Generation (RAG) is a natural language processing technique that
    combines information retrieval with text generation. Rather than relying solely on knowledge
    baked into a language model's parameters at training time, RAG retrieves relevant documents
    from an external knowledge base at inference time and uses them as grounding context for
    the generator. This approach allows models to access up-to-date or domain-specific
    information, significantly reduces hallucinations, and makes the system's knowledge sources
    transparent and updatable without retraining. A RAG system has two key components: a
    retriever (often a dense vector search using embeddings) and a reader or generator (a large
    language model). TF-IDF is a simpler, sparse retrieval alternative that works well for
    keyword-rich queries.
    """,
    """
    NumPy is the fundamental package for scientific and numerical computing in Python. It
    provides support for large, multi-dimensional arrays and matrices, along with a comprehensive
    library of mathematical functions. NumPy's core object is the ndarray, an n-dimensional
    array that supports vectorized operations far faster than Python lists. Key features include
    broadcasting (operating on arrays of different shapes), linear algebra routines, Fourier
    transforms, random number generation, and integration with C and Fortran libraries.
    NumPy is the foundation upon which SciPy, Matplotlib, pandas, scikit-learn, and many
    other scientific Python libraries are built.
    """,
    """
    Prompt caching is a technique used with large language model APIs to reduce latency and
    cost for requests that share a common prefix. When an API request contains a long system
    prompt, a large document, or a set of few-shot examples that do not change across calls,
    the API can cache the computed key-value (KV) attention states for that stable prefix.
    Subsequent requests reusing the same cached prefix are served faster and at a fraction
    of the normal input token cost — often around 10 percent of the uncached price. Cache
    control markers in the request tell the API exactly where to place cache breakpoints.
    Effective caching requires keeping stable content at the start of the prompt and volatile
    content (the user's specific question) after the last breakpoint.
    """,
    """
    TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic
    used in information retrieval and text mining to reflect how important a word is to a
    document within a collection. Term Frequency (TF) measures how often a term appears in
    a document relative to the document length. Inverse Document Frequency (IDF) measures
    how rare or common a term is across all documents in the corpus; terms that appear in
    many documents get a lower IDF weight, reducing the influence of common stop words.
    The TF-IDF score is the product of these two values. It is used to build sparse
    vector representations of documents that enable cosine-similarity-based retrieval
    without requiring dense neural embeddings.
    """,
]


# ── Demo entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  RAG Pipeline Demo  (Anthropic SDK + NumPy TF-IDF)")
    print("=" * 60)
    print()

    pipeline = Pipeline(
        model="claude-sonnet-4-6",
        chunk_size=300,
        overlap=60,
        top_k=3,
    )

    n = pipeline.ingest(_SAMPLE_DOCS)
    print(f"Ingested {len(_SAMPLE_DOCS)} documents  ->  {n} chunks indexed\n")

    questions = [
        "What is RAG and why does it reduce hallucinations?",
        "How does prompt caching lower API costs?",
        "What is TF-IDF and how is it used for retrieval?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer, usage = pipeline.query_with_usage(q)
        print(f"A: {answer.strip()}")
        cache_hit = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        print(
            f"   tokens — in: {usage.get('input_tokens', 0)}, "
            f"out: {usage.get('output_tokens', 0)}, "
            f"cache_write: {cache_write}, "
            f"cache_read: {cache_hit}"
        )
        print()


if __name__ == "__main__":
    main()
