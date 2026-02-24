"""Retriever interface and implementations for Agent-UniRAG search node."""

import tempfile
from pathlib import Path
from typing import Protocol

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = [
    "RetrieverProtocol",
    "MockRetriever",
    "create_mock_retriever_for_demo",
    "ChromaPDFRetriever",
    "load_pdf_and_chunk",
]


class RetrieverProtocol(Protocol):
    """Interface for retrieval: (query, top_k) -> list of source dicts.

    Each source dict should have at least "content" (str) and optionally
    "source_id" for the Evidence Reflector. May include "article_title",
    "section_title" etc. for compatibility with paper's source format.
    """

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        """Retrieve top_k sources for the given query."""
        ...


class MockRetriever:
    """Mock retriever for testing: returns configurable snippets per query."""

    def __init__(self, responses: dict[str, list[dict]] | None = None):
        """Initialize with optional query -> list of sources mapping.

        If responses is None, a default response is used for any query.
        Keys can be lowercased substrings of the query for matching.
        """
        self._responses = responses or {}

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        query_lower = query.lower()
        for key, sources in self._responses.items():
            if key.lower() in query_lower:
                return sources[:top_k]
        default = [
            {
                "source_id": "0",
                "content": "No predefined response for this query. This is a placeholder.",
            }
        ]
        return default[:top_k]


def create_mock_retriever_for_demo() -> MockRetriever:
    """Create a mock retriever with sample responses for demo (e.g. Tim Russert)."""
    return MockRetriever(
        responses={
            "highway renamed in honor of Tim Russert": [
                {
                    "source_id": "0",
                    "content": "On July 23, 2008, U.S. Route 20A leading to the Buffalo Bills' Ralph Wilson Stadium in Orchard Park, New York was renamed the 'Timothy J. Russert Highway' in honor of Tim Russert.",
                }
            ],
        }
    )


def load_pdf_and_chunk(
    pdf_bytes: bytes,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Load a PDF with LangChain PyPDFLoader and split with RecursiveCharacterTextSplitter."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        if not docs:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(docs)
    finally:
        Path(path).unlink(missing_ok=True)


class ChromaPDFRetriever:
    """PDF retriever using LangChain (load + chunk) and ChromaDB (vector store)."""

    COLLECTION_NAME = "agent_unirag"

    def __init__(self, embedding: Embeddings) -> None:
        self._embedding = embedding
        self._client = chromadb.EphemeralClient()
        self._store: Chroma | None = None
        self._chunk_count = 0

    def chunk_count(self) -> int:
        return self._chunk_count

    def replace_documents(
        self,
        pdf_bytes: bytes,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> None:
        """Load PDF with LangChain, chunk, embed and replace Chroma collection."""
        docs = load_pdf_and_chunk(
            pdf_bytes, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not docs:
            self._store = None
            self._chunk_count = 0
            return
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=self._embedding,
            collection_name=self.COLLECTION_NAME,
            client=self._client,
        )
        self._chunk_count = len(docs)

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        if self._store is None or self._chunk_count == 0:
            return []
        docs = self._store.similarity_search(query, k=top_k)
        return [
            {
                "content": d.page_content,
                "source_id": d.metadata.get("source_id", str(i)),
            }
            for i, d in enumerate(docs)
        ]
