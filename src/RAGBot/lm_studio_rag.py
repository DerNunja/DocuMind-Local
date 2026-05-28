from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


CHROMA_PATH = Path(__file__).resolve().parent / "vectordb"
COLLECTION = "projektdokumente"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RETRIEVAL_K = 4
RETRIEVAL_FETCH_K = 12
MAX_TOKENS = 280

RAG_PROMPT = """Du bist ein lokaler Projektmanagement-Assistent.
Beantworte die Frage nur auf Basis der folgenden Dokument-Auszüge.
Wenn keine relevante Information vorhanden ist, sage das klar.
Antworte auf Deutsch, präzise und strukturiert.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:"""


@lru_cache(maxsize=1)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_chroma():
    from langchain_chroma import Chroma

    if not CHROMA_PATH.exists():
        return None
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=load_embeddings(),
        collection_name=COLLECTION,
    )
    return db if db._collection.count() > 0 else None


def chroma_count() -> int:
    if not CHROMA_PATH.exists():
        return 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        return client.get_collection(COLLECTION).count()
    except Exception:
        return 0


def ingest_texts(documents: list[dict[str, Any]], reset: bool = False) -> int:
    import shutil

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if reset and CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    langchain_docs = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if len(text) < 50:
            continue
        metadata = {
            "dateiname": doc.get("dateiname", "unbekannt"),
            "dokument_key": doc.get("dokument_key", ""),
            "quelle": doc.get("quelle", "documind-ui"),
        }
        if doc.get("category"):
            metadata["kategorie"] = doc["category"]
        if doc.get("language"):
            metadata["sprache"] = doc["language"]
        langchain_docs.append(Document(page_content=text, metadata=metadata))

    if not langchain_docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(langchain_docs)
    for chunk in chunks:
        chunk.page_content = "passage: " + chunk.page_content

    Chroma.from_documents(
        documents=chunks,
        embedding=load_embeddings(),
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION,
    )
    return len(chunks)


class LMStudioRagChain:
    def __init__(self, retriever: Any, model: str, base_url: str) -> None:
        self.retriever = retriever
        self.model = model
        self.base_url = base_url.rstrip("/")

    def invoke(self, payload: dict[str, str]) -> dict[str, Any]:
        question = payload.get("query", "").removeprefix("query: ").strip()
        source_documents = self.retriever.invoke("query: " + question)
        context = "\n\n".join(
            f"Quelle: {doc.metadata.get('dateiname', 'unbekannt')}\n"
            f"{doc.page_content.removeprefix('passage: ')}"
            for doc in source_documents
        )
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Du beantwortest Fragen zu lokalen Projektdokumenten."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": MAX_TOKENS,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "result": data["choices"][0]["message"]["content"],
            "source_documents": source_documents,
        }


def build_chain(model_name: str, base_url: str) -> LMStudioRagChain:
    db = load_chroma()
    if db is None:
        raise RuntimeError("Die RAG-Datenbank ist leer.")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_FETCH_K,
            "lambda_mult": 0.7,
        },
    )
    return LMStudioRagChain(retriever=retriever, model=model_name, base_url=base_url)
