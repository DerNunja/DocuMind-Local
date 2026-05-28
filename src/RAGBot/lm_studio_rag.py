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
MAX_TOKENS = 10000

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
        data = self._chat(prompt, max_tokens=MAX_TOKENS)
        answer = extract_chat_response_text(data)
        return {
            "result": answer,
            "raw_response": data,
            "source_documents": source_documents,
        }

    def _chat(self, prompt: str, max_tokens: int, temperature: float = 0.1) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Du beantwortest Fragen zu lokalen Projektdokumenten."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()


def extract_chat_response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                text_parts.append(str(part))
        text = "".join(text_parts).strip()
        if text:
            return text

    for key in ("text", "output_text"):
        value = choice.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Some reasoning models expose the final text inconsistently via extra fields.
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


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
