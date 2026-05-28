from __future__ import annotations

import os
import subprocess
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
RAGBOT_DIR = ROOT / "src" / "RAGBot"
TRANSLATOR_DIR = ROOT / "src" / "translator"
CHROMA_PATH = RAGBOT_DIR / "vectordb"
COLLECTION = "projektdokumente"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
POSTGRES_CONTAINER = "documind-postgres"
POSTGRES_IMAGE = "pgvector/pgvector:pg16"

RAG_MODELS = {
    "Phi-3.5 Mini (schneller)": {
        "ollama_name": "phi3.5",
        "k": 2,
        "fetch_k": 6,
        "num_predict": 180,
        "num_ctx": 1024,
    },
    "Llama 3.1 8B (bessere Qualität)": {
        "ollama_name": "llama3.1:8b",
        "k": 4,
        "fetch_k": 12,
        "num_predict": 280,
        "num_ctx": 2048,
    },
}

RAG_PROMPT = """Du bist ein lokaler Projektmanagement-Assistent.
Beantworte die Frage nur auf Basis der folgenden Dokument-Auszüge.
Wenn keine relevante Information vorhanden ist, sage das klar.
Antworte auf Deutsch, präzise und strukturiert.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:"""


st.set_page_config(
    page_title="DocuMind Local",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def save_uploaded_file(uploaded_file: Any, target_dir: Path) -> Path:
    target = target_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def status_label(ok: bool, label: str) -> None:
    if ok:
        st.success(f"Verfügbar: {label}")
    else:
        st.warning(f"Nicht verbunden: {label}")


@st.cache_resource(show_spinner=False)
def ensure_postgres_docker() -> dict[str, str | bool]:
    if shutil.which("docker") is None:
        return {"ok": False, "message": "Docker CLI wurde nicht gefunden."}

    try:
        inspect = subprocess.run(
            ["docker", "inspect", POSTGRES_CONTAINER],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "message": f"Docker konnte nicht geprüft werden: {exc}"}

    try:
        if inspect.returncode == 0:
            start = subprocess.run(
                ["docker", "start", POSTGRES_CONTAINER],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if start.returncode == 0:
                return {"ok": True, "message": f"Container {POSTGRES_CONTAINER} läuft."}
            message = (start.stderr or start.stdout).strip()
            return {"ok": False, "message": f"Container konnte nicht gestartet werden: {message}"}

        run = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                POSTGRES_CONTAINER,
                "-e",
                "POSTGRES_PASSWORD=postgres",
                "-e",
                "POSTGRES_DB=documind",
                "-p",
                "5432:5432",
                "-d",
                POSTGRES_IMAGE,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if run.returncode == 0:
            return {"ok": True, "message": f"Container {POSTGRES_CONTAINER} wurde angelegt und gestartet."}
        message = (run.stderr or run.stdout).strip()
        return {"ok": False, "message": f"Container konnte nicht angelegt werden: {message}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Docker-Aufruf hat zu lange gedauert."}


@st.cache_resource(show_spinner="Lade Embedding-Modell...")
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


def extract_document(path: Path) -> dict[str, Any]:
    ensure_import_path(RAGBOT_DIR)
    from modul_b_stub import extrahiere_dokument

    result = extrahiere_dokument(str(path))
    if result:
        result["source_path"] = str(path)
        return result
    return {
        "dateiname": path.name,
        "dokument_key": f"DOC-{abs(hash(path.name)) % 10**8:08d}",
        "text": "",
        "seiten": 0,
        "source_path": str(path),
        "error": "Dokument konnte nicht extrahiert werden.",
    }


def get_category_service():
    from psycopg import OperationalError

    from src.categorise.lm_studio import LMStudioClient
    from src.categorise.service import CategorisationService
    from src.categorise.store import DEFAULT_DATABASE_URL, PostgresStore

    try:
        store = PostgresStore(DEFAULT_DATABASE_URL)
    except (OperationalError, RuntimeError) as exc:
        return None, f"PostgreSQL/pgvector nicht erreichbar: {exc}"
    return CategorisationService(store, LMStudioClient()), None


def seed_categories_if_needed(service: Any) -> tuple[int, int]:
    from src.categorise.prompts import SEED_CATEGORIES

    existing = {category.name.lower() for category in service.store.load_categories()}
    added = 0
    for seed in SEED_CATEGORIES:
        if seed["name"].lower() in existing:
            continue
        service.add_category(name=seed["name"], description=seed["description"])
        added += 1
        existing.add(seed["name"].lower())
    return added, len(existing)


def category_name_map(service: Any) -> dict[str, str]:
    return {category.id: category.name for category in service.store.load_categories()}


@st.cache_resource(show_spinner=False)
def load_translation_service():
    ensure_import_path(TRANSLATOR_DIR)
    from service import TranslationService

    return TranslationService()


def translate_text(text: str) -> dict[str, str]:
    service = load_translation_service()
    return service.übersetze_text(text)


def ingest_texts_to_rag(documents: list[dict[str, Any]], reset: bool = False) -> int:
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


def build_rag_chain(model_cfg: dict[str, Any]):
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import OllamaLLM

    db = load_chroma()
    if db is None:
        raise RuntimeError("Die RAG-Datenbank ist leer.")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": model_cfg["k"],
            "fetch_k": model_cfg["fetch_k"],
            "lambda_mult": 0.7,
        },
    )
    llm = OllamaLLM(
        model=model_cfg["ollama_name"],
        temperature=0.1,
        num_predict=model_cfg["num_predict"],
        num_thread=8,
        num_ctx=model_cfg["num_ctx"],
    )
    prompt = PromptTemplate(template=RAG_PROMPT, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def init_session_state() -> None:
    defaults = {
        "documents": [],
        "category_results": {},
        "translation_results": {},
        "chat_messages": [],
        "rag_chain": None,
        "rag_model": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("DocuMind Local")
        st.caption("Zentrale lokale Oberfläche für OCR, Kategorisierung, Übersetzung und RAG")
        st.divider()

        st.subheader("Lokale Dienste")
        docker_status = ensure_postgres_docker()
        if docker_status["ok"]:
            st.success(str(docker_status["message"]))
        else:
            st.warning(str(docker_status["message"]))
        if st.button("PostgreSQL-Container erneut starten", use_container_width=True):
            ensure_postgres_docker.clear()
            st.rerun()

        st.divider()

        st.subheader("Speicher")
        st.metric("Geladene Dokumente", len(st.session_state.documents))
        st.metric("RAG-Chunks", chroma_count())

        if st.button("RAG-Datenbank löschen", use_container_width=True):
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.rerun()

        st.divider()
        st.subheader("RAG-Modell")
        model_name = st.selectbox("Ollama-Modell", list(RAG_MODELS.keys()))
        if st.button("Chat-Modell laden", use_container_width=True):
            try:
                st.session_state.rag_chain = build_rag_chain(RAG_MODELS[model_name])
                st.session_state.rag_model = model_name
                st.session_state.chat_messages = []
                st.success("Chat bereit.")
            except Exception as exc:
                st.error(f"RAG konnte nicht geladen werden: {exc}")
        if st.session_state.rag_model:
            st.info(f"Aktiv: {st.session_state.rag_model}")


def render_upload_pipeline() -> None:
    st.header("Dokument-Pipeline")
    st.write("Dokumente werden einmal hochgeladen und danach an die Module weitergegeben.")

    uploaded_files = st.file_uploader(
        "PDF, DOCX oder TXT hochladen",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Dokumente einlesen", type="primary", use_container_width=True):
        extracted = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            progress = st.progress(0)
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                progress.progress(index / len(uploaded_files), text=f"Extrahiere {uploaded_file.name}")
                file_path = save_uploaded_file(uploaded_file, tmp_path)
                extracted.append(extract_document(file_path))
            progress.empty()
        st.session_state.documents = extracted
        st.session_state.category_results = {}
        st.session_state.translation_results = {}
        st.success(f"{len(extracted)} Dokument(e) eingelesen.")

    if not st.session_state.documents:
        st.info("Noch keine Dokumente geladen.")
        return

    for doc in st.session_state.documents:
        title = f"{doc.get('dateiname', 'Dokument')} · {doc.get('dokument_key', 'ohne Key')}"
        with st.expander(title):
            col_meta, col_text = st.columns([1, 3])
            with col_meta:
                st.metric("Seiten", doc.get("seiten", 0))
                st.metric("Zeichen", len(doc.get("text", "")))
                if doc.get("error"):
                    st.error(doc["error"])
            with col_text:
                st.text_area(
                    "Extrahierter Text",
                    doc.get("text", "")[:4000],
                    height=220,
                    key=f"text_preview_{doc.get('dokument_key')}",
                )

    st.divider()
    col_cat, col_trans, col_rag = st.columns(3)
    with col_cat:
        if st.button("Alle kategorisieren", use_container_width=True):
            run_categorisation(st.session_state.documents)
    with col_trans:
        if st.button("Alle übersetzen", use_container_width=True):
            run_translation(st.session_state.documents)
    with col_rag:
        reset = st.checkbox("RAG vorher leeren")
        if st.button("In RAG indexieren", use_container_width=True):
            chunks = ingest_texts_to_rag(st.session_state.documents, reset=reset)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.success(f"{chunks} Chunk(s) gespeichert. Chat-Modell danach neu laden.")


def run_categorisation(documents: list[dict[str, Any]]) -> None:
    service, error = get_category_service()
    if error:
        st.error(error)
        return
    assert service is not None

    if not service.store.load_categories():
        with st.spinner("Seed-Kategorien werden angelegt..."):
            seed_categories_if_needed(service)

    categories = category_name_map(service)
    results = {}
    progress = st.progress(0)
    for index, doc in enumerate(documents, start=1):
        progress.progress(index / len(documents), text=f"Kategorisiere {doc.get('dateiname')}")
        try:
            record = service.categorise_text(
                text=doc.get("text", ""),
                filename=doc.get("dateiname", "manual-input.txt"),
                source_path=doc.get("source_path"),
            )
            category = categories.get(record.primary_category_id or "", "")
            if category:
                doc["category"] = category
            results[doc.get("dokument_key", doc.get("dateiname", str(index)))] = {
                "record": record,
                "category": category,
            }
        except Exception as exc:
            results[doc.get("dokument_key", doc.get("dateiname", str(index)))] = {"error": str(exc)}
    progress.empty()
    st.session_state.category_results = results


def run_translation(documents: list[dict[str, Any]]) -> None:
    results = {}
    progress = st.progress(0)
    for index, doc in enumerate(documents, start=1):
        progress.progress(index / len(documents), text=f"Übersetze {doc.get('dateiname')}")
        try:
            result = translate_text(doc.get("text", ""))
            doc["translation"] = result.get("translated", "")
            doc["language"] = result.get("language", "en")
            results[doc.get("dokument_key", doc.get("dateiname", str(index)))] = result
        except Exception as exc:
            results[doc.get("dokument_key", doc.get("dateiname", str(index)))] = {"error": str(exc)}
    progress.empty()
    st.session_state.translation_results = results


def render_categorisation() -> None:
    st.header("Modul A – Kategorisierung")
    service, error = get_category_service()
    status_label(error is None, "PostgreSQL/pgvector + LM Studio Client")
    if error:
        st.info("Starte PostgreSQL/pgvector und LM Studio, um die echte Kategorisierung zu verwenden.")
        st.code("docker start documind-postgres\nuv run python -m src.categorise seed-categories")
        return

    assert service is not None
    col_seed, col_count = st.columns([1, 2])
    with col_seed:
        if st.button("Seed-Kategorien anlegen"):
            added, total = seed_categories_if_needed(service)
            st.success(f"{added} neu, {total} insgesamt.")
    with col_count:
        categories = service.store.load_categories()
        st.metric("Kategorien", len(categories))

    if st.session_state.documents and st.button("Geladene Dokumente kategorisieren", type="primary"):
        run_categorisation(st.session_state.documents)

    if not st.session_state.category_results:
        st.info("Keine Kategorisierungsergebnisse vorhanden.")
        return

    for key, result in st.session_state.category_results.items():
        if result.get("error"):
            st.error(f"{key}: {result['error']}")
            continue
        record = result["record"]
        title = f"{record.filename} · {record.status} · {result.get('category') or 'keine Kategorie'}"
        with st.expander(title):
            if record.profile:
                st.write("**Zusammenfassung**")
                st.write(record.profile.summary)
                st.write("**Dokumenttyp**", record.profile.document_type)
            if record.decision:
                st.write("**Begründung**")
                st.write(record.decision.rationale)
                if record.decision.proposed_tags:
                    st.json(record.decision.proposed_tags)
            if record.errors:
                st.error("; ".join(record.errors))


def render_translation() -> None:
    st.header("Modul D – Übersetzung")
    st.caption("Deutsch → Englisch mit lokalem Transformer-Modell")
    text = st.text_area("Text direkt übersetzen", height=180)
    if st.button("Text übersetzen", type="primary"):
        if not text.strip():
            st.warning("Bitte Text eingeben.")
        else:
            with st.spinner("Übersetzung läuft..."):
                result = translate_text(text)
            st.success(result.get("translated", ""))
            st.json(result)

    st.divider()
    if st.session_state.documents and st.button("Geladene Dokumente übersetzen"):
        run_translation(st.session_state.documents)

    for key, result in st.session_state.translation_results.items():
        with st.expander(str(key)):
            if result.get("error"):
                st.error(result["error"])
            else:
                st.text_area("Übersetzung", result.get("translated", ""), height=220, key=f"translation_{key}")


def render_rag() -> None:
    st.header("Modul C – RAG-Chatbot")
    st.caption(f"ChromaDB: {CHROMA_PATH}")

    col_ingest, col_status = st.columns([1, 2])
    with col_ingest:
        if st.session_state.documents and st.button("Geladene Dokumente indexieren"):
            chunks = ingest_texts_to_rag(st.session_state.documents)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.success(f"{chunks} Chunk(s) gespeichert.")
    with col_status:
        st.metric("Aktuelle Chunks", chroma_count())

    if st.session_state.rag_chain is None:
        st.info("Lade zuerst ein Chat-Modell in der Sidebar.")
        return

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Quellen"):
                    for source in message["sources"]:
                        st.caption(source.metadata.get("dateiname", "unbekannt"))
                        st.text(source.page_content.removeprefix("passage: ")[:500])

    question = st.chat_input("Frage zu den Dokumenten...")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Suche relevante Dokumentstellen..."):
            start = time.time()
            try:
                result = st.session_state.rag_chain.invoke({"query": "query: " + question})
                answer = result.get("result", "Keine Antwort.")
                sources = result.get("source_documents", [])
                st.markdown(answer)
                st.caption(f"{time.time() - start:.1f}s · {st.session_state.rag_model}")
                if sources:
                    with st.expander("Quellen"):
                        for source in sources:
                            st.caption(source.metadata.get("dateiname", "unbekannt"))
                            st.text(source.page_content.removeprefix("passage: ")[:500])
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"RAG-Fehler: {exc}")


def render_overview() -> None:
    st.header("Überblick")
    st.write(
        "Diese Oberfläche verbindet die vorhandenen Projektmodule zu einem gemeinsamen Ablauf: "
        "Upload, Textextraktion, Kategorisierung, Übersetzung, RAG-Indexierung und Chat."
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("A Kategorisierung", "PostgreSQL + LM Studio")
    col_b.metric("B OCR", "Stub/Parser")
    col_c.metric("C RAG", "ChromaDB + Ollama")
    col_d.metric("D Übersetzung", "lokales Modell")
    st.info(
        "Die App sendet Dokumentinhalte nicht an externe APIs. Für die KI-Funktionen müssen die lokalen Dienste "
        "wie PostgreSQL, LM Studio und Ollama separat laufen."
    )


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("DocuMind Local")
    tabs = st.tabs([
        "Überblick",
        "Pipeline",
        "Kategorisierung",
        "Übersetzung",
        "RAG-Chatbot",
    ])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_upload_pipeline()
    with tabs[2]:
        render_categorisation()
    with tabs[3]:
        render_translation()
    with tabs[4]:
        render_rag()


if __name__ == "__main__":
    main()
