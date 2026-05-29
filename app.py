from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.RAGBot.lm_studio_rag import CHROMA_PATH, build_chain, chroma_count, ingest_texts
from src.categorise.docker import ensure_postgres_docker
from src.categorise.lm_studio import list_chat_models


ROOT = Path(__file__).resolve().parent
RAGBOT_DIR = ROOT / "src" / "RAGBot"
TRANSLATOR_DIR = ROOT / "src" / "translator"
DEFAULT_LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_CATEGORISER_CHAT_MODEL = os.getenv("LM_STUDIO_CHAT_MODEL", "google/gemma-4-e4b")
DEFAULT_CATEGORISER_EMBEDDING_MODEL = os.getenv(
    "LM_STUDIO_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"
)
DEFAULT_RAG_CHAT_MODEL = os.getenv("LM_STUDIO_RAG_MODEL", DEFAULT_CATEGORISER_CHAT_MODEL)


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


@st.cache_data(ttl=30, show_spinner=False)
def list_lm_studio_models(base_url: str) -> list[str]:
    return list_chat_models(base_url=base_url, timeout=5)


def model_options(models: list[str], selected: str) -> list[str]:
    options = list(models)
    if selected and selected not in options:
        options.insert(0, selected)
    return options or [selected]


def extract_document(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pdf":
        try:
            from src.OCR.docling_scanner import extrahiere_dokument as extrahiere_pdf_mit_docling

            return extrahiere_pdf_mit_docling(str(path))
        except Exception as exc:
            return {
                "dateiname": path.name,
                "dokument_key": f"DOC-{abs(hash(path.name)) % 10**8:08d}",
                "text": "",
                "seiten": 0,
                "source_path": str(path),
                "error": f"Docling-OCR fehlgeschlagen: {exc}",
            }

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


def get_category_service(
    chat_model: str | None = None,
    embedding_model: str | None = None,
    base_url: str | None = None,
):
    from psycopg import OperationalError

    from src.categorise.lm_studio import LMStudioClient
    from src.categorise.service import CategorisationService
    from src.categorise.store import DEFAULT_DATABASE_URL, PostgresStore

    try:
        store = PostgresStore(DEFAULT_DATABASE_URL)
    except (OperationalError, RuntimeError) as exc:
        return None, f"PostgreSQL/pgvector nicht erreichbar: {exc}"
    client = LMStudioClient(
        chat_model=chat_model or DEFAULT_CATEGORISER_CHAT_MODEL,
        embedding_model=embedding_model or DEFAULT_CATEGORISER_EMBEDDING_MODEL,
        base_url=base_url or DEFAULT_LM_STUDIO_BASE_URL,
    )
    return CategorisationService(store, client), None


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


def init_session_state() -> None:
    defaults = {
        "documents": [],
        "category_results": {},
        "translation_results": {},
        "chat_messages": [],
        "rag_chain": None,
        "rag_model": None,
        "categoriser_base_url": DEFAULT_LM_STUDIO_BASE_URL,
        "categoriser_chat_model": DEFAULT_CATEGORISER_CHAT_MODEL,
        "rag_chat_model": DEFAULT_RAG_CHAT_MODEL,
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
            st.rerun()

        st.divider()

        st.subheader("LM Studio")
        st.session_state.categoriser_base_url = st.text_input(
            "Base URL",
            value=st.session_state.categoriser_base_url,
            help="Wird gemeinsam für Kategorisierung und RAG genutzt.",
        )
        try:
            model_count = len(list_lm_studio_models(st.session_state.categoriser_base_url))
            st.success(f"{model_count} Modell(e) verfügbar")
        except Exception as exc:
            st.warning(f"LM Studio nicht erreichbar: {exc}")

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
            chunks = ingest_texts(st.session_state.documents, reset=reset)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.success(f"{chunks} Chunk(s) gespeichert. Chat-Modell danach neu laden.")


def render_ocr() -> None:
    st.header("Modul B – OCR / Digitalisierung")
    st.caption("PDF-Verarbeitung mit Docling, Ausgabe als Markdown und semantische Chunks")

    export_json = st.checkbox("JSON exportieren", value=False)
    export_markdown = st.checkbox("Markdown exportieren", value=False)
    uploaded_files = st.file_uploader(
        "PDFs für OCR hochladen",
        type=["pdf"],
        accept_multiple_files=True,
        key="ocr_upload",
    )

    if uploaded_files and st.button("OCR starten", type="primary"):
        results = []
        export_dir = ROOT / "outputs" / "ocr"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            progress = st.progress(0)
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                progress.progress(index / len(uploaded_files), text=f"Verarbeite {uploaded_file.name}")
                file_path = save_uploaded_file(uploaded_file, tmp_path)
                try:
                    from src.OCR.docling_scanner import extrahiere_dokument as extrahiere_pdf_mit_docling

                    results.append(
                        extrahiere_pdf_mit_docling(
                            str(file_path),
                            export_json=export_json,
                            export_markdown=export_markdown,
                            output_dir=str(export_dir),
                        )
                    )
                except Exception as exc:
                    results.append(
                        {
                            "dateiname": uploaded_file.name,
                            "dokument_key": f"DOC-{abs(hash(uploaded_file.name)) % 10**8:08d}",
                            "text": "",
                            "seiten": 0,
                            "source_path": str(file_path),
                            "error": f"Docling-OCR fehlgeschlagen: {exc}",
                        }
                    )
            progress.empty()
        st.session_state.documents = results
        st.session_state.category_results = {}
        st.session_state.translation_results = {}
        st.success(f"{len(results)} PDF(s) verarbeitet.")
        if export_json or export_markdown:
            st.info(f"Exporte liegen unter `{export_dir}`.")

    if not st.session_state.documents:
        st.info("Noch keine OCR-Ergebnisse vorhanden.")
        return

    for doc in st.session_state.documents:
        with st.expander(f"{doc.get('dateiname', 'Dokument')} · {doc.get('dokument_key', 'ohne Key')}"):
            col_meta, col_text = st.columns([1, 3])
            with col_meta:
                st.metric("Zeichen", len(doc.get("text", "")))
                st.metric("Chunks", doc.get("total_chunks", len(doc.get("chunks", []))))
                if doc.get("error"):
                    st.error(doc["error"])
            with col_text:
                st.text_area("Markdown", doc.get("text", "")[:5000], height=260, key=f"ocr_text_{doc.get('dokument_key')}")

            chunks = doc.get("chunks") or []
            if chunks:
                with st.expander("Docling-Chunks"):
                    for chunk in chunks:
                        st.caption(chunk.get("chunk_id", "chunk"))
                        st.text(chunk.get("content", "")[:1000])

    col_cat, col_rag = st.columns(2)
    with col_cat:
        if st.button("OCR-Ergebnisse kategorisieren", use_container_width=True):
            run_categorisation(st.session_state.documents)
    with col_rag:
        if st.button("OCR-Ergebnisse in RAG indexieren", use_container_width=True):
            chunks = ingest_texts(st.session_state.documents)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.success(f"{chunks} Chunk(s) gespeichert.")


def run_categorisation(documents: list[dict[str, Any]]) -> None:
    service, error = get_category_service(
        chat_model=st.session_state.categoriser_chat_model,
        base_url=st.session_state.categoriser_base_url,
    )
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
    st.subheader("LM Studio Modelle")
    st.caption(f"Provider: `{st.session_state.categoriser_base_url}`")

    try:
        available_models = list_lm_studio_models(st.session_state.categoriser_base_url)
        st.caption(f"{len(available_models)} Modell(e) aus LM Studio geladen.")
    except Exception as exc:
        available_models = []
        st.warning(f"LM Studio Modellliste nicht erreichbar. Details: {exc}")

    chat_options = model_options(available_models, st.session_state.categoriser_chat_model)
    selected_chat = st.selectbox(
        "Chat-/Klassifikationsmodell",
        chat_options,
        index=chat_options.index(st.session_state.categoriser_chat_model),
    )
    st.session_state.categoriser_chat_model = selected_chat

    service, error = get_category_service(
        chat_model=st.session_state.categoriser_chat_model,
        base_url=st.session_state.categoriser_base_url,
    )
    status_label(error is None, "PostgreSQL/pgvector + LM Studio Client")
    if error:
        st.info("Starte PostgreSQL/pgvector und LM Studio, um die echte Kategorisierung zu verwenden.")
        st.code("docker start documind-postgres\nuv run python -m src.categorise seed-categories")
        return

    st.info(
        "Aktiv: "
        f"Chat `{st.session_state.categoriser_chat_model}` · "
        f"Embeddings `{DEFAULT_CATEGORISER_EMBEDDING_MODEL}`"
    )

    assert service is not None
    col_seed, col_count = st.columns([1, 2])
    with col_seed:
        if st.button("Seed-Kategorien anlegen"):
            added, total = seed_categories_if_needed(service)
            st.success(f"{added} neu, {total} insgesamt.")
    with col_count:
        categories = service.store.load_categories()
        st.metric("Kategorien", len(categories))

    st.divider()
    st.subheader("Dokumente kategorisieren")

    if st.session_state.documents and st.button("Geladene Dokumente kategorisieren", type="primary"):
        run_categorisation(st.session_state.documents)

    uploaded_files = st.file_uploader(
        "Dokumente direkt für Modul A hochladen",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="categorise_upload",
    )
    if uploaded_files and st.button("Hochgeladene Dokumente kategorisieren", type="primary"):
        extracted = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file, tmp_path)
                extracted.append(extract_document(file_path))
        run_categorisation(extracted)

    if st.session_state.category_results:
        st.subheader("Aktuelle Kategorisierungsergebnisse")
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
    else:
        st.info("Noch keine aktuellen Kategorisierungsergebnisse vorhanden.")

    st.divider()
    st.subheader("Gespeicherte Dokumente")
    documents = service.store.load_documents()
    categories_by_id = category_name_map(service)
    if not documents:
        st.info("Noch keine Dokumente in PostgreSQL gespeichert.")
    else:
        st.dataframe(
            [
                {
                    "Datei": document.filename,
                    "Status": document.status,
                    "Kategorie": categories_by_id.get(document.primary_category_id or "", ""),
                    "Kategorie-ID": document.primary_category_id or "",
                    "Erstellt": document.created_at,
                }
                for document in documents
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Details gespeicherter Dokumente"):
            selected_filename = st.selectbox(
                "Dokument",
                [document.filename for document in documents],
                key="stored_category_doc",
            )
            selected = next(document for document in documents if document.filename == selected_filename)
            st.write("**Status**", selected.status)
            st.write("**Kategorie**", categories_by_id.get(selected.primary_category_id or "", ""))
            if selected.profile:
                st.write("**Zusammenfassung**")
                st.write(selected.profile.summary)
            if selected.decision:
                st.write("**Begründung**")
                st.write(selected.decision.rationale)
                if selected.decision.proposed_tags:
                    st.json(selected.decision.proposed_tags)
            if selected.errors:
                st.error("; ".join(selected.errors))

    st.divider()
    st.subheader("Taxonomie-Reviews")
    taxonomy_reviews = [document for document in documents if document.status == "needs_taxonomy_review"]
    if not taxonomy_reviews:
        st.info("Keine offenen Taxonomie-Reviews.")
    else:
        st.dataframe(
            [
                {
                    "Datei": document.filename,
                    "Vorgeschlagene Kategorie": (
                        document.decision.none_fits_proposal.name
                        if document.decision and document.decision.none_fits_proposal
                        else ""
                    ),
                    "Beschreibung": (
                        document.decision.none_fits_proposal.description
                        if document.decision and document.decision.none_fits_proposal
                        else ""
                    ),
                    "Begründung": document.decision.rationale if document.decision else "",
                }
                for document in taxonomy_reviews
            ],
            use_container_width=True,
            hide_index=True,
        )


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

    try:
        available_models = list_lm_studio_models(st.session_state.categoriser_base_url)
        st.caption(f"{len(available_models)} Modell(e) aus LM Studio geladen.")
    except Exception as exc:
        available_models = []
        st.warning(f"LM Studio Modellliste nicht erreichbar. Details: {exc}")

    col_model, col_load = st.columns([2, 1])
    with col_model:
        rag_options = model_options(available_models, st.session_state.rag_chat_model)
        selected_rag_model = st.selectbox(
            "RAG-Chatmodell",
            rag_options,
            index=rag_options.index(st.session_state.rag_chat_model),
        )
        st.session_state.rag_chat_model = selected_rag_model
    with col_load:
        st.write("")
        st.write("")
        if st.button("Chat-Modell laden", use_container_width=True):
            try:
                st.session_state.rag_chain = build_chain(
                    st.session_state.rag_chat_model,
                    st.session_state.categoriser_base_url,
                )
                st.session_state.rag_model = st.session_state.rag_chat_model
                st.session_state.chat_messages = []
                st.success("Chat bereit.")
            except Exception as exc:
                st.error(f"RAG konnte nicht geladen werden: {exc}")

    if st.session_state.rag_model:
        st.info(f"Aktives RAG-Modell: `{st.session_state.rag_model}`")

    col_ingest, col_status = st.columns([1, 2])
    with col_ingest:
        if st.session_state.documents and st.button("Geladene Dokumente indexieren"):
            chunks = ingest_texts(st.session_state.documents)
            st.session_state.rag_chain = None
            st.session_state.rag_model = None
            st.success(f"{chunks} Chunk(s) gespeichert.")
    with col_status:
        st.metric("Aktuelle Chunks", chroma_count())

    if st.session_state.rag_chain is None:
        st.info("Lade zuerst ein Chat-Modell in diesem Tab.")
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
                answer = str(result.get("result") or "").strip()
                sources = result.get("source_documents", [])
                if answer:
                    st.markdown(answer)
                else:
                    st.warning(
                        "LM Studio hat keine finale Antwort zurückgegeben. "
                        "Die Quellen wurden gefunden, aber das Modell lieferte leeren Antworttext."
                    )
                    answer = "Keine finale Antwort vom Modell erhalten."
                    raw_response = result.get("raw_response")
                    if raw_response:
                        with st.expander("LM-Studio-Rohantwort"):
                            st.json(raw_response)
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
    col_c.metric("C RAG", "ChromaDB + LM Studio")
    col_d.metric("D Übersetzung", "lokales Modell")
    st.info(
        "Die App sendet Dokumentinhalte nicht an externe APIs. Für die KI-Funktionen müssen die lokalen Dienste "
        "wie PostgreSQL und LM Studio separat laufen."
    )


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("DocuMind Local")
    tabs = st.tabs([
        "Überblick",
        "Pipeline",
        "OCR",
        "Kategorisierung",
        "Übersetzung",
        "RAG-Chatbot",
    ])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_upload_pipeline()
    with tabs[2]:
        render_ocr()
    with tabs[3]:
        render_categorisation()
    with tabs[4]:
        render_translation()
    with tabs[5]:
        render_rag()


if __name__ == "__main__":
    main()
