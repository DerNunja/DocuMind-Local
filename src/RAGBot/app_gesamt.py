"""
app_gesamt.py – Gemeinsame Streamlit-UI für alle 4 Module

Modul A: Kategorisierung     → voll funktionsfähig (Stub-Logik)
Modul B: Digitalisierung/OCR → Stub (Platzhalter)
Modul C: RAG-Chatbot         → voll funktionsfähig 
Modul D: Übersetzer          → Stub (Platzhalter)

Verwendung:
    streamlit run app_gesamt.py

"""

import streamlit as st
import tempfile
import os
import time
import json
import re
from pathlib import Path

# ── Seiteneinstellungen ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PM-Dokumentenmanagementsystem",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# SCHNITTSTELLEN ZU ANDEREN MODULEN
# Hier umschalten wenn Kollegen fertig sind:
#   from modul_b import extrahiere_dokument   ← Modul B echt
#   from modul_d import uebersetze            ← Modul D echt
# ══════════════════════════════════════════════════════════════════════════
try:
    from modul_b_stub import extrahiere_dokument
    MODUL_B_ECHT = False
except ImportError:
    extrahiere_dokument = None
    MODUL_B_ECHT = False

try:
    from modul_d_stub import uebersetze
    MODUL_D_ECHT = False
except ImportError:
    uebersetze = None
    MODUL_D_ECHT = False

# ── Konfiguration ─────────────────────────────────────────────────────────
CHROMA_PFAD      = "./vectordb"
COLLECTION       = "projektdokumente"
EMBEDDING_MODELL = "intfloat/multilingual-e5-small"

MODELLE = {
    "🦙 Llama 3.1 8B (Qualität)": {
        "ollama_name": "llama3.1:8b",
        "k": 4, "fetch_k": 12,
        "num_predict": 250, "num_ctx": 2048,
    },
    "⚡ Phi-3.5 Mini (Geschwindigkeit)": {
        "ollama_name": "phi3.5",
        "k": 2, "fetch_k": 6,
        "num_predict": 150, "num_ctx": 1024,
    },
}

KATEGORIEN = [
    "Protokoll", "Statusbericht", "Rechnung",
    "Vertrag", "Risikoanalyse", "Änderungsantrag",
    "Kommunikationsplan", "Technische Dokumentation", "Sonstiges"
]

PROMPT_TEMPLATE = """Du bist ein hilfreicher Projektmanagement-Assistent.
Beantworte die Frage auf Basis der folgenden Dokument-Auszüge aus dem internen Projektsystem.
Die Dokumente können Vorlagen mit Feldnamen oder ausgefüllte Projektberichte sein.
Beschreibe was die Dokumente zu diesem Thema enthalten – auch Feldnamen und Strukturen sind nützliche Informationen.
Wenn keine relevante Information vorhanden ist, sage dies klar.
Erfinde keine konkreten Werte oder Namen die nicht im Text stehen.
Antworte immer auf Deutsch, präzise und strukturiert.

Dokument-Auszüge:
{context}

Frage: {question}

Antwort:"""

TESTFRAGEN = [
    {"frage": "Wie hoch ist das Gesamtbudget des Projekts und wie viel wurde bisher verbraucht?",
     "keywords": [["113.000","113000","gesamtbudget"],["57.000","57000","verbraucht"],["budget","kosten"],["50%","puffer"]]},
    {"frage": "Welche technischen Probleme oder Risiken wurden im Projekt identifiziert?",
     "keywords": [["risiko","datenmigration","dvs"],["personalausfall","krankheit"],["rot","gelb","score"],["maßnahmen","verantwortlich"]]},
    {"frage": "Wer ist für die Backend-Entwicklung verantwortlich und was wurde zuletzt abgeschlossen?",
     "keywords": [["sarah","müller"],["backend","api","rest"],["postgresql","datenbank"],["abgeschlossen","fertiggestellt"]]},
    {"frage": "Was wurde im letzten Meeting entschieden und welche Aufgaben sind noch offen?",
     "keywords": [["beschluss","entscheidung","genehmigt"],["postgresql","mysql"],["weber","müller","hoffmann"],["fällig","offen"]]},
    {"frage": "Bis wann soll das Projekt abgeschlossen sein und warum gibt es Verzögerungen?",
     "keywords": [["29.05.2026","mai 2026","projektende"],["verzögerung","verschoben"],["krankheit","tobias"],["meilenstein","zeitplan"]]},
]


# ── Hilfsfunktionen ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Lade Embedding-Modell...")
def lade_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def lade_db(embeddings):
    from langchain_chroma import Chroma
    if not Path(CHROMA_PFAD).exists():
        return None
    db = Chroma(persist_directory=CHROMA_PFAD, embedding_function=embeddings, collection_name=COLLECTION)
    return db if db._collection.count() > 0 else None


def baue_chain(db, modell_cfg):
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import RetrievalQA
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": modell_cfg["k"], "fetch_k": modell_cfg["fetch_k"], "lambda_mult": 0.7},
    )
    llm = OllamaLLM(model=modell_cfg["ollama_name"], temperature=0.1,
                    num_predict=modell_cfg["num_predict"], num_thread=8, num_ctx=modell_cfg["num_ctx"])
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                       return_source_documents=True, chain_type_kwargs={"prompt": prompt})


def keyword_score(antwort, keywords):
    antwort_lower = antwort.lower()
    treffer = sum(1 for kw in keywords if
                  (any(s.lower() in antwort_lower for s in kw) if isinstance(kw, list) else kw.lower() in antwort_lower))
    return treffer / len(keywords) if keywords else 0.0


def ingest_dateien(dateipfade, reset=False):
    import shutil
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    try:
        import pdfplumber
        pdfplumber_ok = True
    except ImportError:
        pdfplumber_ok = False

    if reset and Path(CHROMA_PFAD).exists():
        shutil.rmtree(CHROMA_PFAD)

    def bereinige(text):
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def tabelle_zu_text(tabelle):
        if not tabelle or len(tabelle) < 2:
            return ""
        koepfe = [str(z).strip() if z else "-" for z in tabelle[0]]
        zeilen = []
        for zeile in tabelle[1:]:
            if not any(z for z in zeile):
                continue
            werte = [str(z).strip() if z else "-" for z in zeile]
            paare = [f"{k}: {v}" for k, v in zip(koepfe, werte) if v != "-"]
            if paare:
                zeilen.append(" | ".join(paare))
        return "\n".join(zeilen)

    alle_docs = []
    for pfad in dateipfade:
        endung = Path(pfad).suffix.lower()
        try:
            if endung == ".pdf" and pdfplumber_ok:
                with pdfplumber.open(pfad) as pdf:
                    for seite in pdf.pages:
                        text = seite.extract_text() or ""
                        for tabelle in seite.extract_tables():
                            t = tabelle_zu_text(tabelle)
                            if t:
                                text += "\n\n[TABELLE]\n" + t
                        text = bereinige(text)
                        if len(text.strip()) > 50:
                            alle_docs.append(Document(page_content=text, metadata={"dateiname": Path(pfad).name}))
            elif endung == ".pdf":
                loader = PyPDFLoader(pfad)
                for d in loader.load():
                    d.page_content = bereinige(d.page_content)
                    d.metadata["dateiname"] = Path(pfad).name
                    if len(d.page_content.strip()) > 50:
                        alle_docs.append(d)
            elif endung in (".docx", ".doc"):
                loader = Docx2txtLoader(pfad)
                for d in loader.load():
                    d.metadata["dateiname"] = Path(pfad).name
                    alle_docs.append(d)
            elif endung == ".txt":
                loader = TextLoader(pfad, encoding="utf-8")
                for d in loader.load():
                    d.metadata["dateiname"] = Path(pfad).name
                    alle_docs.append(d)
        except Exception as e:
            st.warning(f"Fehler bei {Path(pfad).name}: {e}")

    if not alle_docs:
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                               separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(alle_docs)
    for chunk in chunks:
        chunk.page_content = "passage: " + chunk.page_content

    Chroma.from_documents(documents=chunks, embedding=lade_embeddings(),
                          persist_directory=CHROMA_PFAD, collection_name=COLLECTION)
    return len(alle_docs), len(chunks)


def kategorisiere_dokument_stub(text: str, dateiname: str) -> dict:
    """
    Stub für Modul A – einfache Keyword-basierte Kategorisierung.
    Wird durch echtes Modul A (LLM + Embeddings) ersetzt.
    """
    text_lower = text.lower()
    scores = {}
    regeln = {
        "Protokoll":             ["protokoll", "meeting", "teilnehmer", "tagesordnung", "beschluss"],
        "Statusbericht":         ["status", "fortschritt", "meilenstein", "budget", "zeitplan"],
        "Rechnung":              ["rechnung", "betrag", "mwst", "zahlungsziel", "iban"],
        "Vertrag":               ["vertrag", "vereinbarung", "unterzeichnet", "vertragspartner"],
        "Risikoanalyse":         ["risiko", "wahrscheinlichkeit", "auswirkung", "maßnahme"],
        "Änderungsantrag":       ["änderung", "änderungsantrag", "änderungs-id", "genehmigung"],
        "Kommunikationsplan":    ["kommunikation", "stakeholder", "kanal", "kommunikationsziele"],
        "Technische Dokumentation": ["architektur", "api", "konfiguration", "deployment", "server"],
    }
    for kategorie, keywords in regeln.items():
        scores[kategorie] = sum(1 for kw in keywords if kw in text_lower)

    beste = max(scores, key=scores.get)
    konfidenz = min(scores[beste] / 3.0, 1.0)

    if konfidenz < 0.3:
        beste = "Sonstiges"
        konfidenz = 0.5

    return {
        "kategorie":  beste,
        "konfidenz":  round(konfidenz * 100),
        "dateiname":  dateiname,
        "alle_scores": scores,
    }


# ── Session State ─────────────────────────────────────────────────────────
defaults = {
    "nachrichten": [], "chain": None, "aktives_modell": None,
    "eval_ergebnisse": {}, "kategorisierte_docs": [], "digitalisierte_docs": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 PM-Dokumentensystem")
    st.caption("Lokales KI-gestütztes Dokumentenmanagementsystem")
    st.divider()

    # Modul-Status
    st.subheader("🔌 Modulstatus")
    st.success("✅ Modul A – Kategorisierung (Stub)")
    if MODUL_B_ECHT:
        st.success("✅ Modul B – OCR (echt)")
    else:
        st.warning("⚠️ Modul B – OCR (Stub)")
    st.success("✅ Modul C – RAG-Chatbot")
    if MODUL_D_ECHT:
        st.success("✅ Modul D – Übersetzer (echt)")
    else:
        st.warning("⚠️ Modul D – Übersetzer (Stub)")

    st.divider()

    # DB Status
    st.subheader("🗄️ Datenbank")
    emb = lade_embeddings()
    db_check = lade_db(emb)
    if db_check:
        st.success(f"✅ {db_check._collection.count()} Chunks")
    else:
        st.warning("⚠️ Leer / nicht vorhanden")

    if st.button("🗑️ DB zurücksetzen", use_container_width=True):
        import shutil
        if Path(CHROMA_PFAD).exists():
            shutil.rmtree(CHROMA_PFAD)
        st.session_state.chain = None
        st.session_state.aktives_modell = None
        st.rerun()

    st.divider()

    # RAG Modellauswahl (nur für Tab C)
    st.subheader("🤖 RAG-Modell")
    modell_name = st.selectbox("Modell:", list(MODELLE.keys()))
    modell_cfg  = MODELLE[modell_name]
    st.caption(f"k={modell_cfg['k']} · ctx={modell_cfg['num_ctx']}")

    if st.button("✅ Modell laden", use_container_width=True):
        db = lade_db(lade_embeddings())
        if db is None:
            st.error("Zuerst Dokumente einlesen (Tab B oder C).")
        else:
            with st.spinner("Verbinde mit Ollama..."):
                try:
                    st.session_state.chain = baue_chain(db, modell_cfg)
                    st.session_state.aktives_modell = modell_name
                    st.session_state.nachrichten = []
                    st.success("Bereit!")
                except Exception as e:
                    st.error(f"Ollama Fehler: {e}")

    if st.session_state.aktives_modell:
        st.info(f"Aktiv: {st.session_state.aktives_modell}")


# ── Hauptbereich: 4 Tabs ──────────────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "🏷️ A – Kategorisierung",
    "📷 B – Digitalisierung",
    "💬 C – RAG-Chatbot",
    "🌍 D – Übersetzer",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB A – KATEGORISIERUNG
# ════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.header("🏷️ Modul A – Automatische Dokumentenkategorisierung")

    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Ablauf:** Dokument hochladen → Text extrahieren → LLM analysiert und kategorisiert →
        Embedding-Vergleich mit bestehenden Kategorien → Mensch bestätigt bei niedriger Konfidenz.
        """)
    with col_status:
        st.info("🔧 Stub-Modus\n\nEchtes Modul A\n(LLM + Embeddings)\nfolgt")

    st.divider()

    hochgeladene_a = st.file_uploader(
        "Dokumente zur Kategorisierung hochladen",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        key="upload_a"
    )

    if hochgeladene_a:
        if st.button("🏷️ Kategorisierung starten", type="primary", use_container_width=True):
            ergebnisse_a = []
            fortschritt_a = st.progress(0)

            with tempfile.TemporaryDirectory() as tmp:
                for i, datei in enumerate(hochgeladene_a):
                    fortschritt_a.progress((i+1)/len(hochgeladene_a),
                                           text=f"Analysiere: {datei.name}")
                    tmp_pfad = os.path.join(tmp, datei.name)
                    with open(tmp_pfad, "wb") as f:
                        f.write(datei.getbuffer())

                    # Text extrahieren (wie Modul B)
                    try:
                        from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
                        endung = Path(datei.name).suffix.lower()
                        if endung == ".pdf":
                            docs = PyPDFLoader(tmp_pfad).load()
                        elif endung in (".docx", ".doc"):
                            docs = Docx2txtLoader(tmp_pfad).load()
                        else:
                            docs = TextLoader(tmp_pfad, encoding="utf-8").load()
                        text = " ".join(d.page_content for d in docs)[:2000]
                    except Exception:
                        text = ""

                    # Kategorisierung (Stub → wird durch Modul A ersetzt)
                    kat = kategorisiere_dokument_stub(text, datei.name)
                    ergebnisse_a.append(kat)

            fortschritt_a.empty()
            st.session_state.kategorisierte_docs = ergebnisse_a

        if st.session_state.kategorisierte_docs:
            st.subheader("📋 Kategorisierungsergebnisse")
            for erg in st.session_state.kategorisierte_docs:
                konfidenz = erg["konfidenz"]
                if konfidenz >= 70:
                    icon, farbe = "🟢", "green"
                    hinweis = "Auto-Zuordnung"
                elif konfidenz >= 40:
                    icon, farbe = "🟡", "orange"
                    hinweis = "Mensch prüft"
                else:
                    icon, farbe = "🔴", "red"
                    hinweis = "Neue Kategorie?"

                with st.expander(f"{icon} **{erg['dateiname']}** → {erg['kategorie']} ({konfidenz}% Konfidenz) · {hinweis}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kategorie", erg["kategorie"])
                        st.metric("Konfidenz", f"{konfidenz}%")
                        st.caption(hinweis)

                    with col2:
                        # Manuelle Korrektur
                        neue_kat = st.selectbox(
                            "Kategorie korrigieren:",
                            KATEGORIEN,
                            index=KATEGORIEN.index(erg["kategorie"]) if erg["kategorie"] in KATEGORIEN else 0,
                            key=f"kat_{erg['dateiname']}"
                        )
                        if neue_kat != erg["kategorie"]:
                            st.info(f"✏️ Wird auf '{neue_kat}' geändert")

                    # Score-Details
                    with st.expander("Alle Scores anzeigen"):
                        scores_sorted = sorted(erg["alle_scores"].items(), key=lambda x: x[1], reverse=True)
                        for kat_name, score in scores_sorted:
                            if score > 0:
                                st.progress(min(score/5, 1.0), text=f"{kat_name}: {score}")


# ════════════════════════════════════════════════════════════════════════════
# TAB B – DIGITALISIERUNG / OCR
# ════════════════════════════════════════════════════════════════════════════
with tab_b:
    st.header("📷 Modul B – Digitalisierung & OCR")

    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Ablauf:** Gescanntes Dokument (Bild/PDF) hochladen → KI Vision Modell (OCR + Layout-Analyse) →
        Textextraktion → Dokument-Key generieren → Strukturierte Ausgabe für weitere Module.
        """)
    with col_status:
        if MODUL_B_ECHT:
            st.success("✅ Echtes Modul B")
        else:
            st.warning("🔧 Stub-Modus\n\nEchtes Modul B\n(KI Vision/OCR)\nfolgt")

    st.divider()

    # Pipeline-Diagramm
    st.subheader("📊 Verarbeitungs-Pipeline")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("1️⃣", "Upload", "Scan/PDF")
    col2.metric("2️⃣", "KI Vision", "OCR + Layout")
    col3.metric("3️⃣", "Extraktion", "Reintext")
    col4.metric("4️⃣", "Key", "Dok-ID")
    col5.metric("5️⃣", "Output", "Text + Key")

    st.divider()

    hochgeladene_b = st.file_uploader(
        "Gescannte Dokumente hochladen (PDF, Bild)",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        key="upload_b"
    )

    if hochgeladene_b:
        if st.button("📷 OCR & Digitalisierung starten", type="primary", use_container_width=True):
            ergebnisse_b = []
            with tempfile.TemporaryDirectory() as tmp:
                for datei in hochgeladene_b:
                    tmp_pfad = os.path.join(tmp, datei.name)
                    with open(tmp_pfad, "wb") as f:
                        f.write(datei.getbuffer())

                    with st.spinner(f"Verarbeite {datei.name}..."):
                        if extrahiere_dokument and datei.name.endswith(".pdf"):
                            # Echtes Modul B oder Stub
                            result = extrahiere_dokument(tmp_pfad)
                            if result:
                                ergebnisse_b.append(result)
                        else:
                            # Bildformate → Stub-Ausgabe
                            import hashlib
                            dok_key = "DOC-" + hashlib.md5(datei.name.encode()).hexdigest()[:8].upper()
                            ergebnisse_b.append({
                                "dateiname":    datei.name,
                                "dokument_key": dok_key,
                                "text":         f"[KI Vision OCR – Modul B ausstehend]\nDatei: {datei.name}\nEchtes Modul B wird hier den extrahierten Text liefern.",
                                "seiten":       1,
                            })

            st.session_state.digitalisierte_docs = ergebnisse_b

        if st.session_state.digitalisierte_docs:
            st.subheader("📋 Digitalisierungsergebnisse")
            for doc in st.session_state.digitalisierte_docs:
                with st.expander(f"📄 {doc['dateiname']} · Key: `{doc['dokument_key']}`"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Dokument-Key", doc["dokument_key"])
                        st.metric("Seiten", doc.get("seiten", "–"))
                        st.metric("Zeichen", len(doc.get("text", "")))
                    with col2:
                        st.text_area("Extrahierter Text:", doc.get("text", "")[:1000], height=200,
                                     key=f"text_{doc['dokument_key']}")

                    # Direkt in RAG einpflegen
                    if st.button(f"➡️ In RAG-Datenbank einpflegen", key=f"ingest_{doc['dokument_key']}"):
                        from langchain_core.documents import Document
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        from langchain_chroma import Chroma
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        lc_doc = Document(
                            page_content=doc["text"],
                            metadata={"dateiname": doc["dateiname"], "dokument_key": doc["dokument_key"], "quelle": "modul-b"}
                        )
                        chunks = splitter.split_documents([lc_doc])
                        for c in chunks:
                            c.page_content = "passage: " + c.page_content
                        Chroma.from_documents(documents=chunks, embedding=lade_embeddings(),
                                              persist_directory=CHROMA_PFAD, collection_name=COLLECTION)
                        st.success(f"✅ {len(chunks)} Chunks in RAG-Datenbank gespeichert!")


# ════════════════════════════════════════════════════════════════════════════
# TAB C – RAG CHATBOT (voll funktionsfähig)
# ════════════════════════════════════════════════════════════════════════════
with tab_c:
    st.header("💬 Modul C – RAG-basierter Chatbot")
    st.caption("✅ Voll funktionsfähig · Llama 3.1 / Phi-3.5 · ChromaDB · multilingual-e5-small")

    tab_c1, tab_c2, tab_c3 = st.tabs(["💬 Chat", "📤 Dokumente einlesen", "📊 Evaluation"])

    # ── C1: Chat ──────────────────────────────────────────────────────────
    with tab_c1:
        if not st.session_state.chain:
            st.info("👈 Modell in der Seitenleiste laden.")
        else:
            for msg in st.session_state.nachrichten:
                with st.chat_message(msg["rolle"]):
                    st.markdown(msg["inhalt"])
                    if msg["rolle"] == "assistant" and "quellen" in msg and msg["quellen"]:
                        with st.expander(f"📚 {len(msg['quellen'])} Quellen"):
                            for i, q in enumerate(msg["quellen"], 1):
                                datei = q.metadata.get("dateiname", "–")
                                auszug = q.page_content.removeprefix("passage: ")[:300]
                                st.caption(f"**{i}. {datei}**")
                                st.text(auszug + "…")

            if frage := st.chat_input("Frage zu Projektdokumenten..."):
                st.session_state.nachrichten.append({"rolle": "user", "inhalt": frage})
                with st.chat_message("user"):
                    st.markdown(frage)

                with st.chat_message("assistant"):
                    with st.spinner("🔍 Suche..."):
                        start = time.time()
                        try:
                            ergebnis = st.session_state.chain.invoke({"query": "query: " + frage})
                            antwort  = ergebnis.get("result", "Keine Antwort.")
                            quellen  = ergebnis.get("source_documents", [])
                            dauer    = time.time() - start
                            st.markdown(antwort)
                            st.caption(f"⏱️ {dauer:.1f}s · {st.session_state.aktives_modell}")
                            if quellen:
                                with st.expander(f"📚 {len(quellen)} Quellen"):
                                    for i, q in enumerate(quellen, 1):
                                        st.caption(f"**{i}. {q.metadata.get('dateiname','–')}**")
                                        st.text(q.page_content.removeprefix("passage: ")[:300] + "…")
                            st.session_state.nachrichten.append(
                                {"rolle": "assistant", "inhalt": antwort, "quellen": quellen})
                        except Exception as e:
                            st.error(f"Fehler: {e}")

            if st.session_state.nachrichten:
                if st.button("🗑️ Chat leeren"):
                    st.session_state.nachrichten = []
                    st.rerun()

    # ── C2: Dokumente einlesen ─────────────────────────────────────────────
    with tab_c2:
        hochgeladene_c = st.file_uploader("PDF / DOCX / TXT hochladen",
                                           accept_multiple_files=True, type=["pdf","docx","txt"], key="upload_c")
        reset_c = st.checkbox("DB vorher zurücksetzen", key="reset_c")

        if hochgeladene_c:
            st.write(f"**{len(hochgeladene_c)} Datei(en):**")
            for f in hochgeladene_c:
                st.write(f"  • {f.name} ({f.size//1024} KB)")

            if st.button("🚀 Einlesen", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as tmp:
                    pfade = []
                    for datei in hochgeladene_c:
                        p = os.path.join(tmp, datei.name)
                        with open(p, "wb") as f:
                            f.write(datei.getbuffer())
                        pfade.append(p)

                    with st.spinner("Verarbeite und speichere in ChromaDB..."):
                        docs_n, chunks_n = ingest_dateien(pfade, reset=reset_c)

                if chunks_n > 0:
                    st.success(f"✅ {docs_n} Seite(n) → {chunks_n} Chunks gespeichert!")
                    st.session_state.chain = None
                    st.session_state.aktives_modell = None
                    st.info("👈 Modell neu laden.")
                else:
                    st.error("Keine Dokumente verarbeitet.")

    # ── C3: Evaluation ────────────────────────────────────────────────────
    with tab_c3:
        eval_modell_name = st.selectbox("Modell:", list(MODELLE.keys()), key="eval_sel")

        with st.expander("Testfragen"):
            for i, tf in enumerate(TESTFRAGEN, 1):
                st.write(f"**{i}.** {tf['frage']}")

        if st.button("▶️ Evaluation starten", type="primary", use_container_width=True):
            db = lade_db(lade_embeddings())
            if not db:
                st.error("Keine Datenbank.")
            else:
                try:
                    chain_eval = baue_chain(db, MODELLE[eval_modell_name])
                except Exception as e:
                    st.error(f"Ollama Fehler: {e}")
                    st.stop()

                prog = st.progress(0)
                ergebnisse_eval = []
                for i, test in enumerate(TESTFRAGEN):
                    prog.progress((i+1)/len(TESTFRAGEN), text=f"Frage {i+1}/{len(TESTFRAGEN)}")
                    start = time.time()
                    try:
                        res = chain_eval.invoke({"query": "query: " + test["frage"]})
                        antwort_e = res.get("result", "")
                    except Exception:
                        antwort_e = ""
                    score_e = keyword_score(antwort_e, test["keywords"])
                    ergebnisse_eval.append({"frage": test["frage"], "antwort": antwort_e,
                                            "score": score_e, "zeit_s": round(time.time()-start, 1)})

                prog.empty()
                avg_s = sum(e["score"] for e in ergebnisse_eval) / len(ergebnisse_eval)
                avg_z = sum(e["zeit_s"] for e in ergebnisse_eval) / len(ergebnisse_eval)

                kurzname = "Llama 3.1" if "Llama" in eval_modell_name else "Phi-3.5"
                st.session_state.eval_ergebnisse[kurzname] = {
                    "avg_score_pct": round(avg_s*100, 1),
                    "avg_zeit_s": round(avg_z, 1),
                    "ergebnisse": ergebnisse_eval,
                }

                col1, col2 = st.columns(2)
                col1.metric("Ø Score", f"{avg_s*100:.0f}%")
                col2.metric("Ø Zeit", f"{avg_z:.1f}s")

                for e in ergebnisse_eval:
                    icon = "🟢" if e["score"] >= 0.75 else ("🟡" if e["score"] >= 0.5 else "🔴")
                    with st.expander(f"{icon} {e['score']*100:.0f}% – {e['frage'][:55]}… ({e['zeit_s']}s)"):
                        st.write(e["antwort"])

                if len(st.session_state.eval_ergebnisse) == 2:
                    st.success("✅ Beide Modelle evaluiert – Vergleich verfügbar!")
                    namen = list(st.session_state.eval_ergebnisse.keys())
                    c1, c2 = st.columns(2)
                    for col, name in zip([c1, c2], namen):
                        d = st.session_state.eval_ergebnisse[name]
                        col.metric(name, f"{d['avg_score_pct']}%", f"⏱ {d['avg_zeit_s']}s")


# ════════════════════════════════════════════════════════════════════════════
# TAB D – ÜBERSETZER
# ════════════════════════════════════════════════════════════════════════════
with tab_d:
    st.header("🌍 Modul D – Automatischer Übersetzer")

    col_info, col_status = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Ablauf:** Dokument hochladen → Text extrahieren → Sprache automatisch erkennen →
        Übersetzung (Marian NMT, lokal) → Original + Übersetzung im DMS speichern →
        Für RAG-Chatbot und Suche bereitstellen.
        """)
    with col_status:
        if MODUL_D_ECHT:
            st.success("✅ Echtes Modul D")
        else:
            st.warning("🔧 Stub-Modus\n\nEchtes Modul D\n(Marian NMT)\nfolgt")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Eingabe")
        hochgeladene_d = st.file_uploader("Dokument hochladen", type=["pdf","txt","docx"], key="upload_d")
        zielsprache = st.selectbox("Zielsprache:", ["Deutsch (de)", "Englisch (en)", "Französisch (fr)"])
        zielsprache_code = zielsprache.split("(")[1].replace(")", "")

    with col2:
        st.subheader("📤 Ausgabe")
        if "uebersetzung_result" in st.session_state:
            r = st.session_state.uebersetzung_result
            st.metric("Erkannte Sprache", r.get("ausgangssprache", "–"))
            st.metric("Übersetzt", "✅ Ja" if r.get("uebersetzt") else "⚠️ Stub (Original)")
            st.text_area("Übersetzung:", r.get("uebersetzung", "")[:1000], height=250)
        else:
            st.info("Noch keine Übersetzung.")

    if hochgeladene_d:
        if st.button("🌍 Übersetzen", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_pfad = os.path.join(tmp, hochgeladene_d.name)
                with open(tmp_pfad, "wb") as f:
                    f.write(hochgeladene_d.getbuffer())

                # Text extrahieren
                try:
                    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
                    endung = Path(hochgeladene_d.name).suffix.lower()
                    if endung == ".pdf":
                        docs = PyPDFLoader(tmp_pfad).load()
                    elif endung in (".docx",".doc"):
                        docs = Docx2txtLoader(tmp_pfad).load()
                    else:
                        docs = TextLoader(tmp_pfad, encoding="utf-8").load()
                    text = " ".join(d.page_content for d in docs)[:3000]
                except Exception as e:
                    text = f"Fehler beim Laden: {e}"

                with st.spinner("Übersetze..."):
                    if uebersetze:
                        result = uebersetze(text, zielsprache=zielsprache_code)
                    else:
                        result = {"originaltext": text, "uebersetzung": text,
                                  "ausgangssprache": "unbekannt", "zielsprache": zielsprache_code, "uebersetzt": False}

                st.session_state.uebersetzung_result = result

                if result.get("uebersetzt"):
                    st.success(f"✅ Übersetzt: {result['ausgangssprache']} → {zielsprache_code}")
                else:
                    st.warning("⚠️ Kein Übersetzer verfügbar – Original wird angezeigt. (Marian NMT installieren für echte Übersetzung)")

                # In RAG einpflegen
                if st.button("➡️ Übersetzung in RAG-Datenbank einpflegen"):
                    from langchain_core.documents import Document
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    from langchain_chroma import Chroma
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    lc_doc = Document(
                        page_content=result["uebersetzung"],
                        metadata={"dateiname": hochgeladene_d.name,
                                  "sprache": zielsprache_code, "quelle": "modul-d"}
                    )
                    chunks = splitter.split_documents([lc_doc])
                    for c in chunks:
                        c.page_content = "passage: " + c.page_content
                    Chroma.from_documents(documents=chunks, embedding=lade_embeddings(),
                                          persist_directory=CHROMA_PFAD, collection_name=COLLECTION)
                    st.success(f"✅ {len(chunks)} Chunks gespeichert!")