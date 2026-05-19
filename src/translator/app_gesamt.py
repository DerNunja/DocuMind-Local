# app_gesamt.py
import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

import streamlit as st
from service import TranslationService

@st.cache_resource
def load_service():
    return TranslationService()

service = load_service()

st.set_page_config(
    page_title="Modul D – Übersetzungsmodul",
    layout="centered"
)

st.title("🌍 Modul D – Technisches Übersetzungsmodul")
st.write("Lokale KI-Übersetzung auf deiner RTX 4070")

st.markdown("---")

# ENFIN LE BON MENU DÉROULANT : DE ➔ EN ou DE ➔ FR !
sprach_option = st.selectbox(
    "Bitte Übersetzungmodus wählen / Choisir le mode de traduction :",
    options=["Deutsch ➔ Englisch (DE ➔ EN)", "Deutsch ➔ Französisch (DE ➔ FR)"]
)

# Configuration selon le choix
if "Englisch" in sprach_option:
    aktueller_modus = "de-en"
    ziel_sprache_label = "🇬🇧 Übersetzung (Englisch):"
else:
    aktueller_modus = "de-fr"
    ziel_sprache_label = "🇫🇷 Übersetzung (Französisch):"

# Zone de texte unique (toujours en Allemand au départ !)
eingabe_text = st.text_area(
    "Deutschen technischen Text eingeben:",
    height=150,
    placeholder="Hier den deutschen Text eingeben (z.B. über Flansche, Ventile, Pumpen)..."
)

if st.button("Übersetzen / Traduire", type="primary"):
    if eingabe_text.strip():
        with st.spinner("Modell übersetzt auf GPU (RTX 4070)..."):
            ergebnis = service.übersetze_text(eingabe_text, modus=aktueller_modus)

        st.subheader(ziel_sprache_label)
        st.success(ergebnis["translated"])

        st.subheader("📦 JSON-Ausgabe:")
        st.json(ergebnis)
    else:
        st.warning("Bitte gib zuerst einen Text ein.")
