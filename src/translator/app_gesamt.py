import streamlit as st
from service import TranslationService

# Initialisation du service
service = TranslationService()

# Configuration de la page
st.set_page_config(
    page_title="Modul D – Übersetzungsmodul",
    layout="centered"
)

st.title("🌍 Modul D – Lokales Übersetzungsmodul")
st.write("Deutsch → Englisch (lokale KI-Übersetzung)")

# Zone de texte
eingabe_text = st.text_area(
    "Deutschen Text eingeben:",
    height=200
)

# Bouton
if st.button("Übersetzen"):
    if eingabe_text.strip():
        with st.spinner("Übersetzung läuft..."):
            ergebnis = service.übersetze_text(eingabe_text)

        st.subheader("🇬🇧 Übersetzung")
        st.success(ergebnis["translated"])

        st.subheader("📦 JSON-Ausgabe")
        st.json(ergebnis)

    else:
        st.warning("Bitte Text eingeben.")