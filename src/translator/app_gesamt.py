# app_gesamt.py
import os
os.environ["HF_SKIP_CHECK_TORCH_LOAD_IN_SAFE"] = "True"

import streamlit as st
from service import TranslationService

# ---------------------------------------------------
# Load Translation Service only once
# ---------------------------------------------------

@st.cache_resource
def load_service():
    return TranslationService()

service = load_service()

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Modul D – Übersetzungsmodul",
    layout="centered"
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("🌍 Modul D – Technisches Übersetzungsmodul")

st.write(
    "Lokale KI-Übersetzung auf deiner RTX 4070 "
    "(Deutsch → Englisch / Französisch / Arabisch)"
)

st.markdown("---")

# ---------------------------------------------------
# Translation Mode Selection
# ---------------------------------------------------

sprach_option = st.selectbox(
    "Bitte Übersetzungsmodus wählen:",
    options=[
        "Deutsch ➔ Englisch (DE ➔ EN)",
        "Deutsch ➔ Französisch (DE ➔ FR)",
        "Deutsch ➔ Arabisch (DE ➔ AR)"
    ]
)

# ---------------------------------------------------
# Configure Translation Mode
# ---------------------------------------------------

if "Englisch" in sprach_option:

    aktueller_modus = "de-en"

    ziel_sprache_label = "🇬🇧 Übersetzung (Englisch):"

elif "Französisch" in sprach_option:

    aktueller_modus = "de-fr"

    ziel_sprache_label = "🇫🇷 Übersetzung (Französisch):"

else:

    aktueller_modus = "de-ar"

    ziel_sprache_label = "🇸🇦 Übersetzung (Arabisch):"

# ---------------------------------------------------
# Input Text Area
# ---------------------------------------------------

eingabe_text = st.text_area(
    "Deutschen technischen Text eingeben:",
    height=180,
    placeholder=(
        "Hier den deutschen technischen Text eingeben "
        "(z.B. über Flansche, Ventile, Pumpen, Wartungspläne)..."
    )
)

# ---------------------------------------------------
# Translation Button
# ---------------------------------------------------

if st.button("Übersetzen / Traduire / Translate", type="primary"):

    if eingabe_text.strip():

        with st.spinner("Modell übersetzt auf GPU (RTX 4070)..."):

            ergebnis = service.übersetze_text(
                eingabe_text,
                modus=aktueller_modus
            )

        # -------------------------------------------
        # Output Section
        # -------------------------------------------

        st.subheader(ziel_sprache_label)

        # Arabic needs RTL rendering
        if aktueller_modus == "de-ar":

            st.markdown(
                f"""
                <div dir="rtl"
                     style="
                        text-align: right;
                        font-size: 26px;
                        line-height: 2;
                        background-color: #0e1117;
                        padding: 20px;
                        border-radius: 10px;
                     ">
                    {ergebnis["translated"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.success(ergebnis["translated"])

        # -------------------------------------------
        # JSON Output
        # -------------------------------------------

        st.subheader("📦 JSON-Ausgabe:")

        st.json(ergebnis)

    else:

        st.warning("Bitte gib zuerst einen Text ein.")
