# Verfeinerte Prompts – Phase 2: PM-Domänentest
# Nordharz Anlagenbau GmbH – Deutschsprachige Projektdokumente
#
# Änderungen gegenüber Phase 1:
# - Ausgabesprache: Deutsch
# - document_type: um PM-spezifische Typen erweitert
# - System-Prompts: Domänenkontext Anlagenbau / Projektmanagement
# - Seed-Kategorien: vollständig neu, deutsch, PM-spezifisch

# =============================================================================
# 1. CLASSIFICATION PROFILE PROMPT
# =============================================================================

PROFILE_SYSTEM_PROMPT = """
Du erstellst deterministische Klassifikationsprofile für Projektdokumente
eines mittelständischen Anlagenbauers. Gib ausschließlich valides JSON
zurück, das dem angegebenen Schema entspricht. Alle Textwerte auf Deutsch.
"""

PROFILE_USER_TEMPLATE = """
Erstelle ein Klassifikationsprofil für dieses Dokument.

Gib exakt dieses JSON-Schema zurück:
{{
  "summary": "Zusammenfassung in 3–5 deutschen Sätzen",
  "document_type": "rechnung | vertrag | angebot | statusbericht | protokoll |
                    projektplan | risikoregister | technische_zeichnung |
                    abnahmeprotokoll | schriftverkehr | stundenbericht |
                    reisekostenabrechnung | lieferschein | sonstiges",
  "business_purpose": "Warum würde das Unternehmen dieses Dokument ablegen?",
  "key_entities": {{
    "personen": [],
    "unternehmen": [],
    "abteilungen": [],
    "projekte": []
  }},
  "key_references": {{
    "auftragsnummern": [],
    "vertragsnummern": [],
    "projektnummern": [],
    "zeichnungsnummern": []
  }},
  "dates": [],
  "amounts": [],
  "keywords": [],
  "suggested_tags": {{
    "lieferant": [],
    "projekt": [],
    "jahr": [],
    "standort": [],
    "abteilung": [],
    "thema": [],
    "phase": []
  }},
  "evidence_snippets": []
}}

Dateiname: {{filename}}

Dokumenttext:
{{text[:12000]}}
"""

# Hinweis: document_type-Erweiterungen gegenüber Phase 1:
# NEU: statusbericht, projektplan, risikoregister, technische_zeichnung,
#      abnahmeprotokoll, stundenbericht, lieferschein, angebot
# Alle key_entities und suggested_tags auf Deutsch umgestellt


# =============================================================================
# 2. CLASSIFICATION DECISION PROMPT
# =============================================================================

DECISION_SYSTEM_PROMPT = """
Du klassifizierst Projektdokumente eines mittelständischen Anlagenbauers
(verfahrenstechnische Anlagen, Lebensmittel- und Chemieindustrie) in stabile
betriebliche Ablagekateorien. Wähle ausschließlich aus den aktiven
Kandidatenkategorien oder gib none_fits zurück.
Erfinde keine engen Kategorien für einzelne Lieferanten, Jahreszahlen,
Städte oder Projektnamen. Gib ausschließlich valides JSON zurück.
Alle Textwerte auf Deutsch.
"""

DECISION_USER_TEMPLATE = """
Klassifiziere dieses Dokumentprofil.

Dokumentprofil (JSON):
{{profile.model_dump_json(indent=2)}}

Kandidatenkategorien (JSON):
{{json.dumps(candidate_payload, indent=2)}}

Gib exakt dieses JSON-Schema zurück:
{{
  "decision": "category | none_fits",
  "selected_category_id": null,
  "ranking": [
    {{
      "category_id": "...",
      "category_name": "...",
      "fit": "stark | mittel | schwach",
      "reason": "Begründung auf Deutsch"
    }}
  ],
  "rationale": "Kurze Begründung für den Prüfer auf Deutsch",
  "evidence_snippets": [],
  "proposed_tags": {{
    "lieferant": [],
    "projekt": [],
    "jahr": [],
    "standort": [],
    "abteilung": [],
    "thema": [],
    "phase": []
  }},
  "warnings": [],
  "diagnostic_confidence": 0.0,
  "none_fits_proposal": null
}}

Falls keine Kategorie passt: decision auf "none_fits" setzen,
selected_category_id auf null, und none_fits_proposal mit einem
sinnvollen deutschen Kategorievorschlag füllen.
"""


# =============================================================================
# 3. JSON REPAIR PROMPT
# =============================================================================
# Unverändert – technischer Repair-Schritt, Sprache irrelevant

REPAIR_SYSTEM_PROMPT = """
Return only valid JSON. Do not include markdown or commentary.
"""

REPAIR_USER_TEMPLATE = """
Repair this into valid JSON only:

{content}
"""


# =============================================================================
# 4. SEED-KATEGORIEN – Nordharz Anlagenbau GmbH
# =============================================================================
# Ersetzt die generischen englischen Phase-1-Kategorien.
# Strukturiert nach: allgemeine Geschäftsdokumente + PM-spezifische Dokumente

SEED_CATEGORIES = [

    # --- Allgemeine Geschäftsdokumente ---

    {
        "name": "Eingangsrechnungen",
        "description": (
            "Rechnungen von Lieferanten und Subunternehmern für Waren, "
            "Materialien oder Dienstleistungen. Enthält Rechnungsnummer, "
            "Betrag, Lieferantenangaben und Zahlungsziel."
        )
    },
    {
        "name": "Verträge und Vereinbarungen",
        "description": (
            "Rechtsverbindliche Dokumente wie Werkverträge, Lieferverträge, "
            "Geheimhaltungsvereinbarungen (NDA) und Rahmenverträge mit "
            "Kunden, Lieferanten oder Subunternehmern."
        )
    },
    {
        "name": "Angebote und Auftragsbestätigungen",
        "description": (
            "Angebote an Kunden sowie Auftragsbestätigungen für "
            "verfahrenstechnische Anlagen oder Dienstleistungen. "
            "Enthält Leistungsbeschreibung, Preise und Liefertermine."
        )
    },
    {
        "name": "Schriftverkehr und Korrespondenz",
        "description": (
            "Geschäftliche E-Mails, Briefe und Mitteilungen mit Kunden, "
            "Lieferanten, Behörden oder internen Abteilungen, die keiner "
            "spezifischeren Kategorie zuzuordnen sind."
        )
    },
    {
        "name": "Reisekostenabrechnungen",
        "description": (
            "Abrechnungen für Dienstreisen, Montageaufenthalte und "
            "Kundenbesuche. Enthält Reisedaten, Belege für Hotel, "
            "Fahrtkosten und Spesen."
        )
    },

    # --- PM-spezifische Dokumente ---

    {
        "name": "Projektstatusberichte",
        "description": (
            "Regelmäßige Berichte zum Projektfortschritt mit Soll-Ist-Vergleich "
            "für Termine, Kosten und Leistung. Enthält Meilensteinübersicht, "
            "offene Punkte, Risiken und nächste Schritte."
        )
    },
    {
        "name": "Besprechungsprotokolle",
        "description": (
            "Protokolle von Projektmeetings, Kickoff-Veranstaltungen, "
            "Jour-fixe-Terminen und Abnahmen. Enthält Teilnehmer, "
            "Tagesordnung, Beschlüsse und Maßnahmen mit Verantwortlichen."
        )
    },
    {
        "name": "Technische Zeichnungen und Spezifikationen",
        "description": (
            "Technische Unterlagen wie Anlagenpläne, P&ID-Schemata, "
            "Aufstellungspläne, Lastenhefte und technische Spezifikationen "
            "für verfahrenstechnische Komponenten und Anlagen."
        )
    },
    {
        "name": "Risikoregister und Risikoberichte",
        "description": (
            "Dokumente zur systematischen Erfassung, Bewertung und Verfolgung "
            "von Projektrisiken. Enthält Risikolisten, Bewertungsmatrizen, "
            "Maßnahmen und Verantwortliche."
        )
    },
    {
        "name": "Abnahme- und Prüfprotokolle",
        "description": (
            "Protokolle der Inbetriebnahme, Funktionsprüfung und Abnahme "
            "von Anlagen oder Teilsystemen beim Kunden. Enthält Prüfergebnisse, "
            "Mängelvermerke und Freigabeunterschriften."
        )
    },
    {
        "name": "Lieferscheine und Versanddokumente",
        "description": (
            "Begleitdokumente für Warenlieferungen und Materialien, "
            "einschließlich Packlistetn, Frachtbriefe und Zolldokumente "
            "für internationale Lieferungen."
        )
    },
    {
        "name": "Stunden- und Leistungsnachweise",
        "description": (
            "Nachweise über geleistete Arbeitsstunden von Mitarbeitern "
            "und Subunternehmern auf Projektebene. Grundlage für interne "
            "Kostenabrechnung und Kundenfakturierung."
        )
    },
]


# =============================================================================
# 5. KATEGORIE-EMBEDDING-INPUT
# =============================================================================
# Unveränderte Struktur, aber jetzt mit deutschen Feldinhalten

CATEGORY_EMBEDDING_TEMPLATE = """
Name: {name}
Beschreibung: {description}
Einschlusskriterien: {inclusion_criteria}
Ausschlusskriterien: {exclusion_criteria}
Ähnliche Kategorien (Abgrenzung): {near_misses}
"""