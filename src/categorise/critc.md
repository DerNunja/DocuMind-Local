# Kritik und Korrekturen zur Prototyp-Dokumentation

Die Dokumentation ist inhaltlich nah am aktuellen Prototyp, enthält aber einige Punkte, die nicht mehr zum implementierten Stand passen oder zu stark formuliert sind.

## Wichtige Korrekturen

- "Ohne Datenbankintegration" stimmt nicht mehr. Der aktuelle Prototyp nutzt PostgreSQL + pgvector.
- Es gibt keine separate Vektordatenbank. Die Vektorsuche läuft in PostgreSQL über `pgvector`.
- Es gibt aktuell keine automatische Zuweisung bei `confidence >= 0,85`. Alle erfolgreichen Klassifikationen landen als Vorschlag in `needs_review`.
- Die Schwellenwerte `0,85` und `0,60` wurden nicht empirisch implementiert oder validiert. Sie waren Teil des ursprünglichen Plans, aber nicht des aktuellen Prototyps.
- Der Prototyp verarbeitet aktuell `.txt`-Dateien mit bereits extrahiertem Text. PDF/DOCX/OCR sind nicht Teil dieses Moduls.
- "Upload eines Dokuments" ist zu stark formuliert. Besser: "Übergabe einer Textdatei".
- "Finale Kategoriezuweisung" ist falsch. Besser: "Kategorieempfehlung".
- "Validiert" ist etwas stark. Besser: "getestet" oder "prototypisch erprobt".

## Was ergänzt werden sollte

- `seed-categories` CLI-Befehl für initiale Kategorien.
- `categorise-folder` CLI-Befehl für Batch-Tests.
- Speicherung in Tabellen:
  - `categories`
  - `category_embeddings`
  - `documents`
- Voraussetzung: PostgreSQL mit pgvector, z. B. `pgvector/pgvector:pg16`.
- LM Studio wird über eine OpenAI-kompatible lokale API angesprochen.
- Fehler bei malformiertem JSON werden teilweise durch Reparatur/robusteres Parsing abgefangen.

## Korrigierter Architekturabschnitt

```latex
Nach der Übergabe einer bereits extrahierten Textdatei erzeugt ein lokal betriebenes Sprachmodell zunächst ein strukturiertes Klassifikationsprofil mit Zusammenfassung, Dokumenttyp, relevanten Entitäten, Referenzen, Schlagwörtern und vorgeschlagenen Tags. Aus diesem Profil wird mit einem lokalen Embedding-Modell ein Vektor erzeugt. Die Kategorie-Embeddings werden in PostgreSQL mit pgvector gespeichert und dort per Vektorähnlichkeit durchsucht. Die ähnlichsten Kandidaten werden anschließend zusammen mit dem Dokumentprofil erneut dem Sprachmodell vorgelegt. Das Ergebnis ist keine finale automatische Ablage, sondern eine Kategorieempfehlung mit Begründung, Evidenzstellen und vorgeschlagenen Tags. Der Dokumentstatus wird anschließend auf \texttt{needs\_review} gesetzt.
```

## Korrigierter Abschnitt zur Entscheidungslogik

```latex
\paragraph{Review-basierte Entscheidungslogik}
Im aktuellen Prototyp wird keine automatische finale Kategoriezuweisung vorgenommen. Auch bei hoher modellseitiger Sicherheit wird das Ergebnis zunächst als Vorschlag gespeichert und mit dem Status \texttt{needs\_review} markiert. Der vom Sprachmodell ausgegebene Konfidenzwert wird lediglich als diagnostischer Hinweis betrachtet, da er keine kalibrierte statistische Sicherheit darstellt. Eine spätere Automatisierung soll erst nach Auswertung bestätigter Klassifikationen und kategoriespezifischer Qualitätsmetriken erfolgen.
```

## Korrigierter Abschnitt zu den Testergebnissen

```latex
Der Prototyp wurde mit 28 Beispieltexten aus acht allgemeinen Kategorien getestet. Die Kategorien wurden initial über den CLI-Befehl \texttt{seed-categories} angelegt; anschließend konnten einzelne Dateien oder ganze Ordner mit \texttt{categorise} bzw. \texttt{categorise-folder} verarbeitet werden. Die Ergebnisse wurden in PostgreSQL gespeichert. Bei den Testdokumenten wurden plausible Kategorieempfehlungen erzeugt. Einschränkungen zeigten sich bei sehr kurzen Eingaben sowie bei gelegentlich nicht schema-konformem JSON-Output des Sprachmodells, der durch robusteres Parsing teilweise abgefangen wurde.
```

## Korrektur für Phase-2-Punkt

Der Punkt:

```latex
\item LLM vorgeschlagene Kategorien hinzufügen können
```

sollte ersetzt werden durch:

```latex
\item Workflow zur Übernahme und Freigabe LLM-vorgeschlagener neuer Kategorien
```

Grund: Das Konzept für vorgeschlagene Kategorien ist im Output bereits vorgesehen, aber ein vollständiger Workflow zur Übernahme, Prüfung und Aktivierung ist noch nicht implementiert.
