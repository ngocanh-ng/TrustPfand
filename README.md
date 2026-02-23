# TrustPfand Dashboard

KI-gestütztes Dashboard zur automatischen Pfandflaschen-Erkennung mit YOLOv8.

## Voraussetzungen

- Python 3.11
- YOLOv8-Modell (`best.pt`) im gleichen Ordner wie `dashboard.py`

## Start der Anwendung

```bash
# Virtuelles Environment erstellen
python3.11 -m venv .venv

# Aktivieren (macOS/Linux)
source .venv/bin/activate

# Aktivieren (Windows)
.venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Dashboard starten
streamlit run dashboard.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8501`.

## Bedienung

**Modus wählen** (Sidebar):
- **Live-Webcam** – Kamera per „Start" aktivieren, Flasche vor die Kamera halten, Erkennung wird live angezeigt. Mit „Zum Gesamt hinzufügen" den erkannten Pfand übernehmen, danach „Stop".
- **Bild hochladen** – JPG/PNG-Datei hochladen, Erkennung erfolgt automatisch, Pfand ebenfalls per Button übernehmen.

**Konfidenz-Schwelle** (Sidebar-Slider): Bestimmt, wie sicher das Modell sein muss, bevor es eine Flasche zählt (Standard: 50 %).

**Zurücksetzen** (Sidebar): Löscht alle erfassten Artikel und setzt den Gesamtbetrag auf 0 €.

## Wichtige Hinweise

- Die Datei `best.pt` muss zwingend im selben Ordner wie `dashboard.py` liegen, sonst startet das Modell nicht.
- Kamerazugriff muss erlaubt sein: **Systemeinstellungen → Datenschutz & Sicherheit → Kamera**.
- Das Modell erkennt folgende Kategorien: `PET_Einweg` (0,25 €), `PET_Mehrweg` (0,15 €), `Dose` (0,25 €), `Glas` (0,08 €), `Glas_Buegel` (0,15 €), `Kein_Pfand` (0,00 €).
