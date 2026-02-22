# TrustPfand Dashboard

KI-gestütztes Dashboard zur automatischen Pfandflaschen-Erkennung mit YOLOv8.

## Voraussetzungen

- Python 3.11
- YOLOv8-Modell (`best.pt`) im gleichen Ordner wie `dashboard.py`

## Installation

```bash
# Virtuelles Environment erstellen
python3.11 -m venv .venv

# Aktivieren (macOS/Linux)
source .venv/bin/activate

# Abhängigkeiten installieren
pip install streamlit ultralytics opencv-python numpy
```

## Starten

```bash
streamlit run dashboard.py
```

Das Dashboard öffnet sich automatisch im Browser unter `http://localhost:8501`.
