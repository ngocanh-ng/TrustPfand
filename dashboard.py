"""
TrustPfand Dashboard
=========================
KI-Projekt: TrustPfand - Automatische Pfandflaschen-Erkennung mit YOLOv8

Starten mit: streamlit run dashboard.py
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from collections import Counter
import threading

# ===== KONFIGURATION =====
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

PFAND_WERTE = {
    "PET_Einweg": 0.25,
    "PET_Mehrweg": 0.15,
    "Dose": 0.25,
    "Glas": 0.08,
    "Glas_Buegel": 0.15,
    "Kein_Pfand": 0.00
}

ANZEIGE_NAMEN = {
    "PET_Einweg": "PET Einweg",
    "PET_Mehrweg": "PET Mehrweg",
    "Dose": "Dose",
    "Glas": "Glas",
    "Glas_Buegel": "Glas mit Bügelverschluss",
    "Kein_Pfand": "Kein Pfand"
}

# ===== SEITEN-KONFIGURATION =====
st.set_page_config(
    page_title="TrustPfand",
    layout="wide"
)

# ===== SESSION STATE =====
if 'total_pfand' not in st.session_state:
    st.session_state.total_pfand = 0.0
if 'artikel_liste' not in st.session_state:
    st.session_state.artikel_liste = []
if 'live_active' not in st.session_state:
    st.session_state.live_active = False
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = []
if 'last_pfand' not in st.session_state:
    st.session_state.last_pfand = 0.0

# ===== MODELL LADEN =====
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        return None

class CameraThread:
    """Captures frames in a background thread so the latest frame is always ready."""

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.cap.release()


@st.cache_resource
def get_camera():
    cam = CameraThread(0)
    if cam.isOpened():
        cam.start()
    return cam

def release_camera():
    try:
        cam = get_camera()
        cam.release()
    except Exception:
        pass
    get_camera.clear()

model = load_model()

# ===== LIVE-KAMERA FRAGMENT (aktualisiert nur den Videobereich) =====
@st.fragment(run_every=0.15)
def live_camera_feed(confidence):
    if not st.session_state.live_active:
        return

    cam = get_camera()
    if not cam.isOpened():
        st.error(
            "Kamera nicht verfügbar. Prüfe:\n"
            "- Ist eine Webcam angeschlossen?\n"
            "- Hat die App Kamera-Zugriff? "
            "(Systemeinstellungen → Datenschutz & Sicherheit → Kamera)"
        )
        release_camera()
        return

    ret, frame = cam.read()
    if not ret or frame is None:
        st.warning("Kein Bild von der Kamera empfangen – wird erneut versucht...")
        return

    annotated_img, detections, total_pfand = detect_and_annotate(frame, confidence, imgsz=320)

    st.session_state.last_detections = detections
    st.session_state.last_pfand = total_pfand

    _, jpg = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    st.image(jpg.tobytes(), use_container_width=True)

    if detections:
        for k, ko, p in detections:
            st.markdown(f"**{ANZEIGE_NAMEN.get(k, k)}** {ko:.0%} — {p:.2f}€")
        st.caption(f"Erkannter Pfand: **{total_pfand:.2f} €**")
        if st.button("Zum Gesamt hinzufügen", type="primary", use_container_width=True, key="add_live_fragment"):
            for klasse, _, pfand in detections:
                st.session_state.artikel_liste.append(klasse)
                st.session_state.total_pfand += pfand
            st.session_state.last_detections = []
            st.session_state.last_pfand = 0.0
            st.rerun(scope="app")
    else:
        st.caption("Halte eine Flasche vor die Kamera")

# ===== HILFSFUNKTION: YOLO-Erkennung auf Bild =====
def detect_and_annotate(img, conf_threshold, imgsz=640):
    results = model(img, conf=conf_threshold, imgsz=imgsz, verbose=False)[0]

    detections = []
    total_pfand = 0.0

    for box in results.boxes:
        class_name = model.names[int(box.cls)]
        conf = float(box.conf)
        pfand = PFAND_WERTE.get(class_name, 0)
        detections.append((class_name, conf, pfand))
        total_pfand += pfand

    annotated_img = results.plot()

    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
    annotated_img = cv2.addWeighted(overlay, 0.7, annotated_img, 0.3, 0)
    cv2.putText(annotated_img, f"Pfand: {total_pfand:.2f} EUR", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return annotated_img, detections, total_pfand

# ===== HEADER =====
st.title("TrustPfand Dashboard")
st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("Steuerung")

    # Modell-Status
    if model:
        st.success("Modell geladen")
    else:
        st.error("Modell nicht gefunden")
        st.info("Lege 'best.pt' in den gleichen Ordner!")

    st.divider()

    # Modus-Auswahl
    mode = st.radio(
        "Eingabemodus",
        ["Live-Webcam", "Bild hochladen"],
        index=0
    )

    st.divider()

    # Konfidenz
    confidence = st.slider("Konfidenz-Schwelle", 0.1, 1.0, 0.5, 0.05)

    st.divider()

    # Reset
    if st.button("Zurücksetzen", width="stretch"):
        st.session_state.total_pfand = 0.0
        st.session_state.artikel_liste = []
        st.session_state.last_detections = []
        st.session_state.last_pfand = 0.0
        st.rerun()

    st.divider()

    # Pfandwerte
    st.subheader("Pfandwerte")
    for klasse, wert in PFAND_WERTE.items():
        st.caption(f"{ANZEIGE_NAMEN.get(klasse, klasse)}: {wert:.2f} €")

# ===== HAUPTBEREICH =====
col_cam, col_stats = st.columns([2, 1])

with col_cam:
    if mode == "Live-Webcam":
        st.subheader("Live-Kamera")

        if model:
            # Start/Stop Buttons
            col_start, col_stop = st.columns(2)
            with col_start:
                start = st.button("Start", type="primary", width="stretch")
            with col_stop:
                stop = st.button("Stop", width="stretch")

            if start:
                st.session_state.live_active = True
            if stop:
                st.session_state.live_active = False
                release_camera()

            if st.session_state.live_active:
                live_camera_feed(confidence)
            else:
                if st.session_state.last_detections:
                    st.subheader("Letzte Erkennung")
                    for klasse, konf, pfand in st.session_state.last_detections:
                        st.markdown(f"- **{ANZEIGE_NAMEN.get(klasse, klasse)}**: {konf:.0%} → {pfand:.2f} €")
                    st.metric("Erkannter Pfand", f"{st.session_state.last_pfand:.2f} €")
                    if st.button("Zum Gesamt hinzufügen", type="primary", width="stretch", key="add_after_stop"):
                        for klasse, _, pfand in st.session_state.last_detections:
                            st.session_state.artikel_liste.append(klasse)
                            st.session_state.total_pfand += pfand
                        st.session_state.last_detections = []
                        st.session_state.last_pfand = 0.0
                        st.rerun()
                else:
                    st.info("Drücke Start, um die Kamera zu aktivieren")
        else:
            st.error("Modell nicht geladen")

    else:  # Bild-Upload Modus
        st.subheader("Bild hochladen")

        if model:
            uploaded_file = st.file_uploader(
                "Wähle ein Bild aus",
                type=["jpg", "jpeg", "png"],
            )

            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                annotated_img, detections, total_pfand = detect_and_annotate(img, confidence)

                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        use_container_width=True)

                st.subheader("Erkennungen")
                if detections:
                    for klasse, konf, pfand in detections:
                        st.markdown(f"- **{ANZEIGE_NAMEN.get(klasse, klasse)}**: {konf:.0%} → {pfand:.2f} €")

                    st.metric("Pfand in diesem Bild", f"{total_pfand:.2f} €")

                    if st.button("Zum Gesamt hinzufügen", type="primary", width="stretch"):
                        for klasse, _, pfand in detections:
                            st.session_state.artikel_liste.append(klasse)
                            st.session_state.total_pfand += pfand
                        st.success(f"+{total_pfand:.2f} € hinzugefügt!")
                        st.rerun()
                else:
                    st.info("Keine Pfandflaschen erkannt")
            else:
                st.info("Lade ein Bild hoch")
        else:
            st.error("Modell nicht geladen")

with col_stats:
    st.subheader("Statistiken")

    # Gesamt-Anzeige
    st.metric(
        label="Gesamt-Pfand",
        value=f"{st.session_state.total_pfand:.2f} €"
    )

    st.metric(
        label="Anzahl Artikel",
        value=len(st.session_state.artikel_liste)
    )

    st.divider()

    # Artikel pro Klasse
    st.markdown("**Erfasste Artikel:**")
    if st.session_state.artikel_liste:
        counts = Counter(st.session_state.artikel_liste)
        for klasse, anzahl in counts.most_common():
            wert = PFAND_WERTE.get(klasse, 0) * anzahl
            col1, col2 = st.columns([2, 1])
            col1.write(f"{ANZEIGE_NAMEN.get(klasse, klasse)}")
            col2.write(f"**{anzahl}x** ({wert:.2f}€)")
    else:
        st.info("Noch keine Artikel erfasst")

# ===== FOOTER =====
st.divider()
st.caption("Projekt: TrustPfand - Pfand-Erkennung mit YOLOv8")