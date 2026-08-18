import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import time
import queue
import numpy as np
import cv2
import mediapipe as mp
import pyttsx3

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl, QSize
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QMovie
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QCheckBox, QStackedWidget,
    QGraphicsDropShadowEffect, QSizePolicy, QDialog
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==========================================
# CUSTOM ASSETS & AUDIO CONFIGURATION
# ==========================================
ALERT_AUDIO_PATH = "alert.ogg"
WARNING_AUDIO_PATH = "warning.ogg"
ILLUSTRATION_IMAGE_PATH = "blink_illustration.gif"


# ==========================================
# COLOR PALETTE COMBOS
# ==========================================
ACTIVE_COMBO = 1

COMBOS = {
    1: {
        "primary": "#4351FC",     # Royal Blue (Headers, Primary Buttons, Line Charts)
        "accent": "#E83EA8",      # Vibrant Pink (Active Badges, Warnings, Secondary Highlights)
        "bg_light": "#F2F4FF",    # Light Tint
    },
    2: {
        "primary": "#837EF2",     # Periwinkle (Headers, Primary Buttons, Line Charts)
        "accent": "#B468C3",      # Lavender (Active Badges, Warnings, Secondary Highlights)
        "bg_light": "#F7F4FF",    # Light Tint
    }
}

SELECTED_THEME = COMBOS.get(ACTIVE_COMBO, COMBOS[1])

COLOR_PRIMARY = SELECTED_THEME["primary"]
COLOR_ACCENT  = SELECTED_THEME["accent"]
BG_PRIMARY    = SELECTED_THEME["bg_light"]

NAVY_OUTLINE = "#1A1C23"
WHITE        = "#FFFFFF"

# Fonts
FONT_HEADING = "'Fredoka', 'Sniglet', sans-serif"
FONT_BODY    = "'Nunito', 'Quicksand', sans-serif"


# ==========================================
# CONSTANTS & METRICS
# ==========================================
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 3
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

INITIAL_BLINK_THRESHOLD = 10
MAX_BLINK_THRESHOLD = 30

BLINK_STAGE_DURATION = 60  # Each stage lasts 60 seconds
MAX_BLINK_STAGES = 3        # 60 -> 120 -> 180 seconds

INACTIVITY_WARN_SEC = 300
INACTIVITY_SHUTDOWN_SEC = 600


# ==========================================
# ASYNCHRONOUS TTS WORKER
# ==========================================
class TTSWorker(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = queue.Queue()
        self.running = True

    def run(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        
        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
                if text:
                    engine.say(text)
                    engine.runAndWait()
                    self.queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text: str):
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# EYE TRACKING WORKER
# ==========================================
class EyeTrackingWorker(QThread):
    frame_processed = Signal(QImage)
    metrics_updated = Signal(float, int, int, int, int)
    alert_triggered = Signal(str, str)
    inactivity_warned = Signal(str)
    auto_shutdown_signal = Signal()

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)

        self.camera_index = camera_index
        self.running = False
        self.show_camera = True

        # Session Statistics
        self.total_blinks = 0
        self.minute_blink_history = []
        self.current_minute_blinks = 0

        # Adaptive Blink Monitoring
        self.cycle_blinks = 0
        self.blink_stage = 1
        self.blink_threshold = INITIAL_BLINK_THRESHOLD

        self.cycle_start_time = None
        self.minute_start_time = None

    def reset_blink_cycle(self):
        self.cycle_blinks = 0
        self.blink_stage = 1
        self.blink_threshold = INITIAL_BLINK_THRESHOLD
        self.cycle_start_time = time.time()

    def calculate_ear(self, eye_indices, landmarks):
        try:
            p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
            p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
            p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
            p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h = np.linalg.norm(p1 - p4)

            if h == 0:
                return 0.0

            return (v1 + v2) / (2.0 * h)

        except Exception:
            return 0.0

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.alert_triggered.emit(
                "Camera Error",
                "Unable to open video capture device."
            )
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.running = True
        frame_counter = 0
        session_start_time = time.time()

        self.reset_blink_cycle()
        self.minute_start_time = time.time()

        face_lost_start_time = None
        warned_5min_inactivity = False

        while self.running:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_ear = 0.0
            face_detected = (results.multi_face_landmarks is not None)
            now = time.time()

            # Face lost monitoring
            if not face_detected:
                if face_lost_start_time is None:
                    face_lost_start_time = now
                else:
                    absent_duration = now - face_lost_start_time

                    if absent_duration >= INACTIVITY_WARN_SEC and not warned_5min_inactivity:
                        self.inactivity_warned.emit(
                            "မျက်နှာလေးကို ၅ မိနစ်လောက် "
                            "မတွေ့ရသေးဘူးနော်။ "
                            "ကင်မရာရှေ့မှာ ရှိနေသေးလား "
                            "စစ်ကြည့်ပေးပါ။ "
                            "မသုံးတော့ဘူးဆိုရင် ခဏပိတ်ထားလို့ရတယ်နော်။ 😊"
                        )
                        warned_5min_inactivity = True

                    if absent_duration >= INACTIVITY_SHUTDOWN_SEC:
                        self.auto_shutdown_signal.emit()
                        break
            else:
                face_lost_start_time = None
                warned_5min_inactivity = False

                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = self.calculate_ear(LEFT_EYE_INDICES, landmarks)
                right_ear = self.calculate_ear(RIGHT_EYE_INDICES, landmarks)
                current_ear = (left_ear + right_ear) / 2.0

                if current_ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSEC_FRAMES:
                        self.total_blinks += 1
                        self.current_minute_blinks += 1
                        self.cycle_blinks += 1
                    frame_counter = 0

            # Minute statistics updates
            elapsed_minute = now - self.minute_start_time
            if elapsed_minute >= 60.0:
                self.minute_blink_history.append(self.current_minute_blinks)
                self.current_minute_blinks = 0
                self.minute_start_time = now

            # Adaptive Stage evaluation
            elapsed_cycle = now - self.cycle_start_time

            if self.blink_stage == 1 and elapsed_cycle >= 60.0:
                if self.cycle_blinks >= 10:
                    self.reset_blink_cycle()
                else:
                    self.blink_stage = 2
                    self.blink_threshold = 20

            elif self.blink_stage == 2 and elapsed_cycle >= 120.0:
                if self.cycle_blinks >= 20:
                    self.reset_blink_cycle()
                else:
                    self.blink_stage = 3
                    self.blink_threshold = 30

            elif self.blink_stage == 3 and elapsed_cycle >= 180.0:
                if self.cycle_blinks >= 30:
                    self.reset_blink_cycle()
                else:
                    self.alert_triggered.emit(
                        "👀 မျက်တောင်လေး ခတ်ဖို့ မမေ့နဲ့နော်!",
                        "မျက်တောင်ခတ်ဖို့ အချိန်ရောက်ပြီနော်! "
                        "ခဏလေး အနားယူလိုက်ရအောင်။ 😊"
                    )
                    self.reset_blink_cycle()

            session_duration = int(now - session_start_time)
            self.metrics_updated.emit(
                current_ear,
                self.total_blinks,
                self.current_minute_blinks,
                self.blink_stage,
                session_duration
            )

            # Frame processing
            if self.show_camera:
                if face_detected:
                    cv2.putText(frame, f"EAR: {current_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Cycle: {self.cycle_blinks}/{self.blink_threshold}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Stage: {self.blink_stage}/3", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                bytes_per_line = 3 * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_processed.emit(qt_img)

            self.msleep(30)

        cap.release()
        face_mesh.close()

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# CUSTOM STYLED UI COMPONENTS
# ==========================================
def apply_brutalist_shadow(widget):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(0)
    shadow.setOffset(4, 4)
    shadow.setColor(QColor(NAVY_OUTLINE))
    widget.setGraphicsEffect(shadow)


class IllustrationAlertDialog(QDialog):
    def __init__(self, title, message, image_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setModal(True)
        self.resize(400, 340)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {WHITE};
                border: 4px solid {NAVY_OUTLINE};
                border-radius: 16px;
            }}
        """)
        apply_brutalist_shadow(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel(title.upper())
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 20px; font-weight: 900; font-family: {FONT_HEADING};")
        layout.addWidget(title_label)

        ill_frame = QFrame()
        ill_frame.setStyleSheet(f"background-color: transparent; border: 3px solid {NAVY_OUTLINE}; border-radius: 12px;")
        ill_layout = QVBoxLayout(ill_frame)
        ill_layout.setContentsMargins(15, 15, 15, 15)

        if image_path and os.path.exists(image_path):
            ill_label = QLabel()
            ill_label.setAlignment(Qt.AlignCenter)
            movie = QMovie(image_path)
            movie.setScaledSize(QSize(220, 160)) 
            ill_label.setMovie(movie)
            movie.start()
            self.movie = movie 
        else:
            ill_label = QLabel("👁️ ✨ 😌 ✨ 👁️\n\nTake a Blink Break!")
            ill_label.setAlignment(Qt.AlignCenter)
            ill_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 20px; font-weight: bold; font-family: {FONT_HEADING};")

        ill_layout.addWidget(ill_label)
        layout.addWidget(ill_frame, stretch=1)

        msg_label = QLabel(message)
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 15px; font-weight: bold; font-family: {FONT_BODY}; margin-top: 8px;")
        layout.addWidget(msg_label)

        ok_btn = QPushButton("Got it!")
        ok_btn.setFixedSize(130, 42)
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT}; color: {WHITE}; font-weight: 900;
                font-size: 15px; border-radius: 10px; border: 3px solid {NAVY_OUTLINE};
                font-family: {FONT_HEADING}; margin-top: 8px;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
        """)
        apply_brutalist_shadow(ok_btn)
        ok_btn.clicked.connect(self.accept)

        layout.addWidget(ok_btn, alignment=Qt.AlignCenter)


class StatCard(QFrame):
    def __init__(self, title, value="0", unit="", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            StatCard {{
                background-color: {WHITE};
                border-radius: 12px;
                border: 3px solid {NAVY_OUTLINE};
                padding: 10px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        apply_brutalist_shadow(self)

        layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title.upper())
        self.title_label.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: 12px; font-weight: 900; font-family: {FONT_HEADING};")
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 30px; font-weight: 900; font-family: {FONT_HEADING};")
        
        self.unit_label = QLabel(unit)
        self.unit_label.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 12px; font-family: {FONT_BODY}; font-weight: bold;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.unit_label)

    def set_value(self, val):
        self.value_label.setText(str(val))


class AnalyticsCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=BG_PRIMARY)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor(WHITE)
        super().__init__(fig)

    def plot_analytics(self, history):
        self.ax.clear()

        if not history:
            self.ax.text(
                0.5, 0.5,
                'No Minute Data Logged Yet',
                color=NAVY_OUTLINE,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.ax.transAxes,
                weight='bold'
            )
        else:
            minutes = list(range(1, len(history) + 1))

            self.ax.plot(
                minutes,
                history,
                color=COLOR_PRIMARY,
                marker='o',
                linewidth=3,
                markersize=8,
                label='Blinks / Minute'
            )

            self.ax.axhline(
                y=10,
                color=COLOR_ACCENT,
                linestyle='--',
                linewidth=2,
                label='Base Target (10/min)'
            )

            self.ax.set_xlabel('Minute', color=NAVY_OUTLINE, fontsize=10, weight='bold')
            self.ax.set_ylabel('Blink Count', color=NAVY_OUTLINE, fontsize=10, weight='bold')
            self.ax.legend(facecolor=WHITE, edgecolor=NAVY_OUTLINE, labelcolor=NAVY_OUTLINE)

        self.ax.tick_params(colors=NAVY_OUTLINE)
        for spine in self.ax.spines.values():
            spine.set_color(NAVY_OUTLINE)
            spine.set_linewidth(2)

        self.draw()


# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class EyeAwareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EyeAware - Eye Health & Fatigue Monitor")
        self.resize(1000, 750)

        self.tracking_worker = None
        self.tts_worker = TTSWorker()
        self.tts_worker.start()

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header Title Bar
        header = QFrame()
        header.setStyleSheet(f"background-color: {COLOR_PRIMARY}; border-bottom: 4px solid {NAVY_OUTLINE};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel("EYEAWARE MONITOR")
        title.setStyleSheet(f"color: {WHITE}; letter-spacing: 1px; font-family: {FONT_HEADING}; font-size: 20px; font-weight: 900;")
        
        self.status_badge = QLabel("IDLE")
        self.status_badge.setStyleSheet(f"""
            background-color: {WHITE}; color: {NAVY_OUTLINE}; 
            padding: 6px 14px; border-radius: 12px; font-weight: 900;
            border: 2px solid {NAVY_OUTLINE}; font-family: {FONT_HEADING};
        """)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        self.main_layout.addWidget(header)

        # Main Content Stack
        content_container = QWidget()
        self.content_layout = QVBoxLayout(content_container)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        
        self.stacked_widget = QStackedWidget()
        self.content_layout.addWidget(self.stacked_widget)
        self.main_layout.addWidget(content_container)

        self.setup_home_view()
        self.setup_dashboard_view()
        self.setup_analytics_view()

    def setup_home_view(self):
        home_widget = QWidget()
        layout = QVBoxLayout(home_widget)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel("👁️")
        icon_label.setFont(QFont("Segoe UI Emoji", 68))
        
        welcome_label = QLabel("Welcome to EyeAware")
        welcome_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 32px; font-weight: 900; font-family: {FONT_HEADING};")

        sub_label = QLabel("Real-time blink tracking and ergonomic fatigue prevention.")
        sub_label.setStyleSheet(f"color: {NAVY_OUTLINE}; margin-bottom: 30px; font-size: 16px; font-family: {FONT_BODY}; font-weight: bold;")

        start_btn = QPushButton("Start Monitoring Session")
        start_btn.setFixedSize(280, 56)
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: {WHITE}; font-weight: 900;
                font-size: 17px; border-radius: 12px; border: 3px solid {NAVY_OUTLINE};
                font-family: {FONT_HEADING};
            }}
            QPushButton:hover {{ opacity: 0.95; }}
        """)
        apply_brutalist_shadow(start_btn)
        start_btn.clicked.connect(self.start_session)

        layout.addWidget(icon_label, alignment=Qt.AlignCenter)
        layout.addWidget(welcome_label, alignment=Qt.AlignCenter)
        layout.addWidget(sub_label, alignment=Qt.AlignCenter)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        self.stacked_widget.addWidget(home_widget)

    def setup_dashboard_view(self):
        dash_widget = QWidget()
        layout = QVBoxLayout(dash_widget)

        # Stat Cards
        metrics_layout = QHBoxLayout()
        self.card_ear = StatCard("Current EAR", "0.00", "Eye Aspect Ratio")
        self.card_blinks = StatCard("Total Blinks", "0", "Session Count")
        self.card_bpm = StatCard("Current Min Blinks", "0", "Current 60-second window")
        self.card_time = StatCard("Duration", "00:00", "MM:SS Elapsed")

        metrics_layout.addWidget(self.card_ear)
        metrics_layout.addWidget(self.card_blinks)
        metrics_layout.addWidget(self.card_bpm)
        metrics_layout.addWidget(self.card_time)
        layout.addLayout(metrics_layout)

        # Camera Display Frame
        center_layout = QHBoxLayout()
        self.video_container = QFrame()
        self.video_container.setStyleSheet(f"background-color: {WHITE}; border-radius: 12px; border: 3px solid {NAVY_OUTLINE};")
        apply_brutalist_shadow(self.video_container)
        video_layout = QVBoxLayout(self.video_container)

        self.video_label = QLabel("Camera Feed Active")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 14px; font-weight: bold;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        video_layout.addWidget(self.video_label)

        center_layout.addWidget(self.video_container)
        layout.addLayout(center_layout, stretch=1)

        # Toolbar
        controls = QHBoxLayout()
        self.camera_toggle = QCheckBox("Show Camera Feed")
        self.camera_toggle.setChecked(True)
        self.camera_toggle.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 15px; font-weight: 900; font-family: {FONT_BODY};")
        self.camera_toggle.toggled.connect(self.toggle_camera_view)

        stop_btn = QPushButton("Stop Session")
        stop_btn.setFixedSize(160, 48)
        stop_btn.setCursor(Qt.PointingHandCursor)
        stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT}; color: {WHITE}; font-weight: 900;
                font-size: 16px; border-radius: 12px; border: 3px solid {NAVY_OUTLINE};
                font-family: {FONT_HEADING};
            }}
            QPushButton:hover {{ opacity: 0.95; }}
        """)
        apply_brutalist_shadow(stop_btn)
        stop_btn.clicked.connect(self.stop_session)

        controls.addWidget(self.camera_toggle)
        controls.addStretch()
        controls.addWidget(stop_btn)
        layout.addLayout(controls)

        self.stacked_widget.addWidget(dash_widget)

    def setup_analytics_view(self):
        analytics_widget = QWidget()
        layout = QVBoxLayout(analytics_widget)

        title = QLabel("Session Analytics & Summary")
        title.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 24px; font-weight: 900; font-family: {FONT_HEADING};")
        layout.addWidget(title)

        self.summary_label = QLabel("Total Blinks: 0 | Average BPM: 0.0")
        self.summary_label.setStyleSheet(f"color: {NAVY_OUTLINE}; font-size: 16px; font-weight: bold; margin-bottom: 10px; font-family: {FONT_BODY};")
        layout.addWidget(self.summary_label)

        chart_frame = QFrame()
        chart_frame.setStyleSheet(f"background-color: {WHITE}; border: 3px solid {NAVY_OUTLINE}; border-radius: 12px;")
        apply_brutalist_shadow(chart_frame)
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chart_canvas = AnalyticsCanvas(self)
        chart_layout.addWidget(self.chart_canvas)
        layout.addWidget(chart_frame, stretch=1)

        home_btn = QPushButton("Return to Home")
        home_btn.setFixedSize(180, 48)
        home_btn.setCursor(Qt.PointingHandCursor)
        home_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: {WHITE}; font-weight: 900;
                font-size: 16px; border-radius: 12px; border: 3px solid {NAVY_OUTLINE};
                font-family: {FONT_HEADING}; margin-top: 15px;
            }}
            QPushButton:hover {{ opacity: 0.95; }}
        """)
        apply_brutalist_shadow(home_btn)
        home_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        layout.addWidget(home_btn, alignment=Qt.AlignRight)

        self.stacked_widget.addWidget(analytics_widget)

    # ==========================================
    # LOGIC & EVENT HANDLERS
    # ==========================================
    def play_audio(self, audio_path: str, fallback_message: str):
        """Plays the designated OGG audio file if found, fallback to TTS otherwise."""
        if os.path.exists(audio_path):
            self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(audio_path)))
            self.audio_output.setVolume(1.0)
            self.media_player.play()
        else:
            self.tts_worker.speak(fallback_message)

    def start_session(self):
        self.stacked_widget.setCurrentIndex(1)
        self.status_badge.setText("ACTIVE")
        self.status_badge.setStyleSheet(f"""
            background-color: {COLOR_ACCENT}; color: {WHITE}; 
            padding: 6px 14px; border-radius: 12px; font-weight: 900;
            border: 2px solid {NAVY_OUTLINE}; font-family: {FONT_HEADING};
        """)

        self.tracking_worker = EyeTrackingWorker(camera_index=0)
        self.tracking_worker.frame_processed.connect(self.update_video_frame)
        self.tracking_worker.metrics_updated.connect(self.update_metrics)
        self.tracking_worker.alert_triggered.connect(self.handle_alert)
        self.tracking_worker.inactivity_warned.connect(self.handle_inactivity_warning)
        self.tracking_worker.auto_shutdown_signal.connect(self.handle_auto_shutdown)
        self.tracking_worker.start()

    def stop_session(self):
        if self.tracking_worker and self.tracking_worker.isRunning():
            history = list(self.tracking_worker.minute_blink_history)
            total_blinks = self.tracking_worker.total_blinks
            self.tracking_worker.stop()

            avg_bpm = np.mean(history) if history else 0.0
            self.summary_label.setText(
                f"Total Blinks: {total_blinks} | Monitored Minutes: {len(history)} | Average Rate: {avg_bpm:.1f} BPM"
            )
            self.chart_canvas.plot_analytics(history)

        self.status_badge.setText("IDLE")
        self.status_badge.setStyleSheet(f"""
            background-color: {WHITE}; color: {NAVY_OUTLINE}; 
            padding: 6px 14px; border-radius: 12px; font-weight: 900;
            border: 2px solid {NAVY_OUTLINE}; font-family: {FONT_HEADING};
        """)
        self.stacked_widget.setCurrentIndex(2)

    @Slot(QImage)
    def update_video_frame(self, qt_img):
        if self.camera_toggle.isChecked():
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

    @Slot(float, int, int, int, int)
    def update_metrics(self, ear, total_blinks, current_minute_blinks, stage, duration_sec):
        self.card_ear.set_value(f"{ear:.2f}")
        self.card_blinks.set_value(str(total_blinks))
        self.card_bpm.set_value(str(current_minute_blinks))

        mins, secs = divmod(duration_sec, 60)
        self.card_time.set_value(f"{mins:02d}:{secs:02d}")

    def toggle_camera_view(self, checked):
        if self.tracking_worker:
            self.tracking_worker.show_camera = checked
        if not checked:
            self.video_label.clear()
            self.video_label.setText("Camera Feed Hidden (Running silently in background)")

    @Slot(str, str)
    def handle_alert(self, title, message):
        # Plays alert.ogg for blink reminders
        self.play_audio(ALERT_AUDIO_PATH, message)
        popup = IllustrationAlertDialog(title, message, image_path=ILLUSTRATION_IMAGE_PATH, parent=self)
        popup.exec()

    @Slot(str)
    def handle_inactivity_warning(self, message):
        # Plays warning.ogg for 5-minute inactivity
        self.play_audio(WARNING_AUDIO_PATH, "Inactivity warning. No face detected for 5 minutes.")
        popup = IllustrationAlertDialog("သတိပေးချက်", message, image_path=ILLUSTRATION_IMAGE_PATH, parent=self)
        popup.exec()

    @Slot()
    def handle_auto_shutdown(self):
        # Plays warning.ogg for 10-minute auto shutdown
        self.play_audio(WARNING_AUDIO_PATH, "No face detected for 10 minutes. Automatically shutting down session.")
        self.stop_session()

    def apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {BG_PRIMARY}; }}
            QWidget {{ font-family: {FONT_BODY}; }}
        """)

    def closeEvent(self, event):
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
        if self.tts_worker and self.tts_worker.isRunning():
            self.tts_worker.stop()
        event.accept()


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeAwareApp()
    window.show()
    sys.exit(app.exec())