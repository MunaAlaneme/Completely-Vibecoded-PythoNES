'''
brew install uv
uv venv --python 3.15
source .venv/bin/activate
uv pip install scipy
python -c "import scipy; print(scipy.__version__)"
'''


import sys
import time
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import rubband

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QComboBox,
    QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer


# ---------------- CONFIGURATION ----------------

SR = 48000
CHANNELS = 2
BLOCK = 1024

# ---------------- AUDIO ENGINE -----------------

class DJAudioEngine:

    def __init__(self):
        self.reset_requested = threading.Event()
        self.output_queue = queue.Queue(maxsize=30)
        self.mic_queue = queue.Queue(maxsize=30)

        self.running = False
        self.playing = False

        self.source = "FILE"
        self.audio_file = None
        self.file_position = 0

        self.tempo = 1.0
        self.pitch = 0.0
        self.volume = 0.8

        self.lock = threading.Lock()

        self.stretcher = rubband.Stretcher(
            SR,
            CHANNELS,
            options=rubband.Options(
                process=rubband.ProcessOption.real_time
            )
        )

        self.output_stream = None
        self.input_stream = None
        self.worker = None

    def load_file(self, filename):

        audio, sr = sf.read(
            filename,
            dtype="float32",
            always_2d=True
        )

        # Convert to stereo
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        else:
            audio = audio[:, :2]

        # Resample to 48 kHz
        if sr != SR:
            gcd = np.gcd(sr, SR)
            audio = signal.resample_poly(
                audio,
                SR // gcd,
                sr // gcd,
                axis=0
            )

        with self.lock:
            self.audio_file = np.ascontiguousarray(
                audio, dtype=np.float32
            )
            self.file_position = 0
            self.source = "FILE"

        self.clear_queues()
        self.reset_requested.set()

        print("Loaded:", filename)

    def set_source(self, source):

        with self.lock:
            self.source = source
            self.file_position = 0

        self.clear_queues()
        self.reset_requested.set()

    def clear_queues(self):

        for q in (self.output_queue, self.mic_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def set_tempo(self, value):

        with self.lock:
            self.tempo = float(value)

    def set_pitch(self, value):

        with self.lock:
            self.pitch = float(value)

    def set_volume(self, value):

        with self.lock:
            self.volume = float(value)

    def microphone_callback(self, indata, frames, timing, status):

        if status:
            print("Microphone:", status)

        with self.lock:
            active = self.source == "MIC" and self.playing

        if active:
            block = np.asarray(
                indata[:, :1], dtype=np.float32
            )
            stereo = np.repeat(block, 2, axis=1)

            try:
                self.mic_queue.put_nowait(stereo.copy())
            except queue.Full:
                pass

    def output_callback(self, outdata, frames, timing, status):

        if status:
            print("Output:", status)

        outdata.fill(0)

        try:
            block = self.output_queue.get_nowait()
            count = min(frames, len(block))
            outdata[:count] = block[:count]

            # If a short block is received, leave the rest silent.

        except queue.Empty:
            pass

    def get_input_block(self):

        with self.lock:
            source = self.source
            playing = self.playing
            audio = self.audio_file

            if not playing:
                return None

            if source == "FILE":

                if audio is None:
                    return None

                start = self.file_position
                end = min(start + BLOCK, len(audio))

                if start >= len(audio):
                    self.playing = False
                    return None

                self.file_position = end
                block = audio[start:end]

                if len(block) < BLOCK:
                    block = np.pad(
                        block,
                        ((0, BLOCK - len(block)), (0, 0))
                    )

                return block.astype(np.float32, copy=False)

        try:
            return self.mic_queue.get(timeout=0.01)
        except queue.Empty:
            return None

    def processing_loop(self):
        if self.reset_requested.is_set():
            self.stretcher.reset()
        self.reset_requested.clear()

        while self.running:

            with self.lock:
                playing = self.playing
                tempo = self.tempo
                pitch = self.pitch
                volume = self.volume

            if not playing:
                time.sleep(0.01)
                continue

            block = self.get_input_block()

            if block is None:
                time.sleep(0.002)
                continue

            try:
                # Rubber Band time_ratio is output duration /
                # input duration. Faster tempo means shorter duration.
                self.stretcher.set_time_ratio(1.0 / tempo)

                # Semitone-to-frequency conversion
                pitch_scale = 2.0 ** (pitch / 12.0)
                self.stretcher.set_pitch_scale(pitch_scale)

                self.stretcher.process(
                    np.ascontiguousarray(block, dtype=np.float32),
                    final=False
                )

                available = self.stretcher.available()

                if available > 0:
                    result = self.stretcher.retrieve()
                    processed = np.array(
                        result.numpy(), dtype=np.float32, copy=True
                    )

                    processed *= volume

                    # Prevent clipping
                    np.clip(
                        processed, -1.0, 1.0,
                        out=processed
                    )

                    if len(processed):
                        try:
                            self.output_queue.put_nowait(processed)
                        except queue.Full:
                            # Drop processed block if output is overloaded
                            pass

            except Exception as error:
                print("DSP error:", error)
                time.sleep(0.01)

    def start(self):

        if self.running:
            return

        self.running = True

        self.output_stream = sd.OutputStream(
            samplerate=SR,
            channels=CHANNELS,
            blocksize=BLOCK,
            callback=self.output_callback
        )

        self.input_stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            blocksize=BLOCK,
            callback=self.microphone_callback
        )

        self.output_stream.start()
        self.input_stream.start()

        self.worker = threading.Thread(
            target=self.processing_loop,
            daemon=True
        )
        self.worker.start()

    def play(self):

        with self.lock:
            self.playing = True

    def pause(self):
        self.reset_requested.set()

        with self.lock:
            self.playing = False

        self.clear_queues()
        self.stretcher.reset()

    def stop(self):

        self.pause()

        with self.lock:
            self.file_position = 0

    def close(self):

        self.running = False
        self.playing = False

        if self.worker:
            self.worker.join(timeout=2)

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()


# ---------------- DJ MIXER GUI -----------------

class DJMixer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("PYTHON DJ MIXER")
        self.setMinimumSize(850, 680)

        self.engine = DJAudioEngine()

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #171923;
                color: white;
                font-size: 14px;
            }

            QGroupBox {
                border: 1px solid #41465a;
                border-radius: 12px;
                margin-top: 14px;
                padding: 15px;
                font-weight: bold;
            }

            QPushButton {
                background-color: #30364a;
                border: none;
                border-radius: 9px;
                padding: 12px;
                color: white;
            }

            QPushButton:hover {
                background-color: #505a78;
            }

            QSlider::groove:horizontal {
                height: 8px;
                background: #44495c;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #18d6b3;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }

            QComboBox {
                background: #30364a;
                padding: 9px;
                border-radius: 6px;
            }
        """)

        self.build_interface()

        try:
            self.engine.start()
        except Exception as error:
            QMessageBox.critical(
                self,
                "Audio Error",
                str(error)
            )

    def build_interface(self):

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setSpacing(18)

        title = QLabel("♫  PYTHON DJ MIXER")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #18d6b3;"
        )
        layout.addWidget(title)

        self.status = QLabel("READY")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

        # Source selection

        source_group = QGroupBox("AUDIO INPUT")
        source_layout = QHBoxLayout(source_group)

        self.source_box = QComboBox()
        self.source_box.addItems([
            "FILE",
            "MIC"
        ])

        self.source_box.currentTextChanged.connect(
            self.change_source
        )

        load_button = QPushButton("LOAD AUDIO FILE")
        load_button.clicked.connect(self.load_audio)

        source_layout.addWidget(self.source_box)
        source_layout.addWidget(load_button)

        layout.addWidget(source_group)

        # Playback controls

        transport = QGroupBox("TRANSPORT")
        transport_layout = QHBoxLayout(transport)

        play = QPushButton("▶ PLAY")
        pause = QPushButton("Ⅱ PAUSE")
        stop = QPushButton("■ STOP")

        play.clicked.connect(self.engine.play)
        pause.clicked.connect(self.engine.pause)
        stop.clicked.connect(self.engine.stop)

        transport_layout.addWidget(play)
        transport_layout.addWidget(pause)
        transport_layout.addWidget(stop)

        layout.addWidget(transport)

        # Tempo

        tempo_group = QGroupBox("TEMPO / SPEED")
        tempo_layout = QVBoxLayout(tempo_group)

        self.tempo_value = QLabel("1.00×")
        self.tempo_value.setAlignment(Qt.AlignCenter)
        self.tempo_value.setStyleSheet(
            "font-size: 30px; color: #18d6b3;"
        )

        self.tempo_slider = QSlider(Qt.Horizontal)
        self.tempo_slider.setRange(50, 200)
        self.tempo_slider.setValue(100)

        self.tempo_slider.valueChanged.connect(
            self.change_tempo
        )

        tempo_layout.addWidget(self.tempo_value)
        tempo_layout.addWidget(self.tempo_slider)

        tempo_labels = QHBoxLayout()
        tempo_labels.addWidget(QLabel("50%"))
        tempo_labels.addStretch()
        tempo_labels.addWidget(QLabel("200%"))

        tempo_layout.addLayout(tempo_labels)
        layout.addWidget(tempo_group)

        # Pitch

        pitch_group = QGroupBox("PITCH / KEY")
        pitch_layout = QVBoxLayout(pitch_group)

        self.pitch_value = QLabel("0 semitones")
        self.pitch_value.setAlignment(Qt.AlignCenter)
        self.pitch_value.setStyleSheet(
            "font-size: 30px; color: #ffb347;"
        )

        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)

        self.pitch_slider.valueChanged.connect(
            self.change_pitch
        )

        pitch_layout.addWidget(self.pitch_value)
        pitch_layout.addWidget(self.pitch_slider)

        pitch_labels = QHBoxLayout()
        pitch_labels.addWidget(QLabel("-12"))
        pitch_labels.addStretch()
        pitch_labels.addWidget(QLabel("0"))
        pitch_labels.addStretch()
        pitch_labels.addWidget(QLabel("+12"))

        pitch_layout.addLayout(pitch_labels)
        layout.addWidget(pitch_group)

        # Volume

        volume_group = QGroupBox("MASTER VOLUME")
        volume_layout = QVBoxLayout(volume_group)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)

        self.volume_slider.valueChanged.connect(
            self.change_volume
        )

        volume_layout.addWidget(self.volume_slider)
        layout.addWidget(volume_group)

        footer = QLabel(
            "48 kHz  |  Stereo  |  Rubber Band DSP"
        )
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #888da5;")

        layout.addWidget(footer)

    # ---------------- GUI CALLBACKS ----------------

    def load_audio(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aiff *.aif)"
        )

        if filename:
            try:
                self.engine.load_file(filename)
                self.source_box.setCurrentText("FILE")
                self.status.setText(
                    "LOADED: " + filename.split("/")[-1]
                )
            except Exception as error:
                QMessageBox.critical(
                    self, "Load Error", str(error)
                )

    def change_source(self, value):

        self.engine.set_source(value)
        self.status.setText("SOURCE: " + value)

    def change_tempo(self, value):

        tempo = value / 100.0

        self.engine.set_tempo(tempo)
        self.tempo_value.setText(f"{tempo:.2f}×")

    def change_pitch(self, value):

        self.engine.set_pitch(value)

        self.pitch_value.setText(
            f"{value:+d} semitones"
        )

    def change_volume(self, value):

        self.engine.set_volume(value / 100.0)

    def closeEvent(self, event):

        self.engine.close()
        event.accept()


# ---------------- APPLICATION ------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DJMixer()
    window.show()

    sys.exit(app.exec())
