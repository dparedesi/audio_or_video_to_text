"""
Modern Voice Transcriber - PyQt6 Native macOS App
State-of-the-art desktop application with native widgets
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame, QProgressBar, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
import sys
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
from pathlib import Path
import tempfile
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')


class AudioRecorder(QThread):
    """Thread for recording audio"""
    audio_level_updated = pyqtSignal(float)
    chunk_ready = pyqtSignal(bytes)
    
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.chunk_size = 1024
        self.sample_rate = 16000
        self.chunk_duration = 5
        self.full_audio_frames = []
        
    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while self.is_recording:
            frames = []
            frames_to_record = int(self.sample_rate / self.chunk_size * self.chunk_duration)
            
            for _ in range(frames_to_record):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
                self.full_audio_frames.append(data)
                
                # Calculate audio level
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                level = min(100, (rms / 32768.0 * 200))
                self.audio_level_updated.emit(level)
            
            if frames:
                chunk_audio = b''.join(frames)
                self.chunk_ready.emit(chunk_audio)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def stop(self):
        self.is_recording = False
    
    def get_full_audio(self):
        return b''.join(self.full_audio_frames)
    
    def reset(self):
        self.full_audio_frames = []


class TranscriptionWorker(QThread):
    """Thread for transcribing audio chunks"""
    transcription_ready = pyqtSignal(str)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.audio_data = None
        self.is_final = False
        
    def set_audio(self, audio_data, is_final=False):
        self.audio_data = audio_data
        self.is_final = is_final
        
    def run(self):
        if not self.audio_data:
            return
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filepath = temp_file.name
        temp_file.close()
        
        try:
            # Save audio to file
            with wave.open(temp_filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(self.audio_data)
            
            # Transcribe
            segments, info = self.model.transcribe(
                temp_filepath,
                language="en",
                beam_size=5,
                condition_on_previous_text=self.is_final,
                vad_filter=not self.is_final
            )
            
            text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
            
            if text_parts:
                if self.is_final:
                    transcription = "\n".join(text_parts)
                else:
                    transcription = " ".join(text_parts)
                self.transcription_ready.emit(transcription)
                
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            try:
                os.unlink(temp_filepath)
            except:
                pass


class ModernVoiceTranscriber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_level = 0
        
        # Load Whisper model
        self.setWindowTitle("Loading Whisper Model...")
        QApplication.processEvents()
        
        print("Loading Faster-Whisper model...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Model loaded!")
        
        # Initialize threads
        self.recorder = AudioRecorder()
        self.recorder.audio_level_updated.connect(self.update_audio_level)
        self.recorder.chunk_ready.connect(self.on_chunk_ready)
        
        self.transcription_worker = TranscriptionWorker(self.model)
        self.transcription_worker.transcription_ready.connect(self.on_transcription_ready)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface with modern macOS design"""
        self.setWindowTitle("Voice Transcriber")
        self.setGeometry(100, 100, 700, 750)  # Narrower width, taller height
        
        # Set modern dark theme
        self.set_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Transcription area (moved to top, takes most space)
        transcription_section = self.create_transcription_section()
        main_layout.addLayout(transcription_section)
        
        # Control panel (moved to bottom)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background: #2d2d2d;
                color: #a0a0a0;
                border-top: 1px solid #404040;
            }
        """)
        
    def set_dark_theme(self):
        """Apply modern dark theme"""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(36, 36, 36))
        palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
        palette.setColor(QPalette.ColorRole.Button, QColor(36, 36, 36))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow {
                background: #1a1a1a;
            }
            QWidget {
                font-family: -apple-system, 'SF Pro Display';
                font-size: 14px;
            }
        """)
    
    def create_header(self):
        """Create header with title and status"""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: #242424;
                border-radius: 12px;
                padding: 16px;
            }
        """)
        
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("🎤 Voice Transcriber")
        title.setFont(QFont("SF Pro Display", 24, QFont.Weight.Bold))
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Status badge
        self.status_label = QLabel("● Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                color: #30d158;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: 500;
            }
        """)
        layout.addWidget(self.status_label)
        
        return header
    
    def create_control_panel(self):
        """Create control panel with record button only (clean, minimal)"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: transparent;
                padding: 0px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Record button - bigger, more prominent
        self.record_button = QPushButton("🎤 Start Recording")
        self.record_button.setFixedHeight(64)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a84ff, stop:1 #0066cc);
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 20px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0066cc, stop:1 #004999);
            }
            QPushButton:pressed {
                background: #004999;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)
        
        return panel
    
    def create_transcription_section(self):
        """Create transcription display area"""
        layout = QVBoxLayout()
        
        # Header with buttons
        header_layout = QHBoxLayout()
        
        section_label = QLabel("Transcription")
        section_label.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        header_layout.addWidget(section_label)
        
        header_layout.addStretch()
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(80, 36)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3d3d3d;
                border-color: #0a84ff;
            }
        """)
        clear_btn.clicked.connect(self.clear_transcription)
        header_layout.addWidget(clear_btn)
        
        # Save button
        save_btn = QPushButton("Save")
        save_btn.setFixedSize(80, 36)
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setStyleSheet("""
            QPushButton {
                background: #2d2d2d;
                color: #30d158;
                border: 1px solid #404040;
                border-radius: 8px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #3d3d3d;
                border-color: #30d158;
            }
        """)
        save_btn.clicked.connect(self.save_transcription)
        header_layout.addWidget(save_btn)
        
        layout.addLayout(header_layout)
        
        # Text area
        self.transcription_text = QTextEdit()
        self.transcription_text.setPlaceholderText("🎙️ Ready to transcribe\nClick 'Start Recording' to begin...")
        self.transcription_text.setStyleSheet("""
            QTextEdit {
                background: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 16px;
                font-size: 15px;
                line-height: 1.6;
                font-family: 'SF Pro Text', -apple-system;
            }
        """)
        layout.addWidget(self.transcription_text)
        
        return layout
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.recorder.reset()
        self.recorder.is_recording = True
        self.recorder.start()
        
        # Update UI
        self.record_button.setText("⏹ Stop Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff453a, stop:1 #cc0000);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #cc0000, stop:1 #990000);
            }
        """)
        self.status_label.setText("● Recording")
        self.status_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                color: #0a84ff;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: 500;
            }
        """)
        self.statusBar().showMessage("Recording... Speak now")
        self.transcription_text.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.transcription_text.append("🎤 Recording started...")
        self.transcription_text.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    def stop_recording(self):
        """Stop recording and process final transcription"""
        self.is_recording = False
        self.recorder.stop()
        
        # Update UI
        self.record_button.setText("🎤 Start Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a84ff, stop:1 #0066cc);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0066cc, stop:1 #004999);
            }
        """)
        self.status_label.setText("● Processing...")
        self.status_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                color: #ff9f0a;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: 500;
            }
        """)
        self.statusBar().showMessage("Processing final transcription...")
        
        # Process final transcription
        full_audio = self.recorder.get_full_audio()
        if full_audio:
            self.transcription_text.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            self.transcription_text.append("✨ FINAL COMPLETE TRANSCRIPTION")
            self.transcription_text.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            self.transcription_worker.set_audio(full_audio, is_final=True)
            self.transcription_worker.start()
    
    def update_audio_level(self, level):
        """Update audio level - no visual meter, just internal tracking"""
        self.audio_level = level
        # Audio level is tracked but not displayed for cleaner UI
    
    def on_chunk_ready(self, audio_data):
        """Process audio chunk"""
        if not self.transcription_worker.isRunning():
            self.transcription_worker.set_audio(audio_data, is_final=False)
            self.transcription_worker.start()
    
    def on_transcription_ready(self, text):
        """Display transcription"""
        self.transcription_text.append(text + "\n")
        
        # Update status after final transcription
        if not self.is_recording:
            self.status_label.setText("● Ready")
            self.status_label.setStyleSheet("""
                QLabel {
                    background: #2d2d2d;
                    color: #30d158;
                    border: 1px solid #404040;
                    border-radius: 12px;
                    padding: 8px 16px;
                    font-weight: 500;
                }
            """)
            self.statusBar().showMessage("Transcription complete!")
    
    def clear_transcription(self):
        """Clear transcription text"""
        self.transcription_text.clear()
        self.statusBar().showMessage("Transcription cleared")
    
    def save_transcription(self):
        """Save transcription to file"""
        text = self.transcription_text.toPlainText()
        if not text.strip():
            self.statusBar().showMessage("No transcription to save")
            return
        
        # Use native file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            f"transcription_{datetime.now().strftime('%y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(text)
                self.statusBar().showMessage(f"Saved: {file_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error saving: {e}")


def main():
    app = QApplication(sys.argv)
    
    # Set app name for macOS
    app.setApplicationName("Voice Transcriber")
    app.setOrganizationName("DParedesi")
    
    window = ModernVoiceTranscriber()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
