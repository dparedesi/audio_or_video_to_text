# Audio/Video to Text Processing Toolkit

A collection of Python scripts for transcribing audio/video files to text and creating/merging subtitles using Vosk and OpenAI Whisper.

## Features

- **Video to SRT**: Generate subtitle files from videos using Vosk speech recognition
- **Video + SRT Merger**: Burn subtitles into video files with ffmpeg
- **Audio to Structured Text**: Transcribe audiobooks with automatic chapter detection using Whisper
- **Simple Audio Transcription**: Basic audio-to-text with Whisper CLI
- **Live Voice Transcriber (tkinter)**: Real-time speech-to-text with basic GUI
- **Modern Voice Transcriber (PyQt6)**: Native macOS app with professional design ⭐ NEW!

## Requirements

- Python 3.8+
- ffmpeg (for video processing)
- Vosk models (for `01_create_srt_from_video.py`)
- Microphone access (for `04_live_voice_transcriber.py`)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/dparedesi/audio_or_video_to_text.git
cd audio_or_video_to_text
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install ffmpeg** (if not already installed)
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

5. **Download Vosk models** (for subtitle generation)

Visit [Vosk Models](https://alphacephei.com/vosk/models) and download a model:
- Small EN: `vosk-model-small-en-us-0.15`
- Large EN: `vosk-model-en-us-0.22`
- Spanish: `vosk-model-es-0.42`

Extract the model to your project directory.

## Usage

### 1. Generate SRT Subtitles from Video

Creates timestamped subtitle files using Vosk speech recognition:

```bash
python 01_create_srt_from_video.py \
  --video inputs/my_video.mov \
  --model vosk-model-en-us-0.22 \
  --output outputs/subtitles.srt
```

**Options:**
- `--video`: Path to input video file
- `--model`: Path to Vosk model directory
- `--output`: Output SRT filename (auto-generated if not specified)

### 2. Merge Video with Subtitles

Burns subtitles into video with customizable quality:

```bash
python 01_merge_video_with_srt.py \
  --video inputs/video.mov \
  --subtitles outputs/subtitles.srt \
  --output outputs/final_video.mp4 \
  --quality high
```

**Quality Presets:**
- `low`: CRF 28, 5 Mbps
- `medium`: CRF 23, 10 Mbps
- `high`: CRF 18, 20 Mbps (default)
- `4k`: CRF 18, 40 Mbps, scaled to 3840x2160

### 3. Audiobook to Structured Text

Transcribes audiobooks with automatic chapter detection:

```bash
python 02_mp3_to_text_with_chapters.py \
  --input inputs/audiobook.mp3 \
  --model large-v3 \
  --output outputs/audiobook.txt
```

**Whisper Models:**
- `tiny`, `base`, `small` - Fast, lower accuracy
- `medium` - Balanced
- `large`, `large-v2`, `large-v3` - Best accuracy, slower

### 4. Simple Audio Transcription

Basic transcription with real-time progress:

```bash
python 03_audio_to_text.py \
  --input inputs/audio.m4a \
  --model small \
  --output outputs/transcription.txt
```

### 5. Live Voice Transcriber (GUI)

Real-time speech-to-text transcription with a graphical interface:

```bash
python 04_live_voice_transcriber.py
```

**Features:**
- Real-time voice transcription using faster-whisper
- Dark mode GUI with live audio level meter
- Voice Activity Detection (VAD) for automatic pause detection
- Processes speech in 5-second chunks for quick feedback
- Final complete transcription after stopping recording
- Save transcriptions with timestamps
- Visual feedback for recording status and processing queue

**Controls:**
- Click "Start Recording" to begin capturing audio
- Speak naturally - the app processes speech in real-time
- Watch the audio level meter to ensure proper microphone input
- Click "Stop Recording" to finish and generate final transcription
- "Clear Text" removes current transcription from display
- "Save Transcription" exports final text to `outputs/` folder

**Requirements:**
- `faster-whisper` for efficient speech recognition
- `pyaudio` for microphone access
- `webrtcvad` for voice activity detection
- `tkinter` (included with Python)

**Note:** The app provides two transcriptions:
1. **Real-time chunks**: Quick preview as you speak
2. **Final transcription**: High-quality, context-aware transcription of complete recording

### 6. Modern Voice Transcriber (PyQt6 - Native macOS) ⭐ NEW!

**State-of-the-art native macOS desktop app** with professional design:

```bash
# Install PyQt6 dependencies
pip install -r requirements_pyqt.txt

# Run the modern native app
python 05_modern_voice_transcriber_pyqt.py
```

**Features:**
- 🍎 **True native macOS app** using Qt framework
- 🎨 **Modern dark theme** inspired by macOS Big Sur
- 📊 **Beautiful audio visualizer** with gradient colors
- ⚡ **Multi-threaded** for smooth performance
- 💾 **Native file dialogs** and macOS integration
- 🎯 **Professional-grade** UI used by industry apps
- ✨ **No web technologies** - truly native widgets

**Why PyQt6?**
- Native macOS widgets (not web-based)
- Used by professional apps (Autodesk Maya, Spotify, etc.)
- True desktop performance
- Can build standalone `.app` bundle
- Modern, beautiful UI out of the box

**vs tkinter version (04):**
- Modern gradient buttons vs flat buttons
- Smooth animations and transitions
- Native macOS look and feel
- Better performance (multi-threaded)
- Professional polish

## Directory Structure

```
audio_or_video_to_text/
├── inputs/                  # Place your audio/video files here
├── outputs/                 # Transcriptions and processed files
├── requirements.txt         # Python dependencies (basic)
├── requirements_pyqt.txt    # PyQt6 dependencies (for modern app)
├── 01_*.py                 # Video subtitle scripts
├── 02_*.py                 # Audio processing scripts
├── 03_*.py                 # Simple transcription
├── 04_*.py                 # Live voice transcriber (tkinter)
└── 05_*.py                 # 🆕 Modern native app (PyQt6)
```

## Tips

- **GPU Acceleration**: Whisper models run faster on CUDA-enabled GPUs
- **Model Selection**: Start with smaller models for testing, use larger ones for production
- **Memory Usage**: Large Whisper models require 8GB+ RAM
- **Vosk vs Whisper**: Vosk is faster but less accurate; Whisper is slower but more accurate

## Troubleshooting

**Import errors**: Ensure you've activated the virtual environment and installed requirements

**FFmpeg not found**: Install ffmpeg and ensure it's in your PATH

**Vosk model errors**: Download the correct model from https://alphacephei.com/vosk/models

**Out of memory**: Use smaller Whisper models or process shorter audio segments

## License

MIT License - See [LICENSE](LICENSE) file for details

## Author

Daniel Paredes ([@dparedesi](https://github.com/dparedesi))
