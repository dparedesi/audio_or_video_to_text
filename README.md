# 🎙️ Audio & Video to Text Toolkit

A comprehensive collection of Python tools for audio/video transcription, subtitle generation, and real-time voice recognition. Built with state-of-the-art speech recognition models (Whisper, Vosk, Faster-Whisper).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Table of Contents

- [Features](#-features)
- [Tools Overview](#-tools-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 🎬 **Video to SRT subtitles** - Generate accurate SRT files from videos (offline)
- 🔗 **Subtitle merging** - Embed subtitles directly into videos with customizable styling
- 📚 **Intelligent audiobook transcription** - Automatic chapter detection and formatting
- 🎵 **Simple audio transcription** - Quick and accurate audio-to-text conversion
- 🎤 **Real-time voice transcription** - Modern desktop app with live transcription
- 🌍 **Multi-language support** - Works with multiple languages via Whisper
- 🔒 **Privacy-focused** - Offline processing with Vosk (no API calls required)
- ⚡ **Multiple model options** - Choose between speed and accuracy

## 🛠️ Tools Overview

### 1. Video to SRT Subtitle Generator (`01_create_srt_from_video.py`)

Generates SRT subtitle files from video using **Vosk** speech recognition (fully offline).

**Key Features:**
- Offline speech recognition (no internet required)
- Word-level timestamps for accurate synchronization
- Automatic phrase segmentation (2-second chunks or punctuation breaks)
- Progress tracking with tqdm
- Customizable output paths with timestamps

**Use Cases:**
- Creating subtitles for YouTube videos
- Adding accessibility to video content
- Generating closed captions for presentations
- Transcribing interviews or lectures

### 2. Video & Subtitle Merger (`01_merge_video_with_srt.py`)

Merges SRT subtitle files into video files using **FFmpeg** with professional styling.

**Key Features:**
- Multiple quality presets (low, medium, high, 4K)
- Customizable subtitle styling (font, colors, outline, shadow)
- Hardware-accelerated encoding (H.264 + AAC)
- Resolution scaling support
- Professional subtitle appearance

**Use Cases:**
- Publishing videos with burned-in subtitles
- Creating accessible video content
- Social media content with embedded captions

### 3. Audiobook to Structured Text (`02_mp3_to_text_with_chapters.py`)

Advanced audiobook transcription with **intelligent chapter detection** using OpenAI Whisper.

**Key Features:**
- Automatic chapter detection using acoustic and semantic analysis
- Statistical gap analysis for chapter boundaries
- Pattern recognition (e.g., "Chapter 1", "Part II", "Prologue")
- Smart paragraph formatting and merging
- Robust error handling with partial result recovery
- Multiple Whisper model sizes (tiny to large-v3)

**Use Cases:**
- Transcribing audiobooks with chapter structure
- Converting audio lectures into organized text
- Creating searchable text from podcast series
- Generating study materials from audio courses

### 4. Simple Audio Transcriber (`03_audio_to_text.py`)

Straightforward audio-to-text transcription using **OpenAI Whisper**.

**Key Features:**
- Support for multiple audio formats (MP3, M4A, WAV, etc.)
- Multi-language support with auto-detection
- Multiple model sizes for speed/accuracy trade-offs
- Timestamped output filenames
- Clean, simple API

**Use Cases:**
- Quick transcription of voice memos
- Meeting notes from audio recordings
- Transcribing interviews
- Converting audio notes to text

### 5. Live Voice Transcriber (`04_live_voice_transcriber.py`)

Real-time voice transcription desktop application with **PyQt6** GUI using Faster-Whisper.

**Key Features:**
- Modern, native macOS-style dark theme interface
- Real-time transcription with 5-second chunks
- Live audio level monitoring (internal)
- Final complete transcription after recording stops
- Save transcriptions with native file dialogs
- Clean, minimalist design
- Threaded processing for responsive UI

**Use Cases:**
- Live meeting transcription
- Voice note taking
- Real-time dictation
- Lecture transcription
- Interview recording and transcription

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing)
- macOS, Linux, or Windows

### System Dependencies

#### macOS
```bash
brew install ffmpeg portaudio
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ffmpeg portaudio19-dev python3-pyaudio
```

#### Windows
Download and install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)

### Python Setup

1. **Clone the repository**
```bash
git clone https://github.com/dparedesi/audio_or_video_to_text.git
cd audio_or_video_to_text
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Vosk model** (for offline transcription)

For English (US):
```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
```

Other languages available at: https://alphacephei.com/vosk/models

## 🚀 Quick Start

### Create SRT Subtitles from Video
```bash
python 01_create_srt_from_video.py --video inputs/myvideo.mov --model vosk-model-en-us-0.22
```

### Merge Subtitles into Video
```bash
python 01_merge_video_with_srt.py --video inputs/myvideo.mov --subtitles outputs/subtitles_250105-1430.srt --quality high
```

### Transcribe Audiobook with Chapters
```bash
python 02_mp3_to_text_with_chapters.py --input inputs/audiobook.mp3 --model large-v3
```

### Simple Audio Transcription
```bash
python 03_audio_to_text.py --input inputs/recording.m4a --model small --language en
```

### Launch Real-Time Voice Transcriber
```bash
python 04_live_voice_transcriber.py
```

## 📖 Detailed Usage

### 01_create_srt_from_video.py

**Arguments:**
- `--video` - Path to input video file (default: `inputs/video.mov`)
- `--model` - Path to Vosk model directory (default: `vosk-model-en-us-0.22`)
- `--output` - Output SRT filename (default: auto-generated with timestamp)

**Example:**
```bash
python 01_create_srt_from_video.py \
  --video inputs/presentation.mp4 \
  --model vosk-model-en-us-0.22 \
  --output outputs/presentation_subs.srt
```

**Output:**
- SRT file with word-level timestamps
- Phrases segmented at 2-second intervals or punctuation marks
- Temporary WAV file automatically cleaned up

### 01_merge_video_with_srt.py

**Arguments:**
- `--video` - Path to input video file (**required**)
- `--subtitles` - Path to SRT file (**required**)
- `--output` - Output video path (default: `outputs/merged_video.mp4`)
- `--quality` - Quality preset: `low`, `medium`, `high`, `4k` (default: `high`)

**Quality Presets:**
| Preset | CRF | Bitrate | Scale |
|--------|-----|---------|-------|
| low | 28 | 5M | Original |
| medium | 23 | 10M | Original |
| high | 18 | 20M | Original |
| 4k | 18 | 40M | 3840:2160 |

**Example:**
```bash
python 01_merge_video_with_srt.py \
  --video inputs/lecture.mp4 \
  --subtitles outputs/lecture_subs.srt \
  --quality 4k \
  --output outputs/lecture_final.mp4
```

**Subtitle Styling:**
- Font: Arial
- Color: White with black outline
- Background: Semi-transparent black
- Position: Bottom center

### 02_mp3_to_text_with_chapters.py

**Arguments:**
- `--input` - Path to audio file (default: `inputs/audiobook.mp3`)
- `--model` - Whisper model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (default: `large-v3`)
- `--output` - Output text file (default: `outputs/structured_audiobook.txt`)

**Model Size Comparison:**
| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | ~1GB |
| base | ⚡⚡⚡⚡ | ⭐⭐⭐ | ~1GB |
| small | ⚡⚡⚡ | ⭐⭐⭐⭐ | ~2GB |
| medium | ⚡⚡ | ⭐⭐⭐⭐⭐ | ~5GB |
| large-v3 | ⚡ | ⭐⭐⭐⭐⭐ | ~10GB |

**Example:**
```bash
python 02_mp3_to_text_with_chapters.py \
  --input inputs/becoming.mp3 \
  --model large-v3 \
  --output outputs/becoming_transcript.txt
```

**Chapter Detection Methods:**
- Acoustic: Long silence gaps (statistical analysis)
- Semantic: Pattern matching (e.g., "Chapter 5", "Part III")
- Hybrid: Combines both approaches for best results

**Output Format:**
```
Chapter 1

[First chapter content with proper paragraph formatting...]

Chapter 2

[Second chapter content...]
```

### 03_audio_to_text.py

**Arguments:**
- `--input` - Path to audio file (default: `inputs/New Recording.m4a`)
- `--model` - Whisper model size (default: `small`)
- `--output` - Output text file (default: auto-generated with timestamp)
- `--language` - Language code or `auto` (default: `en`)

**Supported Languages:**
English (`en`), Spanish (`es`), French (`fr`), German (`de`), Chinese (`zh`), Japanese (`ja`), Korean (`ko`), and [90+ more](https://github.com/openai/whisper#available-models-and-languages)

**Example:**
```bash
# English transcription
python 03_audio_to_text.py --input inputs/meeting.m4a --model medium

# Spanish with auto-detection
python 03_audio_to_text.py --input inputs/podcast_es.mp3 --language es --model small

# Automatic language detection
python 03_audio_to_text.py --input inputs/multilingual.wav --language auto
```

### 04_live_voice_transcriber.py

**No command-line arguments** - fully GUI-based application.

**Features:**
- Click "Start Recording" to begin
- Speaks continuously - transcription appears in real-time (5-second chunks)
- Click "Stop Recording" for final complete transcription
- Use "Clear" to reset the text area
- Use "Save" to export transcription with native file dialog

**Keyboard Shortcuts:**
- None currently - GUI button-based interface

**Performance Settings:**
- Model: `small` (hardcoded for speed)
- Compute type: `int8` (optimized for CPU)
- Sample rate: 16kHz
- Chunk duration: 5 seconds

**Customization:**
Edit the script to change model size:
```python
self.model = WhisperModel("small", device="cpu", compute_type="int8")
# Change "small" to "tiny", "base", "medium", or "large"
```

## 📦 Requirements

### Core Dependencies
- **moviepy** - Video editing and audio extraction
- **pydub** - Audio file manipulation
- **ffmpeg-python** - FFmpeg wrapper for video processing
- **vosk** - Offline speech recognition
- **openai-whisper** - State-of-the-art transcription model
- **faster-whisper** - Optimized Whisper inference
- **PyQt6** - GUI framework for desktop app
- **pyaudio** - Audio recording interface
- **pysrt** - SRT subtitle file manipulation
- **torch/torchaudio** - Deep learning framework for Whisper
- **numpy** - Numerical computing
- **tqdm** - Progress bars

See `requirements.txt` for complete list with versions.

## 📁 Project Structure

```
audio_or_video_to_text/
├── 01_create_srt_from_video.py      # Video → SRT subtitle generation (Vosk)
├── 01_merge_video_with_srt.py       # Merge SRT into video (FFmpeg)
├── 02_mp3_to_text_with_chapters.py  # Audiobook with chapter detection
├── 03_audio_to_text.py              # Simple audio transcription
├── 04_live_voice_transcriber.py     # Real-time voice transcriber GUI
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
├── README.md                         # This file
├── inputs/                           # Input files (gitignored)
│   ├── file-1.mp3
│   ├── file-2.mp3
│   └── file-3.mp3
└── outputs/                          # Generated outputs (gitignored)
    └── structured_audiobook.txt
```

## ⚡ Performance Tips

### Choosing the Right Model

**For Speed (Real-time/Interactive):**
- Vosk: Best for offline, fast transcription
- Whisper tiny/base: Quick results, reasonable accuracy
- Faster-Whisper small: Optimized for real-time

**For Accuracy (Batch Processing):**
- Whisper large-v3: Best accuracy, slowest
- Whisper medium: Good balance
- Vosk with larger models: Better offline accuracy

### Hardware Acceleration

**GPU Support (NVIDIA):**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Modify scripts to use GPU:
```python
# For Whisper
model = whisper.load_model("large-v3", device="cuda")

# For Faster-Whisper
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
```

**Apple Silicon (M1/M2/M3):**
```python
# Whisper automatically uses MPS (Metal Performance Shaders)
model = whisper.load_model("large-v3")  # Will use MPS if available
```

### Memory Optimization

For large files:
1. Use smaller models (`small` or `medium`)
2. Process in chunks
3. Enable `fp16` for GPU or `int8` for CPU (Faster-Whisper)
4. Close applications to free RAM

### Batch Processing

Process multiple files:
```bash
# Bash loop for multiple videos
for video in inputs/*.mp4; do
  python 01_create_srt_from_video.py --video "$video" --model vosk-model-en-us-0.22
done

# Transcribe all audio files
for audio in inputs/*.mp3; do
  python 03_audio_to_text.py --input "$audio" --model small
done
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "FFmpeg not found"
**Solution:**
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org and add to PATH
```

#### 2. "Vosk model not found"
**Solution:**
Download the model and extract it:
```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
```

#### 3. PyAudio installation fails
**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Ubuntu:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
Download wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

#### 4. "CUDA out of memory"
**Solutions:**
- Use smaller Whisper model
- Enable `fp16` or `int8` precision
- Process shorter audio segments
- Use CPU instead of GPU

#### 5. Poor transcription quality
**Solutions:**
- Use larger Whisper model (`large-v3`)
- Ensure audio quality is good (minimal background noise)
- Specify correct language with `--language`
- Try different Vosk models for your language

#### 6. Real-time transcriber lag
**Solutions:**
- Use `tiny` or `base` Whisper model
- Reduce chunk duration
- Close other applications
- Use GPU if available

### Debug Mode

Enable verbose logging:
```bash
# Add to any script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Ideas for Contributions
- [ ] Add support for more languages
- [ ] Implement GPU acceleration options
- [ ] Create batch processing scripts
- [ ] Add speaker diarization
- [ ] Improve chapter detection algorithms
- [ ] Add more subtitle styling options
- [ ] Create web interface
- [ ] Add Docker support
- [ ] Implement progress persistence
- [ ] Add audio preprocessing options

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** - State-of-the-art speech recognition
- **Vosk** - Offline speech recognition toolkit
- **FFmpeg** - Multimedia processing framework
- **PyQt6** - Cross-platform GUI toolkit
- **Faster-Whisper** - Optimized Whisper implementation

## 📧 Contact

Daniel Paredes - [@dparedesi](https://github.com/dparedesi)

Project Link: [https://github.com/dparedesi/audio_or_video_to_text](https://github.com/dparedesi/audio_or_video_to_text)

---

**Star ⭐ this repository if you find it helpful!**
