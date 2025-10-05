# Audio/Video to Text Processing Toolkit

A collection of Python scripts for transcribing audio/video files to text and creating/merging subtitles using Vosk and OpenAI Whisper.

## Features

- **Video to SRT**: Generate subtitle files from videos using Vosk speech recognition
- **Video + SRT Merger**: Burn subtitles into video files with ffmpeg
- **Audio to Structured Text**: Transcribe audiobooks with automatic chapter detection using Whisper
- **Simple Audio Transcription**: Basic audio-to-text with Whisper CLI

## Requirements

- Python 3.8+
- ffmpeg (for video processing)
- Vosk models (for `01_create_srt_from_video.py`)

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

## Directory Structure

```
audio_or_video_to_text/
├── inputs/          # Place your audio/video files here
├── outputs/         # Transcriptions and processed files
├── requirements.txt # Python dependencies
└── *.py            # Processing scripts
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
