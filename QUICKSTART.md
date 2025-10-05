# Quick Start Guide

## Installation (3 minutes)

```bash
# 1. Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (if not installed)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
```

## Common Use Cases

### Generate subtitles from a video
```bash
# Download Vosk model first: https://alphacephei.com/vosk/models
python 01_create_srt_from_video.py \
  --video inputs/my_video.mov \
  --model vosk-model-en-us-0.22
```

### Add subtitles to video
```bash
python 01_merge_video_with_srt.py \
  --video inputs/video.mov \
  --subtitles outputs/subtitles.srt \
  --quality high
```

### Transcribe audiobook with chapters
```bash
python 02_mp3_to_text_with_chapters.py \
  --input inputs/audiobook.mp3 \
  --model large-v3
```

### Simple audio transcription
```bash
python 03_audio_to_text.py \
  --input inputs/recording.m4a \
  --model small
```

## Directory Setup

Place your files in `inputs/`:
```
inputs/
  ├── video.mov
  ├── audiobook.mp3
  └── recording.m4a
```

Results appear in `outputs/`:
```
outputs/
  ├── subtitles_251005_1430.srt
  ├── merged_video.mp4
  └── transcription.txt
```

## Model Recommendations

| Use Case | Vosk Model | Whisper Model |
|----------|-----------|---------------|
| Quick test | small-en-us | tiny/base |
| Production | en-us-0.22 | large-v3 |
| Non-English | Download language-specific | medium+ |

## Troubleshooting

**"Module not found"** → Activate venv: `source venv/bin/activate`

**"FFmpeg not found"** → Install ffmpeg for your OS

**"Vosk model not found"** → Download from https://alphacephei.com/vosk/models

**Out of memory** → Use smaller Whisper model (`small` instead of `large`)

See README.md for full documentation.
