import os
import gc
import sys
import argparse
from moviepy.editor import VideoFileClip
from vosk import Model, KaldiRecognizer
import wave
import json
import pysrt
from pydub import AudioSegment
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from video using Vosk speech recognition."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="inputs/video.mov",
        help="Path to the input video file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vosk-model-en-us-0.22",
        help="Path to Vosk model directory (download from https://alphacephei.com/vosk/models)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output SRT filename (default: outputs/subtitles_YYMMDD-HHMM.srt)."
    )
    return parser.parse_args()

# Step 1: Extract and convert audio to WAV mono PCM

def extract_and_convert_audio(video, audio_path, temp_dir="outputs"):
    """Extract audio from video and convert to mono 16kHz WAV."""
    os.makedirs(temp_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
    
    video.audio.write_audiofile(temp_audio_path)
    
    audio_segment = AudioSegment.from_file(temp_audio_path)
    mono_audio = audio_segment.set_channels(1).set_frame_rate(16000)
    mono_audio.export(audio_path, format="wav")
    
    os.remove(temp_audio_path)  # Remove temporary audio file

# Step 2: Transcribe audio using Vosk
def transcribe_audio_vosk(audio_path, model):
    wf = wave.open(audio_path, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    total_frames = wf.getnframes()
    frames_per_chunk = 4000

    with tqdm(total=total_frames // frames_per_chunk, desc="Transcribing") as pbar:
        while True:
            data = wf.readframes(frames_per_chunk)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
            pbar.update(1)

    results.append(json.loads(rec.FinalResult()))
    wf.close()

    # Extract transcribed text and word-level timestamps
    words = []
    for result in results:
        if 'result' in result:
            words.extend(result['result'])

    return words

transcriptions = transcribe_audio_vosk(audio_path)

# Step 3: Create SRT file with accurate timings
def create_srt(transcriptions):
    subs = pysrt.SubRipFile()
    index = 1
    current_phrase = []
    phrase_start_time = None

    def add_phrase_to_subs():
        nonlocal index, current_phrase, phrase_start_time
        if current_phrase:
            start_time = pysrt.SubRipTime.from_ordinal(int(current_phrase[0]['start'] * 1000))
            end_time = pysrt.SubRipTime.from_ordinal(int(current_phrase[-1]['end'] * 1000))
            text = " ".join([word['word'] for word in current_phrase])
            subs.append(pysrt.SubRipItem(index, start=start_time, end=end_time, text=text))
            index += 1
            current_phrase = []
            phrase_start_time = None

    for word in transcriptions:
        if phrase_start_time is None:
            phrase_start_time = word['start']

        current_phrase.append(word)

        # Add phrase to subs if the phrase duration exceeds 2 seconds or contains a punctuation mark
        if (word['end'] - phrase_start_time >= 2) or word['word'].endswith(('.', '?', '!', ',')):
            add_phrase_to_subs()

    # Add any remaining words as the last phrase
    add_phrase_to_subs()
    
    return subs

def save_srt(subs, output_path):
    """Save SRT file to specified path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subs.save(output_path, encoding='utf-8')
    print(f"Subtitles saved to '{output_path}'")

def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Validate model
    if not os.path.exists(args.model):
        print(f"Error: Vosk model not found: {args.model}")
        print("Download models from: https://alphacephei.com/vosk/models")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        timestamp = datetime.now().strftime("%y%m%d-%H%M")
        args.output = f"outputs/subtitles_{timestamp}.srt"
    
    try:
        # Load video and prepare audio path
        print(f"Loading video: {args.video}")
        video = VideoFileClip(args.video)
        audio_path = "outputs/audio_mono.wav"
        
        # Extract and convert audio
        print("Extracting audio...")
        extract_and_convert_audio(video, audio_path)
        
        # Load Vosk model
        print(f"Loading model: {args.model}")
        model = Model(args.model)
        
        # Transcribe
        transcriptions = transcribe_audio_vosk(audio_path, model)
        
        # Create and save SRT
        subs = create_srt(transcriptions)
        save_srt(subs, args.output)
        
        # Clean up
        os.remove(audio_path)
        video.close()
        gc.collect()
        
        print("Processing complete.")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

