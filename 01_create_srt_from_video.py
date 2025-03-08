import os
import gc
from moviepy.editor import VideoFileClip
from vosk import Model, KaldiRecognizer
import wave
import json
import pysrt
from pydub import AudioSegment
from tqdm import tqdm
from datetime import datetime

# Make sure you have the files
video_path = "inputs-outputs/4. Daniel y Desi. Discursos.mov"

# Select which model to use (Models available in https://alphacephei.com/vosk/models)
#model_path = "vosk-model-small-en-us-0.15" # Small model EN
#model_path = "vosk-model-small-es-0.42" # Small model ES
model_path = "vosk-model-en-us-0.22"  # Large model EN
#model_path = "vosk-model-es-0.42" # Large model ES

# Step 1: Extract and convert audio to WAV mono PCM
# Load video file
video = VideoFileClip(video_path)
# Extract audio (this file is deleted once the srt is created)
audio_path = "inputs-outputs/audio_mono.wav"
audio = video.audio

def extract_and_convert_audio(video, audio_path):
    temp_audio_path = "inputs-outputs/temp_audio.wav"
    audio = video.audio
    audio.write_audiofile(temp_audio_path)

    audio_segment = AudioSegment.from_file(temp_audio_path)
    mono_audio = audio_segment.set_channels(1).set_frame_rate(16000)
    mono_audio.export(audio_path, format="wav")

    os.remove(temp_audio_path)  # Remove temporary audio file

extract_and_convert_audio(video, audio_path)

# Step 2: Transcribe audio using Vosk
if not os.path.exists(model_path):
    raise ValueError(
        "Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")

model = Model(model_path)

def transcribe_audio_vosk(audio_path):
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

    # Create a timestamp for the file name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    subtitle_filename = f'inputs-outputs/subtitles_{timestamp}.srt'

    subs.save(subtitle_filename, encoding='utf-8')
    print(f"Subtitles saved to '{subtitle_filename}'")

create_srt(transcriptions)

# Clean up temporary files and memory
os.remove(audio_path)  # Remove audio file

# Release resources and force garbage collection
video.close()
audio.close()

# Force garbage collection to free memory
gc.collect()

print("Cleanup complete.")

