import re  # <-- Missing import added here
import whisper
import numpy as np
from pydub import AudioSegment

# Configuration
MP3_PATH = "inputs-outputs/file-3.mp3"
OUTPUT_TXT = "inputs-outputs/structured_audiobook.txt"
MODEL_SIZE = "large-v3"

def transcribe_with_context():
    """Optimized Whisper transcription with memory management"""
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE)
    
    print("Processing audio...")
    result = model.transcribe(
        MP3_PATH,
        word_timestamps=True,
        verbose=True,
        condition_on_previous_text=True,
        no_speech_threshold=0.45,
        compression_ratio_threshold=2.4,
        fp16=False  # Disable if using CPU
    )
    
    return result

def analyze_structure(segments):
    """Robust structure analysis with error handling"""
    print("Identifying narrative structure...")
    
    if not segments:
        raise ValueError("No transcription segments found")
    
    structured_content = []
    current_chapter = []
    chapter_counter = 1
    
    # Calculate gaps between segments safely
    gaps = []
    prev_end = segments[0]['start'] if segments else 0  # Handle first segment
    
    for seg in segments:
        gaps.append(seg['start'] - prev_end)
        prev_end = seg['end']
    
    # Robust statistical analysis
    try:
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        chapter_threshold = max(2.5, gap_mean + 2*gap_std)  # Ensure minimum threshold
    except Exception as e:
        print(f"Statistical analysis failed: {e}, using fallback threshold")
        chapter_threshold = 2.5
    
    # Initialize first chapter
    structured_content.append(f"Chapter {chapter_counter}\n")
    
    for i, seg in enumerate(segments):
        try:
            # Chapter detection logic
            if i > 0:
                current_gap = gaps[i]
                prev_seg = segments[i-1]
                
                # Combined acoustic and semantic detection
                if current_gap > chapter_threshold and is_new_chapter(seg, prev_seg):
                    if current_chapter:
                        structured_content.append("\n\n".join(current_chapter))
                        chapter_counter += 1
                        structured_content.append(f"\n\nChapter {chapter_counter}\n")
                        current_chapter = []
            
            # Add text with clean formatting
            clean_text = re.sub(r'\s+', ' ', seg['text']).strip()
            if clean_text:
                current_chapter.append(clean_text)
                
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            continue
    
    # Add final content
    if current_chapter:
        structured_content.append("\n\n".join(current_chapter))
    
    return "".join(structured_content)

def is_new_chapter(current_seg, previous_seg):
    """Enhanced chapter detection with validation"""
    text = f"{previous_seg['text']} {current_seg['text']}".lower()
    
    # Chapter number patterns
    patterns = [
        r"\b(chapter|part|section|book)\s+[xiv\d]+\b",
        r"\b(prologue|epilogue)\b",
        r"\b\d{1,2}\s*:\s*\d{1,2}\b"  # Timecode patterns (e.g., "00:15:23")
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

def format_advanced(text):
    """Professional formatting with validation"""
    # Clean special characters
    text = re.sub(r'\[\s*\]', '', text)  # Remove empty brackets
    text = re.sub(r'([.!?])"', r'\1"', text)  # Fix punctuation inside quotes
    
    # Paragraph optimization
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Merge short paragraphs
    merged = []
    current_para = []
    for para in paragraphs:
        current_para.append(para)
        if sum(len(p) for p in current_para) > 250:
            merged.append(" ".join(current_para))
            current_para = []
    
    if current_para:
        merged.append(" ".join(current_para))
    
    return "\n\n".join(merged)

def main():
    try:
        result = transcribe_with_context()
        structured_text = analyze_structure(result['segments'])
        final_text = format_advanced(structured_text)
        
        with open(OUTPUT_TXT, 'w') as f:
            f.write(final_text)
            
        print(f"Successfully created audiobook text: {OUTPUT_TXT}")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        if 'result' in locals():
            print("Attempting to save partial results...")
            with open(OUTPUT_TXT, 'w') as f:
                f.write("\n".join([seg['text'] for seg in result.get('segments', [])]))

if __name__ == "__main__":
    main()