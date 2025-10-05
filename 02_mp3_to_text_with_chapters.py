import re
import sys
import os
import argparse
import whisper
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to structured text with chapter detection."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="inputs/audiobook.mp3",
        help="Path to the input audio file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size to use."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/structured_audiobook.txt",
        help="Path to the output text file."
    )
    return parser.parse_args()

def transcribe_with_context(audio_path, model_size):
    """Optimized Whisper transcription with memory management"""
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    
    print("Processing audio...")
    result = model.transcribe(
        audio_path,
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
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Audio file not found: {args.input}")
        sys.exit(1)
    
    try:
        result = transcribe_with_context(args.input, args.model)
        structured_text = analyze_structure(result['segments'])
        final_text = format_advanced(structured_text)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        with open(args.output, 'w') as f:
            f.write(final_text)
            
        print(f"Successfully created audiobook text: {args.output}")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        if 'result' in locals():
            print("Attempting to save partial results...")
            fallback_path = args.output.replace('.txt', '_partial.txt')
            with open(fallback_path, 'w') as f:
                f.write("\n".join([seg['text'] for seg in result.get('segments', [])]))
            print(f"Partial results saved to: {fallback_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()