#!/usr/bin/env python
# pip install openai-whisper torch torchaudio
import argparse
import logging
import os
import sys
from datetime import datetime
import whisper

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="inputs/New Recording.m4a",
        help="Path to the input audio file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size to use."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output transcription file."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (e.g., 'en', 'es', 'fr'). Use 'auto' for automatic detection."
    )
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def transcribe_audio(input_path, model_name, language):
    """
    Transcribe audio using Whisper Python API.
    """
    logging.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    logging.info(f"Transcribing: {input_path}")
    
    # Set language option
    kwargs = {"verbose": True}
    if language != "auto":
        kwargs["language"] = language
    
    result = model.transcribe(input_path, **kwargs)
    
    return result["text"]

def main():
    args = parse_args()
    setup_logging()
    
    # Validate input file
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        args.output = f"outputs/transcription_{timestamp}.txt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Transcribe
        text = transcribe_audio(args.input, args.model, args.language)
        
        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logging.info(f"Transcription saved to: {args.output}")
        logging.info(f"Total length: {len(text)} characters")
        
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()