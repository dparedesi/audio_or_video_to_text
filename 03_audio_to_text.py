#!/usr/bin/env python
# pip install openai-whisper torch torchaudio
import argparse
import logging
import os
import pty
import sys
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Whisper CLI with real-time progress via a pseudo-terminal."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="inputs/New_Recording.m4a",
        help="Path to the input audio file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"outputs/transcription_{datetime.now().strftime('%y%m%d_%H%M')}.txt",
        help="Path to the output transcription file."
    )
    # Use parse_known_args to ignore extra Colab arguments.
    args, _ = parser.parse_known_args()
    return args

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def transcribe_with_cli(input_path, model, output_path):
    """
    Runs the Whisper CLI command using a pseudo-terminal so that
    its built-in progress bar and log output are visible in real time.
    After completion, renames the default output file to the provided output_path.
    """
    cmd = [
        "whisper",
        input_path,
        "--model", model,
        "--language", "en",            # Force language to English
        "--output_format", "txt",
        "--output_dir", ".",
        "--verbose", "True"
    ]
    logging.info("Starting transcription with CLI command: %s", " ".join(cmd))
    
    # Use pty.spawn so the CLI thinks it's attached to a terminal.
    exit_status = pty.spawn(cmd)
    
    if exit_status != 0:
        raise Exception("Transcription failed with exit code {}".format(exit_status))
    
    # The CLI writes the output file using the base name of the input.
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    default_output = f"{base_name}.txt"
    if os.path.exists(default_output):
        os.rename(default_output, output_path)
        logging.info("Transcription saved to '%s'.", output_path)
    else:
        raise Exception("Expected output file not found: " + default_output)

def main():
    args = parse_args()
    setup_logging()
    try:
        transcribe_with_cli(args.input, args.model, args.output)
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()