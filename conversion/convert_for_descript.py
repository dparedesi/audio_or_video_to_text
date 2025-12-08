#!/usr/bin/env python3
"""Convert multi-channel recordings into Descript-friendly stereo M4A files.

Usage:
    # Convert ALL audio files in the script's folder (default behaviour):
    python conversion/convert_for_descript.py

    # Convert a specific file:
    python conversion/convert_for_descript.py path/to/input.opus

Outputs are written to an `output/` sub-folder next to the script (created
automatically). Each output file keeps the original stem with an `.m4a` extension.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Audio extensions we will attempt to convert
AUDIO_EXTENSIONS = {".opus", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma"}


def ensure_tool(name: str) -> None:
    if shutil.which(name):
        return
    raise SystemExit(
        f"Required tool '{name}' is not available on PATH. "
        "Install ffmpeg (which includes ffprobe) and re-run the script."
    )


def run_ffprobe(path: Path) -> Dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_name,channels,channel_layout,sample_rate",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"ffprobe failed for '{path}':\n{result.stderr.strip() or result.stdout}"
        )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise SystemExit(f"No audio streams found in '{path}'.")
    return streams[0]


def run_command(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        joined = " ".join(cmd)
        raise SystemExit(f"Command failed ({joined}): {exc}") from exc


def convert_to_stereo_mix(
    src: Path, dst: Path, sample_rate: int, overwrite: bool
) -> None:
    if dst.exists() and not overwrite:
        print(f"[skip] Target already exists: {dst}")
        return
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "2",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-ar",
        str(sample_rate),
        str(dst),
    ]
    print(f"[convert] {dst.name}")
    run_command(cmd)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Descript-friendly stereo M4A files from audio sources."
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        type=Path,
        help="Audio file(s) to convert. If omitted, converts all audio files in the script's folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the converted files (defaults to 'output/' next to the script).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate for the outputs (default: 48000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args(argv)


def convert_single_file(src: Path, output_dir: Path, sample_rate: int, overwrite: bool) -> bool:
    """Convert one audio file. Returns True on success, False on skip/failure."""
    info = run_ffprobe(src)
    channels = info.get("channels")
    layout = info.get("channel_layout") or "unknown"
    print(f"Detected audio stream: {channels} channels, layout '{layout}'.")
    if channels and channels <= 2:
        print(
            "Warning: source already has 2 or fewer channels; Descript should accept it as-is."
        )

    output_path = output_dir / f"{src.stem}.m4a"
    convert_to_stereo_mix(src, output_path, sample_rate=sample_rate, overwrite=overwrite)
    return True


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    # Determine the script's own directory (where we look for files)
    script_dir = Path(__file__).resolve().parent

    # Default output directory: `output/` inside script's directory
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else script_dir / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather input files
    if args.input_files:
        sources = [p.expanduser().resolve() for p in args.input_files]
    else:
        # Auto-discover audio files in the script's directory
        sources = sorted(
            p for p in script_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not sources:
            print(f"No audio files found in {script_dir}")
            return
        print(f"Found {len(sources)} audio file(s) to convert.\n")

    for src in sources:
        if not src.exists():
            print(f"[skip] Input file not found: {src}")
            continue
        print(f"--- {src.name}")
        try:
            convert_single_file(src, output_dir, args.sample_rate, args.overwrite)
            print("Done.\n")
        except SystemExit as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
