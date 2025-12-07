#!/usr/bin/env python3
"""Convert multi-channel recordings into Descript-friendly stereo WAV files.

Usage:
    python conversion/convert_for_descript.py path/to/input.m4a

The script produces a single stereo WAV that lets ffmpeg downmix every channel.
Outputs live next to the source file and reuse its stem with `_stereo.wav`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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
        description="Create a Descript-friendly stereo WAV from a source file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Audio file with >2 channels (e.g. .m4a) to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the converted files (defaults to the source file directory).",
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


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    src = args.input_file.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else src.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    info = run_ffprobe(src)
    channels = info.get("channels")
    layout = info.get("channel_layout") or "unknown"
    print(f"Detected audio stream: {channels} channels, layout '{layout}'.")
    if channels and channels <= 2:
        print(
            "Warning: source already has 2 or fewer channels; Descript should accept it as-is."
        )

    base = output_dir / src.stem
    output_path = base.with_suffix(".m4a")
    convert_to_stereo_mix(
        src,
        output_path,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
