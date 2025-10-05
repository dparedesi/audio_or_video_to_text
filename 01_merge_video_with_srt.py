import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge video with SRT subtitles using ffmpeg."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--subtitles",
        type=str,
        required=True,
        help="Path to the SRT subtitle file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/merged_video.mp4",
        help="Path to the output video file."
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["low", "medium", "high", "4k"],
        help="Output quality preset."
    )
    return parser.parse_args()

def get_quality_settings(quality):
    """Return ffmpeg quality settings based on preset."""
    settings = {
        "low": {"crf": "28", "bitrate": "5M", "scale": None},
        "medium": {"crf": "23", "bitrate": "10M", "scale": None},
        "high": {"crf": "18", "bitrate": "20M", "scale": None},
        "4k": {"crf": "18", "bitrate": "40M", "scale": "3840:2160"}
    }
    return settings[quality]

def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.subtitles):
        print(f"Error: Subtitle file not found: {args.subtitles}")
        sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get quality settings
    quality = get_quality_settings(args.quality)
    
    try:
        # Build subtitle filter with styling
        subtitle_filter = f"subtitles={args.subtitles}:force_style='FontName=Arial,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H64000000,BorderStyle=0.5,Outline=0.5,Shadow=1'"
        
        # Add scaling if needed
        if quality["scale"]:
            vf_filter = f"{subtitle_filter},scale={quality['scale']}"
        else:
            vf_filter = subtitle_filter
        
        # Construct the ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', args.video,
            '-vf', vf_filter,
            '-c:v', 'libx264',  # Use the H.264 codec for video
            '-crf', quality["crf"],
            '-preset', 'slow',  # Use a slower preset for better compression
            '-b:v', quality["bitrate"],
            '-c:a', 'aac',  # Use the AAC codec for audio
            '-b:a', '320k',  # Set the audio bitrate to 320 kbps
            args.output
        ]
        
        print(f"Processing video with {args.quality} quality settings...")
        # Execute the command
        subprocess.run(command, check=True)
        
        print(f"Video with subtitles saved to '{args.output}'")
    
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
