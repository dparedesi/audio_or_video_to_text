import os
import subprocess

# Paths to the input video and subtitle files
video_path = "inputs-outputs/6. Daniel y Desi. Vídeo largo con más barra libre.mov"
subtitle_path = "inputs-outputs/2. Daniel & Desi. Video largo-translation.srt"
output_path = "inputs-outputs/merged_video_4k.mp4"

try:
    # Construct the ffmpeg command to burn subtitles into the video with styling and upscale to 4K
    command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', video_path,
        '-vf',
        f"subtitles={subtitle_path}:force_style='FontName=Arial,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H64000000,BorderStyle=0.5,Outline=0.5,Shadow=1',scale=3840:2160",
        '-c:v', 'libx264',  # Use the H.264 codec for video
        '-crf', '18',  # Set the Constant Rate Factor to 18 for high quality
        '-preset', 'slow',  # Use a slower preset for better compression
        '-b:v', '40M',  # Set the video bitrate to 40 Mbps for high quality
        '-c:a', 'aac',  # Use the AAC codec for audio
        '-b:a', '320k',  # Set the audio bitrate to 320 kbps for high quality
        '-color_primaries', 'bt2020',  # Preserve the BT.2020 color profile
        '-color_trc', 'arib-std-b67',  # Preserve the HLG transfer characteristics
        '-colorspace', 'bt2020nc',  # Preserve the BT.2020 non-constant luminance colorspace
        output_path
    ]

    # Execute the command
    subprocess.run(command, check=True)

    print(f"Video with subtitles saved to '{output_path}'")

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")