"""
Video processing utilities.
"""
import subprocess
import supervision as sv
from pathlib import Path
from typing import Optional


def compress_video(
    input_path: Path,
    output_path: Path,
    crf: int = 28,
    overwrite: bool = True,
) -> bool:
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-crf", str(crf),
        str(output_path),
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")


def get_video_info(video_path: Path) -> sv.VideoInfo:
    
    return sv.VideoInfo.from_video_path(str(video_path))


def get_total_frames(video_info: sv.VideoInfo) -> Optional[int]:
    
    return getattr(video_info, "total_frames", getattr(video_info, "frame_count", None))
