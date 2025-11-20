"""
Application configuration and settings.
"""
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application-wide configuration."""
    
    # Shot tracking parameters (in frames)
    shot_reset_time_seconds: float = 1.7
    minimum_time_between_starts_seconds: float = 0.5
    cooldown_after_made_seconds: float = 0.5
    
    # Keypoint smoothing
    smoothing_buffer_length: int = 3
    
    # Court visualization
    court_scale: float = 20.0
    court_padding: int = 50
    court_line_thickness: int = 4
    
    # Video compression
    ffmpeg_crf: int = 28
    
    # Minimum keypoints required
    min_keypoints_required: int = 4
    
    def get_reset_time_frames(self, fps: float) -> int:
        """Calculate reset time in frames based on FPS."""
        return int(fps * self.shot_reset_time_seconds)
    
    def get_min_frames_between_starts(self, fps: float) -> int:
        """Calculate minimum frames between starts based on FPS."""
        return int(fps * self.minimum_time_between_starts_seconds)
    
    def get_cooldown_frames_after_made(self, fps: float) -> int:
        """Calculate cooldown frames after made shot based on FPS."""
        return int(fps * self.cooldown_after_made_seconds)
