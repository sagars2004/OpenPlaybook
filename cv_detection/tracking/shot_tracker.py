"""
Shot event tracking for detecting shot starts, makes, and misses.
"""
from typing import List, Dict, Optional, Any


class ShotEventTracker:
    """Tracks shot events (START, MADE, MISSED) based on detection signals."""
    
    def __init__(
        self,
        reset_time_frames: int,
        minimum_frames_between_starts: int,
        cooldown_frames_after_made: int,
    ):
        """
        Initialize the shot event tracker.
        
        Args:
            reset_time_frames: Frames to wait before resetting state if no activity
            minimum_frames_between_starts: Minimum frames between shot start events
            cooldown_frames_after_made: Frames to wait after a made shot before allowing new start
        """
        self.reset_time_frames = reset_time_frames
        self.minimum_frames_between_starts = minimum_frames_between_starts
        self.cooldown_frames_after_made = cooldown_frames_after_made
        
        self.state = "IDLE"  # IDLE, SHOT_STARTED, SHOT_MADE, SHOT_MISSED
        self.last_start_frame = -1
        self.last_made_frame = -1
        self.last_frame = -1
        
        # Counters for consecutive detections
        self.jump_shot_count = 0
        self.layup_dunk_count = 0
        self.ball_in_basket_count = 0
    
    def update(
        self,
        frame_index: int,
        has_jump_shot: bool,
        has_layup_dunk: bool,
        has_ball_in_basket: bool,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with current frame detections and return any events.
        
        Args:
            frame_index: Current frame number
            has_jump_shot: Whether jump shot action detected
            has_layup_dunk: Whether layup/dunk action detected
            has_ball_in_basket: Whether ball in basket detected
            
        Returns:
            List of events, each with 'event' (START/MADE/MISSED) and 'frame' keys
        """
        events = []
        frames_since_last_start = frame_index - self.last_start_frame
        frames_since_last_made = frame_index - self.last_made_frame
        frames_since_last_update = frame_index - self.last_frame
        
        # Update consecutive detection counters
        if has_jump_shot or has_layup_dunk:
            if has_jump_shot:
                self.jump_shot_count += 1
            if has_layup_dunk:
                self.layup_dunk_count += 1
        else:
            self.jump_shot_count = 0
            self.layup_dunk_count = 0
        
        if has_ball_in_basket:
            self.ball_in_basket_count += 1
        else:
            self.ball_in_basket_count = 0
        
        # Check for reset condition
        if frames_since_last_update >= self.reset_time_frames:
            self.state = "IDLE"
        
        # State machine logic
        if self.state == "IDLE":
            # Check for shot start
            if (self.jump_shot_count >= 3 or self.layup_dunk_count >= 3) and \
               frames_since_last_start >= self.minimum_frames_between_starts and \
               frames_since_last_made >= self.cooldown_frames_after_made:
                self.state = "SHOT_STARTED"
                self.last_start_frame = frame_index
                events.append({"event": "START", "frame": frame_index})
        
        elif self.state == "SHOT_STARTED":
            # Check for made shot
            if self.ball_in_basket_count >= 2:
                self.state = "SHOT_MADE"
                self.last_made_frame = frame_index
                events.append({"event": "MADE", "frame": frame_index})
                self.state = "IDLE"
            
            # Check for missed shot (no ball in basket after timeout)
            elif frames_since_last_start >= self.reset_time_frames:
                self.state = "SHOT_MISSED"
                events.append({"event": "MISSED", "frame": frame_index})
                self.state = "IDLE"
        
        self.last_frame = frame_index
        return events
