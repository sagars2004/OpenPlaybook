"""
Model configuration and constants for detection models.
"""
import os
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for detection models."""
    
    # Model paths/IDs
    player_model_path: Optional[str] = None
    player_model_id: Optional[str] = None
    court_model_path: Optional[str] = None
    court_model_id: Optional[str] = None
    
    # API key
    roboflow_api_key: Optional[str] = None
    
    # Detection thresholds
    player_confidence_threshold: float = 0.3
    player_iou_threshold: float = 0.7
    court_confidence_threshold: float = 0.3
    keypoint_confidence_threshold: float = 0.5
    
    # Class IDs
    ball_in_basket_class_id: int = 1
    jump_shot_class_id: int = 5
    layup_dunk_class_id: int = 6
    
    def __post_init__(self):
        """Load from environment variables if not provided."""
        if self.roboflow_api_key is None:
            self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
        
        if self.player_model_id is None:
            self.player_model_id = os.getenv("PLAYER_MODEL_ID")
        
        if self.court_model_id is None:
            self.court_model_id = os.getenv("COURT_MODEL_ID")
    
    def get_player_model_source(self) -> Tuple[str, Optional[str]]:
        """
        Get player model source (path or ID).
        
        Returns:
            Tuple of (source_type, value) where source_type is 'path' or 'id'
        """
        if self.player_model_path:
            return ("path", self.player_model_path)
        elif self.player_model_id:
            return ("id", self.player_model_id)
        else:
            raise ValueError("Either player_model_path or player_model_id must be set")
    
    def get_court_model_source(self) -> Tuple[str, Optional[str]]:
        """
        Get court model source (path or ID).
        
        Returns:
            Tuple of (source_type, value) where source_type is 'path' or 'id'
        """
        if self.court_model_path:
            return ("path", self.court_model_path)
        elif self.court_model_id:
            return ("id", self.court_model_id)
        else:
            raise ValueError("Either court_model_path or court_model_id must be set")


# Default model IDs (from example code)
DEFAULT_PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
