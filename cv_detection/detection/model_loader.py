"""
Model loading utilities for player and court detection.
"""
import os
from pathlib import Path
from typing import Optional
from inference import get_model


class ModelLoader:
    """Loads and manages detection models."""
    
    def __init__(
        self,
        player_model_path: Optional[str] = None,
        player_model_id: Optional[str] = None,
        court_model_path: Optional[str] = None,
        court_model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize model loader.
        
        Args:
            player_model_path: Path to local player detection model file
            player_model_id: Roboflow model ID for player detection (e.g., "basketball-player-detection-3-ycjdo/4")
            court_model_path: Path to local court detection model file
            court_model_id: Roboflow model ID for court detection
            api_key: Roboflow API key (can also be set via ROBOFLOW_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        self.player_model = None
        self.court_model = None
        
        # Load player model
        if player_model_path:
            self.player_model = self._load_from_path(player_model_path)
        elif player_model_id:
            self.player_model = self._load_from_roboflow(player_model_id)
        else:
            raise ValueError("Either player_model_path or player_model_id must be provided")
        
        # Load court model
        if court_model_path:
            self.court_model = self._load_from_path(court_model_path)
        elif court_model_id:
            self.court_model = self._load_from_roboflow(court_model_id)
        else:
            raise ValueError("Either court_model_path or court_model_id must be provided")
    
    def _load_from_path(self, model_path: str):
        """Load model from local file path."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return get_model(model_path=model_path)
    
    def _load_from_roboflow(self, model_id: str):
        """Load model from Roboflow using model ID."""
        if not self.api_key:
            raise ValueError(
                "Roboflow API key required. Set ROBOFLOW_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return get_model(model_id=model_id, api_key=self.api_key)
    
    def get_player_model(self):
        """Get the loaded player detection model."""
        return self.player_model
    
    def get_court_model(self):
        """Get the loaded court detection model."""
        return self.court_model
