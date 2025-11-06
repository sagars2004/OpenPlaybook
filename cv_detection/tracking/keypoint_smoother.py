"""
Keypoint smoothing for stable court detection across video frames.
"""
from collections import deque
from typing import Optional
import numpy as np


class KeyPointsSmoother:
    """Smooths keypoints over multiple frames using a rolling buffer."""
    
    def __init__(self, length: int):
        """
        Initialize the smoother with a buffer of specified length.
        
        Args:
            length: Maximum number of frames to keep in buffer for smoothing
        """
        self.length = length
        self.buffer = deque(maxlen=length)
    
    def update(
        self,
        xy: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        conf_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Update the smoother with new keypoint coordinates and return smoothed result.
        
        Args:
            xy: Keypoint coordinates with shape (1, N, 2) where N is number of keypoints
            confidence: Optional confidence scores with shape (1, N)
            conf_threshold: Minimum confidence threshold for valid keypoints
            
        Returns:
            Smoothed keypoint coordinates with shape (1, N, 2)
        """
        assert xy.ndim == 3 and xy.shape[0] == 1
        xy_f = xy.astype(np.float32, copy=True)

        if confidence is not None:
            assert confidence.shape[:2] == xy.shape[:2]
            mask = (confidence >= conf_threshold)[..., None]
            xy_f = np.where(mask, xy_f, np.nan)

        self.buffer.append(xy_f)
        stacked = np.stack(self.buffer, axis=0)

        if np.isnan(stacked).any():
            mean_xy = np.nanmean(stacked, axis=0)
        else:
            mean_xy = stacked.mean(axis=0)

        return mean_xy
