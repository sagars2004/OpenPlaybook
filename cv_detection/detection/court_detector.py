"""
Court keypoint detection wrapper.
"""
import numpy as np
import supervision as sv


class CourtDetector:
    """Wrapper for court keypoint detection."""
    
    def __init__(self, model, confidence_threshold: float = 0.3):
        """
        Initialize court detector.
        
        Args:
            model: Loaded detection model (from inference.get_model)
            confidence_threshold: Minimum confidence for keypoint detection
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
    
    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """
        Detect court keypoints in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            KeyPoints object with detected court keypoints
        """
        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold
        )[0]
        
        return sv.KeyPoints.from_inference(result)
    
    def has_enough_points(self, key_points: sv.KeyPoints, min_points: int = 4, 
                         confidence_threshold: float = 0.5) -> bool:
        """
        Check if enough keypoints are detected with sufficient confidence.
        
        Args:
            key_points: KeyPoints object
            min_points: Minimum number of points required
            confidence_threshold: Minimum confidence for valid points
            
        Returns:
            True if enough valid points are detected
        """
        if key_points.confidence is None:
            return False
        
        key_mask = key_points.confidence[0] > confidence_threshold
        return np.count_nonzero(key_mask) >= min_points
