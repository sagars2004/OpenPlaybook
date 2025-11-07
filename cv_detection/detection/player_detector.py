"""
Player and action detection wrapper.
"""
import numpy as np
import supervision as sv


class PlayerDetector:
    """Wrapper for player detection and action classification."""
    
    # Class IDs (from the example code)
    BALL_IN_BASKET_CLASS_ID = 1
    JUMP_SHOT_CLASS_ID = 5
    LAYUP_DUNK_CLASS_ID = 6
    
    def __init__(self, model, confidence_threshold: float = 0.3, iou_threshold: float = 0.7):
        """
        Initialize player detector.
        
        Args:
            model: Loaded detection model (from inference.get_model)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players and actions in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            Detections object with bounding boxes and class IDs
        """
        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )[0]
        
        return sv.Detections.from_inference(result)
    
    def has_jump_shot(self, detections: sv.Detections) -> bool:
        """Check if jump shot action is detected."""
        return len(detections[detections.class_id == self.JUMP_SHOT_CLASS_ID]) > 0
    
    def has_layup_dunk(self, detections: sv.Detections) -> bool:
        """Check if layup/dunk action is detected."""
        return len(detections[detections.class_id == self.LAYUP_DUNK_CLASS_ID]) > 0
    
    def has_ball_in_basket(self, detections: sv.Detections) -> bool:
        """Check if ball in basket is detected."""
        return len(detections[detections.class_id == self.BALL_IN_BASKET_CLASS_ID]) > 0
    
    def get_shot_anchors(self, detections: sv.Detections, anchor: sv.Position = sv.Position.BOTTOM_CENTER) -> np.ndarray:
        """
        Get anchor coordinates for shot detection (player positions).
        
        Args:
            detections: Detections object
            anchor: Position to extract (default: BOTTOM_CENTER for player feet)
            
        Returns:
            Array of anchor coordinates as Nx2
        """
        shot_detections = detections[
            (detections.class_id == self.JUMP_SHOT_CLASS_ID) |
            (detections.class_id == self.LAYUP_DUNK_CLASS_ID)
        ]
        
        if len(shot_detections) == 0:
            return np.empty((0, 2), dtype=float)
        
        return shot_detections.get_anchors_coordinates(anchor=anchor)
