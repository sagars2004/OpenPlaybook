"""
Court drawing utilities for basketball visualization.
"""
import numpy as np
import cv2
import supervision as sv
from typing import Optional

# Import will be relative or absolute depending on usage
try:
    from config.court_config import CourtConfiguration
except ImportError:
    from cv_detection.config.court_config import CourtConfiguration


def draw_court(
    config: CourtConfiguration,
    scale: float = 20.0,
    padding: int = 50,
    line_thickness: int = 4,
    background_color: sv.Color = sv.Color.WHITE,
    line_color: sv.Color = sv.Color.BLACK,
) -> np.ndarray:
    """
    Draw a basketball court based on configuration.
    
    Args:
        config: CourtConfiguration object with court dimensions
        scale: Scaling factor for court dimensions (pixels per unit)
        padding: Padding around court in pixels
        line_thickness: Thickness of court lines
        background_color: Background color of court
        line_color: Color of court lines
        
    Returns:
        Court image as numpy array (BGR format)
    """
    # Calculate court dimensions in pixels
    court_w = int(config.court_width * scale)
    court_h = int(config.court_length * scale)
    
    # Create canvas with padding
    canvas_w = court_w + 2 * padding
    canvas_h = court_h + 2 * padding
    court_image = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * np.array(background_color.as_bgr(), dtype=np.uint8)
    
    # Offset for padding
    offset_x = padding
    offset_y = padding
    
    # Draw outer court boundary
    cv2.rectangle(
        court_image,
        (offset_x, offset_y),
        (offset_x + court_w, offset_y + court_h),
        line_color.as_bgr(),
        line_thickness
    )
    
    # Draw center line
    center_x = offset_x + court_w // 2
    cv2.line(
        court_image,
        (center_x, offset_y),
        (center_x, offset_y + court_h),
        line_color.as_bgr(),
        line_thickness
    )
    
    # Draw center circle
    center_y = offset_y + court_h // 2
    radius = int(min(court_w, court_h) * 0.15)
    cv2.circle(
        court_image,
        (center_x, center_y),
        radius,
        line_color.as_bgr(),
        line_thickness
    )
    
    # Draw key areas (paint) on both sides
    key_w = int(config.key_width * scale)
    key_h = int(config.key_length * scale)
    
    # Left key
    cv2.rectangle(
        court_image,
        (offset_x, center_y - key_w // 2),
        (offset_x + key_h, center_y + key_w // 2),
        line_color.as_bgr(),
        line_thickness
    )
    
    # Right key
    cv2.rectangle(
        court_image,
        (offset_x + court_w - key_h, center_y - key_w // 2),
        (offset_x + court_w, center_y + key_w // 2),
        line_color.as_bgr(),
        line_thickness
    )
    
    # Draw free throw circles
    free_throw_radius = int(key_w * 0.5)
    free_throw_x_left = offset_x + key_h
    free_throw_x_right = offset_x + court_w - key_h
    
    # Left free throw circle (semicircle)
    cv2.ellipse(
        court_image,
        (free_throw_x_left, center_y),
        (free_throw_radius, free_throw_radius),
        0, 270, 90,
        line_color.as_bgr(),
        line_thickness
    )
    
    # Right free throw circle (semicircle)
    cv2.ellipse(
        court_image,
        (free_throw_x_right, center_y),
        (free_throw_radius, free_throw_radius),
        0, 90, 270,
        line_color.as_bgr(),
        line_thickness
    )
    
    return court_image


def draw_made_and_miss_on_court(
    config: CourtConfiguration,
    made_xy: Optional[np.ndarray] = None,
    miss_xy: Optional[np.ndarray] = None,
    made_size: int = 25,
    miss_size: int = 25,
    made_color: sv.Color = sv.Color.from_hex("#007A33"),
    miss_color: sv.Color = sv.Color.from_hex("#850101"),
    made_thickness: int = 6,
    miss_thickness: int = 6,
    scale: float = 20.0,
    padding: int = 50,
    line_thickness: int = 4,
) -> np.ndarray:
    """
    Draw court with made and missed shots.
    
    Args:
        config: CourtConfiguration object
        made_xy: Array of made shot positions as Nx2 (court coordinates)
        miss_xy: Array of missed shot positions as Nx2 (court coordinates)
        made_size: Size of made shot markers
        miss_size: Size of missed shot markers
        made_color: Color for made shots
        miss_color: Color for missed shots
        made_thickness: Thickness of made shot markers
        miss_thickness: Thickness of missed shot markers
        scale: Scaling factor for court dimensions
        padding: Padding around court
        line_thickness: Thickness of court lines
        
    Returns:
        Court image with shots drawn on it
    """
    # Draw base court
    court_image = draw_court(
        config=config,
        scale=scale,
        padding=padding,
        line_thickness=line_thickness
    )
    
    offset_x = padding
    offset_y = padding
    
    # Draw made shots
    if made_xy is not None and len(made_xy) > 0:
        for x, y in made_xy:
            pixel_x = int(offset_x + x * scale)
            pixel_y = int(offset_y + y * scale)
            cv2.circle(
                court_image,
                (pixel_x, pixel_y),
                made_size,
                made_color.as_bgr(),
                made_thickness
            )
    
    # Draw missed shots
    if miss_xy is not None and len(miss_xy) > 0:
        for x, y in miss_xy:
            pixel_x = int(offset_x + x * scale)
            pixel_y = int(offset_y + y * scale)
            cv2.circle(
                court_image,
                (pixel_x, pixel_y),
                miss_size,
                miss_color.as_bgr(),
                miss_thickness
            )
    
    return court_image
