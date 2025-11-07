"""
Annotation setup for shot visualization on video frames.
"""
import supervision as sv
from typing import List


def create_triangle_annotator(
    colors: List[str],
    base: int = 25,
    height: int = 21,
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
) -> sv.TriangleAnnotator:
    """
    Create a triangle annotator for shot markers.
    
    Args:
        colors: List of hex color strings
        base: Base width of triangle
        height: Height of triangle
        color_lookup: Color lookup method
        
    Returns:
        TriangleAnnotator instance
    """
    color_palette = sv.ColorPalette.from_hex(colors)
    return sv.TriangleAnnotator(
        color=color_palette,
        base=base,
        height=height,
        color_lookup=color_lookup,
    )


def create_text_annotator(
    font_path: str,
    font_size: int = 60,
    colors: List[str] = None,
    text_color: sv.Color = sv.Color.WHITE,
    text_offset: tuple = (0, -30),
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
    text_position: sv.Position = sv.Position.TOP_CENTER,
) -> sv.RichLabelAnnotator:
    """
    Create a text annotator for shot labels.
    
    Args:
        font_path: Path to font file
        font_size: Font size in pixels
        colors: List of hex color strings for text background
        text_color: Color of text
        text_offset: Offset for text position (x, y)
        color_lookup: Color lookup method
        text_position: Position of text relative to marker
        
    Returns:
        RichLabelAnnotator instance
    """
    color_palette = sv.ColorPalette.from_hex(colors) if colors else None
    
    return sv.RichLabelAnnotator(
        font_path=font_path,
        font_size=font_size,
        color=color_palette,
        text_color=text_color,
        text_offset=text_offset,
        color_lookup=color_lookup,
        text_position=text_position,
    )


