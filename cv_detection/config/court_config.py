"""
Court configuration and geometry definitions for basketball shot detection.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np


class League(Enum):
    """Basketball league types."""
    NBA = "nba"
    NCAA = "ncaa"
    FIBA = "fiba"


class MeasurementUnit(Enum):
    """Measurement units for court dimensions."""
    FEET = "feet"
    METERS = "meters"


@dataclass
class CourtConfiguration:
    """Configuration for basketball court dimensions and layout."""
    
    league: League
    measurement_unit: MeasurementUnit
    
    # Court dimensions (in feet for NBA)
    court_length: float = 94.0
    court_width: float = 50.0
    
    # Key areas
    key_length: float = 19.0
    key_width: float = 16.0
    
    # Three-point line
    three_point_distance: float = 23.75
    three_point_corner_distance: float = 22.0
    
    # Free throw line
    free_throw_distance: float = 15.0
    
    # Basket
    basket_height: float = 10.0
    basket_diameter: float = 1.5
    
    # Court vertices for transformation (normalized coordinates)
    vertices: List[Tuple[float, float]] = None
    
    # Basket indices in vertices
    left_basket_index: int = 0
    right_basket_index: int = 1
    
    def __post_init__(self):
        """Initialize court vertices after object creation."""
        if self.vertices is None:
            self.vertices = self._get_default_vertices()
    
    def _get_default_vertices(self) -> List[Tuple[float, float]]:
        """Get default court vertices for NBA court."""
        if self.league == League.NBA:
            return [
                # Left basket (backboard center)
                (self.key_length / 2, self.court_width / 2),
                # Right basket (backboard center)
                (self.court_length - self.key_length / 2, self.court_width / 2),
                # Left corner baseline
                (0, 0),
                # Right corner baseline
                (self.court_length, 0),
                # Left corner sideline
                (0, self.court_width),
                # Right corner sideline
                (self.court_length, self.court_width),
                # Center court
                (self.court_length / 2, self.court_width / 2),
                # Left free throw line
                (self.free_throw_distance, self.court_width / 2),
                # Right free throw line
                (self.court_length - self.free_throw_distance, self.court_width / 2),
            ]
        else:
            # Default to NBA dimensions
            return self._get_default_vertices()
    
    def get_court_bounds(self) -> Tuple[float, float, float, float]:
        """Get court bounds as (min_x, min_y, max_x, max_y)."""
        return (0, 0, self.court_length, self.court_width)
    
    def is_inside_court(self, x: float, y: float) -> bool:
        """Check if a point is inside the court bounds."""
        min_x, min_y, max_x, max_y = self.get_court_bounds()
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def get_distance_to_basket(self, x: float, y: float, basket_side: str = "left") -> float:
        """Calculate distance from point to basket."""
        if basket_side == "left":
            basket_x, basket_y = self.vertices[self.left_basket_index]
        else:
            basket_x, basket_y = self.vertices[self.right_basket_index]
        
        return np.sqrt((x - basket_x) ** 2 + (y - basket_y) ** 2)
