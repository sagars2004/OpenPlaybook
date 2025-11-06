"""
Distance calculation utilities for basketball shot detection.
"""
import numpy as np
from typing import Union, Sequence


def euclidean_distance(
    start_point: Union[Sequence[float], np.ndarray],
    end_point: Union[Sequence[float], np.ndarray]
) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        start_point: Starting point as (x, y) or array of shape (2,)
        end_point: Ending point as (x, y) or array of shape (2,)
        
    Returns:
        Euclidean distance as float
        
    Raises:
        ValueError: If points don't have shape (2,)
    """
    start_point_array = np.asarray(start_point, dtype=float)
    end_point_array = np.asarray(end_point, dtype=float)

    if start_point_array.shape != (2,) or end_point_array.shape != (2,):
        raise ValueError("Both points must have shape (2,).")

    return float(np.linalg.norm(end_point_array - start_point_array))
