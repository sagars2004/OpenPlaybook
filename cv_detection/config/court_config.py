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
