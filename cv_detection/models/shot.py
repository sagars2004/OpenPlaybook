from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Shot:
	"""Represents a basketball shot event on the court."""
	x: float
	y: float
	distance: float
	result: bool  # True = made, False = missed
	team: int  # team/class id for coloring
