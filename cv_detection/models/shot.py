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
	
	def __str__(self) -> str:
		"""Return string representation of shot."""
		result_str = "MADE" if self.result else "MISSED"
		return f"Shot({result_str}, {self.distance:.1f}ft, team={self.team}, pos=({self.x:.1f}, {self.y:.1f}))"


def extract_made(shots: List[Shot]) -> List[Shot]:
	"""Return only shots that were made (result == True)."""
	return [shot for shot in shots if shot.result]


def extract_xy(shots: List[Shot]) -> np.ndarray:
	"""Return Nx2 array of shot (x, y) coordinates."""
	return np.array([[shot.x, shot.y] for shot in shots], dtype=float)


def extract_class_id(shots: List[Shot]) -> np.ndarray:
	"""Return N array of team/class ids."""
	return np.array([shot.team for shot in shots], dtype=int)


def extract_label(shots: List[Shot]) -> np.ndarray:
	"""Return N array of human-readable distance labels in feet."""
	return np.array([f"{shot.distance:.2f} ft" for shot in shots], dtype=str)
