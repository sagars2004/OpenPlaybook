"""
View transformation between image coordinates and court coordinates.
"""
import numpy as np
from typing import Union, Sequence


class ViewTransformer:
    """Transforms points between two coordinate systems using homography."""
    
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize transformer with source and target point correspondences.
        
        Args:
            source: Source points as Nx2 array (e.g., image coordinates)
            target: Target points as Nx2 array (e.g., court coordinates)
        """
        self.source = np.array(source, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        
        if self.source.shape != self.target.shape:
            raise ValueError("Source and target must have the same shape")
        if self.source.shape[0] < 4:
            raise ValueError("At least 4 point correspondences are required")
        
        # Compute homography matrix
        self.homography = self._compute_homography(self.source, self.target)
        self.inverse_homography = self._compute_homography(self.target, self.source)
    
    def _compute_homography(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Compute homography matrix using Direct Linear Transform (DLT) algorithm.
        
        Args:
            src: Source points as Nx2 array
            dst: Destination points as Nx2 array
            
        Returns:
            3x3 homography matrix
        """
        n = src.shape[0]
        A = np.zeros((2 * n, 9))
        
        for i in range(n):
            x, y = src[i]
            u, v = dst[i]
            
            A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]
        
        # Solve using SVD
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        
        # Normalize
        H = H / H[2, 2]
        
        return H
    
    def _apply_transform(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Helper method to apply transformation matrix to points."""
        n = points.shape[0]
        homogeneous = np.ones((n, 3))
        homogeneous[:, :2] = points
        transformed = (matrix @ homogeneous.T).T
        return transformed[:, :2] / transformed[:, 2:3]
    
    def transform_points(self, points: Union[np.ndarray, Sequence], inverse: bool = False) -> np.ndarray:
        """
        Transform points between coordinate systems.
        
        Args:
            points: Points to transform as Nx2 array or list of (x, y) tuples
            inverse: If True, transform from target to source (default: False)
            
        Returns:
            Transformed points as Nx2 array
        """
        points = np.array(points, dtype=np.float32)
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2)")
        
        matrix = self.inverse_homography if inverse else self.homography
        return self._apply_transform(points, matrix)
