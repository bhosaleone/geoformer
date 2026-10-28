"""
Domain-Specific Language (DSL) of Discrete Primitives for ARC-AGI.
Provides pure, verifiable spatial, topological, and color transformations.
"""

from typing import List, Tuple, Optional
import numpy as np
import scipy.ndimage as ndi


class ARCPrimitives:
    """Discrete transformations for ARC-AGI grids."""

    # ---------------------------------------------------------
    # 1. Rigid Geometric Transformations
    # ---------------------------------------------------------
    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        return grid.copy()

    @staticmethod
    def rot90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1).copy()

    @staticmethod
    def rot180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2).copy()

    @staticmethod
    def rot270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3).copy()

    @staticmethod
    def flip_h(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid).copy()

    @staticmethod
    def flip_v(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid).copy()

    @staticmethod
    def flip_diag(grid: np.ndarray) -> np.ndarray:
        return grid.T.copy()

    # ---------------------------------------------------------
    # 2. Symmetry Operations
    # ---------------------------------------------------------
    @staticmethod
    def symmetrize_h(grid: np.ndarray) -> np.ndarray:
        """Completes horizontal reflection symmetry (left reflected to right)."""
        h, w = grid.shape
        out = grid.copy()
        half_w = w // 2
        for r in range(h):
            for c in range(half_w):
                if out[r, c] != 0 and out[r, w - 1 - c] == 0:
                    out[r, w - 1 - c] = out[r, c]
                elif out[r, w - 1 - c] != 0 and out[r, c] == 0:
                    out[r, c] = out[r, w - 1 - c]
        return out

    @staticmethod
    def symmetrize_v(grid: np.ndarray) -> np.ndarray:
        """Completes vertical reflection symmetry (top reflected to bottom)."""
        h, w = grid.shape
        out = grid.copy()
        half_h = h // 2
        for r in range(half_h):
            for c in range(w):
                if out[r, c] != 0 and out[h - 1 - r, c] == 0:
                    out[h - 1 - r, c] = out[r, c]
                elif out[h - 1 - r, c] != 0 and out[r, c] == 0:
                    out[r, c] = out[h - 1 - r, c]
        return out

    # ---------------------------------------------------------
    # 3. Bounding Boxes & Cropping
    # ---------------------------------------------------------
    @staticmethod
    def crop_nonzero(grid: np.ndarray) -> np.ndarray:
        """Crops grid to the tightest bounding box of non-zero pixels."""
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return grid.copy()
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return grid[rmin:rmax + 1, cmin:cmax + 1].copy()

    # ---------------------------------------------------------
    # 4. Color Manipulations
    # ---------------------------------------------------------
    @staticmethod
    def recolor(grid: np.ndarray, c_from: int, c_to: int) -> np.ndarray:
        out = grid.copy()
        out[out == c_from] = c_to
        return out

    @staticmethod
    def invert_colors(grid: np.ndarray) -> np.ndarray:
        """Inverts non-zero foreground colors with dominant foreground."""
        vals, counts = np.unique(grid[grid != 0], return_counts=True)
        if len(vals) == 0:
            return grid.copy()
        dom = vals[np.argmax(counts)]
        out = grid.copy()
        out[grid == dom] = vals[np.argmin(counts)]
        return out

    @staticmethod
    def fill_holes(grid: np.ndarray) -> np.ndarray:
        """Fills enclosed 0-holes inside non-zero boundaries with boundary color."""
        out = grid.copy()
        binary_mask = (grid != 0)
        filled = ndi.binary_fill_holes(binary_mask)
        holes = filled & (~binary_mask)
        if np.any(holes):
            # Find surrounding color
            vals, counts = np.unique(grid[grid != 0], return_counts=True)
            fill_c = vals[np.argmax(counts)] if len(vals) > 0 else 1
            out[holes] = fill_c
        return out

    # ---------------------------------------------------------
    # 5. Connected Components & Filtering
    # ---------------------------------------------------------
    @staticmethod
    def keep_largest_component(grid: np.ndarray) -> np.ndarray:
        """Keeps only the largest connected non-zero component."""
        binary = (grid != 0)
        labeled, num_features = ndi.label(binary)
        if num_features <= 1:
            return grid.copy()
        sizes = ndi.sum(binary, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        out = grid.copy()
        out[labeled != largest_label] = 0
        return out

    @staticmethod
    def keep_smallest_component(grid: np.ndarray) -> np.ndarray:
        """Keeps only the smallest connected non-zero component."""
        binary = (grid != 0)
        labeled, num_features = ndi.label(binary)
        if num_features <= 1:
            return grid.copy()
        sizes = ndi.sum(binary, labeled, range(1, num_features + 1))
        smallest_label = np.argmin(sizes) + 1
        out = grid.copy()
        out[labeled != smallest_label] = 0
        return out

    # ---------------------------------------------------------
    # 6. Gravity Transformations
    # ---------------------------------------------------------
    @staticmethod
    def gravity_down(grid: np.ndarray) -> np.ndarray:
        """Drops all non-zero pixels down to the bottom boundary."""
        h, w = grid.shape
        out = np.zeros_like(grid)
        for c in range(w):
            col_vals = grid[:, c]
            nonzero = col_vals[col_vals != 0]
            if len(nonzero) > 0:
                out[h - len(nonzero):, c] = nonzero
        return out

    @staticmethod
    def gravity_up(grid: np.ndarray) -> np.ndarray:
        """Pulls all non-zero pixels up to the top boundary."""
        h, w = grid.shape
        out = np.zeros_like(grid)
        for c in range(w):
            col_vals = grid[:, c]
            nonzero = col_vals[col_vals != 0]
            if len(nonzero) > 0:
                out[:len(nonzero), c] = nonzero
        return out

    # ---------------------------------------------------------
    # 7. Advanced Structural Primitives
    # ---------------------------------------------------------
    @staticmethod
    def kronecker_fractal(grid: np.ndarray) -> np.ndarray:
        """Kronecker fractal tiling: replaces non-zero pixels with full input grid."""
        mask = (grid != 0).astype(int)
        return np.kron(mask, grid).copy()

    @staticmethod
    def shift_right(grid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(grid)
        out[:, 1:] = grid[:, :-1]
        return out

    @staticmethod
    def shift_left(grid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(grid)
        out[:, :-1] = grid[:, 1:]
        return out

    @staticmethod
    def shift_down(grid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(grid)
        out[1:, :] = grid[:-1, :]
        return out

    @staticmethod
    def shift_up(grid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(grid)
        out[:-1, :] = grid[1:, :]
        return out

    @staticmethod
    def point_symmetry_4fold(grid: np.ndarray) -> np.ndarray:
        """Completes 4-fold rotational symmetry about grid center."""
        out = grid.copy()
        for k in (1, 2, 3):
            rotated = np.rot90(grid, k=k)
            mask = (out == 0) & (rotated != 0)
            out[mask] = rotated[mask]
        return out

    @staticmethod
    def enclosed_loop_fill(grid: np.ndarray, boundary_c: int, fill_c: int) -> np.ndarray:
        """Fills holes enclosed by boundary_c with fill_c."""
        out = grid.copy()
        boundary = (grid == boundary_c)
        filled = ndi.binary_fill_holes(boundary)
        interior = filled & (~boundary)
        out[interior] = fill_c
        return out

    @staticmethod
    def apply_color_map(grid: np.ndarray, mapping: dict) -> np.ndarray:
        """Permutes colors according to mapping dictionary."""
        out = grid.copy()
        for src_c, dst_c in mapping.items():
            out[grid == src_c] = dst_c
        return out

    @staticmethod
    def divider_subgrid_and(grid: np.ndarray, fill_color: int = 2) -> np.ndarray:
        """Detects single-color vertical divider column and computes boolean AND between left and right."""
        h, w = grid.shape
        # Check for vertical divider column of uniform color != 0
        for c in range(1, w - 1):
            col_vals = grid[:, c]
            if np.all(col_vals == col_vals[0]) and col_vals[0] != 0:
                left = grid[:, :c]
                right = grid[:, c + 1:]
                if left.shape == right.shape:
                    and_mask = (left != 0) & (right != 0)
                    out = np.zeros_like(left)
                    out[and_mask] = fill_color
                    return out
        return grid.copy()

    @staticmethod
    def rank_recolor_vertical_bars(grid: np.ndarray) -> np.ndarray:
        """Ranks non-zero vertical bars by height and recolors with rank index 1, 2, 3..."""
        out = np.zeros_like(grid)
        h, w = grid.shape
        col_lengths = []
        for c in range(w):
            col_vals = grid[:, c]
            length = np.sum(col_vals != 0)
            if length > 0:
                col_lengths.append((length, c))
        if not col_lengths:
            return grid.copy()
        col_lengths.sort(key=lambda x: -x[0])
        for rank, (length, c) in enumerate(col_lengths, 1):
            for r in range(h):
                if grid[r, c] != 0:
                    out[r, c] = rank
        return out

    @staticmethod
    def geodesic_highway_bridge(grid: np.ndarray) -> np.ndarray:
        """Connects disconnected blocks of the same color along horizontal rows (bridges gaps)."""
        out = grid.copy()
        h, w = grid.shape
        for r in range(h):
            row = grid[r]
            unique_colors = np.unique(row[row != 0])
            for color in unique_colors:
                cols = np.where(row == color)[0]
                if len(cols) >= 2:
                    c_first, c_last = cols[0], cols[-1]
                    # Fill bridge between first and last if separated
                    out[r, c_first:c_last + 1] = color
        return out



    # ---------------------------------------------------------
    # 7. GeoFormer Metric Heuristic Evaluator
    # ---------------------------------------------------------
    @staticmethod
    def compute_dirichlet_energy(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
        """
        Computes the discrete Dirichlet energy distance on the 4-neighbor Von Neumann lattice:
            E(A, B) = ||A - B||_F^2 + lambda * ||L(A) - L(B)||_F^2
        Used by the synthesizer as the sub-nanojoule heuristic cost h(s).
        """
        if grid_a.shape != grid_b.shape:
            return float('inf')
        
        diff = (grid_a != grid_b).astype(np.float32)
        pixel_error = np.sum(diff)
        
        # Spatial smoothness error via 4-neighbor gradient
        dr_a = np.diff(grid_a.astype(np.float32), axis=0)
        dr_b = np.diff(grid_b.astype(np.float32), axis=0)
        grad_error = np.sum((dr_a != dr_b).astype(np.float32))
        
        return float(pixel_error + 0.5 * grad_error)
