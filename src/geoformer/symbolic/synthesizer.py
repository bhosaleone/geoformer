"""
Neuro-Symbolic Program Synthesizer for ARC-AGI.
Uses GeoFormer Metric Dirichlet Energy to prune search space over DSL compositions.
"""

from typing import List, Tuple, Callable, Optional, Dict
import numpy as np
from .dsl import ARCPrimitives


class ARCSynthesizer:
    def __init__(self, max_depth: int = 3, beam_width: int = 12):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.primitives = self._build_primitive_catalog()

    def _build_primitive_catalog(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        catalog = {
            "identity": ARCPrimitives.identity,
            "rot90": ARCPrimitives.rot90,
            "rot180": ARCPrimitives.rot180,
            "rot270": ARCPrimitives.rot270,
            "flip_h": ARCPrimitives.flip_h,
            "flip_v": ARCPrimitives.flip_v,
            "flip_diag": ARCPrimitives.flip_diag,
            "symmetrize_h": ARCPrimitives.symmetrize_h,
            "symmetrize_v": ARCPrimitives.symmetrize_v,
            "crop_nonzero": ARCPrimitives.crop_nonzero,
            "fill_holes": ARCPrimitives.fill_holes,
            "keep_largest": ARCPrimitives.keep_largest_component,
            "keep_smallest": ARCPrimitives.keep_smallest_component,
            "gravity_down": ARCPrimitives.gravity_down,
            "gravity_up": ARCPrimitives.gravity_up,
            "invert_colors": ARCPrimitives.invert_colors,
            "kronecker_fractal": ARCPrimitives.kronecker_fractal,
            "shift_right": ARCPrimitives.shift_right,
            "shift_left": ARCPrimitives.shift_left,
            "shift_down": ARCPrimitives.shift_down,
            "shift_up": ARCPrimitives.shift_up,
            "point_symmetry_4fold": ARCPrimitives.point_symmetry_4fold,
            "divider_subgrid_and": ARCPrimitives.divider_subgrid_and,
            "rank_recolor_vertical_bars": ARCPrimitives.rank_recolor_vertical_bars,
            "geodesic_highway_bridge": ARCPrimitives.geodesic_highway_bridge,
        }
        return catalog

    def _generate_dynamic_primitives(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        dyn_catalog = {}
        all_colors = set()
        for x, y in train_pairs:
            all_colors.update(np.unique(x).tolist())
            all_colors.update(np.unique(y).tolist())
        all_colors = list(all_colors)

        # 1. Individual recoloring rules
        for c1 in all_colors:
            for c2 in all_colors:
                if c1 != c2:
                    name = f"recolor({c1}->{c2})"
                    dyn_catalog[name] = (lambda grid, from_c=c1, to_c=c2: ARCPrimitives.recolor(grid, from_c, to_c))
                    
                    # 2. Enclosed boundary flood fills
                    fill_name = f"enclosed_fill({c1}->{c2})"
                    dyn_catalog[fill_name] = (lambda grid, b_c=c1, f_c=c2: ARCPrimitives.enclosed_loop_fill(grid, b_c, f_c))

        # 3. Inductive Full Color Map Permutation
        color_map = {}
        consistent_map = True
        for x, y in train_pairs:
            if x.shape != y.shape:
                consistent_map = False
                break
            for cx, cy in zip(x.flat, y.flat):
                if cx in color_map and color_map[cx] != cy:
                    consistent_map = False
                    break
                color_map[cx] = cy
            if not consistent_map:
                break
        if consistent_map and len(color_map) > 1:
            captured_map = dict(color_map)
            dyn_catalog["inductive_full_color_map"] = (lambda grid, m=captured_map: ARCPrimitives.apply_color_map(grid, m))

        return dyn_catalog

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        """
        Synthesizes a transformation program that satisfies all train_pairs,
        then evaluates it on test_input.
        
        Returns:
            (solved, program_name, test_prediction)
        """
        all_prims = dict(self.primitives)
        dyn_prims = self._generate_dynamic_primitives(train_pairs)
        all_prims.update(dyn_prims)

        # -------------------------------------------------------------
        # Level 1: Single Primitive Search
        # -------------------------------------------------------------
        for name, fn in all_prims.items():
            if self._verify_program([fn], train_pairs):
                pred = fn(test_input)
                return True, name, pred

        # -------------------------------------------------------------
        # Level 2: Two-Step Composition with GeoFormer Metric Pruning
        # -------------------------------------------------------------
        # Evaluate Level 1 candidates by average Dirichlet energy on training targets
        ranked_l1 = []
        for name, fn in all_prims.items():
            total_energy = 0.0
            valid = True
            for x, y in train_pairs:
                try:
                    out = fn(x)
                    energy = ARCPrimitives.compute_dirichlet_energy(out, y)
                    total_energy += energy
                except Exception:
                    valid = False
                    break
            if valid:
                ranked_l1.append((total_energy, name, fn))

        ranked_l1.sort(key=lambda item: item[0])
        # Keep top beam candidates
        beam_candidates = ranked_l1[:self.beam_width]

        for _, name1, fn1 in beam_candidates:
            for name2, fn2 in all_prims.items():
                if self._verify_program([fn1, fn2], train_pairs):
                    prog_name = f"{name2} o {name1}"
                    pred = fn2(fn1(test_input))
                    return True, prog_name, pred

        # -------------------------------------------------------------
        # Level 3: Three-Step Composition
        # -------------------------------------------------------------
        if self.max_depth >= 3:
            # Re-rank level 2 candidates
            ranked_l2 = []
            for _, name1, fn1 in beam_candidates[:6]:
                for name2, fn2 in all_prims.items():
                    total_energy = 0.0
                    valid = True
                    for x, y in train_pairs:
                        try:
                            out = fn2(fn1(x))
                            energy = ARCPrimitives.compute_dirichlet_energy(out, y)
                            total_energy += energy
                        except Exception:
                            valid = False
                            break
                    if valid:
                        ranked_l2.append((total_energy, (name1, name2), (fn1, fn2)))

            ranked_l2.sort(key=lambda item: item[0])
            for _, (name1, name2), (fn1, fn2) in ranked_l2[:self.beam_width]:
                for name3, fn3 in all_prims.items():
                    if self._verify_program([fn1, fn2, fn3], train_pairs):
                        prog_name = f"{name3} o {name2} o {name1}"
                        pred = fn3(fn2(fn1(test_input)))
                        return True, prog_name, pred

        # Default fallback: lowest energy level-1 prediction
        if len(ranked_l1) > 0:
            best_fn = ranked_l1[0][2]
            best_name = f"heuristic_fallback({ranked_l1[0][1]})"
            return False, best_name, best_fn(test_input)

        return False, "identity_fallback", test_input.copy()

    def _verify_program(self, pipeline: List[Callable[[np.ndarray], np.ndarray]], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for x, y in train_pairs:
            curr = x
            try:
                for fn in pipeline:
                    curr = fn(curr)
                if curr.shape != y.shape or not np.array_equal(curr, y):
                    return False
            except Exception:
                return False
        return True
