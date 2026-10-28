"""
src/geoformer/autonomous/schedules.py
=====================================
Multi-scale relaxation step schedules for continuous Q-TAC dynamics.
Implements Continued-Fraction and Fibonacci Diophantine convergence steps.
"""

from typing import List, Sequence
import math


def continued_fraction_convergents(alpha: float, max_terms: int = 12) -> List[int]:
    """
    Return denominators q_n of convergents to alpha (positive integers).
    
    alpha = [a_0; a_1, a_2, ...]
    p_{-1} = 1, p_0 = a_0, p_n = a_n * p_{n-1} + p_{n-2}
    q_{-1} = 0, q_0 = 1,   q_n = a_n * q_{n-1} + q_{n-2}
    """
    if max_terms <= 0:
        return []

    # Handle float near-zero or negative
    x = abs(float(alpha))
    
    # Continued fraction expansion
    cf = []
    curr = x
    for _ in range(max_terms):
        a = int(math.floor(curr))
        cf.append(a)
        frac = curr - a
        if frac < 1e-12:
            break
        curr = 1.0 / frac

    if not cf:
        return [1]

    # Compute convergents denominators
    q_prev = 0
    q_curr = 1
    q_list = [q_curr]

    for a in cf[1:]:
        q_next = a * q_curr + q_prev
        q_prev = q_curr
        q_curr = q_next
        if q_curr > 0:
            q_list.append(int(q_curr))
        if len(q_list) >= max_terms:
            break

    return q_list


def fibonacci_schedule(max_steps: int = 21) -> List[int]:
    """
    Return deduplicated increasing Fibonacci step counts <= max_steps.
    Example for max_steps=13: [1, 2, 3, 5, 8, 13]
    """
    if max_steps < 1:
        return [1]

    fibs = [1]
    a, b = 1, 2
    while b <= max_steps:
        fibs.append(b)
        a, b = b, a + b

    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in fibs:
        if f not in seen and f > 0:
            seen.add(f)
            result.append(f)

    return result


def make_relaxation_schedule(
    mode: str = "fibonacci",            # "fibonacci" | "cf" | "fixed"
    alpha: float = (math.sqrt(5) - 1) / 2, # for "cf" (golden ratio conjugate)
    max_steps: int = 13,
    fixed_steps: int = 4,
) -> List[int]:
    """
    Returns a strictly increasing list of cumulative relaxation step counts
    to be used by Q-TAC.
    Example: [1, 2, 3, 5, 8, 13]
    """
    mode = mode.lower()
    if mode == "fixed":
        steps = max(1, int(fixed_steps))
        return [steps]

    elif mode == "fibonacci":
        sched = fibonacci_schedule(max_steps=max_steps)
        if not sched:
            sched = [1]
        return sorted(list(set(sched)))

    elif mode == "cf":
        raw_qs = continued_fraction_convergents(alpha, max_terms=16)
        filtered = [q for q in raw_qs if 1 <= q <= max_steps]
        if not filtered:
            filtered = [1]
        return sorted(list(set(filtered)))

    else:
        raise ValueError(f"Unknown schedule mode: {mode}. Choose 'fibonacci', 'cf', or 'fixed'.")
