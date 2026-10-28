"""
tests/autonomous/test_schedules.py
==================================
Unit tests for multi-scale Q-TAC relaxation schedules.
"""

import math
import pytest
from geoformer.autonomous.schedules import (
    continued_fraction_convergents,
    fibonacci_schedule,
    make_relaxation_schedule,
)


def test_fibonacci_schedule():
    sched = fibonacci_schedule(13)
    # Deduplicated increasing version
    assert sched == [1, 2, 3, 5, 8, 13]
    
    # Boundary tests
    assert fibonacci_schedule(1) == [1]
    assert fibonacci_schedule(0) == [1]
    sched_21 = fibonacci_schedule(21)
    assert sched_21 == [1, 2, 3, 5, 8, 13, 21]


def test_continued_fraction_convergents_phi():
    phi = (math.sqrt(5) - 1) / 2 # golden ratio conjugate ≈ 0.6180339887
    qs = continued_fraction_convergents(phi, max_terms=8)
    # Known Fibonacci denominators for [0; 1, 1, 1, 1, 1, 1, ...]
    # q_0=1, q_1=1, q_2=2, q_3=3, q_4=5, q_5=8, q_6=13, q_7=21
    assert len(qs) == 8
    assert qs[0] == 1
    assert qs[1] == 1
    assert qs[2] == 2
    assert qs[3] == 3
    assert qs[4] == 5
    assert qs[5] == 8
    assert qs[6] == 13
    assert qs[7] == 21


def test_make_relaxation_schedule_fixed():
    sched = make_relaxation_schedule("fixed", fixed_steps=4)
    assert sched == [4]
    sched_8 = make_relaxation_schedule("fixed", fixed_steps=8)
    assert sched_8 == [8]


def test_make_relaxation_schedule_properties():
    modes = ["fibonacci", "cf", "fixed"]
    for mode in modes:
        sched = make_relaxation_schedule(mode=mode, max_steps=21, fixed_steps=4)
        assert isinstance(sched, list)
        assert len(sched) > 0
        # All elements must be strictly positive integers
        assert all(isinstance(x, int) and x > 0 for x in sched)
        # Strictly increasing (no duplicates)
        assert sched == sorted(list(set(sched)))
