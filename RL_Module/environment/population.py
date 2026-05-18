"""
Generate 10,000 synthetic students with personality parameters.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from RL_Module.config import POPULATION_SIZE
from RL_Module.environment.student_model import SyntheticStudent


def generate_population(
    n: int = POPULATION_SIZE,
    seed: Optional[int] = None,
) -> List[SyntheticStudent]:
    rng = np.random.default_rng(seed)
    population: List[SyntheticStudent] = []

    for i in range(n):
        student = SyntheticStudent(
            gamma_s=float(rng.uniform(0.1, 1.0)),
            beta_s=float(rng.uniform(0.1, 1.0)),
            lambda_s=float(rng.uniform(0.01, 0.3)),
            rho_s=float(rng.uniform(0.1, 1.0)),
            student_id=i,
        )
        student.student_type = student.classify_type()
        population.append(student)

    return population
