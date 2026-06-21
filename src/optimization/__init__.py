"""
Módulo de otimização do projeto.

Contém:
- algoritmo genético multiobjetivo (NSGA-II)
"""

from .ga_nsga2 import NSGA2Optimizer

__all__ = ["NSGA2Optimizer"]