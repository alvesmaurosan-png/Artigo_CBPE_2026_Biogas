"""
Módulos econômicos do projeto.
"""

from .lcoe import build_economics_summary, compute_lcoe, compute_capex_total

__all__ = ["build_economics_summary", "compute_lcoe", "compute_capex_total"]