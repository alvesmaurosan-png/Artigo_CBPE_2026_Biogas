"""
Métricas de desempenho energético.
"""

from .peak_metrics import build_peak_metrics_summary, compute_r_peak, compute_peak_energy_kwh

__all__ = ["build_peak_metrics_summary", "compute_r_peak", "compute_peak_energy_kwh"]