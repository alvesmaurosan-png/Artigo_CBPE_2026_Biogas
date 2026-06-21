from __future__ import annotations

from enum import StrEnum


class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    TIME_LIMIT = "TIME_LIMIT"
    EXECUTION_ERROR = "EXECUTION_ERROR"


def classify_outcome(
    solver_status: str | None,
    *,
    process_returncode: int = 0,
    artifacts_complete: bool = False,
) -> SolverStatus:
    """Classify scientific status independently from wrapper-process failures.

    A process may fail after writing a valid solver result (for example, while
    printing to a Windows console). In that case the solver result remains the
    scientific outcome and must not be relabeled as infeasible.
    """

    normalized = (solver_status or "").strip().upper()
    if normalized in {"OPTIMAL", "FEASIBLE", "INFEASIBLE"}:
        return SolverStatus(normalized)
    if normalized in {"NOT_SOLVED", "TIME_LIMIT"}:
        return SolverStatus.TIME_LIMIT
    if process_returncode != 0 or not artifacts_complete:
        return SolverStatus.EXECUTION_ERROR
    return SolverStatus.TIME_LIMIT

