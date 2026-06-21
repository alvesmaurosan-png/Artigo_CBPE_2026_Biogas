from cbpe.status import SolverStatus, classify_outcome


def test_console_failure_does_not_override_valid_solver_result() -> None:
    assert classify_outcome("OPTIMAL", process_returncode=1, artifacts_complete=True) is SolverStatus.OPTIMAL


def test_execution_failure_is_not_infeasibility() -> None:
    assert classify_outcome(None, process_returncode=1, artifacts_complete=False) is SolverStatus.EXECUTION_ERROR


def test_time_limit_is_explicit() -> None:
    assert classify_outcome("NOT_SOLVED", artifacts_complete=True) is SolverStatus.TIME_LIMIT
