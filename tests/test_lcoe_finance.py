from src.economics.lcoe import capital_recovery_factor


def test_capital_recovery_factor_standard_case():
    crf = capital_recovery_factor(0.08, 20)
    assert abs(crf - 0.10185220882315059) < 1e-12


def test_capital_recovery_factor_zero_rate():
    crf = capital_recovery_factor(0.0, 20)
    assert abs(crf - 0.05) < 1e-12
