"""Tests focused on CVC5 plugin helper structure."""

from smart_inference_ai_fusion.verification.plugins.cvc5_plugin import CVC5Verifier


def test_cvc5_robustness_handlers_have_expected_keys():
    """Robustness handlers map should include all supported test types."""
    verifier = CVC5Verifier()
    handlers = verifier._get_robustness_test_handlers()  # pylint: disable=protected-access

    assert set(handlers.keys()) == {
        "adversarial_robustness",
        "noise_robustness",
        "parameter_sensitivity",
        "distributional_robustness",
    }


def test_cvc5_determine_bounds_violation_below_minimum():
    """Below-minimum values should produce structured violation details."""
    verifier = CVC5Verifier()
    violation = verifier._determine_bounds_violation(  # pylint: disable=protected-access
        value=-1.0,
        min_val=0.0,
        max_val=10.0,
        strict=False,
        index=2,
    )

    assert violation["type"] == "below_minimum"
    assert violation["index"] == 2
    assert violation["expected_min"] == 0.0


def test_cvc5_determine_bounds_violation_above_maximum():
    """Above-maximum values should produce structured violation details."""
    verifier = CVC5Verifier()
    violation = verifier._determine_bounds_violation(  # pylint: disable=protected-access
        value=12.0,
        min_val=0.0,
        max_val=10.0,
        strict=False,
        index=0,
    )

    assert violation["type"] == "above_maximum"
    assert violation["index"] == 0
    assert violation["expected_max"] == 10.0


def test_cvc5_supported_constraints_success():
    """✅ SUCCESS: CVC5 should support core constraint types."""
    verifier = CVC5Verifier()
    constraints = verifier.supported_constraints()
    assert isinstance(constraints, list)
    assert "bounds" in constraints
    assert "postcondition" in constraints


def test_cvc5_supported_constraints_failure_missing_would_be_invalid():
    """❌ FAILURE: If core constraints were missing, verification would be incomplete."""
    verifier = CVC5Verifier()
    constraints = verifier.supported_constraints()
    # This test verifies we DO have essential constraint types
    assert len(constraints) > 5
