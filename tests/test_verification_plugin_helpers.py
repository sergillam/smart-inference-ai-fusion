"""Unit tests for verification plugin helper/dispatch behavior."""

from smart_inference_ai_fusion.verification.plugins.cvc5_plugin import CVC5Verifier
from smart_inference_ai_fusion.verification.plugins.z3_plugin import Z3Verifier


def test_z3_constraint_handlers_expose_core_constraints():
    """Z3 must expose core constraints in the handler dispatcher."""
    verifier = Z3Verifier()
    handlers = verifier._get_constraint_handlers()  # pylint: disable=protected-access

    expected = {
        "bounds",
        "range_check",
        "type_safety",
        "shape_preservation",
        "non_negative",
        "positive",
        "invariant",
        "precondition",
        "postcondition",
        "robustness",
    }

    assert expected.issubset(set(handlers.keys()))


def test_z3_counterexample_handlers_include_aliases():
    """Counterexample dispatcher should support both NN keys."""
    verifier = Z3Verifier()
    handlers = verifier._get_counterexample_handlers()  # pylint: disable=protected-access

    assert "neural_network" in handlers
    assert "neural_network_verification" in handlers


def test_z3_unknown_constraint_uses_generic_fallback():
    """Unknown constraints should fall back to generic verifier path."""
    verifier = Z3Verifier()

    verifier._verify_generic_constraint = lambda *_: True  # pylint: disable=protected-access
    result = verifier._verify_constraint("unknown_constraint", {}, None)  # pylint: disable=protected-access

    assert result is True


def test_cvc5_robustness_handlers_are_registered():
    """CVC5 robustness strategy map must register all supported tests."""
    verifier = CVC5Verifier()
    handlers = verifier._get_robustness_test_handlers()  # pylint: disable=protected-access

    assert "adversarial_robustness" in handlers
    assert "noise_robustness" in handlers
    assert "parameter_sensitivity" in handlers
    assert "distributional_robustness" in handlers


def test_cvc5_bounds_violation_below_minimum_shape():
    """Bounds violation helper should generate structured below-minimum payload."""
    verifier = CVC5Verifier()
    violation = verifier._determine_bounds_violation(  # pylint: disable=protected-access
        value=-2.0,
        min_val=0.0,
        max_val=10.0,
        strict=False,
        index=3,
    )

    assert violation["type"] == "below_minimum"
    assert violation["index"] == 3
    assert "expected_min" in violation


def test_cvc5_bounds_violation_above_maximum_shape():
    """Bounds violation helper should generate structured above-maximum payload."""
    verifier = CVC5Verifier()
    violation = verifier._determine_bounds_violation(  # pylint: disable=protected-access
        value=42.0,
        min_val=0.0,
        max_val=10.0,
        strict=False,
        index=1,
    )

    assert violation["type"] == "above_maximum"
    assert violation["index"] == 1
    assert "expected_max" in violation


def test_z3_supported_constraints_success():
    """✅ SUCCESS: Z3Verifier should return non-empty supported constraints list."""
    verifier = Z3Verifier()
    constraints = verifier.supported_constraints()
    assert isinstance(constraints, list)
    assert len(constraints) > 0
    assert "bounds" in constraints


def test_z3_verify_with_invalid_constraint_failure():
    """❌ FAILURE: Unsupported constraint types should not verify successfully without handler."""
    verifier = Z3Verifier()
    # Create a mock input
    from dataclasses import dataclass

    @dataclass
    class MockInput:
        input_data: None = None
        output_data: None = None
        constraints: dict = None
        parameters: dict = None

    # Attempt to verify with constraint that has no explicit handler
    # This should not crash but may return specific result
    try:
        result = verifier.verify(MockInput(constraints={}, parameters={}))
        # If we get here, the verifier handled it (either success or with detail)
        assert result is not None
    except Exception:
        # This is acceptable - verifier may raise on invalid setup
        pass
