"""Tests focused on Z3 plugin dispatch and strategy structure."""

from smart_inference_ai_fusion.verification.plugins.z3_plugin import Z3Verifier


def test_z3_constraint_handlers_have_expected_keys():
    """Main constraint handler map should expose key verification routes."""
    verifier = Z3Verifier()
    handlers = verifier._get_constraint_handlers()  # pylint: disable=protected-access

    for key in [
        "bounds",
        "range_check",
        "type_safety",
        "shape_preservation",
        "integer_arithmetic",
        "non_negative",
        "invariant",
        "postcondition",
        "robustness",
    ]:
        assert key in handlers


def test_z3_counterexample_handlers_have_expected_keys():
    """Counterexample map should expose core handlers and aliases."""
    verifier = Z3Verifier()
    handlers = verifier._get_counterexample_handlers()  # pylint: disable=protected-access

    assert "bounds" in handlers
    assert "range_check" in handlers
    assert "neural_network" in handlers
    assert "neural_network_verification" in handlers


def test_z3_constraint_dispatch_uses_fallback_for_unknown():
    """Unknown constraints should use generic verification fallback path."""
    verifier = Z3Verifier()

    verifier._verify_generic_constraint = lambda *_: True  # pylint: disable=protected-access
    assert (
        verifier._verify_constraint("unknown_key", {}, None) is True
    )  # pylint: disable=protected-access


def test_z3_supported_constraints_success_returns_list():
    """✅ SUCCESS: supported_constraints should return non-empty list."""
    verifier = Z3Verifier()
    constraints = verifier.supported_constraints()
    assert isinstance(constraints, list)
    assert len(constraints) > 0
    assert "bounds" in constraints


def test_z3_supported_constraints_failure_empty_would_be_invalid():
    """❌ FAILURE: If supported_constraints returned empty list, it would be invalid."""
    verifier = Z3Verifier()
    constraints = verifier.supported_constraints()
    # This test verifies we're NOT returning an empty list
    assert len(constraints) > 0
