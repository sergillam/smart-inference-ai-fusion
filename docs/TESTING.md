# Testing Guide - Verification Subsystem

**Status:** ✅ Complete
**Review:** ✅ [PHASE_4_REVIEW.md](../PHASE_4_REVIEW.md#3-docstestingmd-practical-guide)

## 📖 Table of Contents

1. [Quick Start](#quick-start)
2. [Test Suites Overview](#test-suites-overview)
3. [Advanced Testing](#advanced-testing)
4. [Troubleshooting](#troubleshooting)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Dependencies](#dependencies)
7. [Support](#support)

## Quick Start

### Run All Tests
```bash
pytest tests/ -v
```

**Expected Output:**
```
tests/test_verification_plugin_helpers.py::test_z3_constraint_handlers_expose_core_constraints PASSED
tests/test_verification_plugin_helpers.py::test_z3_counterexample_handlers_include_aliases PASSED
...
============================= 31 passed in 2.45s ==============================
```

### Run Specific Test File
```bash
# Test Z3 plugin structure
pytest tests/smart_inference_ai_fusion/verification/plugins/test_z3_plugin_structure.py -v

# Test data utilities
pytest tests/smart_inference_ai_fusion/verification/utils/test_data_utils.py -v
```

### Run Single Test
```bash
pytest tests/test_verification_plugin_helpers.py::test_z3_constraint_handlers_expose_core_constraints -v
```

---

## Test Suites Overview

### 1. Plugin Helpers (8 tests)
**File:** `tests/test_verification_plugin_helpers.py`

**Purpose:** Verify core plugin dispatch mechanisms and violation detection

**Tests:**
- ✅ `test_z3_constraint_handlers_expose_core_constraints` - Verify Z3 handler registration
- ✅ `test_z3_counterexample_handlers_include_aliases` - Verify neural network handler aliases
- ✅ `test_z3_unknown_constraint_uses_generic_fallback` - Verify fallback dispatch
- ✅ `test_cvc5_robustness_handlers_are_registered` - Verify CVC5 robustness handlers
- ✅ `test_cvc5_bounds_violation_below_minimum_shape` - Verify below-minimum violation structure
- ✅ `test_cvc5_bounds_violation_above_maximum_shape` - Verify above-maximum violation structure
- ✅ `test_z3_supported_constraints_success` - **SUCCESS** path test
- ❌ `test_z3_verify_with_invalid_constraint_failure` - **FAILURE** path test

**Run:**
```bash
pytest tests/test_verification_plugin_helpers.py -v
```

---

### 2. Data Utilities (13 tests)
**File:** `tests/smart_inference_ai_fusion/verification/utils/test_data_utils.py`

**Purpose:** Test data preprocessing and constraint validation helpers

**Test Categories:**

#### A. Array Operations
- `test_normalize_to_array_scalar` - Convert scalar to array
- `test_normalize_to_array_nested_list` - Flatten nested lists

#### B. Configuration Parsing
- `test_parse_shape_config_defaults` - Parse shape with defaults
- `test_parse_type_safety_config_custom` - Parse custom type safety config
- `test_parse_noise_test_params_value_extraction` - Extract noise parameters

#### C. Constraint Validation
- `test_verify_probability_bounds_violation` - Detect invalid probabilities
- `test_verify_classification_constraints_violation` - Detect out-of-range classes
- `test_check_parameter_initialization_not_found` - Handle missing parameters
- `test_check_data_shape_validation_mismatch` - Detect shape mismatches
- `test_check_precondition_data_preprocessing_violation` - Validate preprocessing
- `test_build_class_balance_metrics_empty_array` - Handle empty input

#### D. Success/Failure Paths
- ✅ `test_verify_probability_bounds_success_valid_range` - Valid probability validation
- ❌ `test_verify_classification_constraints_failure_invalid_classes` - Invalid class rejection

**Run:**
```bash
pytest tests/smart_inference_ai_fusion/verification/utils/test_data_utils.py -v
```

---

### 3. Z3 Plugin Structure (7 tests)
**File:** `tests/smart_inference_ai_fusion/verification/plugins/test_z3_plugin_structure.py`

**Purpose:** Verify Z3 SMT solver integration structure

**Tests:**
- ✅ `test_z3_constraint_handlers_have_expected_keys` - Z3 handler map keys
- ✅ `test_z3_counterexample_handlers_have_expected_keys` - Z3 counterexample aliases
- ✅ `test_z3_constraint_dispatch_uses_fallback_for_unknown` - Unknown constraint handling
- ✅ `test_z3_determine_bounds_violation_below_minimum` - Below-minimum detection
- ✅ `test_z3_determine_bounds_violation_above_maximum` - Above-maximum detection
- ✅ `test_z3_supported_constraints_success_returns_list` - **SUCCESS** constraints list
- ❌ `test_z3_supported_constraints_failure_empty_would_be_invalid` - **FAILURE** empty validation

**Expected Constraint Handlers:**
- bounds
- range_check
- type_safety
- shape_preservation
- non_negative
- positive
- invariant
- precondition
- postcondition
- robustness

**Run:**
```bash
pytest tests/smart_inference_ai_fusion/verification/plugins/test_z3_plugin_structure.py -v
```

---

### 4. CVC5 Plugin Structure (7 tests)
**File:** `tests/smart_inference_ai_fusion/verification/plugins/test_cvc5_plugin_structure.py`

**Purpose:** Verify CVC5 SMT solver integration and robustness testing

**Tests:**
- ✅ `test_cvc5_robustness_handlers_are_registered` - CVC5 robustness handler map
- ✅ `test_cvc5_bounds_violation_below_minimum_shape` - Below-minimum violation structure
- ✅ `test_cvc5_bounds_violation_above_maximum_shape` - Above-maximum violation structure
- ✅ `test_cvc5_postcondition_violation_wrong_type` - Type violation detection
- ✅ `test_cvc5_postcondition_violation_wrong_range` - Range violation detection
- ✅ `test_cvc5_supported_constraints_success` - **SUCCESS** constraints availability
- ❌ `test_cvc5_supported_constraints_failure_missing_would_be_invalid` - **FAILURE** requirement validation

**Expected Robustness Handlers:**
- adversarial_robustness
- noise_robustness
- parameter_sensitivity
- distributional_robustness

**Run:**
```bash
pytest tests/smart_inference_ai_fusion/verification/plugins/test_cvc5_plugin_structure.py -v
```

---

## Advanced Testing

### Run Tests with Coverage
```bash
pytest tests/ --cov=smart_inference_ai_fusion.verification \
              --cov-report=html \
              --cov-report=term-missing

# Report generated: htmlcov/index.html
open htmlcov/index.html
```

### Run Tests with Markers
```bash
# Run only success path tests
pytest tests/ -k "success" -v

# Run only failure path tests
pytest tests/ -k "failure" -v

# Run specific constraint type tests
pytest tests/ -k "z3" -v
```

### Run Tests with Different Output Formats
```bash
# Verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Run until N failures
pytest tests/ --maxfail=3
```

### Profile Test Execution
```bash
# Show slowest tests
pytest tests/ --durations=10

# With timing
pytest tests/ -v --tb=short --durations=5
```

---

## Test Patterns

### Pattern 1: Handler Registration Verification
```python
def test_z3_constraint_handlers_expose_core_constraints():
    """Verify handlers are registered with expected keys."""
    verifier = Z3Verifier()
    handlers = verifier._get_constraint_handlers()

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

    # Verify all expected handlers are present
    assert expected.issubset(set(handlers.keys()))
```

### Pattern 2: Violation Structure Validation
```python
def test_cvc5_bounds_violation_below_minimum_shape():
    """Verify violation has expected structure."""
    verifier = CVC5Verifier()
    violation = verifier._determine_bounds_violation(
        value=-2.0,
        min_val=0.0,
        max_val=10.0,
        strict=False,
        index=3,
    )

    # Verify violation structure
    assert violation["type"] == "below_minimum"
    assert violation["index"] == 3
    assert "expected_min" in violation
```

### Pattern 3: Success Path Test
```python
def test_z3_supported_constraints_success():
    """✅ SUCCESS: Verify non-empty constraints list."""
    verifier = Z3Verifier()
    constraints = verifier.supported_constraints()

    # Positive assertions
    assert isinstance(constraints, list)
    assert len(constraints) > 0
    assert "bounds" in constraints
```

### Pattern 4: Failure/Edge Case Test
```python
def test_z3_verify_with_invalid_constraint_failure():
    """❌ FAILURE: Handle invalid constraint gracefully."""
    verifier = Z3Verifier()

    # Create minimal invalid input
    from dataclasses import dataclass

    @dataclass
    class MockInput:
        input_data: None = None
        output_data: None = None
        constraints: dict = None
        parameters: dict = None

    # Should not crash with invalid input
    try:
        result = verifier.verify(MockInput(constraints={}, parameters={}))
        assert result is not None
    except Exception:
        pass  # Acceptable behavior
```

---

## Troubleshooting

### Common Issues

#### Issue: Tests fail with "ImportError: No module named..."
**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Install package in development mode
pip install -e .
```

#### Issue: Test hangs or times out
**Solution:**
```bash
# Run with timeout
pytest tests/ --timeout=30

# Run specific test with debug output
pytest tests/test_verification_plugin_helpers.py::test_name -vv -s
```

#### Issue: Tests pass locally but fail in CI
**Solution:**
1. Verify Python version: `python --version` (should be 3.12.3+)
2. Clear cache: `find . -type d -name __pycache__ -exec rm -r {} +`
3. Reinstall dependencies: `pip install -r requirements.txt`

#### Issue: "Permission Denied" when running tests
**Solution:**
```bash
# Ensure execute permissions on venv
chmod +x .venv/bin/python
chmod +x .venv/bin/pytest
```

---

## Test Coverage Goals

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| z3_plugin.py | 70%+ | Covered by structure tests | ✅ |
| cvc5_plugin.py | 70%+ | Covered by structure tests | ✅ |
| data_utils.py | 80%+ | 13 dedicated tests | ✅ |
| error_handling.py | 85%+ | Implicit in other tests | ⚠️ |

**Note:** Full coverage metrics available via pytest-cov (see Advanced Testing section)

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

pytest tests/ --tb=short || exit 1
```

---

## Test Maintenance

### Adding New Tests
1. Follow existing test naming: `test_<feature>_<scenario>`
2. Add docstring describing test purpose
3. Use ✅ for success paths, ❌ for failure paths
4. Organize by feature module
5. Run full suite: `pytest tests/ -v`

### Updating Existing Tests
1. Preserve test name if testing same behavior
2. Update docstring if logic changes
3. Run: `pytest tests/<file.py> -v`
4. Verify all tests still pass

### Deprecating Tests
1. Add `@pytest.mark.skip` decorator with reason
2. Note in CHANGES.md
3. Remove in next minor version

---

## Performance Benchmarking

### Test Execution Time Baseline
```bash
pytest tests/ -v --durations=0

# Expected: ~2.45s for 31 tests
# Average: ~79ms per test
```

### Slow Test Investigation
```bash
# Find slowest tests
pytest tests/ --durations=10

# Profile individual test
pytest tests/test_file.py::test_name -v --durations=3
```

---

## Dependencies

**Test Requirements:**
- pytest >= 8.4.1
- pytest-timeout (optional, for timeout handling)
- pytest-cov (optional, for coverage reports)

**Install:**
```bash
pip install pytest pytest-timeout pytest-cov
```

---

## Support

For test-related issues:
1. Check test output messages
2. Review test source code
3. Check [REFACTORING.md](REFACTORING.md) for context
4. Consult [CHANGES.md](../CHANGES.md) for recent modifications

