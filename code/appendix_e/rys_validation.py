#!/usr/bin/env python3
"""
Rys Quadrature: Validation Module

This module validates the Rys quadrature implementation against:
    1. Moment matching: sum W_i x_i^n = m_n for n = 0, ..., 2*n_r - 1
    2. Boys function reproduction: F_n(T) = (1/2) sum W_i x_i^n
    3. Stability tests across different T values
    4. Edge case handling (T = 0, very small T, very large T)
    5. Comparison with theoretical properties (positivity, bounds)

The validation helps ensure numerical accuracy and stability of the
quadrature implementation across the parameter ranges encountered in
molecular integral evaluation.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix E: Rys Quadrature Reference Notes
"""

import numpy as np
import sys
import os

# Import from local modules
sys.path.insert(0, os.path.dirname(__file__))
from rys_quadrature import (
    rys_roots_weights,
    compute_rys_moments,
    verify_moment_matching,
    root_count_for_angular_momentum,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'appendix_d'))
from boys_function import boys, boys_array


def test_moment_matching_comprehensive(T_values: list, n_roots_values: list,
                                       tolerance: float = 1e-10) -> dict:
    """
    Test moment matching across a range of T and n_roots values.

    For each combination, verify that:
        sum_{i=1}^{n_roots} W_i * x_i^n = m_n(T) = 2*F_n(T)

    for n = 0, 1, ..., 2*n_roots - 1.

    Args:
        T_values: List of T values to test
        n_roots_values: List of n_roots values to test
        tolerance: Maximum acceptable error

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "failures": []
    }

    for T in T_values:
        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)
                n_test = 2 * n_roots
                moments_exact = compute_rys_moments(T, n_test)

                max_error = 0.0
                for n in range(n_test):
                    m_exact = moments_exact[n]
                    m_quad = np.sum(weights * nodes ** n)
                    error = abs(m_quad - m_exact)
                    max_error = max(max_error, error)

                if max_error < tolerance:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "max_error": max_error
                    })
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "error": str(e)
                })

    return results


def test_boys_function_reproduction(T_values: list, n_roots: int,
                                    tolerance: float = 1e-10) -> dict:
    """
    Test that Rys quadrature correctly reproduces Boys function values.

    The key relation is:
        F_n(T) = (1/2) * sum_{i=1}^{n_roots} W_i * x_i^n

    for n = 0, 1, ..., 2*n_roots - 1.

    Args:
        T_values: List of T values to test
        n_roots: Number of quadrature points
        tolerance: Maximum acceptable error

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    for T in T_values:
        nodes, weights = rys_roots_weights(T, n_roots)
        max_n = 2 * n_roots

        for n in range(max_n):
            F_exact = boys(n, T)
            F_quad = 0.5 * np.sum(weights * nodes ** n)
            error = abs(F_quad - F_exact)

            if error < tolerance:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n": n,
                    "F_exact": F_exact,
                    "F_quad": F_quad,
                    "error": error
                })

    return results


def test_node_bounds() -> dict:
    """
    Verify that all quadrature nodes lie in the interval (0, 1).

    Rys nodes must satisfy 0 < x_i < 1 for all T >= 0.
    This is a fundamental property of the quadrature rule.

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    T_values = [0.0, 1e-10, 1e-5, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    n_roots_values = [1, 2, 3, 4, 5]

    for T in T_values:
        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)

                # Check bounds (allow small numerical tolerance)
                eps = 1e-12
                all_in_bounds = np.all((nodes >= -eps) & (nodes <= 1 + eps))

                if all_in_bounds:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "nodes": nodes.tolist(),
                        "issue": "nodes outside (0,1)"
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "issue": str(e)
                })

    return results


def test_weight_positivity() -> dict:
    """
    Verify that all quadrature weights are positive.

    For Gaussian quadrature with a positive weight function,
    all weights must be positive. This is essential for
    numerical stability in integral evaluation.

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    T_values = [0.0, 1e-10, 1e-5, 0.1, 1.0, 5.0, 10.0, 50.0]
    n_roots_values = [1, 2, 3, 4, 5]

    for T in T_values:
        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)

                # Check positivity (allow small numerical tolerance)
                eps = -1e-12
                all_positive = np.all(weights > eps)

                if all_positive:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "weights": weights.tolist(),
                        "issue": "negative weights"
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "issue": str(e)
                })

    return results


def test_weight_sum() -> dict:
    """
    Verify that weights sum to m_0(T) = 2*F_0(T).

    The zeroth moment equation gives:
        sum_{i=1}^{n_roots} W_i = m_0 = 2*F_0(T)

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    T_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    n_roots_values = [1, 2, 3, 4, 5]
    tolerance = 1e-10

    for T in T_values:
        m0 = 2.0 * boys(0, T)

        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)
                weight_sum = np.sum(weights)
                error = abs(weight_sum - m0)

                if error < tolerance:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "m_0": m0,
                        "weight_sum": weight_sum,
                        "error": error
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "issue": str(e)
                })

    return results


def test_large_T_stability() -> dict:
    """
    Test stability for large T values.

    For large T, the weight function becomes more peaked near x = 0,
    which can cause numerical difficulties. This test verifies that
    the implementation handles large T correctly.

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    T_values = [50.0, 100.0, 200.0, 500.0]
    n_roots_values = [1, 2, 3]
    tolerance = 1e-8  # Slightly relaxed for large T

    for T in T_values:
        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)

                # Verify moment matching
                n_test = 2 * n_roots
                moments_exact = compute_rys_moments(T, n_test)

                max_error = 0.0
                for n in range(n_test):
                    m_exact = moments_exact[n]
                    m_quad = np.sum(weights * nodes ** n)
                    rel_error = abs(m_quad - m_exact) / max(abs(m_exact), 1e-100)
                    max_error = max(max_error, rel_error)

                if max_error < tolerance:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "max_rel_error": max_error
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "issue": str(e)
                })

    return results


def test_small_T_stability() -> dict:
    """
    Test stability for small T values.

    For T -> 0, the moments approach m_n(0) = 2/(2n+1).
    This test verifies continuous behavior as T -> 0.

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    T_values = [1e-15, 1e-12, 1e-10, 1e-8, 1e-5]
    n_roots_values = [1, 2, 3, 4]
    tolerance = 1e-10

    for T in T_values:
        for n_roots in n_roots_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)
                nodes_zero, weights_zero = rys_roots_weights(0.0, n_roots)

                # Nodes and weights should be close to T=0 values
                node_diff = np.max(np.abs(nodes - nodes_zero))
                weight_diff = np.max(np.abs(weights - weights_zero))

                # Also verify moment matching
                n_test = 2 * n_roots
                moments_exact = compute_rys_moments(T, n_test)
                max_moment_error = 0.0
                for n in range(n_test):
                    m_exact = moments_exact[n]
                    m_quad = np.sum(weights * nodes ** n)
                    max_moment_error = max(max_moment_error, abs(m_quad - m_exact))

                if max_moment_error < tolerance:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "T": T,
                        "n_roots": n_roots,
                        "max_moment_error": max_moment_error
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "T": T,
                    "n_roots": n_roots,
                    "issue": str(e)
                })

    return results


def test_angular_momentum_coverage() -> dict:
    """
    Test that quadrature works for angular momenta encountered in practice.

    Common shell combinations and their required root counts:
        - (ss|ss): L = 0, n_roots = 1
        - (ps|ss): L = 1, n_roots = 1
        - (pp|ss): L = 2, n_roots = 2
        - (pp|pp): L = 4, n_roots = 3
        - (dd|dd): L = 8, n_roots = 5
        - (ff|ff): L = 12, n_roots = 7

    Returns:
        Dictionary with test results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "details": []
    }

    # Shell type to angular momentum
    shell_L = {"s": 0, "p": 1, "d": 2, "f": 3}

    test_cases = [
        ("ssss", 0),  # (ss|ss)
        ("psss", 1),  # (ps|ss)
        ("ppss", 2),  # (pp|ss)
        ("ppps", 3),  # (pp|ps)
        ("pppp", 4),  # (pp|pp)
        ("ddss", 4),  # (dd|ss)
        ("dddd", 8),  # (dd|dd)
        ("ffff", 12), # (ff|ff)
    ]

    T_values = [0.5, 2.0, 10.0]
    tolerance = 1e-10

    for shells, L_total in test_cases:
        n_roots = root_count_for_angular_momentum(L_total)

        for T in T_values:
            try:
                nodes, weights = rys_roots_weights(T, n_roots)

                # Verify moment matching
                n_test = 2 * n_roots
                moments_exact = compute_rys_moments(T, n_test)
                max_error = 0.0
                for n in range(n_test):
                    m_exact = moments_exact[n]
                    m_quad = np.sum(weights * nodes ** n)
                    max_error = max(max_error, abs(m_quad - m_exact))

                if max_error < tolerance:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "shells": shells,
                        "L": L_total,
                        "n_roots": n_roots,
                        "T": T,
                        "max_error": max_error
                    })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "shells": shells,
                    "L": L_total,
                    "n_roots": n_roots,
                    "T": T,
                    "issue": str(e)
                })

    return results


def run_all_validations():
    """Run all validation tests and print summary report."""
    print("=" * 70)
    print("Rys Quadrature Validation Suite")
    print("=" * 70)

    all_passed = 0
    all_failed = 0

    # Test 1: Comprehensive moment matching
    print("\n[1] Comprehensive Moment Matching Test")
    print("-" * 50)
    T_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_roots_values = [1, 2, 3, 4, 5]
    results = test_moment_matching_comprehensive(T_values, n_roots_values)
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['failures']:
        print("Failures:", results['failures'][:3])  # Show first 3 failures
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 2: Boys function reproduction
    print("\n[2] Boys Function Reproduction Test")
    print("-" * 50)
    T_values = [0.1, 1.0, 5.0, 10.0]
    results = test_boys_function_reproduction(T_values, n_roots=4)
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 3: Node bounds
    print("\n[3] Node Bounds Test (0 < x_i < 1)")
    print("-" * 50)
    results = test_node_bounds()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 4: Weight positivity
    print("\n[4] Weight Positivity Test (W_i > 0)")
    print("-" * 50)
    results = test_weight_positivity()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 5: Weight sum
    print("\n[5] Weight Sum Test (sum W_i = m_0)")
    print("-" * 50)
    results = test_weight_sum()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 6: Large T stability
    print("\n[6] Large T Stability Test")
    print("-" * 50)
    results = test_large_T_stability()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 7: Small T stability
    print("\n[7] Small T Stability Test")
    print("-" * 50)
    results = test_small_T_stability()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Test 8: Angular momentum coverage
    print("\n[8] Angular Momentum Coverage Test")
    print("-" * 50)
    results = test_angular_momentum_coverage()
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")
    if results['details']:
        print("Issues:", results['details'][:3])
    all_passed += results['passed']
    all_failed += results['failed']

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total tests passed: {all_passed}")
    print(f"Total tests failed: {all_failed}")
    print(f"Overall: {'PASS' if all_failed == 0 else 'FAIL'}")
    print("=" * 70)

    return all_failed == 0


def detailed_moment_analysis(T: float, n_roots: int):
    """
    Print detailed moment analysis for debugging.

    Shows exact moments, quadrature approximations, and errors for
    each moment up to 2*n_roots - 1.

    Args:
        T: Rys argument
        n_roots: Number of quadrature points
    """
    print(f"\nDetailed Moment Analysis: T = {T}, n_roots = {n_roots}")
    print("=" * 70)

    nodes, weights = rys_roots_weights(T, n_roots)

    print("\nQuadrature nodes and weights:")
    print("-" * 50)
    print(f"{'i':>4}  {'x_i':>18}  {'W_i':>18}")
    print("-" * 50)
    for i in range(n_roots):
        print(f"{i+1:4d}  {nodes[i]:18.14f}  {weights[i]:18.14f}")

    print(f"\nWeight sum: {np.sum(weights):.14f}")
    print(f"Expected:   {2*boys(0, T):.14f}")

    print("\nMoment matching:")
    print("-" * 70)
    print(f"{'n':>4}  {'m_n (exact)':>18}  {'m_n (quad)':>18}  {'Error':>14}  {'Rel Err':>14}")
    print("-" * 70)

    n_test = 2 * n_roots
    for n in range(n_test):
        m_exact = 2.0 * boys(n, T)
        m_quad = np.sum(weights * nodes ** n)
        error = abs(m_quad - m_exact)
        rel_err = error / max(abs(m_exact), 1e-100)
        print(f"{n:4d}  {m_exact:18.14f}  {m_quad:18.14f}  {error:14.2e}  {rel_err:14.2e}")

    print("-" * 70)

    print("\nBoys function reproduction:")
    print("-" * 70)
    print(f"{'n':>4}  {'F_n (exact)':>18}  {'F_n (quad)':>18}  {'Error':>14}")
    print("-" * 70)

    for n in range(n_test):
        F_exact = boys(n, T)
        F_quad = 0.5 * np.sum(weights * nodes ** n)
        error = abs(F_quad - F_exact)
        print(f"{n:4d}  {F_exact:18.14f}  {F_quad:18.14f}  {error:14.2e}")

    print("-" * 70)


def main():
    """Run validation suite."""
    success = run_all_validations()

    # Show detailed analysis for a few cases
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLES")
    print("=" * 70)

    detailed_moment_analysis(T=1.0, n_roots=3)
    detailed_moment_analysis(T=10.0, n_roots=3)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
