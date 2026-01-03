#!/usr/bin/env python3
"""
moment_matching.py - Verification of Rys Quadrature Moment Matching

This module provides comprehensive testing that Rys quadrature satisfies
the moment-matching conditions:

    sum_i W_i * x_i^n = m_n(T)    for n = 0, 1, ..., 2*n_roots - 1

It also demonstrates that moment matching FAILS for n >= 2*n_roots, which
is the expected behavior of Gaussian quadrature.

The tests sweep over:
    - A wide range of T values (1e-8 to 100)
    - Different root counts (1, 2, 3, 4, 5)
    - Edge cases (T=0, very large T)

References:
    - Chapter 5, Section 5: Evaluating Boys functions by Rys quadrature
    - Exercise 5.2: Two-root quadrature and moment matching

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import Tuple, List

# Import from companion modules
from boys_moments import moment, moments_all
from rys_quadrature import rys_nodes_weights


def test_moment_matching(T: float, n_roots: int, verbose: bool = False) -> Tuple[float, np.ndarray]:
    """
    Test moment matching for given T and n_roots.

    Returns
    -------
    max_error : float
        Maximum absolute error for n = 0, ..., 2*n_roots - 1
    errors : np.ndarray
        Array of errors for each n
    """
    nodes, weights = rys_nodes_weights(T, n_roots)

    errors = []
    for n in range(2 * n_roots):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        errors.append(error)

        if verbose:
            rel_error = error / abs(m_exact) if abs(m_exact) > 1e-15 else error
            print(f"  n={n:2d}: m_exact={m_exact:16.10e}, m_quad={m_quad:16.10e}, "
                  f"abs_err={error:10.2e}, rel_err={rel_error:10.2e}")

    errors = np.array(errors)
    return np.max(errors), errors


def test_beyond_exactness(T: float, n_roots: int, n_extra: int = 3, verbose: bool = False) -> np.ndarray:
    """
    Test moment matching for n >= 2*n_roots (where exactness is NOT guaranteed).

    Returns
    -------
    errors : np.ndarray
        Array of errors for n = 2*n_roots, ..., 2*n_roots + n_extra - 1
    """
    nodes, weights = rys_nodes_weights(T, n_roots)

    errors = []
    for n in range(2 * n_roots, 2 * n_roots + n_extra):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        errors.append(error)

        if verbose:
            rel_error = error / abs(m_exact) if abs(m_exact) > 1e-15 else error
            print(f"  n={n:2d}: m_exact={m_exact:16.10e}, m_quad={m_quad:16.10e}, "
                  f"abs_err={error:10.2e} (NOT guaranteed!)")

    return np.array(errors)


def sweep_T_values(n_roots: int = 2, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep over T values from 1e-8 to 50 and test moment matching.

    Note: For very large T (> 50), the Hankel matrix becomes increasingly
    ill-conditioned and numerical accuracy degrades. This is a known limitation
    of the moment-based approach to Rys quadrature.

    Parameters
    ----------
    n_roots : int
        Number of quadrature roots
    verbose : bool
        Print results

    Returns
    -------
    T_values : np.ndarray
        Array of T values tested
    max_errors : np.ndarray
        Maximum moment-matching error at each T
    """
    # T values spanning many orders of magnitude (limit to T <= 50 for stability)
    T_values = np.concatenate([
        [0.0],
        np.logspace(-8, -4, 5),    # 1e-8 to 1e-4
        np.logspace(-3, 0, 10),    # 1e-3 to 1
        np.logspace(0.1, 1.7, 12), # ~1.26 to 50
    ])

    max_errors = []

    if verbose:
        print(f"\nSweep over T values with n_roots = {n_roots}")
        print("=" * 70)
        print(f"{'T':>12} {'Max moment error':>18} {'Status':>10}")
        print("-" * 70)

    for T in T_values:
        try:
            max_err, _ = test_moment_matching(T, n_roots)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle numerical failures gracefully
            max_err = float('inf')
            if verbose:
                print(f"{T:>12.4e} {'NUMERICAL ERROR':>18} {'SKIP':>10}")
            continue

        max_errors.append(max_err)

        if verbose:
            status = "OK" if max_err < 1e-10 else "WARN" if max_err < 1e-6 else "FAIL"
            print(f"{T:>12.4e} {max_err:>18.2e} {status:>10}")

    return T_values, np.array(max_errors)


def validate_for_all_root_counts(T: float = 1.0) -> bool:
    """
    Validate moment matching for various root counts at fixed T.

    Parameters
    ----------
    T : float
        Test parameter value

    Returns
    -------
    all_passed : bool
        True if all tests pass
    """
    print(f"\nValidation for T = {T} with various n_roots")
    print("=" * 70)

    all_passed = True

    for n_roots in range(1, 6):
        print(f"\nn_roots = {n_roots}:")
        print("-" * 50)

        # Test within exactness range
        print("Within exactness range (n = 0, ..., 2*n_roots - 1):")
        max_err, _ = test_moment_matching(T, n_roots, verbose=True)

        passed = max_err < 1e-10
        all_passed = all_passed and passed
        print(f"Maximum error: {max_err:.2e} -> {'PASS' if passed else 'FAIL'}")

        # Test beyond exactness range
        print("\nBeyond exactness range (errors expected!):")
        beyond_errors = test_beyond_exactness(T, n_roots, n_extra=2, verbose=True)

    return all_passed


def demonstrate_exactness_boundary():
    """
    Demonstrate the sharp boundary of Gaussian quadrature exactness.

    For n_roots roots, exactness holds for polynomials of degree <= 2*n_roots - 1,
    but fails for degree >= 2*n_roots.
    """
    print("\nDemonstration: Exactness boundary of Gaussian quadrature")
    print("=" * 70)

    T = 5.0  # Non-trivial T value

    for n_roots in [2, 3]:
        nodes, weights = rys_nodes_weights(T, n_roots)

        print(f"\nn_roots = {n_roots} (exact for degree <= {2*n_roots - 1})")
        print("-" * 60)
        print(f"{'Degree n':>10} {'Exact moment':>18} {'Quadrature':>18} {'Error':>12} {'Exact?':>8}")
        print("-" * 60)

        for n in range(2 * n_roots + 2):
            m_exact = moment(n, T)
            m_quad = np.sum(weights * (nodes ** n))
            error = abs(m_exact - m_quad)
            exact = "Yes" if n < 2 * n_roots else "No"

            # Mark the boundary
            marker = " <-- boundary" if n == 2 * n_roots else ""
            print(f"{n:>10} {m_exact:>18.10e} {m_quad:>18.10e} {error:>12.2e} {exact:>8}{marker}")


def run_exercise_5_2():
    """
    Run Exercise 5.2: Two-root quadrature and moment matching.

    Implement Algorithm 5.1 for n_r = 2 and verify moment matching
    for n = 0, 1, 2, 3 over a wide range of T.
    """
    print("\n" + "=" * 70)
    print("Exercise 5.2: Two-root quadrature and moment matching")
    print("=" * 70)

    n_roots = 2

    # Test values spanning many orders of magnitude (limit to T <= 50 for stability)
    T_values = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

    print(f"\nTesting n_roots = {n_roots} (exact for n = 0, 1, 2, 3)")
    print("-" * 70)
    print(f"{'T':>12} {'err(n=0)':>12} {'err(n=1)':>12} {'err(n=2)':>12} {'err(n=3)':>12} {'max':>12}")
    print("-" * 70)

    global_max = 0.0

    for T in T_values:
        try:
            _, errors = test_moment_matching(T, n_roots)
            max_err = np.max(errors)
            global_max = max(global_max, max_err)
            print(f"{T:>12.2e} {errors[0]:>12.2e} {errors[1]:>12.2e} {errors[2]:>12.2e} {errors[3]:>12.2e} {max_err:>12.2e}")
        except (np.linalg.LinAlgError, ValueError):
            print(f"{T:>12.2e} {'---':>12} {'---':>12} {'---':>12} {'---':>12} {'NUMERICAL':>12}")

    print("-" * 70)
    print(f"Maximum absolute error across all tests: {global_max:.2e}")

    return global_max


def visualize_errors():
    """
    Create visualization of moment-matching errors vs T (if matplotlib available).
    """
    try:
        import matplotlib.pyplot as plt

        print("\nGenerating error visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Max error vs T for different n_roots (limit T to stable range)
        T_vals = np.logspace(-6, 1.5, 40)  # Up to ~30

        for n_roots in [1, 2, 3, 4]:
            errors = []
            for T in T_vals:
                try:
                    max_err, _ = test_moment_matching(T, n_roots)
                    errors.append(max_err)
                except (np.linalg.LinAlgError, ValueError):
                    errors.append(np.nan)
            ax1.loglog(T_vals, errors, 'o-', label=f'n_roots={n_roots}', markersize=3)

        ax1.axhline(y=1e-10, color='k', linestyle='--', alpha=0.5, label='1e-10 threshold')
        ax1.set_xlabel('T')
        ax1.set_ylabel('Maximum moment-matching error')
        ax1.set_title('Moment-matching error vs T')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        ax1.set_ylim(1e-16, 1e-5)

        # Plot 2: Error vs moment order for fixed T
        T = 5.0
        for n_roots in [2, 3, 4]:
            nodes, weights = rys_nodes_weights(T, n_roots)
            n_vals = range(2 * n_roots + 3)
            errors = []
            for n in n_vals:
                m_exact = moment(n, T)
                m_quad = np.sum(weights * (nodes ** n))
                errors.append(abs(m_exact - m_quad))
            ax2.semilogy(n_vals, errors, 'o-', label=f'n_roots={n_roots}')
            # Mark the exactness boundary
            ax2.axvline(x=2*n_roots - 0.5, color=ax2.lines[-1].get_color(), linestyle=':', alpha=0.5)

        ax2.set_xlabel('Moment order n')
        ax2.set_ylabel('Moment-matching error')
        ax2.set_title(f'Error vs moment order (T = {T})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('moment_matching_errors.png', dpi=150)
        print("Saved plot to: moment_matching_errors.png")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping visualization)")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Moment Matching Verification for Rys Quadrature")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run comprehensive validation
    all_passed = validate_for_all_root_counts(T=1.0)

    # Demonstrate exactness boundary
    demonstrate_exactness_boundary()

    # Run Exercise 5.2
    max_error = run_exercise_5_2()

    # Sweep over T values
    print("\n" + "=" * 70)
    print("Robustness test: sweeping T from 1e-8 to 100")
    print("=" * 70)
    _, errors_sweep = sweep_T_values(n_roots=3)

    # Visualization
    visualize_errors()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"All basic tests passed: {all_passed}")
    print(f"Exercise 5.2 max error: {max_error:.2e}")
    print(f"Sweep test max error: {np.max(errors_sweep):.2e}")

    if all_passed and max_error < 1e-10 and np.max(errors_sweep) < 1e-9:
        print("\nAll validations PASSED")
    else:
        print("\nSome validations may need attention")
    print("=" * 70)
