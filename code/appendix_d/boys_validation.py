#!/usr/bin/env python3
"""
Boys Function Validation Against SciPy

The Boys function is related to the incomplete gamma function:

    F_n(T) = (1/2) * T^{-(n+1/2)} * gamma(n+1/2) * gammainc(n+1/2, T)

where gammainc is the lower regularized incomplete gamma function P(a,x).
Note: scipy.special.gammainc returns P(a,x) = gamma(a,x)/Gamma(a).

This module validates our Boys implementation against SciPy's gammainc.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix D: Boys Functions
"""

import math
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc

# Import our implementation
from boys_function import boys, boys_array


def boys_scipy(n: int, T: float) -> float:
    """
    Compute F_n(T) using SciPy's incomplete gamma function.

    The relationship is:
        F_n(T) = (1/2) * T^{-(n+0.5)} * Gamma(n+0.5) * P(n+0.5, T)

    where P(a, x) = gammainc(a, x) is the regularized lower incomplete gamma.

    For T = 0:
        F_n(0) = 1/(2n+1)

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        F_n(T) computed via scipy.special.gammainc
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    # Handle very small T to avoid numerical issues
    if T < 1e-15:
        return 1.0 / (2 * n + 1)

    a = n + 0.5
    # gammainc(a, T) = P(a, T) = gamma(a, T) / Gamma(a)
    # We need: (1/2) * T^{-a} * Gamma(a) * P(a, T)
    #        = (1/2) * T^{-a} * gamma(a, T)
    #        = (1/2) * T^{-a} * Gamma(a) * gammainc(a, T)
    result = 0.5 * (T**(-a)) * gamma_func(a) * gammainc(a, T)
    return result


def test_special_cases():
    """Test F_n(0) = 1/(2n+1) for n = 0, 1, 2, ..."""
    print("\n" + "=" * 70)
    print("Test 1: Special Case F_n(0) = 1/(2n+1)")
    print("=" * 70)

    all_passed = True
    print(f"\n{'n':>4}  {'Our boys(n,0)':>18}  {'Exact 1/(2n+1)':>18}  {'Status':>10}")
    print("-" * 60)

    for n in range(10):
        computed = boys(n, 0)
        exact = 1.0 / (2 * n + 1)
        error = abs(computed - exact)
        passed = error < 1e-14
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{n:4d}  {computed:18.14f}  {exact:18.14f}  {status:>10}")

    return all_passed


def test_against_scipy():
    """Compare boys() against scipy gammainc reference."""
    print("\n" + "=" * 70)
    print("Test 2: Validation Against SciPy (gammainc)")
    print("=" * 70)

    # Test grid: various n and T values
    n_values = [0, 1, 2, 3, 5, 10]
    T_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    all_passed = True
    max_error = 0.0

    print(f"\n{'n':>4}  {'T':>8}  {'Our boys':>16}  {'SciPy ref':>16}  "
          f"{'Abs Error':>12}  {'Status':>8}")
    print("-" * 80)

    for n in n_values:
        for T in T_values:
            ours = boys(n, T)
            ref = boys_scipy(n, T)
            error = abs(ours - ref)
            max_error = max(max_error, error)

            # Tolerance: relative error < 1e-10 or absolute < 1e-14
            rel_tol = 1e-10
            abs_tol = 1e-14
            passed = error < max(abs_tol, rel_tol * abs(ref))
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"

            print(f"{n:4d}  {T:8.2f}  {ours:16.12f}  {ref:16.12f}  "
                  f"{error:12.2e}  {status:>8}")

    print("-" * 80)
    print(f"Maximum absolute error: {max_error:.2e}")
    return all_passed


def test_recurrence_consistency():
    """Verify that upward and downward recurrence give consistent results."""
    print("\n" + "=" * 70)
    print("Test 3: Recurrence Consistency")
    print("=" * 70)
    print("\nVerify: F_{n+1} = [(2n+1)*F_n - exp(-T)] / (2T)  (upward)")
    print("        F_n = [2T*F_{n+1} + exp(-T)] / (2n+1)      (downward)")

    all_passed = True
    T_values = [0.5, 2.0, 10.0, 50.0]

    for T in T_values:
        print(f"\nT = {T}:")
        print(f"{'n':>4}  {'F_n':>16}  {'F_{n+1} (up)':>16}  "
              f"{'F_{n+1} (direct)':>16}  {'Error':>12}")
        print("-" * 75)

        vals = boys_array(10, T)
        exp_mT = math.exp(-T)

        for n in range(8):
            F_n = vals[n]
            F_np1_direct = vals[n + 1]

            # Upward recurrence
            F_np1_up = ((2 * n + 1) * F_n - exp_mT) / (2 * T)
            error = abs(F_np1_up - F_np1_direct)

            passed = error < 1e-10
            all_passed = all_passed and passed

            print(f"{n:4d}  {F_n:16.12f}  {F_np1_up:16.12f}  "
                  f"{F_np1_direct:16.12f}  {error:12.2e}")

    return all_passed


def test_derivative_identity():
    """Verify: dF_n/dT = -F_{n+1}"""
    print("\n" + "=" * 70)
    print("Test 4: Derivative Identity dF_n/dT = -F_{n+1}")
    print("=" * 70)

    all_passed = True
    h = 1e-6  # Step size for numerical derivative

    print(f"\n{'n':>4}  {'T':>8}  {'-F_{n+1}':>16}  {'Numerical dF/dT':>16}  "
          f"{'Error':>12}")
    print("-" * 70)

    for n in [0, 1, 2, 3]:
        for T in [0.5, 2.0, 10.0]:
            # Numerical derivative using central difference
            F_plus = boys(n, T + h)
            F_minus = boys(n, T - h)
            dF_dT_numerical = (F_plus - F_minus) / (2 * h)

            # Analytic: dF_n/dT = -F_{n+1}
            dF_dT_analytic = -boys(n + 1, T)

            error = abs(dF_dT_numerical - dF_dT_analytic)
            passed = error < 1e-8
            all_passed = all_passed and passed

            print(f"{n:4d}  {T:8.2f}  {dF_dT_analytic:16.12f}  "
                  f"{dF_dT_numerical:16.12f}  {error:12.2e}")

    return all_passed


def test_asymptotic_behavior():
    """Test asymptotic approximation for large T."""
    print("\n" + "=" * 70)
    print("Test 5: Asymptotic Behavior for Large T")
    print("=" * 70)
    print("\nFor large T: F_0(T) -> (1/2)*sqrt(pi/T)")
    print("             F_n(T) -> (2n-1)!! / [2^{n+1} * T^{n+1/2}] * sqrt(pi)")

    print(f"\n{'n':>4}  {'T':>8}  {'F_n(T)':>16}  {'Asymptotic':>16}  {'Ratio':>10}")
    print("-" * 65)

    for n in [0, 1, 2]:
        for T in [50.0, 100.0, 200.0]:
            val = boys(n, T)

            # Asymptotic: (2n-1)!! / (2^{n+1} * T^{n+0.5}) * sqrt(pi)
            double_fact = 1.0
            for k in range(1, 2 * n, 2):
                double_fact *= k
            asymp = double_fact / (2**(n + 1) * T**(n + 0.5)) * math.sqrt(math.pi)

            ratio = val / asymp if asymp > 0 else float('nan')
            print(f"{n:4d}  {T:8.1f}  {val:16.12e}  {asymp:16.12e}  {ratio:10.6f}")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("Boys Function Validation Suite")
    print("=" * 70)
    print("\nComparing our implementation against SciPy's gammainc reference")
    print("and verifying mathematical identities.")

    results = {}

    results['special_cases'] = test_special_cases()
    results['scipy'] = test_against_scipy()
    results['recurrence'] = test_recurrence_consistency()
    results['derivative'] = test_derivative_identity()
    test_asymptotic_behavior()  # Informational, no pass/fail

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        marker = "[OK]" if passed else "[!!]"
        print(f"  {marker} {test_name}: {status}")
        all_passed = all_passed and passed

    print("-" * 70)
    if all_passed:
        print("All validation tests PASSED.")
    else:
        print("Some tests FAILED. Review output above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
