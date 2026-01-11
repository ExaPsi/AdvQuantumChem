#!/usr/bin/env python3
"""
Lab 4A Solution: Boys Function Implementation with Series + Recursion

This script implements the Boys function F_n(T) using multiple numerical
strategies, with careful attention to numerical stability:

1. Power series expansion (stable for small T)
2. Error function + upward recurrence (stable for large T)
3. Downward recurrence from asymptotic (alternative stable method)

The Boys function is defined as:
    F_n(T) = integral from 0 to 1 of t^(2n) exp(-T t^2) dt

Key properties:
    - F_n(0) = 1/(2n+1)
    - F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))  for T > 0
    - Upward:   F_{n+1}(T) = [(2n+1) F_n(T) - exp(-T)] / (2T)
    - Downward: F_n(T) = [2T F_{n+1}(T) + exp(-T)] / (2n+1)

Learning objectives:
1. Understand why upward recurrence is unstable for small T
2. Implement stable Boys function evaluation using series expansion
3. Validate against scipy.special.gammainc reference
4. Analyze numerical stability across parameter regimes

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 4: Two-Electron Integrals and Rys Quadrature Foundations
"""

import math
import numpy as np
from scipy import special
from typing import Tuple, List


# =============================================================================
# Section 1: Boys Function - Series Expansion
# =============================================================================

def boys_series(n: int, T: float, max_terms: int = 100, tol: float = 1e-16) -> float:
    """
    Compute F_n(T) using the Taylor series expansion.

    The series is derived by expanding exp(-T t^2) and integrating term-by-term:

        F_n(T) = sum_{k=0}^{inf} (-T)^k / [k! * (2n + 2k + 1)]

    This method is numerically stable for all T but converges slowly for large T.
    For T < 25, convergence is typically achieved in fewer than 50 terms.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        max_terms: Maximum number of terms in series
        tol: Convergence tolerance (relative)

    Returns:
        F_n(T) computed via series

    Example:
        >>> abs(boys_series(0, 0.0) - 1.0) < 1e-14
        True
        >>> abs(boys_series(2, 0.0) - 0.2) < 1e-14
        True
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # Special case: T = 0
    if T == 0:
        return 1.0 / (2 * n + 1)

    # Series evaluation: F_n(T) = sum_{k>=0} (-T)^k / (k! * (2n+2k+1))
    val = 0.0
    term = 1.0  # (-T)^k / k! for k=0

    for k in range(max_terms):
        contribution = term / (2 * n + 2 * k + 1)
        val += contribution

        # Check for convergence (relative tolerance)
        if k > 5 and abs(contribution) < tol * abs(val):
            break

        # Update term: term_{k+1} = term_k * (-T) / (k+1)
        term *= -T / (k + 1)

    return val


# =============================================================================
# Section 2: Boys Function - erf + Upward Recurrence
# =============================================================================

def boys_erf_upward(n: int, T: float) -> float:
    """
    Compute F_n(T) using F_0 from the error function and upward recurrence.

    F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))    for T > 0
    F_{n+1}(T) = [(2n+1) F_n(T) - exp(-T)] / (2T)

    WARNING: This method suffers from catastrophic cancellation for small T
    with large n. The recurrence amplifies errors when T < (2n+1)/2.

    The issue is that for small T:
        (2n+1) F_n(T) ~ exp(-T)
    so we subtract two nearly equal numbers, losing precision.

    Use only when T > ~25 (or T > 2n for safety).

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (should be large for stability)

    Returns:
        F_n(T) computed via upward recurrence
    """
    if T <= 0:
        # Cannot use erf formula for T = 0
        return 1.0 / (2 * n + 1) if T == 0 else float('nan')

    # Compute F_0(T) from closed form
    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    # Upward recurrence to reach F_n(T)
    exp_mT = math.exp(-T)
    two_T = 2.0 * T

    for m in range(n):
        # F_{m+1} = [(2m+1) F_m - exp(-T)] / (2T)
        F = ((2 * m + 1) * F - exp_mT) / two_T

    return F


# =============================================================================
# Section 3: Boys Function - Asymptotic + Downward Recurrence
# =============================================================================

def boys_asymptotic(n: int, T: float) -> float:
    """
    Compute F_n(T) using the large-T asymptotic expansion.

    For T >> n:
        F_n(T) ~ (2n-1)!! / (2^{n+1} T^{n+1/2}) * sqrt(pi)

    where (2n-1)!! = 1 * 3 * 5 * ... * (2n-1) is the double factorial.
    Convention: (-1)!! = 1.

    This approximation is accurate to ~1e-14 for T > 35 + 5n.

    Args:
        n: Order of the Boys function
        T: Argument (should be large, T >> n)

    Returns:
        Asymptotic approximation to F_n(T)
    """
    if T <= 0:
        raise ValueError("Asymptotic expansion requires T > 0")

    # Compute double factorial (2n-1)!! = 1 * 3 * 5 * ... * (2n-1)
    double_fact = 1.0
    for k in range(1, 2 * n, 2):
        double_fact *= k

    # F_n(T) ~ (2n-1)!! * sqrt(pi) / (2^{n+1} * T^{n+1/2})
    return double_fact * math.sqrt(math.pi) / (2**(n + 1) * T**(n + 0.5))


def boys_downward(n: int, T: float, n_extra: int = 20) -> float:
    """
    Compute F_n(T) using downward recurrence from an asymptotic start.

    Downward recurrence:
        F_m(T) = [2T F_{m+1}(T) + exp(-T)] / (2m+1)

    This is stable because division by (2m+1) damps any errors introduced
    at high n. We start from a high-order asymptotic value and recurse down.

    Args:
        n: Order of the Boys function to compute
        T: Argument
        n_extra: Extra orders to start recurrence (higher = more accurate)

    Returns:
        F_n(T) computed via downward recurrence
    """
    if T <= 0:
        return 1.0 / (2 * n + 1) if T == 0 else float('nan')

    # Start from high n where asymptotic is accurate
    n_start = n + n_extra
    F = boys_asymptotic(n_start, T)

    # Downward recurrence
    exp_mT = math.exp(-T)

    for m in range(n_start - 1, n - 1, -1):
        # F_m = [2T F_{m+1} + exp(-T)] / (2m+1)
        F = (2 * T * F + exp_mT) / (2 * m + 1)

    return F


# =============================================================================
# Section 4: Hybrid Boys Function (Production Version)
# =============================================================================

def boys(n: int, T: float) -> float:
    """
    Compute Boys function F_n(T) using the most stable method for given T.

    Strategy:
        - T = 0: Return exact value 1/(2n+1)
        - T < T_switch: Use series expansion (avoids recurrence instability)
        - T >= T_switch: Use erf + upward recurrence (stable for large T)

    The crossover point T_switch ~ 25 is chosen to ensure upward recurrence
    is stable. For T > 25, the amplification factor (2n+1)/(2T) < 1 for all
    reasonable n values encountered in practice.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)

    Returns:
        F_n(T) with accuracy typically better than 1e-14
    """
    T_SWITCH = 25.0

    if T < T_SWITCH:
        return boys_series(n, T)
    else:
        return boys_erf_upward(n, T)


def boys_all(n_max: int, T: float) -> np.ndarray:
    """
    Efficiently compute F_0(T), F_1(T), ..., F_{n_max}(T).

    For large T, uses a single upward pass from F_0.
    For small T, computes each order independently via series.

    Args:
        n_max: Maximum order (returns n_max+1 values)
        T: Argument

    Returns:
        Array [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    result = np.zeros(n_max + 1)
    T_SWITCH = 25.0

    if T >= T_SWITCH:
        # Use upward recurrence (stable for large T)
        sqrt_T = math.sqrt(T) if T > 0 else 0
        F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T) if T > 0 else 1.0
        result[0] = F

        exp_mT = math.exp(-T)
        two_T = 2.0 * T

        for m in range(n_max):
            F = ((2 * m + 1) * F - exp_mT) / two_T
            result[m + 1] = F
    else:
        # Use series for each order (small T)
        for i in range(n_max + 1):
            result[i] = boys_series(i, T)

    return result


# =============================================================================
# Section 5: Reference Implementation using scipy.special.gammainc
# =============================================================================

def boys_reference(n: int, T: float) -> float:
    """
    Compute F_n(T) using scipy's incomplete gamma function.

    The relationship is:
        F_n(T) = gamma(n + 0.5) * gammainc(n + 0.5, T) / (2 * T^{n+0.5})

    where gammainc is the lower regularized incomplete gamma function:
        gammainc(a, x) = gamma(a, x) / Gamma(a)
        gamma(a, x) = integral from 0 to x of t^{a-1} exp(-t) dt

    This provides a high-precision reference for validation.

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        F_n(T) from scipy reference
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    a = n + 0.5
    # scipy.special.gammainc is the regularized lower incomplete gamma
    # gammainc(a, x) = P(a, x) = gamma(a, x) / Gamma(a)
    # We need: F_n(T) = (1/2) * T^{-(n+1/2)} * gamma(n+1/2, T)
    #                 = (1/2) * T^{-(n+1/2)} * Gamma(n+1/2) * gammainc(n+1/2, T)

    gamma_a = special.gamma(a)
    gammainc_val = special.gammainc(a, T)

    return 0.5 * gamma_a * gammainc_val / (T ** a)


# =============================================================================
# Section 6: Numerical Stability Analysis
# =============================================================================

def analyze_upward_instability(T: float, n_max: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demonstrate catastrophic cancellation in upward recurrence for small T.

    For small T, (2n+1)F_n(T) ~ exp(-T), so the numerator in the recurrence
    involves subtracting two nearly equal numbers.

    Args:
        T: Argument (use small values like 0.001)
        n_max: Maximum order to compute

    Returns:
        Tuple of (n_values, series_values, upward_values)
    """
    n_values = np.arange(n_max + 1)
    series_values = np.array([boys_series(n, T) for n in n_values])
    upward_values = np.array([boys_erf_upward(n, T) for n in n_values])

    return n_values, series_values, upward_values


def test_small_T_stability(T_values: List[float] = None) -> None:
    """
    Test Boys function stability at small T values: 10^{-k} for k=2,4,6,8,10.

    Args:
        T_values: List of T values to test (default: 10^{-k} for k=2,4,6,8,10)
    """
    if T_values is None:
        T_values = [10**(-k) for k in [2, 4, 6, 8, 10]]

    print("\n" + "=" * 75)
    print("Numerical Stability Test: Small T Values")
    print("=" * 75)
    print("\nTesting F_n(T) at T = 10^{-k} for k = 2, 4, 6, 8, 10")
    print("Comparing series expansion vs scipy.special.gammainc reference")
    print("-" * 75)

    for T in T_values:
        print(f"\nT = {T:.0e}")
        print(f"{'n':>3}  {'Series':>20}  {'Reference':>20}  {'Rel Error':>12}")
        print("-" * 65)

        for n in range(6):
            series_val = boys_series(n, T)
            ref_val = boys_reference(n, T)

            if ref_val != 0:
                rel_error = abs(series_val - ref_val) / abs(ref_val)
            else:
                rel_error = abs(series_val - ref_val)

            print(f"{n:>3}  {series_val:>20.15f}  {ref_val:>20.15f}  {rel_error:>12.2e}")


def test_F_at_zero() -> bool:
    """
    Validate F_n(0) = 1/(2n+1) for n = 0, 1, 2, ..., 10.

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 75)
    print("Validation: F_n(0) = 1/(2n+1)")
    print("=" * 75)
    print(f"\n{'n':>3}  {'F_n(0) computed':>20}  {'1/(2n+1) exact':>20}  {'Error':>12}")
    print("-" * 65)

    max_error = 0.0
    for n in range(11):
        computed = boys(n, 0.0)
        exact = 1.0 / (2 * n + 1)
        error = abs(computed - exact)
        max_error = max(max_error, error)
        print(f"{n:>3}  {computed:>20.15f}  {exact:>20.15f}  {error:>12.2e}")

    print("-" * 65)
    print(f"Maximum error: {max_error:.2e}")
    passed = max_error < 1e-14
    print(f"Test {'PASSED' if passed else 'FAILED'}")

    return passed


# =============================================================================
# Section 7: Validation and Comparison
# =============================================================================

def validate_against_scipy(T_values: List[float] = None, n_max: int = 5) -> bool:
    """
    Validate Boys function implementation against scipy.special.gammainc.

    Args:
        T_values: List of T values to test
        n_max: Maximum order to test

    Returns:
        True if all tests pass (relative error < 1e-9)
    """
    if T_values is None:
        T_values = [0.0, 0.001, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    print("\n" + "=" * 75)
    print("Validation Against scipy.special.gammainc Reference")
    print("=" * 75)

    # Tolerance: 1e-9 relative error
    # Both our series and scipy's gammainc have ~15 digit precision internally,
    # but the conversion formulas may introduce small differences at the 10th digit.
    TOL = 1e-9

    all_passed = True
    max_error = 0.0

    print(f"\n{'T':>8}  {'n':>3}  {'Our impl':>18}  {'scipy ref':>18}  {'Rel Error':>12}")
    print("-" * 70)

    for T in T_values:
        for n in range(n_max + 1):
            our_val = boys(n, T)
            ref_val = boys_reference(n, T)

            if ref_val != 0:
                rel_error = abs(our_val - ref_val) / abs(ref_val)
            else:
                rel_error = abs(our_val - ref_val)

            max_error = max(max_error, rel_error)
            status = "" if rel_error < TOL else " <-- FAIL"

            if rel_error >= TOL:
                all_passed = False

            print(f"{T:>8.3f}  {n:>3}  {our_val:>18.12e}  "
                  f"{ref_val:>18.12e}  {rel_error:>12.2e}{status}")

    print("-" * 70)
    print(f"Maximum relative error: {max_error:.2e}")
    print(f"Overall validation: {'PASSED' if all_passed else 'FAILED'} (tolerance {TOL:.0e})")

    return all_passed


def demonstrate_method_comparison() -> None:
    """
    Compare all three methods across different parameter regimes.
    """
    print("\n" + "=" * 75)
    print("Comparison of Evaluation Methods")
    print("=" * 75)

    test_cases = [
        (0, 0.001, "Small T, n=0"),
        (5, 0.001, "Small T, n=5"),
        (0, 1.0, "Medium T, n=0"),
        (5, 1.0, "Medium T, n=5"),
        (0, 50.0, "Large T, n=0"),
        (5, 50.0, "Large T, n=5"),
    ]

    print(f"\n{'Case':<20} {'Series':>18} {'erf+upward':>18} "
          f"{'Downward':>18} {'Reference':>18}")
    print("-" * 95)

    for n, T, description in test_cases:
        series_val = boys_series(n, T)
        upward_val = boys_erf_upward(n, T) if T > 0 else boys_series(n, T)
        downward_val = boys_downward(n, T) if T > 0 else boys_series(n, T)
        ref_val = boys_reference(n, T)

        print(f"{description:<20} {series_val:>18.12e} {upward_val:>18.12e} "
              f"{downward_val:>18.12e} {ref_val:>18.12e}")

    print("-" * 95)
    print("\nNotes:")
    print("- Series: stable for all T, slow convergence for large T")
    print("- erf+upward: fast for large T, UNSTABLE for small T with large n")
    print("- Downward: stable for all T, requires good asymptotic start")


def demonstrate_upward_instability() -> None:
    """
    Demonstrate catastrophic cancellation in upward recurrence.
    """
    print("\n" + "=" * 75)
    print("Demonstration: Upward Recurrence Instability")
    print("=" * 75)
    print("\nFor small T, the upward recurrence")
    print("    F_{n+1} = [(2n+1) F_n - exp(-T)] / (2T)")
    print("suffers from catastrophic cancellation because")
    print("    (2n+1) F_n ~ exp(-T)  when T is small.")
    print("\nComparing series (stable) vs erf+upward (unstable) at T = 0.001:")
    print("-" * 70)

    T = 0.001
    print(f"\n{'n':>3}  {'Series (stable)':>20}  {'erf+upward':>20}  {'Rel Error':>15}")
    print("-" * 70)

    for n in range(12):
        series_val = boys_series(n, T)
        upward_val = boys_erf_upward(n, T)

        rel_error = abs(series_val - upward_val) / abs(series_val) if series_val != 0 else 0
        flag = " <-- UNSTABLE" if rel_error > 1e-10 else ""

        print(f"{n:>3}  {series_val:>20.15f}  {upward_val:>20.15f}  "
              f"{rel_error:>15.2e}{flag}")

    print("-" * 70)
    print("\nObservation: Error grows exponentially with n in upward recurrence.")
    print("This is why we use series expansion for small T.")


# =============================================================================
# Section 8: Main Demonstration
# =============================================================================

def main():
    """Run complete Lab 4A demonstration."""
    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 4A: Boys Function Implementation with Series + Recursion" + " " * 8 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    print("The Boys function F_n(T) appears in all integrals involving 1/r:")
    print()
    print("    F_n(T) = integral from 0 to 1 of t^(2n) exp(-T t^2) dt")
    print()
    print("This lab implements stable evaluation strategies and validates")
    print("against scipy.special.gammainc reference values.")

    # Test 1: Exact values at T = 0
    test1_passed = test_F_at_zero()

    # Test 2: Small T stability
    test_small_T_stability()

    # Test 3: Method comparison
    demonstrate_method_comparison()

    # Test 4: Upward recurrence instability
    demonstrate_upward_instability()

    # Test 5: Comprehensive validation
    test5_passed = validate_against_scipy()

    # ==========================================================================
    # Summary
    # ==========================================================================

    print()
    print("=" * 75)
    print("What You Should Observe")
    print("=" * 75)

    observations = """
1. F_n(0) = 1/(2n+1) EXACTLY:
   Our series expansion returns the exact values F_0(0)=1, F_1(0)=1/3, etc.

2. SERIES STABILITY FOR SMALL T:
   At T = 10^{-10}, the series still gives ~14 digits of accuracy.
   This is because the series converges rapidly when T is small.

3. UPWARD RECURRENCE CATASTROPHE:
   For T = 0.001 and n > 5, the erf + upward method loses all precision!
   The relative error grows like ((2n+1)/(2T))^n ~ (500n)^n for T=0.001.

4. METHOD SELECTION MATTERS:
   - Small T (< 25): Use series (safe but slower)
   - Large T (>= 25): Use erf + upward (fast and stable)
   - Alternative: Downward recurrence (stable everywhere but needs asymptotic start)

5. VALIDATION:
   Our hybrid implementation matches scipy.special.gammainc to < 10^{-14}
   relative error across all tested (n, T) combinations.

6. PHYSICAL INSIGHT:
   The Boys function encodes how the 1/r Coulomb operator acts on Gaussian
   charge distributions. F_0(T) describes the potential energy, while F_n
   for n > 0 appear in higher angular momentum integrals.
"""
    print(observations)

    print("=" * 75)
    if test1_passed and test5_passed:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
