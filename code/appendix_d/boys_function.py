#!/usr/bin/env python3
"""
Boys Function: Core Implementation

The Boys function is fundamental to all integrals involving 1/r operators
(nuclear attraction, electron repulsion). It is defined as:

    F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt

Key formulas:
    - F_n(0) = 1/(2n+1)
    - F_0(T) = (1/2)*sqrt(pi/T)*erf(sqrt(T))  for T > 0
    - Series: F_n(T) = sum_{k=0}^inf (-T)^k / [k! * (2n+2k+1)]
    - Upward:   F_{n+1}(T) = [(2n+1)*F_n(T) - exp(-T)] / (2T)
    - Downward: F_n(T) = [2T*F_{n+1}(T) + exp(-T)] / (2n+1)

This module provides a stable implementation with proper handling of:
    - T = 0 exact case
    - Small T via Taylor series
    - Large T via asymptotic start + downward recurrence
    - Intermediate T via erf + upward recurrence

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix D: Boys Functions
"""

import math
import numpy as np
from typing import Union


def _boys_series(n: int, T: float, max_terms: int = 50) -> float:
    """
    Compute F_n(T) using Taylor series expansion.

    The series is:
        F_n(T) = sum_{k=0}^inf (-T)^k / [k! * (2n+2k+1)]

    This is numerically stable for small T where the erf-based approach
    suffers from cancellation in the upward recurrence.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        max_terms: Maximum number of terms to sum

    Returns:
        F_n(T) computed via series
    """
    val = 0.0
    term = 1.0  # (-T)^k / k! for k=0
    for k in range(max_terms):
        contribution = term / (2 * n + 2 * k + 1)
        val += contribution
        # Check for convergence (relative tolerance)
        if abs(contribution) < 1e-16 * abs(val) and k > 5:
            break
        # Update: term_{k+1} = term_k * (-T) / (k+1)
        term *= -T / (k + 1)
    return val


def _boys_erf_upward(n: int, T: float) -> float:
    """
    Compute F_n(T) using F_0 from erf and upward recurrence.

    F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))
    F_{n+1}(T) = [(2n+1)*F_n(T) - exp(-T)] / (2T)

    This is stable for moderate to large T, but loses precision for small T
    due to catastrophic cancellation: (2n+1)*F_n(T) approx exp(-T) when T->0.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T > 0)

    Returns:
        F_n(T) computed via upward recurrence
    """
    # Compute F_0(T) from closed form
    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    # Upward recurrence to reach F_n(T)
    exp_mT = math.exp(-T)
    two_T = 2.0 * T
    for m in range(n):
        F = ((2 * m + 1) * F - exp_mT) / two_T
    return F


def _boys_downward(n_target: int, n_start: int, F_start: float, T: float) -> float:
    """
    Compute F_{n_target}(T) using downward recurrence from F_{n_start}(T).

    Downward recurrence:
        F_n(T) = [2T * F_{n+1}(T) + exp(-T)] / (2n+1)

    Downward recurrence is stable for all T because it averages rather than
    subtracts similar quantities. The key is to start from a high enough n
    where F_n(T) can be approximated asymptotically.

    Args:
        n_target: Desired order (n_target <= n_start)
        n_start: Starting order for recurrence
        F_start: Value of F_{n_start}(T)
        T: Argument (T >= 0)

    Returns:
        F_{n_target}(T) computed via downward recurrence
    """
    if n_target > n_start:
        raise ValueError("n_target must be <= n_start for downward recurrence")

    F = F_start
    exp_mT = math.exp(-T)
    two_T = 2.0 * T

    # Recurse downward from n_start to n_target
    for m in range(n_start - 1, n_target - 1, -1):
        F = (two_T * F + exp_mT) / (2 * m + 1)
    return F


def _boys_asymptotic(n: int, T: float) -> float:
    """
    Compute F_n(T) using asymptotic expansion for large T.

    For large T:
        F_n(T) -> (2n-1)!! / (2^{n+1} * T^{n+1/2}) * sqrt(pi)

    This provides a starting point for downward recurrence when T is large.

    Args:
        n: Order of the Boys function
        T: Argument (should be large, T >> n)

    Returns:
        F_n(T) from asymptotic approximation
    """
    # double_factorial(2n-1) = (2n-1)!! = 1 * 3 * 5 * ... * (2n-1)
    double_fact = 1.0
    for k in range(1, 2 * n, 2):
        double_fact *= k

    return double_fact / (2**(n + 1) * T**(n + 0.5)) * math.sqrt(math.pi)


def boys(n: int, T: float) -> float:
    """
    Compute Boys function F_n(T) with stable evaluation.

    Strategy:
        1. T = 0 exact: return 1/(2n+1)
        2. Small T < 1e-10: use Taylor series (avoids cancellation)
        3. Large T > 35 + 5*n: use asymptotic start + downward recurrence
        4. Intermediate T: use F_0 via erf + upward recurrence

    Args:
        n: Order of the Boys function (integer >= 0)
        T: Argument (float >= 0)

    Returns:
        F_n(T), the Boys function of order n at argument T

    Examples:
        >>> boys(0, 0)
        1.0
        >>> boys(1, 0)
        0.3333333333333333
        >>> abs(boys(0, 1.0) - 0.746824) < 1e-5
        True
    """
    # Input validation
    if n < 0:
        raise ValueError(f"Order n must be non-negative, got {n}")
    if T < 0:
        raise ValueError(f"Argument T must be non-negative, got {T}")

    # Case 1: T = 0 exactly
    if T == 0:
        return 1.0 / (2 * n + 1)

    # Case 2: Small T - use series expansion
    # The series is stable and fast for small T
    # Also use series when upward recurrence would be unstable
    # Upward amplification = (2n+1)/(2T), so unstable when T < (2n+1)/2
    SMALL_T_ABS = 1e-10  # Always use series below this
    SMALL_T_REL = (2 * n + 1) / 4.0  # Use series when upward would amplify errors
    if T < SMALL_T_ABS or T < SMALL_T_REL:
        return _boys_series(n, T)

    # Case 3: Large T - use asymptotic + downward recurrence
    # Threshold from libcint: roughly 35 + 5*n
    LARGE_T_BASE = 35.0
    LARGE_T_SLOPE = 5.0
    large_T_threshold = LARGE_T_BASE + LARGE_T_SLOPE * n

    if T > large_T_threshold:
        # Start from high n where asymptotic is accurate
        n_start = n + 20  # extra buffer for accuracy
        F_start = _boys_asymptotic(n_start, T)
        return _boys_downward(n, n_start, F_start, T)

    # Case 4: Intermediate T - use erf + upward recurrence
    # At this point T is large enough that upward is stable
    return _boys_erf_upward(n, T)


def boys_array(n_max: int, T: float) -> np.ndarray:
    """
    Compute Boys functions F_0(T) through F_{n_max}(T) efficiently.

    For intermediate T, computes all values using a single upward pass.
    For other regimes, uses the stable single-value function.

    Args:
        n_max: Maximum order (returns n_max+1 values)
        T: Argument (T >= 0)

    Returns:
        Array of shape (n_max+1,) with F_0, F_1, ..., F_{n_max}
    """
    result = np.zeros(n_max + 1)

    # For simplicity and robustness, use the single-value function
    # A more optimized version would share computation
    for n in range(n_max + 1):
        result[n] = boys(n, T)
    return result


def main():
    """Demonstrate Boys function implementation with key test cases."""
    print("=" * 70)
    print("Boys Function Implementation - Core Module")
    print("=" * 70)

    # Test 1: Exact values at T = 0
    print("\n[1] Special case F_n(0) = 1/(2n+1):")
    print("-" * 50)
    print(f"{'n':>4}  {'Computed':>14}  {'Exact':>14}  {'Error':>12}")
    print("-" * 50)
    for n in range(6):
        computed = boys(n, 0)
        exact = 1.0 / (2 * n + 1)
        error = abs(computed - exact)
        print(f"{n:4d}  {computed:14.10f}  {exact:14.10f}  {error:12.2e}")

    # Test 2: Series expansion for small T
    print("\n[2] Small T (series expansion):")
    print("-" * 50)
    T_small = 1e-12
    print(f"T = {T_small:.2e}")
    print(f"{'n':>4}  {'F_n(T)':>20}")
    print("-" * 50)
    for n in range(5):
        val = boys(n, T_small)
        # Compare with F_n(0) = 1/(2n+1) - should be very close
        approx_zero = 1.0 / (2 * n + 1)
        print(f"{n:4d}  {val:20.15f}  (F_n(0) = {approx_zero:.15f})")

    # Test 3: Moderate T values
    print("\n[3] Moderate T values:")
    print("-" * 50)
    T_values = [0.1, 1.0, 5.0, 10.0]
    print(f"{'T':>6}  {'F_0':>12}  {'F_1':>12}  {'F_2':>12}  {'F_3':>12}")
    print("-" * 50)
    for T in T_values:
        vals = boys_array(3, T)
        print(f"{T:6.1f}  {vals[0]:12.8f}  {vals[1]:12.8f}  "
              f"{vals[2]:12.8f}  {vals[3]:12.8f}")

    # Test 4: Large T (asymptotic + downward)
    print("\n[4] Large T (asymptotic behavior):")
    print("-" * 50)
    print("For T -> infinity: F_0(T) -> (1/2)*sqrt(pi/T)")
    print(f"{'T':>6}  {'F_0(T)':>14}  {'Asymptotic':>14}  {'Ratio':>10}")
    print("-" * 50)
    for T in [20, 50, 100, 200]:
        computed = boys(0, T)
        asymptotic = 0.5 * math.sqrt(math.pi / T)
        ratio = computed / asymptotic
        print(f"{T:6d}  {computed:14.10f}  {asymptotic:14.10f}  {ratio:10.6f}")

    # Test 5: Recurrence consistency check
    print("\n[5] Recurrence consistency check:")
    print("-" * 50)
    print("Verify: F_{n+1} = [(2n+1)*F_n - exp(-T)] / (2T)")
    T_test = 2.5
    print(f"T = {T_test}")
    vals = boys_array(5, T_test)
    exp_mT = math.exp(-T_test)
    print(f"{'n':>4}  {'F_n(T)':>16}  {'F_n+1 (recurrence)':>20}  {'F_n+1 (direct)':>18}")
    print("-" * 70)
    for n in range(4):
        F_n = vals[n]
        F_np1_recur = ((2 * n + 1) * F_n - exp_mT) / (2 * T_test)
        F_np1_direct = vals[n + 1]
        print(f"{n:4d}  {F_n:16.12f}  {F_np1_recur:20.12f}  {F_np1_direct:18.12f}")

    print("\n" + "=" * 70)
    print("All demonstrations completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
