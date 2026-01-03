#!/usr/bin/env python3
"""
boys_function.py - Boys Function Evaluator (Lab 4A)

This module implements the Boys function F_n(T) using multiple strategies:
1. Series expansion (stable for small T)
2. F_0 from error function + upward recursion (stable for large T)
3. Downward recursion from asymptotic (alternative stable method)

The Boys function is defined as:
    F_n(T) = integral_0^1 t^(2n) exp(-T t^2) dt

Key properties:
    - F_n(0) = 1/(2n+1)
    - F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))  for T > 0
    - dF_n/dT = -F_{n+1}(T)

References:
    - Chapter 4, Section 6: The Boys Function
    - Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory", Section 9.2

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import List, Tuple


def boys_series(n: int, T: float, max_terms: int = 50, tol: float = 1e-15) -> float:
    """
    Evaluate F_n(T) using the series expansion.

    F_n(T) = sum_{k>=0} (-T)^k / (k! (2n + 2k + 1))

    This method is stable for all n but converges slowly for large T.

    Parameters
    ----------
    n : int
        Order of the Boys function (n >= 0)
    T : float
        Argument (T >= 0)
    max_terms : int
        Maximum number of terms in series
    tol : float
        Convergence tolerance

    Returns
    -------
    float
        Value of F_n(T)
    """
    val = 0.0
    term = 1.0 / (2*n + 1)

    for k in range(max_terms):
        val += term
        if abs(term) < tol * abs(val) and k > 0:
            break
        term *= -T / (k + 1) / (2*n + 2*k + 3) * (2*n + 2*k + 1)

    return val


def boys_erf_recursion(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using F_0 from erf + upward recursion.

    F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))
    F_{m+1}(T) = ((2m+1) F_m(T) - exp(-T)) / (2T)

    WARNING: This method suffers from catastrophic cancellation for small T
    with large n. Use only when T > ~25.

    Parameters
    ----------
    n : int
        Order of the Boys function
    T : float
        Argument (must be > 0 for F_0 formula)

    Returns
    -------
    float
        Value of F_n(T)
    """
    if T < 1e-10:
        # For very small T, F_0 ~ 1
        F = 1.0
    else:
        F = 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))

    if n == 0:
        return F

    # Upward recursion
    exp_neg_T = math.exp(-T)
    for m in range(n):
        F = ((2*m + 1) * F - exp_neg_T) / (2*T) if T > 1e-10 else 1.0 / (2*m + 3)

    return F


def boys_asymptotic(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using the large-T asymptotic expansion.

    F_n(T) ~ (2n-1)!! / 2^{n+1} * sqrt(pi / T^{2n+1})

    where (2n-1)!! = 1 * 3 * 5 * ... * (2n-1) with (-1)!! = 1.

    This is accurate for T > 30 + 5n approximately.

    Parameters
    ----------
    n : int
        Order of the Boys function
    T : float
        Argument (should be large)

    Returns
    -------
    float
        Asymptotic approximation to F_n(T)
    """
    # Compute double factorial (2n-1)!!
    double_fact = 1.0
    for k in range(1, 2*n, 2):
        double_fact *= k

    return double_fact / (2**(n+1)) * math.sqrt(math.pi / (T**(2*n + 1)))


def boys_downward(n_max: int, T: float, n_extra: int = 10) -> List[float]:
    """
    Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) using downward recursion.

    Start from F_{n_max + n_extra}(T) using asymptotic approximation,
    then recurse downward:
        F_m(T) = (2T F_{m+1}(T) + exp(-T)) / (2m+1)

    Downward recursion is numerically stable because division by (2m+1) damps errors.

    Parameters
    ----------
    n_max : int
        Maximum order needed
    T : float
        Argument
    n_extra : int
        Extra orders to start recursion (higher = more accurate)

    Returns
    -------
    list of float
        [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    n_start = n_max + n_extra

    # Initialize with asymptotic approximation
    F_current = boys_asymptotic(n_start, T) if T > 1e-10 else 1.0 / (2*n_start + 1)

    exp_neg_T = math.exp(-T) if T < 700 else 0.0  # Avoid overflow

    # Storage for results
    F_values = [0.0] * (n_max + 1)

    # Downward recursion
    for m in range(n_start - 1, -1, -1):
        F_current = (2*T * F_current + exp_neg_T) / (2*m + 1)
        if m <= n_max:
            F_values[m] = F_current

    return F_values


def boys(n: int, T: float, T_switch: float = 25.0) -> float:
    """
    Evaluate the Boys function F_n(T) using a hybrid strategy.

    Strategy:
        - T < T_switch: Use series expansion (stable for all n)
        - T >= T_switch: Use F_0 from erf + upward recursion (stable for large T)

    This is the recommended general-purpose evaluator.

    Parameters
    ----------
    n : int
        Order of the Boys function (n >= 0)
    T : float
        Argument (T >= 0)
    T_switch : float
        Crossover point between series and recursion methods

    Returns
    -------
    float
        Value of F_n(T)

    Examples
    --------
    >>> boys(0, 0.0)
    1.0
    >>> boys(2, 0.0)
    0.2
    >>> abs(boys(0, 1.0) - 0.7468241328) < 1e-9
    True
    """
    if T < T_switch:
        return boys_series(n, T)
    else:
        return boys_erf_recursion(n, T)


def boys_all(n_max: int, T: float, T_switch: float = 25.0) -> np.ndarray:
    """
    Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) efficiently.

    Uses the most stable method for the given T value.

    Parameters
    ----------
    n_max : int
        Maximum order needed
    T : float
        Argument
    T_switch : float
        Crossover point between methods

    Returns
    -------
    np.ndarray
        Array of [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    if T < T_switch:
        # Use series for each order
        return np.array([boys_series(n, T) for n in range(n_max + 1)])
    else:
        # Use downward recursion for stability
        return np.array(boys_downward(n_max, T))


# =============================================================================
# Validation and demonstration
# =============================================================================

def validate_boys_at_zero():
    """Verify F_n(0) = 1/(2n+1) for n = 0, 1, 2, ..., 10."""
    print("Validation: F_n(0) = 1/(2n+1)")
    print("-" * 50)
    print(f"{'n':>3}  {'F_n(0) computed':>18}  {'1/(2n+1) exact':>18}  {'Error':>12}")
    print("-" * 50)

    max_error = 0.0
    for n in range(11):
        computed = boys(n, 0.0)
        exact = 1.0 / (2*n + 1)
        error = abs(computed - exact)
        max_error = max(max_error, error)
        print(f"{n:>3}  {computed:>18.15f}  {exact:>18.15f}  {error:>12.2e}")

    print("-" * 50)
    print(f"Maximum error: {max_error:.2e}")
    return max_error < 1e-14


def validate_boys_at_one():
    """Verify F_0(1.0) against known value."""
    print("\nValidation: F_0(1.0)")
    print("-" * 50)

    # Reference value: F_0(1) = (1/2) sqrt(pi) erf(1) = 0.746824132812427...
    reference = 0.5 * math.sqrt(math.pi) * math.erf(1.0)
    computed = boys(0, 1.0)
    error = abs(computed - reference)

    print(f"F_0(1.0) computed:  {computed:.15f}")
    print(f"F_0(1.0) reference: {reference:.15f}")
    print(f"Error: {error:.2e}")

    return error < 1e-14


def compare_methods():
    """Compare different evaluation methods across parameter ranges."""
    print("\nComparison of evaluation methods")
    print("=" * 80)

    test_cases = [
        (0, 0.001),
        (0, 1.0),
        (0, 10.0),
        (0, 50.0),
        (5, 0.001),
        (5, 1.0),
        (5, 10.0),
        (5, 50.0),
        (10, 0.001),
        (10, 1.0),
        (10, 10.0),
        (10, 50.0),
    ]

    print(f"{'n':>3} {'T':>8} {'Series':>18} {'Erf+Recur':>18} {'Asymptotic':>18}")
    print("-" * 80)

    for n, T in test_cases:
        series_val = boys_series(n, T)
        erf_val = boys_erf_recursion(n, T)
        asymp_val = boys_asymptotic(n, T) if T > 1 else float('nan')

        print(f"{n:>3} {T:>8.3f} {series_val:>18.12e} {erf_val:>18.12e} {asymp_val:>18.12e}")


def demonstrate_instability():
    """Demonstrate catastrophic cancellation in upward recursion for small T."""
    print("\nDemonstration: Upward recursion instability for small T")
    print("=" * 70)
    print("For T = 0.001, comparing series (stable) vs erf+recursion (unstable)")
    print("-" * 70)

    T = 0.001
    print(f"{'n':>3} {'Series (stable)':>20} {'Erf+Recur':>20} {'Relative Error':>15}")
    print("-" * 70)

    for n in range(11):
        series_val = boys_series(n, T)
        erf_val = boys_erf_recursion(n, T)
        rel_error = abs(series_val - erf_val) / abs(series_val) if series_val != 0 else 0

        flag = " <-- UNSTABLE" if rel_error > 1e-10 else ""
        print(f"{n:>3} {series_val:>20.15f} {erf_val:>20.15f} {rel_error:>15.2e}{flag}")


if __name__ == "__main__":
    print("=" * 80)
    print("Lab 4A: Boys Function Evaluator")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 80)

    # Run validations
    success1 = validate_boys_at_zero()
    success2 = validate_boys_at_one()

    # Compare methods
    compare_methods()

    # Demonstrate instability
    demonstrate_instability()

    print("\n" + "=" * 80)
    if success1 and success2:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
    print("=" * 80)
