#!/usr/bin/env python3
"""
boys_moments.py - Boys Function and Moments for Rys Quadrature (Lab 5A)

This module computes Boys function moments m_n(T) = 2*F_n(T) which arise from
the change of variables t^2 = x in the Boys integral, leading to the weight
function w_T(x) = x^{-1/2} exp(-Tx) on [0,1].

The connection between Boys functions and moments is:
    F_n(T) = (1/2) * integral_0^1 x^n * x^{-1/2} * exp(-Tx) dx
           = (1/2) * integral_0^1 x^n * w_T(x) dx
           = (1/2) * m_n(T)

Thus: m_n(T) = 2 * F_n(T)

These moments are the input to Rys quadrature construction (Algorithm 5.1).

References:
    - Chapter 5, Section 2: From Boys functions to moments
    - Equation (5.8): m_n(T) = integral_0^1 x^n w_T(x) dx = 2*F_n(T)

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import List, Tuple
from scipy import special


def boys_series(n: int, T: float, max_terms: int = 60, tol: float = 1e-16) -> float:
    """
    Evaluate F_n(T) using the series expansion (stable for small T).

    F_n(T) = exp(-T) * sum_{k>=0} (2T)^k / [(2n+1)(2n+3)...(2n+2k+1)]
           = sum_{k>=0} (-T)^k / [k! * (2n + 2k + 1)]

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
        # Next term: term *= (-T) / (k+1) * (2n+2k+1) / (2n+2k+3)
        term *= -T / (k + 1) * (2*n + 2*k + 1) / (2*n + 2*k + 3)

    return val


def boys_erf_recursion(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using F_0 from erf + upward recursion.

    F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))
    F_{m+1}(T) = [(2m+1) F_m(T) - exp(-T)] / (2T)

    WARNING: This method can suffer from catastrophic cancellation for small T
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
        if T > 1e-10:
            F = ((2*m + 1) * F - exp_neg_T) / (2*T)
        else:
            # For T -> 0, F_n(0) = 1/(2n+1)
            F = 1.0 / (2*m + 3)

    return F


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


def boys_all(n_max: int, T: float) -> np.ndarray:
    """
    Evaluate F_0(T), F_1(T), ..., F_{n_max}(T) efficiently.

    Uses the most stable method for the given T value.

    Parameters
    ----------
    n_max : int
        Maximum order needed
    T : float
        Argument

    Returns
    -------
    np.ndarray
        Array of [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    return np.array([boys(n, T) for n in range(n_max + 1)])


def moment(n: int, T: float) -> float:
    """
    Compute the n-th moment of the weight function w_T(x) = x^{-1/2} exp(-Tx).

    m_n(T) = integral_0^1 x^n * w_T(x) dx = 2 * F_n(T)

    This is Eq. (5.8) from the lecture notes.

    Parameters
    ----------
    n : int
        Moment order (n >= 0)
    T : float
        Parameter in the weight function

    Returns
    -------
    float
        Value of m_n(T)
    """
    return 2.0 * boys(n, T)


def moments_all(n_max: int, T: float) -> np.ndarray:
    """
    Compute moments m_0(T), m_1(T), ..., m_{n_max}(T).

    Parameters
    ----------
    n_max : int
        Maximum moment order
    T : float
        Parameter in the weight function

    Returns
    -------
    np.ndarray
        Array of [m_0(T), m_1(T), ..., m_{n_max}(T)]
    """
    return 2.0 * boys_all(n_max, T)


def boys_hyp1f1(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using the hypergeometric function (reference).

    F_n(T) = (1/(2n+1)) * exp(-T) * 1F1(1, n+3/2, T)
           = (1/(2n+1)) * 1F1(n+1/2, n+3/2, -T)

    This serves as an independent reference for validation.

    Parameters
    ----------
    n : int
        Order of the Boys function
    T : float
        Argument

    Returns
    -------
    float
        Value of F_n(T)
    """
    # Use scipy's hypergeometric function
    # F_n(T) = (1/(2n+1)) * hyp1f1(n + 0.5, n + 1.5, -T)
    return special.hyp1f1(n + 0.5, n + 1.5, -T) / (2*n + 1)


def validate_against_scipy():
    """
    Validate Boys function against scipy's hypergeometric function.
    """
    print("Validation: Boys function against scipy hyp1f1")
    print("=" * 70)
    print(f"{'n':>3} {'T':>8} {'Our boys()':>20} {'scipy hyp1f1':>20} {'Diff':>12}")
    print("-" * 70)

    test_cases = [
        (0, 0.0), (0, 0.001), (0, 1.0), (0, 10.0), (0, 50.0),
        (3, 0.0), (3, 0.001), (3, 1.0), (3, 10.0), (3, 50.0),
        (6, 0.0), (6, 0.001), (6, 1.0), (6, 10.0), (6, 50.0),
    ]

    max_diff = 0.0
    for n, T in test_cases:
        our_val = boys(n, T)
        scipy_val = boys_hyp1f1(n, T)
        diff = abs(our_val - scipy_val)
        max_diff = max(max_diff, diff)
        print(f"{n:>3} {T:>8.3f} {our_val:>20.14e} {scipy_val:>20.14e} {diff:>12.2e}")

    print("-" * 70)
    print(f"Maximum difference: {max_diff:.2e}")
    return max_diff < 1e-12


def validate_moments_direct_integration():
    """
    Validate moments against direct numerical integration (scipy.integrate).
    """
    from scipy import integrate

    print("\nValidation: Moments against direct numerical integration")
    print("=" * 70)
    print(f"{'n':>3} {'T':>8} {'m_n = 2*F_n':>20} {'Direct quad':>20} {'Diff':>12}")
    print("-" * 70)

    def integrand(x, n, T):
        """Integrand for moment: x^n * x^{-1/2} * exp(-T*x)"""
        if x < 1e-15:
            return 0.0  # Avoid singularity at x=0
        return (x**n) * (x**(-0.5)) * math.exp(-T * x)

    test_cases = [
        (0, 0.0), (0, 1.0), (0, 10.0),
        (3, 0.0), (3, 1.0), (3, 10.0),
        (5, 0.5), (5, 5.0),
    ]

    max_diff = 0.0
    for n, T in test_cases:
        our_val = moment(n, T)
        # Direct integration (with singularity handling at x=0)
        quad_val, _ = integrate.quad(lambda x: integrand(x, n, T), 0, 1, limit=100)
        diff = abs(our_val - quad_val)
        max_diff = max(max_diff, diff)
        print(f"{n:>3} {T:>8.3f} {our_val:>20.14e} {quad_val:>20.14e} {diff:>12.2e}")

    print("-" * 70)
    print(f"Maximum difference: {max_diff:.2e}")
    return max_diff < 1e-10


def demonstrate_moment_properties():
    """
    Demonstrate key properties of moments m_n(T).
    """
    print("\nKey properties of moments m_n(T) = 2*F_n(T)")
    print("=" * 70)

    # Property 1: m_n(0) = 2/(2n+1)
    print("\n1. At T=0: m_n(0) = 2/(2n+1)")
    print("-" * 40)
    print(f"{'n':>3} {'m_n(0) computed':>18} {'2/(2n+1) exact':>18}")
    print("-" * 40)
    for n in range(6):
        computed = moment(n, 0.0)
        exact = 2.0 / (2*n + 1)
        print(f"{n:>3} {computed:>18.14f} {exact:>18.14f}")

    # Property 2: For large T, m_n(T) -> 0
    print("\n2. Large T limit: m_n(T) -> 0 as T -> infinity")
    print("-" * 40)
    print(f"{'n':>3} {'m_n(1.0)':>14} {'m_n(10.0)':>14} {'m_n(100.0)':>14}")
    print("-" * 40)
    for n in range(4):
        print(f"{n:>3} {moment(n, 1.0):>14.6e} {moment(n, 10.0):>14.6e} {moment(n, 100.0):>14.6e}")

    # Property 3: m_0(T) is the total weight (normalization of quadrature)
    print("\n3. m_0(T) = total weight of distribution")
    print("-" * 40)
    for T in [0.0, 0.1, 1.0, 5.0, 10.0]:
        m0 = moment(0, T)
        print(f"   T = {T:6.2f}:  m_0 = {m0:.10f}")


def visualize_moments():
    """
    Create visualization of moments vs T (if matplotlib available).
    """
    try:
        import matplotlib.pyplot as plt

        print("\nGenerating moment visualization...")

        T_vals = np.linspace(0.01, 20.0, 200)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: m_n(T) vs T for different n
        for n in range(5):
            m_vals = [moment(n, T) for T in T_vals]
            ax1.plot(T_vals, m_vals, label=f'n={n}')

        ax1.set_xlabel('T')
        ax1.set_ylabel('m_n(T)')
        ax1.set_title('Boys Moments m_n(T) = 2F_n(T)')
        ax1.legend()
        ax1.set_ylim(0, 2.1)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Log scale for decay behavior
        for n in range(5):
            m_vals = [moment(n, T) for T in T_vals]
            ax2.semilogy(T_vals, m_vals, label=f'n={n}')

        ax2.set_xlabel('T')
        ax2.set_ylabel('m_n(T) [log scale]')
        ax2.set_title('Boys Moments (log scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('boys_moments_plot.png', dpi=150)
        print("Saved plot to: boys_moments_plot.png")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping visualization)")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 5A: Boys Function Moments for Rys Quadrature")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run validations
    success1 = validate_against_scipy()
    success2 = validate_moments_direct_integration()

    # Demonstrate properties
    demonstrate_moment_properties()

    # Optional visualization
    visualize_moments()

    print("\n" + "=" * 70)
    if success1 and success2:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
    print("=" * 70)
