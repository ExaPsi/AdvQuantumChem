#!/usr/bin/env python3
"""
Lab 4C Solution: Numerical Quadrature Check for F_n(T)

This script demonstrates why generic numerical quadrature (Gauss-Legendre)
is inefficient for evaluating the Boys function, motivating the specialized
Rys quadrature developed in Chapter 5.

The Boys function is defined as:
    F_n(T) = integral from 0 to 1 of t^(2n) exp(-T t^2) dt

We compare:
1. Generic Gauss-Legendre quadrature (treats whole integrand as polynomial-like)
2. Dedicated Boys evaluation (series/recursion, exploits analytical structure)

Key observations:
- Gauss-Legendre requires many points (64-128+) for high accuracy
- Efficiency degrades for large T (sharply peaked integrand)
- Rys quadrature (Chapter 5) achieves exactness with n_r = floor(L/2)+1 points

Learning objectives:
1. Understand integrand shape changes with T
2. Implement Gauss-Legendre quadrature for Boys function
3. Analyze convergence vs number of quadrature points
4. Appreciate why Rys quadrature is superior

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 4: Two-Electron Integrals and Rys Quadrature Foundations
"""

import numpy as np
from scipy import special
from typing import Tuple, List
import math


# =============================================================================
# Section 1: Reference Boys Function Implementation
# =============================================================================

def boys_series(n: int, T: float, max_terms: int = 100, tol: float = 1e-16) -> float:
    """
    Compute F_n(T) using Taylor series expansion (reference implementation).

    Args:
        n: Order of Boys function
        T: Argument
        max_terms: Maximum terms in series
        tol: Convergence tolerance

    Returns:
        F_n(T)
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    val = 0.0
    term = 1.0

    for k in range(max_terms):
        contribution = term / (2 * n + 2 * k + 1)
        val += contribution

        if k > 5 and abs(contribution) < tol * abs(val):
            break

        term *= -T / (k + 1)

    return val


def boys_erf_upward(n: int, T: float) -> float:
    """
    Compute F_n(T) using F_0 from erf and upward recurrence.

    Args:
        n: Order of Boys function
        T: Argument (should be > 0)

    Returns:
        F_n(T)
    """
    if T <= 0:
        return 1.0 / (2 * n + 1)

    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    exp_mT = math.exp(-T)
    two_T = 2.0 * T

    for m in range(n):
        F = ((2 * m + 1) * F - exp_mT) / two_T

    return F


def boys_reference(n: int, T: float) -> float:
    """
    Compute F_n(T) using hybrid series/recursion (high-precision reference).

    Args:
        n: Order of Boys function
        T: Argument

    Returns:
        F_n(T) with ~15 digits accuracy
    """
    T_SWITCH = 25.0
    if T < T_SWITCH:
        return boys_series(n, T)
    else:
        return boys_erf_upward(n, T)


# =============================================================================
# Section 2: Gauss-Legendre Quadrature Implementation
# =============================================================================

def boys_gauss_legendre(n: int, T: float, n_pts: int) -> float:
    """
    Approximate F_n(T) using Gauss-Legendre quadrature.

    Gauss-Legendre nodes and weights are designed for integrals of the form:
        integral_{-1}^{1} f(x) dx

    We map to [0, 1] using the transformation:
        t = (x + 1) / 2,  dt = dx / 2

    Then:
        F_n(T) = integral_0^1 t^{2n} exp(-T t^2) dt
               = (1/2) integral_{-1}^{1} ((x+1)/2)^{2n} exp(-T ((x+1)/2)^2) dx
               approx (1/2) sum_i w_i f((x_i + 1)/2)

    Args:
        n: Order of Boys function
        T: Argument
        n_pts: Number of Gauss-Legendre quadrature points

    Returns:
        Approximation to F_n(T)
    """
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_pts)

    # Map to [0, 1]: t = (x + 1)/2
    t = 0.5 * (x_gl + 1.0)

    # Weights on [0, 1]: w = w_gl / 2
    w = 0.5 * w_gl

    # Evaluate integrand: t^{2n} * exp(-T * t^2)
    integrand = (t ** (2 * n)) * np.exp(-T * t**2)

    return float(np.sum(w * integrand))


def boys_gauss_legendre_all(n_max: int, T: float, n_pts: int) -> np.ndarray:
    """
    Compute F_0(T), ..., F_{n_max}(T) using Gauss-Legendre quadrature.

    More efficient than calling boys_gauss_legendre repeatedly since
    we reuse the nodes, weights, and exponential.

    Args:
        n_max: Maximum order
        T: Argument
        n_pts: Number of quadrature points

    Returns:
        Array [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    # Get Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_pts)
    t = 0.5 * (x_gl + 1.0)
    w = 0.5 * w_gl

    # Compute exponential once
    exp_factor = np.exp(-T * t**2)

    # Compute integrals for each n
    result = np.zeros(n_max + 1)
    t_power = np.ones_like(t)  # t^0 = 1

    for k in range(n_max + 1):
        integrand = t_power * exp_factor
        result[k] = np.sum(w * integrand)
        t_power = t_power * t * t  # Update for next n: t^{2k} -> t^{2(k+1)}

    return result


# =============================================================================
# Section 3: Convergence Analysis
# =============================================================================

def analyze_gl_convergence(n: int, T: float, n_pts_list: List[int] = None) -> List[Tuple[int, float, float]]:
    """
    Study convergence of Gauss-Legendre quadrature for F_n(T).

    Args:
        n: Order of Boys function
        T: Argument
        n_pts_list: List of point counts to test

    Returns:
        List of (n_pts, value, absolute_error)
    """
    if n_pts_list is None:
        n_pts_list = [2, 4, 8, 16, 32, 64, 128, 256]

    ref = boys_reference(n, T)
    results = []

    for n_pts in n_pts_list:
        approx = boys_gauss_legendre(n, T, n_pts)
        error = abs(approx - ref)
        results.append((n_pts, approx, error))

    return results


def find_min_points_for_accuracy(n: int, T: float, tol: float = 1e-10,
                                  max_pts: int = 512) -> int:
    """
    Find minimum number of GL points needed for given accuracy.

    Args:
        n: Order of Boys function
        T: Argument
        tol: Target accuracy
        max_pts: Maximum points to try

    Returns:
        Minimum n_pts achieving tolerance, or -1 if max_pts insufficient
    """
    ref = boys_reference(n, T)

    for n_pts in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if n_pts > max_pts:
            break
        approx = boys_gauss_legendre(n, T, n_pts)
        if abs(approx - ref) < tol:
            return n_pts

    return -1  # Insufficient points


# =============================================================================
# Section 4: Integrand Shape Analysis
# =============================================================================

def analyze_integrand_shape(n: int, T_values: List[float]) -> None:
    """
    Analyze how the Boys integrand shape changes with T.

    The integrand t^{2n} exp(-T t^2) has different characteristics:
    - Small T: Nearly polynomial (easy for GL)
    - Large T: Sharply peaked near t = 0 (hard for GL)

    Args:
        n: Order of Boys function
        T_values: List of T values to analyze
    """
    print(f"\nIntegrand shape analysis: f(t) = t^{{{2*n}}} exp(-T t^2)")
    print("-" * 70)
    print(f"{'T':>8} {'Peak location':>15} {'Peak value':>15} {'FWHM':>12} {'Difficulty':>12}")
    print("-" * 70)

    t = np.linspace(1e-10, 1, 10000)

    for T in T_values:
        integrand = (t ** (2 * n)) * np.exp(-T * t**2)

        # Find maximum
        max_idx = np.argmax(integrand)
        max_loc = t[max_idx]
        max_val = integrand[max_idx]

        # Estimate full width at half maximum (FWHM)
        half_max = max_val / 2
        above_half = integrand > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = t[indices[-1]] - t[indices[0]]
        else:
            fwhm = 0.0

        # Assess difficulty: narrow peak = hard for GL
        if fwhm > 0.5:
            difficulty = "Easy"
        elif fwhm > 0.1:
            difficulty = "Moderate"
        else:
            difficulty = "Hard"

        print(f"{T:>8.2f} {max_loc:>15.6f} {max_val:>15.6e} {fwhm:>12.4f} {difficulty:>12}")


def demonstrate_integrand_shapes(n: int = 2) -> None:
    """
    Print text visualization of integrand shapes for different T.
    """
    print("\n" + "=" * 70)
    print(f"Integrand Visualization: t^{{{2*n}}} * exp(-T*t^2)")
    print("=" * 70)

    T_values = [0.1, 1.0, 10.0, 50.0]
    t = np.linspace(0, 1, 50)

    for T in T_values:
        integrand = (t ** (2 * n)) * np.exp(-T * t**2)
        integrand_norm = integrand / (np.max(integrand) + 1e-15)

        print(f"\nT = {T}:")
        print("+" + "-" * 48 + "+")

        # Simple ASCII plot
        for row in range(10, -1, -1):
            threshold = row / 10
            line = "|"
            for i in range(48):
                idx = int(i * len(t) / 48)
                if integrand_norm[idx] >= threshold:
                    line += "*"
                else:
                    line += " "
            line += "|"
            if row == 10:
                line += " max"
            elif row == 0:
                line += " 0"
            print(line)

        print("+" + "-" * 48 + "+")
        print(" t=0" + " " * 42 + "t=1")


# =============================================================================
# Section 5: Why Rys Quadrature?
# =============================================================================

def explain_why_rys() -> None:
    """
    Explain the motivation for Rys quadrature.
    """
    explanation = """
Why Rys Quadrature is Needed
============================

The key insight is to rewrite the Boys function integral:

  F_n(T) = integral_0^1 t^{2n} exp(-T t^2) dt

Using the substitution x = t^2 (so t = sqrt(x), dt = dx/(2*sqrt(x))):

  F_n(T) = (1/2) integral_0^1 x^{n-1/2} exp(-T x) dx
         = (1/2) integral_0^1 x^n * [x^{-1/2} exp(-T x)] dx

Now we have a polynomial x^n times a WEIGHT FUNCTION w_T(x) = x^{-1/2} exp(-Tx).

GAUSS-LEGENDRE PROBLEM:
-----------------------
Standard Gauss-Legendre treats exp(-T t^2) as part of the "function" to integrate,
requiring many points to capture its shape. As T increases, the exponential
becomes sharply peaked, and GL needs more points.

RYS QUADRATURE SOLUTION:
------------------------
Rys quadrature builds w_T(x) = x^{-1/2} exp(-Tx) into the WEIGHT FUNCTION.
Then only the polynomial x^n needs to be integrated!

Gaussian quadrature with n_r points is EXACT for polynomials up to degree 2n_r - 1.
So we need only:
    n_r = ceil((n + 1) / 2) points

For (ss|ss) integrals, n = 0, so n_r = 1 point suffices!
For higher angular momentum L = l_a + l_b + l_c + l_d:
    n_r = floor(L/2) + 1

EFFICIENCY COMPARISON:
----------------------
| Integral type | Angular momentum L | Rys points | GL points (1e-10) |
|---------------|--------------------|-----------:|------------------:|
| (ss|ss)       | 0                  | 1          | 16-32             |
| (ps|ss)       | 1                  | 1          | 32-64             |
| (pp|pp)       | 4                  | 3          | 64-128            |
| (dd|dd)       | 8                  | 5          | 128-256           |

The Rys approach converts a difficult quadrature problem into:
1. Computing moments m_k = 2 F_k(T) for k = 0, ..., 2n_r - 1
2. Finding orthogonal polynomial roots (Rys nodes)
3. Computing weights from the orthogonal polynomial

This is Chapter 5's topic: Rys Quadrature in Practice.
"""
    print(explanation)


def compare_efficiency_comprehensive() -> None:
    """
    Comprehensive comparison showing GL inefficiency.
    """
    print("\n" + "=" * 75)
    print("Comprehensive Efficiency Comparison")
    print("=" * 75)

    print("\nMinimum GL points for 1e-10 accuracy:")
    print("-" * 60)
    print(f"{'n':>3} {'T':>8} {'F_n(T)':>18} {'GL points':>12} {'Rys pts':>10}")
    print("-" * 60)

    test_cases = [
        (0, 0.1), (0, 1.0), (0, 10.0), (0, 50.0),
        (2, 0.1), (2, 1.0), (2, 10.0), (2, 50.0),
        (5, 0.1), (5, 1.0), (5, 10.0), (5, 50.0),
    ]

    for n, T in test_cases:
        ref = boys_reference(n, T)
        min_pts = find_min_points_for_accuracy(n, T, tol=1e-10)
        rys_pts = (n + 1 + 1) // 2  # ceil((n+1)/2)

        pts_str = str(min_pts) if min_pts > 0 else ">512"
        print(f"{n:>3} {T:>8.1f} {ref:>18.12e} {pts_str:>12} {rys_pts:>10}")

    print("-" * 60)
    print("Note: Rys points = ceil((n+1)/2), exact for polynomial degree 2*n_r - 1")


# =============================================================================
# Section 6: Detailed Convergence Study
# =============================================================================

def detailed_convergence_study() -> None:
    """
    Detailed study of GL convergence for selected cases.
    """
    print("\n" + "=" * 75)
    print("Detailed Convergence Study")
    print("=" * 75)

    cases = [
        (0, 1.0, "Easy: n=0, T=1"),
        (2, 10.0, "Moderate: n=2, T=10"),
        (5, 50.0, "Hard: n=5, T=50"),
    ]

    for n, T, description in cases:
        print(f"\n{description}")
        print(f"Reference F_{n}({T}) = {boys_reference(n, T):.15e}")
        print("-" * 55)
        print(f"{'n_pts':>6} {'Approximation':>20} {'Error':>12} {'Digits':>8}")
        print("-" * 55)

        for n_pts in [2, 4, 8, 16, 32, 64, 128, 256]:
            approx = boys_gauss_legendre(n, T, n_pts)
            error = abs(approx - boys_reference(n, T))

            if error > 0:
                digits = -np.log10(error / abs(boys_reference(n, T) + 1e-50))
            else:
                digits = 15.0

            print(f"{n_pts:>6} {approx:>20.15e} {error:>12.2e} {digits:>8.1f}")


# =============================================================================
# Section 7: Main Demonstration
# =============================================================================

def main():
    """Run complete Lab 4C demonstration."""
    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 4C: Numerical Quadrature Check for Boys Function" + " " * 14 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    print("Comparing generic Gauss-Legendre quadrature vs dedicated Boys evaluation")
    print("to motivate the specialized Rys quadrature (Chapter 5).")

    # ==========================================================================
    # Basic Comparison
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Basic Comparison: Series/Recursion vs Gauss-Legendre")
    print("=" * 75)

    test_cases = [(0, 0.1), (0, 10.0), (2, 0.1), (2, 10.0)]

    print(f"\n{'n':>3} {'T':>8} {'Series/Recur':>20} {'GL-64':>20} {'Difference':>12}")
    print("-" * 70)

    for n, T in test_cases:
        ref = boys_reference(n, T)
        gl64 = boys_gauss_legendre(n, T, 64)
        diff = abs(gl64 - ref)
        print(f"{n:>3} {T:>8.1f} {ref:>20.15e} {gl64:>20.15e} {diff:>12.2e}")

    # ==========================================================================
    # Integrand Shape Analysis
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Integrand Shape Analysis")
    print("=" * 75)

    print("\nAs T increases, the integrand t^{2n} exp(-T t^2) becomes:")
    print("  - More sharply peaked near t = 0")
    print("  - Harder to approximate with fixed GL quadrature")

    analyze_integrand_shape(n=2, T_values=[0.1, 1.0, 5.0, 10.0, 25.0, 50.0])

    # ASCII visualization
    demonstrate_integrand_shapes(n=2)

    # ==========================================================================
    # Detailed Convergence
    # ==========================================================================

    detailed_convergence_study()

    # ==========================================================================
    # Efficiency Comparison
    # ==========================================================================

    compare_efficiency_comprehensive()

    # ==========================================================================
    # Why Rys Quadrature
    # ==========================================================================

    explain_why_rys()

    # ==========================================================================
    # Summary
    # ==========================================================================

    print()
    print("=" * 75)
    print("What You Should Observe")
    print("=" * 75)

    observations = """
1. INTEGRAND SHAPE MATTERS:
   - Small T: Integrand is broad, polynomial-like -> GL works well
   - Large T: Integrand peaks sharply near t=0 -> GL needs many points

2. GL CONVERGENCE IS SLOW:
   - Need 64-128 points for 10 digits of accuracy
   - More points needed as T or n increases

3. EFFICIENCY GAP:
   - GL: O(64-256) function evaluations per integral
   - Rys: O(1-5) points for the same accuracy (depending on L)

4. PHYSICAL INSIGHT:
   The exponential exp(-T t^2) encodes the Coulomb 1/r interaction.
   Generic quadrature doesn't "know" this structure. Rys quadrature
   builds it into the weight function, requiring only polynomial moments.

5. CHAPTER 5 PREVIEW:
   Rys quadrature computes nodes/weights from Boys moments:
       m_k = 2 F_k(T) for k = 0, 1, ..., 2n_r - 1
   These are the first 2n_r moments of the weight function w_T(x).
   The Golub-Welsch algorithm then gives exact quadrature nodes/weights.

6. PRACTICAL IMPLICATION:
   Modern quantum chemistry codes use Rys quadrature (or variants)
   for ERI evaluation. This makes the difference between feasible
   and infeasible calculations for large molecules.
"""
    print(observations)

    print("=" * 75)
    print("Lab 4C Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
