#!/usr/bin/env python3
"""
boys_quadrature_comparison.py - Numerical Quadrature for Boys Function (Lab 4C)

This module compares different numerical quadrature approaches for evaluating
the Boys function:

    F_n(T) = integral_0^1 t^(2n) exp(-T t^2) dt

Demonstrates why generic Gauss-Legendre quadrature requires many points,
motivating specialized Rys quadrature developed in Chapter 5.

References:
    - Chapter 4, Section 8: Lab 4C
    - Chapter 5: Rys Quadrature in Practice

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import Tuple, List

# Import Boys function from companion module
from boys_function import boys, boys_series


def boys_gauss_legendre(n: int, T: float, npts: int = 64) -> float:
    """
    Approximate F_n(T) using Gauss-Legendre quadrature.

    The integrand t^(2n) exp(-Tt^2) is approximated on [0,1] by
    mapping from the standard Gauss-Legendre nodes on [-1,1].

    Parameters
    ----------
    n : int
        Order of Boys function
    T : float
        Argument
    npts : int
        Number of quadrature points

    Returns
    -------
    float
        Approximation to F_n(T)
    """
    # Gauss-Legendre nodes/weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(npts)

    # Map to [0, 1]: t = (x + 1)/2, dt = dx/2
    t = 0.5 * (x_gl + 1.0)
    w = 0.5 * w_gl

    # Evaluate integrand
    integrand = (t ** (2*n)) * np.exp(-T * t**2)

    return float(np.sum(w * integrand))


def boys_gauss_legendre_convergence(
    n: int, T: float, max_pts: int = 128
) -> List[Tuple[int, float, float]]:
    """
    Study convergence of Gauss-Legendre quadrature for F_n(T).

    Parameters
    ----------
    n : int
        Order of Boys function
    T : float
        Argument
    max_pts : int
        Maximum number of quadrature points

    Returns
    -------
    list of (npts, value, error)
        Convergence data
    """
    ref = boys(n, T)
    results = []

    for npts in [2, 4, 8, 16, 32, 64, 128]:
        if npts > max_pts:
            break
        approx = boys_gauss_legendre(n, T, npts)
        error = abs(approx - ref)
        results.append((npts, approx, error))

    return results


def analyze_integrand_shape(n: int, T_values: List[float]) -> None:
    """
    Analyze how the Boys integrand shape changes with T.

    Parameters
    ----------
    n : int
        Order of Boys function
    T_values : list
        T values to analyze
    """
    t = np.linspace(0, 1, 1000)

    print(f"\nIntegrand shape analysis for n={n}")
    print("-" * 60)
    print(f"{'T':>8} {'Max location':>15} {'Max value':>15} {'Width (FWHM)':>15}")
    print("-" * 60)

    for T in T_values:
        integrand = (t ** (2*n)) * np.exp(-T * t**2)

        # Find maximum
        max_idx = np.argmax(integrand)
        max_loc = t[max_idx]
        max_val = integrand[max_idx]

        # Estimate width (FWHM)
        half_max = max_val / 2
        above_half = integrand > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            width = t[indices[-1]] - t[indices[0]]
        else:
            width = 0.0

        print(f"{T:>8.2f} {max_loc:>15.6f} {max_val:>15.6e} {width:>15.6f}")


def quadrature_difficulty_vs_T(n: int, T_values: List[float]) -> None:
    """
    Show how quadrature difficulty changes with T.

    Parameters
    ----------
    n : int
        Order of Boys function
    T_values : list
        T values to analyze
    """
    print(f"\nQuadrature points needed for 1e-10 accuracy (n={n})")
    print("-" * 60)
    print(f"{'T':>8} {'Ref value':>18} {'Min pts for 1e-10':>20}")
    print("-" * 60)

    for T in T_values:
        ref = boys(n, T)

        # Find minimum points needed
        min_pts = None
        for npts in [2, 4, 8, 16, 32, 64, 128, 256]:
            approx = boys_gauss_legendre(n, T, npts)
            if abs(approx - ref) < 1e-10:
                min_pts = npts
                break

        pts_str = str(min_pts) if min_pts else ">256"
        print(f"{T:>8.2f} {ref:>18.12e} {pts_str:>20}")


def compare_methods_comprehensive():
    """
    Comprehensive comparison of Boys evaluation methods.
    """
    print("\nComprehensive method comparison")
    print("=" * 80)

    test_cases = [
        (0, 0.1),
        (0, 1.0),
        (0, 10.0),
        (0, 50.0),
        (2, 0.1),
        (2, 1.0),
        (2, 10.0),
        (2, 50.0),
        (5, 0.1),
        (5, 1.0),
        (5, 10.0),
        (5, 50.0),
    ]

    print(f"{'n':>3} {'T':>8} {'Series/Recur':>18} {'GL-16':>18} "
          f"{'GL-64':>18} {'GL-128':>18}")
    print("-" * 80)

    for n, T in test_cases:
        ref = boys(n, T)
        gl16 = boys_gauss_legendre(n, T, 16)
        gl64 = boys_gauss_legendre(n, T, 64)
        gl128 = boys_gauss_legendre(n, T, 128)

        # Show errors relative to series/recursion reference
        err16 = abs(gl16 - ref)
        err64 = abs(gl64 - ref)
        err128 = abs(gl128 - ref)

        print(f"{n:>3} {T:>8.1f} {ref:>18.12e} {err16:>18.2e} "
              f"{err64:>18.2e} {err128:>18.2e}")

    print("-" * 80)
    print("Note: GL-N columns show error vs series/recursion reference")


# =============================================================================
# Main demonstration
# =============================================================================

def main():
    """Run Boys quadrature comparison for Lab 4C."""
    print("=" * 70)
    print("Lab 4C: Numerical Quadrature Check for Boys Function")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Basic comparison
    print("\nBasic comparison: Series/Recursion vs Gauss-Legendre")
    print("-" * 70)

    test_cases = [(0, 0.1), (0, 10.0), (2, 0.1), (2, 10.0)]

    print(f"{'n':>3} {'T':>8} {'Series/Recur':>18} {'GLQ-64':>18} {'Diff':>12}")
    print("-" * 70)

    for n, T in test_cases:
        ref = boys(n, T)
        approx = boys_gauss_legendre(n, T, npts=64)
        diff = abs(approx - ref)
        print(f"{n:>3} {T:>8.1f} {ref:>18.12e} {approx:>18.12e} {diff:>12.2e}")

    # Checkpoint (a): Integrand shape analysis
    print("\n" + "=" * 70)
    print("Checkpoint (a): Integrand shape t^(2n) exp(-T t^2)")
    print("=" * 70)

    analyze_integrand_shape(2, [0.1, 1.0, 10.0])

    # Checkpoint (b) & (c): T regime analysis
    print("\n" + "=" * 70)
    print("Checkpoint (b)-(c): Quadrature difficulty vs T")
    print("=" * 70)

    print("\nFor small T: integrand is nearly polynomial -> easy for GLQ")
    print("For large T: integrand peaks sharply near t=0 -> harder for GLQ")

    quadrature_difficulty_vs_T(2, [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0])

    # Convergence study
    print("\n" + "=" * 70)
    print("Convergence of Gauss-Legendre quadrature")
    print("=" * 70)

    for n, T in [(0, 1.0), (2, 10.0), (5, 50.0)]:
        print(f"\nn={n}, T={T}")
        print(f"{'npts':>6} {'Approx':>18} {'Error':>12}")
        print("-" * 40)

        conv = boys_gauss_legendre_convergence(n, T)
        for npts, val, err in conv:
            print(f"{npts:>6} {val:>18.12e} {err:>12.2e}")

    # Checkpoint (d): Why Rys quadrature?
    print("\n" + "=" * 70)
    print("Checkpoint (d): Why Rys quadrature?")
    print("=" * 70)

    print("""
Rys quadrature is 'customized' for the Boys integrand because:

1. WEIGHT FUNCTION: The standard Boys integral can be rewritten as
   F_n(T) = (1/2) integral_0^1 x^(n-1/2) exp(-Tx) dx
   with weight function w_T(x) = x^(-1/2) exp(-Tx).

2. OPTIMAL NODES: Rys quadrature computes nodes and weights that are
   EXACT for all polynomials up to degree 2n_r - 1 under this specific
   weight function.

3. EFFICIENCY: For (ss|ss) ERIs, only n_r = 1 root is needed.
   For higher angular momentum, n_r = floor(L/2) + 1 roots suffice.
   This is far fewer than the 64-128 points needed by generic GLQ.

4. T-DEPENDENCE: The nodes and weights depend on T, adapting to
   wherever the weight function concentrates. This is why we compute
   them from the moments m_k = 2 F_k(T).

Generic Gauss-Legendre treats exp(-T t^2) as part of the integrand,
requiring many points to capture its shape. Rys quadrature builds
this exponential into the weight function, so only polynomial
moments remain to be integrated.
""")

    # Comprehensive comparison
    compare_methods_comprehensive()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
- Generic Gauss-Legendre requires 64-128+ points for high accuracy
- Efficiency decreases for large T (sharply peaked integrand)
- Rys quadrature exploits the specific integrand structure
- Rys with n_r points is exact for appropriate polynomial degrees
- This motivates Chapter 5: Rys Quadrature in Practice
""")


if __name__ == "__main__":
    main()
