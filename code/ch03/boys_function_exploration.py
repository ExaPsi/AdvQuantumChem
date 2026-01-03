#!/usr/bin/env python3
"""
Boys Function Exploration

The Boys function F_n(T) is central to molecular integrals involving
the Coulomb operator 1/r:

    F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt

This script explores:
1. Evaluation methods (series, recurrence, special values)
2. Numerical stability considerations
3. Connection to nuclear attraction and ERI integrals
4. Asymptotic behavior

Supports Exercise 3.8 from Chapter 3.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
import math
from scipy import special


def boys_series(n: int, T: float, max_terms: int = 50, tol: float = 1e-15) -> float:
    """
    Compute Boys function using power series expansion.

    F_n(T) = sum_{k=0}^{infinity} (-T)^k / (k! * (2n + 2k + 1))

    Converges well for small T (T < 10-15 typically).

    Parameters
    ----------
    n : int
        Order of Boys function
    T : float
        Argument
    max_terms : int
        Maximum number of terms
    tol : float
        Convergence tolerance

    Returns
    -------
    float
        F_n(T) value
    """
    result = 0.0
    term = 1.0 / (2 * n + 1)  # First term

    for k in range(max_terms):
        result += term
        # Next term: multiply by -T/(k+1) and adjust denominator
        term *= -T / (k + 1)
        term *= (2 * n + 2 * k + 1) / (2 * n + 2 * k + 3)
        if abs(term) < tol:
            break

    return result


def boys_erf(T: float) -> float:
    """
    Compute F_0(T) using error function formula.

    F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))  for T > 0
    F_0(0) = 1

    This is exact for F_0 and serves as starting point for recurrence.
    """
    if T < 1e-15:
        # Series: F_0(T) = 1 - T/3 + T^2/10 - ...
        return 1.0 - T / 3.0 + T**2 / 10.0 - T**3 / 42.0
    return 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))


def boys_upward_recurrence(n_max: int, T: float) -> np.ndarray:
    """
    Compute F_n(T) for n = 0, 1, ..., n_max using upward recurrence.

    F_{n+1}(T) = [(2n+1) * F_n(T) - exp(-T)] / (2T)

    WARNING: Numerically unstable for large n and small T.

    Parameters
    ----------
    n_max : int
        Maximum order to compute
    T : float
        Argument

    Returns
    -------
    np.ndarray
        Array of F_n(T) for n = 0, ..., n_max
    """
    F = np.zeros(n_max + 1)

    if T < 1e-15:
        # Use special values at T = 0: F_n(0) = 1/(2n+1)
        for n in range(n_max + 1):
            F[n] = 1.0 / (2 * n + 1)
        return F

    # Start with F_0 from error function
    F[0] = boys_erf(T)

    # Upward recurrence
    exp_mT = math.exp(-T)
    for n in range(n_max):
        F[n + 1] = ((2 * n + 1) * F[n] - exp_mT) / (2 * T)

    return F


def boys_downward_recurrence(n_max: int, T: float, n_start: int = None) -> np.ndarray:
    """
    Compute F_n(T) using downward recurrence (numerically stable).

    F_n(T) = [2T * F_{n+1}(T) + exp(-T)] / (2n+1)

    Starts from large n using asymptotic expansion, recurs downward.

    Parameters
    ----------
    n_max : int
        Maximum order needed
    T : float
        Argument
    n_start : int
        Starting order for recurrence (default: n_max + 15)

    Returns
    -------
    np.ndarray
        Array of F_n(T) for n = 0, ..., n_max
    """
    if n_start is None:
        n_start = n_max + 15

    if T < 1e-15:
        # Use special values at T = 0: F_n(0) = 1/(2n+1)
        F = np.zeros(n_max + 1)
        for n in range(n_max + 1):
            F[n] = 1.0 / (2 * n + 1)
        return F

    # Asymptotic value for large n
    # F_n(T) ~ (2n-1)!! / (2T)^{n+1/2} * sqrt(pi/2) for large n or T
    # For starting the recurrence, use series at n_start
    F_high = boys_series(n_start, T)

    # Downward recurrence
    F = np.zeros(n_start + 1)
    F[n_start] = F_high

    exp_mT = math.exp(-T)
    for n in range(n_start - 1, -1, -1):
        F[n] = (2 * T * F[n + 1] + exp_mT) / (2 * n + 1)

    return F[:n_max + 1]


def boys_scipy(n: int, T: float) -> float:
    """
    Compute F_n(T) using scipy's incomplete gamma function.

    F_n(T) = gamma(n+1/2) * gammainc(n+1/2, T) / (2 * T^(n+1/2))

    This is a reference implementation for validation.
    """
    if T < 1e-15:
        return 1.0 / (2 * n + 1)

    a = n + 0.5
    # gammainc is the regularized incomplete gamma: P(a, x) = gamma(a, x) / Gamma(a)
    # We need the lower incomplete gamma: gamma(a, x) = P(a, x) * Gamma(a)
    return 0.5 * special.gamma(a) * special.gammainc(a, T) / (T ** a)


def main():
    print("=" * 70)
    print("Boys Function Exploration")
    print("=" * 70)

    # Section 1: Special values at T = 0
    print("\n" + "-" * 50)
    print("1. SPECIAL VALUES AT T = 0")
    print("-" * 50)

    print("\n   F_n(0) = 1/(2n+1)")
    print("\n   n    F_n(0)      Computed")
    print("   " + "-" * 30)

    for n in range(6):
        exact = 1.0 / (2 * n + 1)
        computed = boys_series(n, 0.0)
        print(f"   {n}    {exact:.6f}    {computed:.6f}")

    # Section 2: Values at T = 1.0 (validation points)
    print("\n" + "-" * 50)
    print("2. VALUES AT T = 1.0 (Validation Points)")
    print("-" * 50)

    T = 1.0
    F_upward = boys_upward_recurrence(5, T)
    F_downward = boys_downward_recurrence(5, T)

    print(f"\n   T = {T}")
    print("\n   n    Upward       Downward     Scipy (ref)")
    print("   " + "-" * 45)

    for n in range(6):
        ref = boys_scipy(n, T)
        print(f"   {n}    {F_upward[n]:.8f}   {F_downward[n]:.8f}   {ref:.8f}")

    print("\n   Reference values for Exercise 3.8 validation:")
    print(f"   F_0(1) = {boys_scipy(0, 1.0):.4f}")
    print(f"   F_1(1) = {boys_scipy(1, 1.0):.4f}")
    print(f"   F_2(1) = {boys_scipy(2, 1.0):.4f}")

    # Section 3: Behavior over T range
    print("\n" + "-" * 50)
    print("3. BEHAVIOR OVER T RANGE")
    print("-" * 50)

    print("\n   F_n(T) for T = 0, 1, 5, 10, 20:")
    print("\n   n    T=0       T=1       T=5       T=10      T=20")
    print("   " + "-" * 55)

    for n in range(5):
        values = [boys_scipy(n, T) for T in [0, 1, 5, 10, 20]]
        print(f"   {n}    " + "   ".join(f"{v:.4f}" for v in values))

    print("""
   Observations:
   - F_n(T) decreases monotonically with T for fixed n
   - F_n(T) decreases with n for fixed T
   - F_n(T) -> 0 as T -> infinity
   - Decay is faster for larger n
    """)

    # Section 4: Numerical stability of recurrences
    print("-" * 50)
    print("4. NUMERICAL STABILITY OF RECURRENCES")
    print("-" * 50)

    print("\n   Testing at T = 1e-6 (small T, where upward is unstable):")
    T_small = 1e-6

    F_upward = boys_upward_recurrence(10, T_small)
    F_downward = boys_downward_recurrence(10, T_small)

    print("\n   n    Upward         Downward       Exact (series)")
    print("   " + "-" * 50)

    for n in range(11):
        exact = boys_series(n, T_small)
        up_err = abs(F_upward[n] - exact)
        down_err = abs(F_downward[n] - exact)
        print(f"   {n:2d}   {F_upward[n]:.10f}   {F_downward[n]:.10f}   {exact:.10f}")

    print("""
   Key insight:
   - Upward recurrence is UNSTABLE for small T and large n
   - The error grows exponentially with n
   - Downward recurrence is STABLE because errors diminish

   Practical implementations use:
   - Series expansion for small T
   - Asymptotic expansion for large T
   - Downward recurrence in intermediate regions
    """)

    # Section 5: Asymptotic behavior
    print("-" * 50)
    print("5. ASYMPTOTIC BEHAVIOR (Large T)")
    print("-" * 50)

    print("\n   For large T: F_n(T) ~ (2n-1)!! / (2T)^{n+1/2} * sqrt(pi/2)")
    print("\n   T = 50:")

    T_large = 50.0
    for n in range(4):
        exact = boys_scipy(n, T_large)
        # Asymptotic: (2n-1)!! * sqrt(pi/2) / (2T)^(n+0.5)
        # (2n-1)!! = 1 for n=0, 1 for n=1, 3 for n=2, 15 for n=3
        double_fact = [1, 1, 3, 15][n]
        asymp = double_fact * math.sqrt(math.pi / 2) / (2 * T_large) ** (n + 0.5)
        print(f"   F_{n}(50) = {exact:.6e}, asymptotic = {asymp:.6e}")

    # Section 6: Physical interpretation
    print("\n" + "-" * 50)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 50)

    print("""
   The Boys function appears in Coulomb integrals:

   1. NUCLEAR ATTRACTION (s-s):
      V_ab = -Z * (2*pi/p) * exp(-mu*R_AB^2) * F_0(p*|P-C|^2)

      - F_0 modulates the Coulomb interaction
      - Large T = large |P-C| -> F_0 -> 0 (distant nucleus)
      - Small T = small |P-C| -> F_0 -> 1 (nucleus at overlap center)

   2. TWO-ELECTRON INTEGRALS:
      For (ab|cd), we need F_n(T) for n = 0, 1, ..., L
      where L = l_a + l_b + l_c + l_d (sum of angular momenta)

      - (ss|ss): only F_0 needed
      - (ps|ss): F_0 and F_1 needed
      - (pp|pp): F_0, F_1, F_2, F_3, F_4 needed

   3. WHY F_n(T) -> 0 AS T -> infinity:
      - Large T means large separation between charge distributions
      - The integral of t^(2n) * exp(-T*t^2) becomes negligible
      - Physically: distant electrons interact weakly

   4. ROLE IN RYS QUADRATURE (Chapter 4):
      - F_n(T) are the "moments" of the Rys weight function
      - Rys quadrature uses these to compute integrals exactly
      - Key relation: Boys moments -> orthogonal polynomials -> quadrature
    """)

    # Section 7: Exercise 3.8 validation points
    print("-" * 50)
    print("7. EXERCISE 3.8 VALIDATION POINTS")
    print("-" * 50)

    print("\n   At T = 1.0:")
    print(f"   F_0(1) = {boys_scipy(0, 1.0):.4f}  (expected: 0.7468)")
    print(f"   F_1(1) = {boys_scipy(1, 1.0):.4f}  (expected: 0.1894)")
    print(f"   F_2(1) = {boys_scipy(2, 1.0):.4f}  (expected: 0.1003)")

    print("\n   Upward recurrence verification:")
    F = boys_upward_recurrence(3, 1.0)
    print(f"   F_0(1) from erf:       {F[0]:.6f}")
    print(f"   F_1(1) from recurrence: {F[1]:.6f}")
    print(f"   F_2(1) from recurrence: {F[2]:.6f}")

    print("\n" + "=" * 70)
    print("Boys Function Exploration Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
