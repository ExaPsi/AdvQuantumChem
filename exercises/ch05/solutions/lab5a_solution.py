#!/usr/bin/env python3
"""
Lab 5A Solution: Construct Rys Nodes/Weights and Verify Moment Matching

This script implements Algorithm 5.1 for constructing Gaussian quadrature nodes
and weights for the Rys weight function w_T(x) = x^{-1/2} exp(-Tx) on [0,1].

Algorithm 5.1 (Hankel + Cholesky + Golub-Welsch):
    1. Compute moments m_k = 2*F_k(T) for k = 0, 1, ..., 2*n_roots - 1
    2. Build Hankel matrix H with H_ij = m_{i+j}
    3. Build shifted Hankel H^(1) with H^(1)_ij = m_{i+j+1}
    4. Cholesky factorize: H = L L^T, then C = L^{-1}
    5. Build Jacobi matrix: J = C H^(1) C^T
    6. Diagonalize J: eigenvalues are nodes, weights from eigenvectors

The quadrature satisfies moment matching:
    sum_i W_i * x_i^n = m_n    for n = 0, 1, ..., 2*n_roots - 1

Learning objectives:
1. Implement the moment-based construction of Gaussian quadrature
2. Understand the connection between Boys functions and Rys moments
3. Verify moment matching to floating-point precision
4. Test numerical stability across a wide range of T values

Test values: T = 0.0, 1e-6, 0.1, 1.0, 10.0

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 5: Rys Quadrature in Practice
"""

import math
import numpy as np
from scipy import special
from typing import Tuple

# =============================================================================
# Section 1: Boys Function Implementation
# =============================================================================

def boys_series(n: int, T: float, max_terms: int = 60, tol: float = 1e-16) -> float:
    """
    Evaluate F_n(T) using the series expansion (stable for small T).

    F_n(T) = sum_{k>=0} (-T)^k / [k! * (2n + 2k + 1)]

    This series converges rapidly for small T and is numerically stable
    because all terms have the same sign structure.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        max_terms: Maximum number of terms in series
        tol: Convergence tolerance

    Returns:
        Value of F_n(T)
    """
    val = 0.0
    term = 1.0 / (2 * n + 1)

    for k in range(max_terms):
        val += term
        if abs(term) < tol * abs(val) and k > 0:
            break
        # Next term: term *= (-T) / (k+1) * (2n+2k+1) / (2n+2k+3)
        term *= -T / (k + 1) * (2 * n + 2 * k + 1) / (2 * n + 2 * k + 3)

    return val


def boys_erf_recursion(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using F_0 from erf + upward recursion.

    F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))
    F_{m+1}(T) = [(2m+1) F_m(T) - exp(-T)] / (2T)

    This method is stable for moderate to large T but suffers from
    catastrophic cancellation for small T with large n.

    Args:
        n: Order of the Boys function
        T: Argument (must be > 0 for F_0 formula)

    Returns:
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
            F = ((2 * m + 1) * F - exp_neg_T) / (2 * T)
        else:
            # For T -> 0, F_n(0) = 1/(2n+1)
            F = 1.0 / (2 * m + 3)

    return F


def boys(n: int, T: float, T_switch: float = 25.0) -> float:
    """
    Evaluate the Boys function F_n(T) using a hybrid strategy.

    Strategy:
        - T < T_switch: Use series expansion (stable for all n)
        - T >= T_switch: Use F_0 from erf + upward recursion (stable for large T)

    The Boys function appears in all integrals involving 1/r operators:
        F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt

    Key properties:
        - F_n(0) = 1/(2n+1)
        - F_0(T) = (1/2)*sqrt(pi/T)*erf(sqrt(T)) for T > 0
        - Recurrence: F_{n+1}(T) = [(2n+1)*F_n(T) - exp(-T)] / (2T)

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        T_switch: Crossover point between series and recursion methods

    Returns:
        Value of F_n(T)
    """
    if T < T_switch:
        return boys_series(n, T)
    else:
        return boys_erf_recursion(n, T)


def boys_hyp1f1(n: int, T: float) -> float:
    """
    Evaluate F_n(T) using the hypergeometric function (reference).

    F_n(T) = (1/(2n+1)) * hyp1f1(n + 0.5, n + 1.5, -T)

    This serves as an independent reference for validation using scipy.

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        Value of F_n(T)
    """
    return special.hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)


# =============================================================================
# Section 2: Moment Computation
# =============================================================================

def moment(n: int, T: float) -> float:
    """
    Compute the n-th moment of the Rys weight function w_T(x) = x^{-1/2} exp(-Tx).

    m_n(T) = integral from 0 to 1 of x^n * w_T(x) dx = 2 * F_n(T)

    The factor of 2 arises from the change of variables t^2 = x in the
    Boys integral:
        F_n(T) = integral_0^1 t^{2n} exp(-T*t^2) dt
               = (1/2) * integral_0^1 x^n * x^{-1/2} * exp(-Tx) dx
               = (1/2) * m_n(T)

    Args:
        n: Moment order (n >= 0)
        T: Parameter in the weight function

    Returns:
        Value of m_n(T) = 2 * F_n(T)
    """
    return 2.0 * boys(n, T)


def moments_array(n_max: int, T: float) -> np.ndarray:
    """
    Compute moments m_0(T), m_1(T), ..., m_{n_max}(T).

    Args:
        n_max: Maximum moment order
        T: Parameter in the weight function

    Returns:
        Array of [m_0(T), m_1(T), ..., m_{n_max}(T)]
    """
    return np.array([moment(n, T) for n in range(n_max + 1)])


# =============================================================================
# Section 3: Hankel Matrix Construction
# =============================================================================

def build_hankel_matrices(moments: np.ndarray, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Hankel matrices H and H^(1) from moments.

    The Hankel matrix H is defined as:
        H_ij = m_{i+j}    for i, j = 0, 1, ..., n_roots - 1

    The shifted Hankel matrix H^(1) is:
        H^(1)_ij = m_{i+j+1}

    Physical interpretation:
    - H is the Gram matrix of monomials {1, x, x^2, ...} under the inner product
      <f, g> = integral f(x) g(x) w_T(x) dx
    - H^(1) represents multiplication by x in this basis

    Args:
        moments: Array of moments [m_0, m_1, ..., m_{2*n_roots-1}]
        n_roots: Number of quadrature roots

    Returns:
        H: Hankel matrix of shape (n_roots, n_roots)
        H1: Shifted Hankel matrix of shape (n_roots, n_roots)
    """
    H = np.zeros((n_roots, n_roots))
    H1 = np.zeros((n_roots, n_roots))

    for i in range(n_roots):
        for j in range(n_roots):
            H[i, j] = moments[i + j]
            H1[i, j] = moments[i + j + 1]

    return H, H1


# =============================================================================
# Section 4: Rys Quadrature Construction (Algorithm 5.1)
# =============================================================================

def rys_nodes_weights(T: float, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Rys quadrature nodes and weights (Algorithm 5.1).

    This implements the Golub-Welsch algorithm for constructing Gaussian
    quadrature from moments:

    Algorithm 5.1:
        1. Compute moments m_k = 2*F_k(T) for k = 0, ..., 2*n_roots - 1
        2. Build Hankel matrices H and H^(1)
        3. Cholesky factorize H = L L^T; set C = L^{-1}
        4. Build Jacobi matrix J = C H^(1) C^T
        5. Diagonalize J -> eigenvalues are nodes x_i
        6. Weights: W_i = m_0 * (V_{0,i})^2

    The resulting quadrature is exact for polynomials up to degree 2*n_roots - 1:
        sum_i W_i x_i^n = m_n    for n = 0, 1, ..., 2*n_roots - 1

    Args:
        T: Parameter in the weight function (T >= 0)
        n_roots: Number of quadrature roots (n_roots >= 1)

    Returns:
        nodes: Quadrature nodes x_i in (0, 1), sorted in ascending order
        weights: Quadrature weights W_i > 0 (same order as nodes)
    """
    if n_roots < 1:
        raise ValueError("n_roots must be >= 1")

    # Step 1: Compute moments m_k = 2*F_k(T) for k = 0, ..., 2*n_roots - 1
    m = moments_array(2 * n_roots - 1, T)

    # Step 2: Build Hankel matrices
    H, H1 = build_hankel_matrices(m, n_roots)

    # Step 3: Cholesky factorization of H and compute C = L^{-1}
    # The Hankel matrix H should be positive definite for T >= 0
    try:
        L = np.linalg.cholesky(H)
        # C = L^{-1} (solve L @ C = I)
        C = np.linalg.solve(L, np.eye(n_roots))
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue decomposition for ill-conditioned cases
        # H = V @ diag(d) @ V^T, then H^{-1/2} = V @ diag(1/sqrt(d)) @ V^T
        eigvals, V = np.linalg.eigh(H)

        # Threshold small/negative eigenvalues for numerical stability
        thresh = max(1e-12 * eigvals.max(), 1e-30)
        eigvals = np.maximum(eigvals, thresh)

        # Compute H^{-1/2} via eigendecomposition
        d_inv_sqrt = 1.0 / np.sqrt(eigvals)
        C = V @ np.diag(d_inv_sqrt) @ V.T

    # Step 4: Build Jacobi matrix J = C @ H1 @ C^T
    J = C @ H1 @ C.T

    # Symmetrize to remove tiny numerical asymmetry
    J = 0.5 * (J + J.T)

    # Step 5: Golub-Welsch eigendecomposition
    eigenvalues, V = np.linalg.eigh(J)

    # Nodes are the eigenvalues (clamp to [0, 1] for numerical safety)
    nodes = np.clip(eigenvalues, 0.0, 1.0)

    # Step 6: Weights from eigenvectors
    # W_i = m_0 * (V_{0,i})^2
    weights = m[0] * (V[0, :] ** 2)

    # Sort by ascending node value
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]

    return nodes, weights


# =============================================================================
# Section 5: Moment Matching Verification
# =============================================================================

def verify_moment_matching(T: float, n_roots: int, verbose: bool = True) -> Tuple[float, np.ndarray]:
    """
    Verify that nodes and weights satisfy moment matching.

    Tests: sum_i W_i * x_i^n = m_n for n = 0, 1, ..., 2*n_roots - 1

    This is the fundamental property of Gaussian quadrature: an n_roots-point
    rule is exact for polynomials of degree up to 2*n_roots - 1.

    Args:
        T: Parameter value
        n_roots: Number of roots
        verbose: Print detailed results

    Returns:
        max_error: Maximum absolute error in moment matching
        errors: Array of errors for each moment
    """
    nodes, weights = rys_nodes_weights(T, n_roots)

    if verbose:
        print(f"\nVerification for T = {T}, n_roots = {n_roots}")
        print("-" * 70)
        print(f"{'n':>4} {'m_n (exact)':>20} {'sum W*x^n':>20} {'Error':>14}")
        print("-" * 70)

    errors = []
    for n in range(2 * n_roots):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        errors.append(error)

        if verbose:
            print(f"{n:>4} {m_exact:>20.14e} {m_quad:>20.14e} {error:>14.2e}")

    errors = np.array(errors)
    max_error = np.max(errors)

    if verbose:
        print("-" * 70)
        print(f"Maximum error: {max_error:.2e}")
        status = "PASS" if max_error < 1e-10 else "FAIL"
        print(f"Status: {status}")

    return max_error, errors


def verify_beyond_exactness(T: float, n_roots: int, n_extra: int = 3, verbose: bool = True) -> np.ndarray:
    """
    Test moment matching for n >= 2*n_roots (where exactness is NOT guaranteed).

    This demonstrates the sharp boundary of Gaussian quadrature: exactness holds
    for polynomials of degree <= 2*n_roots - 1, but fails for higher degrees.

    Args:
        T: Parameter value
        n_roots: Number of roots
        n_extra: How many moments beyond exactness range to check
        verbose: Print results

    Returns:
        errors: Array of errors for n = 2*n_roots, ..., 2*n_roots + n_extra - 1
    """
    nodes, weights = rys_nodes_weights(T, n_roots)

    if verbose:
        print(f"\nBeyond exactness range (n >= {2*n_roots}):")
        print("-" * 70)
        print(f"{'n':>4} {'m_n (exact)':>20} {'sum W*x^n':>20} {'Error':>14}")
        print("-" * 70)

    errors = []
    for n in range(2 * n_roots, 2 * n_roots + n_extra):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        errors.append(error)

        if verbose:
            print(f"{n:>4} {m_exact:>20.14e} {m_quad:>20.14e} {error:>14.2e}")

    if verbose:
        print("-" * 70)
        print("Note: Errors expected for n >= 2*n_roots (beyond exactness range)")

    return np.array(errors)


# =============================================================================
# Section 6: Comprehensive Testing
# =============================================================================

def test_all_T_values(n_roots: int = 2) -> bool:
    """
    Test moment matching for the specified T values: 0.0, 1e-6, 0.1, 1.0, 10.0

    Args:
        n_roots: Number of quadrature roots

    Returns:
        all_passed: True if all tests pass within tolerance
    """
    T_values = [0.0, 1e-6, 0.1, 1.0, 10.0]

    print("=" * 75)
    print(f"Moment Matching Tests for n_roots = {n_roots}")
    print("=" * 75)
    print(f"{'T':>12} {'Max Error':>16} {'Status':>10}")
    print("-" * 75)

    all_passed = True
    results = []

    for T in T_values:
        max_err, _ = verify_moment_matching(T, n_roots, verbose=False)
        status = "PASS" if max_err < 1e-10 else "FAIL"
        results.append((T, max_err, status))
        print(f"{T:>12.2e} {max_err:>16.2e} {status:>10}")

        if max_err >= 1e-10:
            all_passed = False

    print("-" * 75)
    return all_passed


def print_nodes_weights_table(T_values: list, n_roots: int = 2) -> None:
    """
    Print a table of Rys nodes and weights for reference.

    Args:
        T_values: List of T values to display
        n_roots: Number of quadrature roots
    """
    print("\n" + "=" * 75)
    print(f"Rys Nodes and Weights Table (n_roots = {n_roots})")
    print("=" * 75)

    for T in T_values:
        nodes, weights = rys_nodes_weights(T, n_roots)

        print(f"\nT = {T}")
        print("-" * 50)
        print(f"{'i':>4} {'x_i (node)':>20} {'W_i (weight)':>20}")
        print("-" * 50)

        for i, (x, W) in enumerate(zip(nodes, weights)):
            print(f"{i+1:>4} {x:>20.15f} {W:>20.15f}")

        # Verify sum of weights = m_0
        m0 = moment(0, T)
        W_sum = np.sum(weights)
        print(f"\nSum of weights: {W_sum:.15f}")
        print(f"m_0 = 2*F_0(T):  {m0:.15f}")
        print(f"Difference:      {abs(W_sum - m0):.2e}")


# =============================================================================
# Section 7: Physical Interpretation
# =============================================================================

def explain_rys_quadrature() -> None:
    """Explain the physical meaning and applications of Rys quadrature."""
    explanation = """
Physical Interpretation of Rys Quadrature
==========================================

The Rys quadrature arises from evaluating electron repulsion integrals (ERIs).
The Boys function F_n(T) appears in all ERIs and can be written as a moment:

    F_n(T) = (1/2) * integral_0^1 x^n * w_T(x) dx

where w_T(x) = x^{-1/2} * exp(-T*x) is the Rys weight function.

GAUSSIAN QUADRATURE PROPERTY:
-----------------------------
An n_roots-point Gaussian quadrature rule with nodes {x_i} and weights {W_i}
satisfies:

    integral_0^1 x^n * w_T(x) dx = sum_i W_i * x_i^n

exactly for n = 0, 1, ..., 2*n_roots - 1.

ROOT COUNT RULE FOR ERIs:
-------------------------
For a shell quartet with total angular momentum L = l_a + l_b + l_c + l_d,
the number of Rys roots needed is:

    n_roots = floor(L/2) + 1

Examples:
  - (ss|ss): L = 0, n_roots = 1
  - (ps|ss): L = 1, n_roots = 1
  - (pp|ss): L = 2, n_roots = 2
  - (pp|pp): L = 4, n_roots = 3
  - (dd|dd): L = 8, n_roots = 5

WHY THIS WORKS:
---------------
The ERI for Cartesian Gaussians involves integrals of the form:

    integral x^n * w_T(x) dx

where n comes from the polynomial prefactors (x-A)^l, etc. The maximum
polynomial degree is L, so we need exactness for n = 0, 1, ..., L.
Since 2*n_roots - 1 >= L, we need n_roots >= (L+1)/2, i.e., floor(L/2) + 1.

NUMERICAL ADVANTAGES:
---------------------
1. Exact for the polynomial integrands that appear in ERIs
2. Number of roots independent of basis set size
3. Nodes and weights computed once per shell quartet argument T
4. Enables efficient recursive integral schemes (McMurchie-Davidson, etc.)
"""
    print(explanation)


# =============================================================================
# Section 8: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 5A demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 5A: Rys Quadrature Nodes/Weights and Moment Matching" + " " * 13 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Test values specified in the lab
    test_T_values = [0.0, 1e-6, 0.1, 1.0, 10.0]

    # ==========================================================================
    # Section A: Boys Function Validation
    # ==========================================================================

    print("=" * 75)
    print("Section A: Boys Function Validation")
    print("=" * 75)
    print("\nComparing our Boys function against scipy.special.hyp1f1:")
    print("-" * 75)
    print(f"{'n':>4} {'T':>10} {'Our boys()':>22} {'scipy hyp1f1':>22} {'Diff':>12}")
    print("-" * 75)

    boys_passed = True
    for n in [0, 2, 5]:
        for T in [0.0, 0.1, 1.0, 10.0, 50.0]:
            our_val = boys(n, T)
            ref_val = boys_hyp1f1(n, T)
            diff = abs(our_val - ref_val)
            print(f"{n:>4} {T:>10.2e} {our_val:>22.15e} {ref_val:>22.15e} {diff:>12.2e}")
            if diff > 1e-12:
                boys_passed = False

    print("-" * 75)
    print(f"Boys function validation: {'PASS' if boys_passed else 'FAIL'}")

    # ==========================================================================
    # Section B: Moment Properties
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section B: Moment Properties (m_n = 2*F_n)")
    print("=" * 75)

    print("\nAt T = 0: m_n(0) = 2/(2n+1)")
    print("-" * 40)
    print(f"{'n':>4} {'m_n(0) computed':>20} {'2/(2n+1) exact':>20}")
    print("-" * 40)
    for n in range(6):
        computed = moment(n, 0.0)
        exact = 2.0 / (2 * n + 1)
        print(f"{n:>4} {computed:>20.15f} {exact:>20.15f}")

    # ==========================================================================
    # Section C: Hankel Matrix Construction
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section C: Hankel Matrix Construction")
    print("=" * 75)

    T_demo = 1.0
    n_roots_demo = 2
    moments = moments_array(2 * n_roots_demo - 1, T_demo)
    H, H1 = build_hankel_matrices(moments, n_roots_demo)

    print(f"\nFor T = {T_demo}, n_roots = {n_roots_demo}:")
    print(f"\nMoments: m = {moments}")
    print(f"\nHankel matrix H (H_ij = m_{{i+j}}):")
    print(H)
    print(f"\nShifted Hankel matrix H^(1) (H^(1)_ij = m_{{i+j+1}}):")
    print(H1)

    # Verify H is positive definite
    eigvals_H = np.linalg.eigvalsh(H)
    print(f"\nEigenvalues of H: {eigvals_H}")
    print(f"H is positive definite: {np.all(eigvals_H > 0)}")

    # ==========================================================================
    # Section D: Rys Quadrature Construction
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section D: Algorithm 5.1 - Rys Quadrature Construction")
    print("=" * 75)

    nodes, weights = rys_nodes_weights(T_demo, n_roots_demo)
    print(f"\nFor T = {T_demo}, n_roots = {n_roots_demo}:")
    print(f"Nodes:   {nodes}")
    print(f"Weights: {weights}")

    # ==========================================================================
    # Section E: Moment Matching Verification
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section E: Moment Matching Verification")
    print("=" * 75)

    # Detailed verification for one case
    verify_moment_matching(T_demo, n_roots_demo, verbose=True)

    # Test beyond exactness range
    verify_beyond_exactness(T_demo, n_roots_demo, n_extra=3, verbose=True)

    # ==========================================================================
    # Section F: Comprehensive T Value Testing
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section F: Comprehensive Testing (T = 0.0, 1e-6, 0.1, 1.0, 10.0)")
    print("=" * 75)

    all_passed_2root = test_all_T_values(n_roots=2)
    all_passed_3root = test_all_T_values(n_roots=3)

    # ==========================================================================
    # Section G: Nodes and Weights Tables
    # ==========================================================================

    print_nodes_weights_table(test_T_values, n_roots=2)

    # ==========================================================================
    # Section H: Single-Root Special Case
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Section H: Single-Root Special Case (Exercise 5.1)")
    print("=" * 75)
    print("\nFor n_roots = 1, analytic formulas apply:")
    print("  W_1 = m_0 = 2*F_0(T)")
    print("  x_1 = m_1/m_0 = F_1(T)/F_0(T)")
    print("-" * 60)
    print(f"{'T':>10} {'x_1':>18} {'F_1/F_0':>18} {'W_1':>18} {'m_0':>18}")
    print("-" * 60)

    for T in test_T_values:
        nodes, weights = rys_nodes_weights(T, 1)
        F0 = boys(0, T)
        F1 = boys(1, T)
        m0 = 2.0 * F0
        x1_expected = F1 / F0 if F0 > 1e-15 else 0.5  # Limit as T -> inf

        print(f"{T:>10.2e} {nodes[0]:>18.14f} {x1_expected:>18.14f} "
              f"{weights[0]:>18.14f} {m0:>18.14f}")

    # ==========================================================================
    # Section I: Physical Interpretation
    # ==========================================================================

    explain_rys_quadrature()

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("=" * 75)
    print("Lab 5A Summary")
    print("=" * 75)
    print(f"Boys function validation:     {'PASS' if boys_passed else 'FAIL'}")
    print(f"2-root moment matching:       {'PASS' if all_passed_2root else 'FAIL'}")
    print(f"3-root moment matching:       {'PASS' if all_passed_3root else 'FAIL'}")

    all_tests_passed = boys_passed and all_passed_2root and all_passed_3root

    print("-" * 75)
    if all_tests_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED - Review output above")
    print("=" * 75)


if __name__ == "__main__":
    main()
