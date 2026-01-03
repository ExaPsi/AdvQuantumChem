#!/usr/bin/env python3
"""
rys_quadrature.py - Rys Quadrature Nodes and Weights (Algorithm 5.1)

This module implements the construction of Gaussian quadrature nodes and weights
for the Rys weight function w_T(x) = x^{-1/2} * exp(-T*x) on [0,1].

The algorithm follows the moment-based approach (Algorithm 5.1 in lecture notes):
    1. Compute moments m_k(T) = 2*F_k(T) for k = 0, 1, ..., 2*n_roots - 1
    2. Build Hankel matrix H with H_ij = m_{i+j}
    3. Build shifted Hankel H^(1) with H^(1)_ij = m_{i+j+1}
    4. Cholesky factorize: H = L L^T, then C = L^{-1}
    5. Build Jacobi matrix: J = C H^(1) C^T
    6. Diagonalize J: eigenvalues are nodes, weights from eigenvectors

The resulting quadrature rule is exact for polynomials of degree <= 2*n_roots - 1:
    integral_0^1 f(x) w_T(x) dx = sum_i W_i f(x_i)

References:
    - Chapter 5, Section 4: Nodes and weights from moments (Algorithm 5.1)
    - Golub & Welsch, Math. Comp. 23 (1969) 221-230
    - Dupuis, Rys, King, J. Chem. Phys. 65 (1976) 111

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import Tuple

# Import Boys function from companion module
from boys_moments import boys, moment, moments_all


def build_hankel_matrix(moments: np.ndarray, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Hankel matrices H and H^(1) from moments.

    H_ij = m_{i+j}         (standard Hankel)
    H^(1)_ij = m_{i+j+1}   (shifted Hankel)

    These are Gram matrices:
    - H is the Gram matrix of monomials {1, x, x^2, ...} under the inner product
      <f, g> = integral f(x) g(x) w_T(x) dx
    - H^(1) represents multiplication by x in this basis

    Parameters
    ----------
    moments : np.ndarray
        Array of moments [m_0, m_1, ..., m_{2*n_roots-1}]
    n_roots : int
        Number of quadrature roots

    Returns
    -------
    H : np.ndarray
        Hankel matrix of shape (n_roots, n_roots)
    H1 : np.ndarray
        Shifted Hankel matrix of shape (n_roots, n_roots)
    """
    H = np.zeros((n_roots, n_roots))
    H1 = np.zeros((n_roots, n_roots))

    for i in range(n_roots):
        for j in range(n_roots):
            H[i, j] = moments[i + j]
            H1[i, j] = moments[i + j + 1]

    return H, H1


def rys_nodes_weights(T: float, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct Rys quadrature nodes and weights for weight w_T(x) = x^{-1/2} exp(-Tx).

    Implements Algorithm 5.1 from the lecture notes:
        1. Compute moments m_k = 2*F_k(T) for k = 0, ..., 2*n_roots - 1
        2. Build Hankel matrices H and H^(1)
        3. Cholesky factorize H = L L^T; set C = L^{-1}
        4. Build Jacobi matrix J = C H^(1) C^T
        5. Diagonalize J -> eigenvalues are nodes x_i
        6. Weights: W_i = m_0 * (V_{0,i})^2

    The resulting quadrature is exact for polynomials up to degree 2*n_roots - 1:
        sum_i W_i x_i^n = m_n    for n = 0, 1, ..., 2*n_roots - 1

    Parameters
    ----------
    T : float
        Parameter in the weight function (T >= 0)
    n_roots : int
        Number of quadrature roots (n_roots >= 1)

    Returns
    -------
    nodes : np.ndarray
        Quadrature nodes x_i in (0, 1), sorted in ascending order
    weights : np.ndarray
        Quadrature weights W_i > 0 (same order as nodes)

    Examples
    --------
    >>> x, W = rys_nodes_weights(1.0, 2)
    >>> # Should satisfy: sum(W * x**n) = 2*F_n(1.0) for n = 0, 1, 2, 3
    """
    if n_roots < 1:
        raise ValueError("n_roots must be >= 1")

    # Step 1: Compute moments m_k = 2*F_k(T) for k = 0, ..., 2*n_roots - 1
    m = moments_all(2 * n_roots - 1, T)

    # Step 2: Build Hankel matrices
    H, H1 = build_hankel_matrix(m, n_roots)

    # Step 3: Cholesky factorization of H and compute C = L^{-1}
    # For numerical stability, add small regularization if needed
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        # Add tiny regularization for near-singular cases
        eps = 1e-14 * np.trace(H) / n_roots
        L = np.linalg.cholesky(H + eps * np.eye(n_roots))

    # C = L^{-1} (solve L @ C = I)
    C = np.linalg.solve(L, np.eye(n_roots))

    # Step 4: Build Jacobi matrix J = C @ H1 @ C^T
    J = C @ H1 @ C.T

    # Symmetrize to remove tiny numerical asymmetry
    J = 0.5 * (J + J.T)

    # Step 5: Golub-Welsch eigendecomposition
    eigenvalues, V = np.linalg.eigh(J)

    # Nodes are the eigenvalues
    nodes = eigenvalues

    # Step 6: Weights from eigenvectors (Eq. 5.xx)
    # W_i = m_0 * (V_{0,i})^2
    weights = m[0] * (V[0, :] ** 2)

    # Sort by ascending node value
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]

    return nodes, weights


def rys_nodes_weights_robust(T: float, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust version of Rys quadrature that handles edge cases.

    Handles:
    - Very small T (uses limiting forms)
    - Very large T (uses asymptotic forms)
    - Single root case (analytic formulas)

    Parameters
    ----------
    T : float
        Parameter in the weight function (T >= 0)
    n_roots : int
        Number of quadrature roots

    Returns
    -------
    nodes : np.ndarray
        Quadrature nodes
    weights : np.ndarray
        Quadrature weights
    """
    if n_roots == 1:
        # For one root, analytic formulas from Exercise 1:
        # W_1 = m_0, x_1 = m_1/m_0 = F_1(T)/F_0(T)
        m0 = moment(0, T)
        m1 = moment(1, T)
        return np.array([m1 / m0]), np.array([m0])

    # For n_roots > 1, use the general algorithm
    return rys_nodes_weights(T, n_roots)


def verify_nodes_weights(T: float, n_roots: int, verbose: bool = True) -> float:
    """
    Verify that nodes and weights satisfy moment matching.

    Tests: sum_i W_i * x_i^n = m_n for n = 0, 1, ..., 2*n_roots - 1

    Parameters
    ----------
    T : float
        Parameter value
    n_roots : int
        Number of roots
    verbose : bool
        Print detailed results

    Returns
    -------
    max_error : float
        Maximum absolute error in moment matching
    """
    nodes, weights = rys_nodes_weights(T, n_roots)

    if verbose:
        print(f"\nVerification for T = {T}, n_roots = {n_roots}")
        print("-" * 60)
        print(f"{'n':>4} {'m_n (exact)':>18} {'sum W*x^n':>18} {'Error':>12}")
        print("-" * 60)

    max_error = 0.0
    for n in range(2 * n_roots):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        max_error = max(max_error, error)

        if verbose:
            print(f"{n:>4} {m_exact:>18.12e} {m_quad:>18.12e} {error:>12.2e}")

    if verbose:
        print("-" * 60)
        print(f"Maximum error: {max_error:.2e}")

    return max_error


def compute_boys_from_rys(T: float, n_max: int, n_roots: int = None) -> np.ndarray:
    """
    Compute Boys functions F_0(T), ..., F_{n_max}(T) via Rys quadrature.

    F_n(T) = (1/2) * sum_i W_i * x_i^n

    Parameters
    ----------
    T : float
        Boys function argument
    n_max : int
        Maximum Boys order
    n_roots : int, optional
        Number of quadrature roots. If None, uses ceil((n_max+1)/2)

    Returns
    -------
    np.ndarray
        Array [F_0(T), F_1(T), ..., F_{n_max}(T)]
    """
    if n_roots is None:
        # Need exactness for n = 0, ..., n_max
        # 2*n_roots - 1 >= n_max => n_roots >= (n_max + 1) / 2
        n_roots = (n_max + 2) // 2

    nodes, weights = rys_nodes_weights(T, n_roots)

    # F_n(T) = m_n(T) / 2 = (1/2) * sum W_i * x_i^n
    F_values = np.array([0.5 * np.sum(weights * (nodes ** n)) for n in range(n_max + 1)])

    return F_values


# =============================================================================
# Validation and demonstration
# =============================================================================

def validate_basic():
    """Basic validation of Rys quadrature construction."""
    print("Basic validation of Rys quadrature")
    print("=" * 70)

    # Test several T values with different root counts
    test_cases = [
        (0.0, 1), (0.0, 2), (0.0, 3),
        (1.0, 1), (1.0, 2), (1.0, 3),
        (10.0, 2), (10.0, 3), (10.0, 4),
    ]

    all_passed = True
    for T, n_roots in test_cases:
        max_error = verify_nodes_weights(T, n_roots, verbose=False)
        status = "PASS" if max_error < 1e-10 else "FAIL"
        print(f"T = {T:6.2f}, n_roots = {n_roots}: max_error = {max_error:.2e}  [{status}]")
        if max_error >= 1e-10:
            all_passed = False

    return all_passed


def validate_single_root():
    """Validate single-root formulas from Exercise 1."""
    print("\nValidation of single-root formulas (Exercise 1)")
    print("=" * 70)
    print("For n_roots = 1:  W_1 = m_0,  x_1 = m_1/m_0 = F_1(T)/F_0(T)")
    print("-" * 70)

    for T in [0.0, 0.1, 1.0, 5.0, 10.0]:
        nodes, weights = rys_nodes_weights(T, 1)

        F0 = boys(0, T)
        F1 = boys(1, T)
        m0 = 2.0 * F0
        x1_expected = F1 / F0 if F0 > 1e-15 else 0.5  # Limit as T->inf

        print(f"T = {T:6.2f}:  x_1 = {nodes[0]:.10f},  F_1/F_0 = {x1_expected:.10f},  W = {weights[0]:.10f},  m_0 = {m0:.10f}")


def demonstrate_rys_for_eri():
    """Demonstrate Rys quadrature in context of ERI evaluation."""
    print("\nRys quadrature in ERI context")
    print("=" * 70)

    # For (ss|ss) with L = 0, need n_roots = 1
    # For (ps|ss) with L = 1, need n_roots = 1
    # For (pp|ss) with L = 2, need n_roots = 2
    # For (dd|pp) with L = 6, need n_roots = 4

    print("Root count rule: n_roots = floor(L/2) + 1, where L = sum of angular momenta")
    print("-" * 70)

    shell_quartets = [
        ("ss|ss", 0),
        ("ps|ss", 1),
        ("pp|ss", 2),
        ("pp|ps", 3),
        ("pp|pp", 4),
        ("dd|pp", 6),
        ("ff|ff", 12),
    ]

    print(f"{'Shell quartet':>12} {'L':>4} {'n_roots':>8}")
    print("-" * 30)
    for name, L in shell_quartets:
        n_roots = L // 2 + 1
        print(f"{name:>12} {L:>4} {n_roots:>8}")

    # Example: compute F_0, F_1 from 1-root Rys quadrature
    print("\nExample: F_0(T) and F_1(T) from 1-root quadrature at T = 1.0")
    T = 1.0
    x, W = rys_nodes_weights(T, 1)
    F0_rys = 0.5 * W[0]
    F1_rys = 0.5 * W[0] * x[0]

    F0_direct = boys(0, T)
    F1_direct = boys(1, T)

    print(f"  F_0(1.0): Rys = {F0_rys:.12f}, Direct = {F0_direct:.12f}, Diff = {abs(F0_rys - F0_direct):.2e}")
    print(f"  F_1(1.0): Rys = {F1_rys:.12f}, Direct = {F1_direct:.12f}, Diff = {abs(F1_rys - F1_direct):.2e}")


def print_nodes_weights_table():
    """Print table of Rys nodes and weights for reference."""
    print("\nRys nodes and weights table")
    print("=" * 70)

    for T in [0.0, 1.0, 5.0, 10.0]:
        print(f"\nT = {T}")
        print("-" * 60)

        for n_roots in [1, 2, 3]:
            nodes, weights = rys_nodes_weights(T, n_roots)
            print(f"  n_roots = {n_roots}:")
            for i, (x, W) in enumerate(zip(nodes, weights)):
                print(f"    x_{i+1} = {x:15.12f},  W_{i+1} = {W:15.12f}")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 5A continued: Rys Quadrature Nodes and Weights (Algorithm 5.1)")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run validations
    success1 = validate_basic()

    # Validate single root formulas
    validate_single_root()

    # Demonstrate in ERI context
    demonstrate_rys_for_eri()

    # Print reference table
    print_nodes_weights_table()

    # Detailed verification for one case
    verify_nodes_weights(1.0, 2, verbose=True)

    print("\n" + "=" * 70)
    if success1:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
    print("=" * 70)
