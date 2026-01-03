#!/usr/bin/env python3
"""
Rys Quadrature: Core Implementation

Rys quadrature is a specialized Gaussian quadrature designed for evaluating
molecular integrals. It uses the weight function:

    w_T(x) = x^(-1/2) * exp(-T*x)   on [0, 1]

whose moments are related to the Boys function:

    m_n(T) = integral_0^1 x^n * w_T(x) dx = 2 * F_n(T)

The quadrature rule with n_r nodes exactly reproduces moments 0 through 2*n_r - 1:

    m_n(T) = sum_{i=1}^{n_r} W_i * x_i^n   for n = 0, 1, ..., 2*n_r - 1

This module implements the moment-based Golub-Welsch algorithm:
    1. Compute moments m_k = 2*F_k(T) for k = 0, ..., 2*n_r - 1
    2. Form Hankel matrices: H_ij = m_{i+j}, H^(1)_ij = m_{i+j+1}
    3. Cholesky factorize: H = L * L^T, define C = L^(-1)
    4. Form Jacobi matrix: J = C * H^(1) * C^T
    5. Diagonalize J: eigenvalues are nodes x_i
    6. Weights: W_i = m_0 * (V_{0i})^2 where V is the eigenvector matrix

Reference: Golub & Welsch, Math. Comp. 23 (1969) 221-230

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix E: Rys Quadrature Reference Notes
"""

import numpy as np
from numpy.linalg import cholesky, eigh, solve
from typing import Tuple
import sys
import os

# Import Boys function from appendix_d
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'appendix_d'))
from boys_function import boys, boys_array


def compute_rys_moments(T: float, n_moments: int) -> np.ndarray:
    """
    Compute Rys quadrature moments m_k(T) = 2 * F_k(T).

    The moments of the weight function w_T(x) = x^(-1/2) * exp(-T*x) on [0,1]
    are directly related to Boys functions:

        m_n(T) = integral_0^1 x^n * x^(-1/2) * exp(-T*x) dx
               = integral_0^1 x^(n-1/2) * exp(-T*x) dx

    Via substitution x = t^2:

        m_n(T) = 2 * integral_0^1 t^(2n) * exp(-T*t^2) dt = 2 * F_n(T)

    Args:
        T: Rys argument (T >= 0)
        n_moments: Number of moments needed (k = 0, 1, ..., n_moments-1)

    Returns:
        Array of moments [m_0, m_1, ..., m_{n_moments-1}]
    """
    # Use Boys function array for efficiency
    F_vals = boys_array(n_moments - 1, T)
    return 2.0 * F_vals


def build_hankel_matrices(moments: np.ndarray, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Hankel and shifted Hankel matrices from moments.

    The Hankel matrix H and shifted Hankel matrix H^(1) are defined as:

        H_ij     = m_{i+j}       for i,j = 0, ..., n_roots-1
        H^(1)_ij = m_{i+j+1}     for i,j = 0, ..., n_roots-1

    These matrices encode the moment information needed to construct
    orthogonal polynomials for the weight function.

    Args:
        moments: Array of moments [m_0, m_1, ..., m_{2*n_roots-1}]
        n_roots: Number of quadrature nodes

    Returns:
        Tuple (H, H1) where:
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


def golub_welsch(H: np.ndarray, H1: np.ndarray, m0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quadrature nodes and weights via Golub-Welsch algorithm.

    Given the Hankel matrix H and shifted Hankel matrix H^(1), this algorithm
    computes the nodes and weights of the Gaussian quadrature rule.

    Algorithm steps:
        1. Cholesky factorize H = L * L^T (H must be positive definite)
        2. Compute C = L^(-1)
        3. Form Jacobi matrix J = C * H^(1) * C^T
        4. Diagonalize J: J = V * Lambda * V^T
        5. Nodes are eigenvalues: x_i = Lambda_ii
        6. Weights from eigenvectors: W_i = m_0 * (V_{0i})^2

    The Jacobi matrix J is symmetric tridiagonal and its eigenvalues are
    the nodes of the quadrature rule.

    Args:
        H: Hankel matrix (n_roots x n_roots)
        H1: Shifted Hankel matrix (n_roots x n_roots)
        m0: Zeroth moment (= m_0 = 2*F_0(T))

    Returns:
        Tuple (nodes, weights) where:
            nodes: Array of quadrature nodes x_i in (0, 1)
            weights: Array of quadrature weights W_i > 0
    """
    n_roots = H.shape[0]

    # Step 1: Cholesky factorization H = L * L^T
    # L is lower triangular
    L = cholesky(H)

    # Step 2: Compute C = L^(-1)
    # We solve L * C = I for C
    C = solve(L, np.eye(n_roots))

    # Step 3: Form Jacobi matrix J = C * H^(1) * C^T
    # This is symmetric and tridiagonal (in exact arithmetic)
    J = C @ H1 @ C.T

    # Ensure symmetry (numerical clean-up)
    J = 0.5 * (J + J.T)

    # Step 4: Diagonalize J
    # eigenvalues = nodes, eigenvectors used for weights
    eigenvalues, eigenvectors = eigh(J)

    # Sort by eigenvalue (should already be sorted, but ensure it)
    idx = np.argsort(eigenvalues)
    nodes = eigenvalues[idx]
    V = eigenvectors[:, idx]

    # Step 5: Compute weights
    # W_i = m_0 * (V_{0,i})^2
    # where V_{0,i} is the first component of the i-th eigenvector
    weights = m0 * V[0, :] ** 2

    return nodes, weights


def rys_roots_weights(T: float, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Rys quadrature nodes and weights for argument T.

    This is the main interface function for Rys quadrature. Given the argument T
    and the desired number of quadrature points n_roots, it returns nodes x_i
    in (0,1) and weights W_i > 0 such that:

        F_n(T) = (1/2) * sum_{i=1}^{n_roots} W_i * x_i^n

    is exact for n = 0, 1, ..., 2*n_roots - 1.

    Special cases:
        - T = 0: Nodes and weights are analytic
        - n_roots = 1: Closed-form solution available

    Args:
        T: Rys argument (T >= 0)
        n_roots: Number of quadrature points (n_roots >= 1)

    Returns:
        Tuple (nodes, weights) where:
            nodes: Array of quadrature nodes x_i in (0, 1)
            weights: Array of quadrature weights W_i > 0

    Raises:
        ValueError: If T < 0 or n_roots < 1
    """
    if T < 0:
        raise ValueError(f"Argument T must be non-negative, got {T}")
    if n_roots < 1:
        raise ValueError(f"Number of roots must be >= 1, got {n_roots}")

    # Special case: n_roots = 1 has closed-form solution
    if n_roots == 1:
        return _rys_one_root(T)

    # Special case: T = 0
    if T == 0.0:
        return _rys_T_zero(n_roots)

    # Special case: very small T (avoid numerical issues with Hankel matrix)
    if T < 1e-15:
        return _rys_T_zero(n_roots)

    # General case: moment-based Golub-Welsch
    # Need 2*n_roots moments: m_0, m_1, ..., m_{2*n_roots-1}
    n_moments = 2 * n_roots
    moments = compute_rys_moments(T, n_moments)

    # Build Hankel matrices
    H, H1 = build_hankel_matrices(moments, n_roots)

    # Check conditioning - Hankel matrices can be ill-conditioned
    # for large n_roots or extreme T values
    try:
        nodes, weights = golub_welsch(H, H1, moments[0])
    except np.linalg.LinAlgError:
        # Fallback: try with regularization or report failure
        raise RuntimeError(
            f"Cholesky factorization failed for T={T}, n_roots={n_roots}. "
            "Hankel matrix may be ill-conditioned."
        )

    # Sanity checks (nodes should be in (0,1), weights should be positive)
    if not np.all((nodes > -1e-10) & (nodes < 1 + 1e-10)):
        # Nodes slightly outside [0,1] can happen due to numerics
        # Clip to valid range
        nodes = np.clip(nodes, 0.0, 1.0)

    if not np.all(weights > -1e-10):
        # Negative weights indicate numerical issues
        weights = np.maximum(weights, 0.0)

    return nodes, weights


def _rys_one_root(T: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form solution for one-root Rys quadrature.

    For n_roots = 1, moment matching for n = 0, 1 gives:

        W_1 = m_0 = 2*F_0(T)
        x_1 = m_1 / m_0 = F_1(T) / F_0(T)

    This satisfies:
        - m_0 = W_1 * x_1^0 = W_1
        - m_1 = W_1 * x_1^1 = m_0 * (m_1/m_0) = m_1

    Args:
        T: Rys argument (T >= 0)

    Returns:
        Tuple (nodes, weights) with single node and weight
    """
    F0 = boys(0, T)
    F1 = boys(1, T)

    m0 = 2.0 * F0
    m1 = 2.0 * F1

    W1 = m0
    x1 = m1 / m0 if m0 > 1e-100 else 1.0 / 3.0  # F_1(0)/F_0(0) = (1/3)/(1) = 1/3

    return np.array([x1]), np.array([W1])


def _rys_T_zero(n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rys quadrature nodes and weights for T = 0.

    At T = 0, the weight function becomes:
        w_0(x) = x^(-1/2)   on [0, 1]

    The moments are:
        m_n(0) = integral_0^1 x^(n-1/2) dx = 1 / (n + 1/2) = 2 / (2n + 1)

    Note: m_n(0) = 2*F_n(0) = 2/(2n+1), which checks out.

    This is Gauss-Jacobi quadrature with weight x^(-1/2).

    Args:
        n_roots: Number of quadrature points

    Returns:
        Tuple (nodes, weights) for T = 0 case
    """
    # Build moments at T = 0: m_n = 2/(2n+1)
    n_moments = 2 * n_roots
    moments = np.array([2.0 / (2 * k + 1) for k in range(n_moments)])

    # Use Golub-Welsch with these moments
    H, H1 = build_hankel_matrices(moments, n_roots)
    nodes, weights = golub_welsch(H, H1, moments[0])

    return nodes, weights


def verify_moment_matching(T: float, n_roots: int, nodes: np.ndarray, weights: np.ndarray,
                           verbose: bool = False) -> bool:
    """
    Verify that computed nodes/weights reproduce moments correctly.

    The quadrature rule should satisfy:
        sum_{i=1}^{n_roots} W_i * x_i^n = m_n(T) = 2*F_n(T)

    for n = 0, 1, ..., 2*n_roots - 1.

    Args:
        T: Rys argument
        n_roots: Number of quadrature points
        nodes: Quadrature nodes
        weights: Quadrature weights
        verbose: If True, print comparison table

    Returns:
        True if all moments match within tolerance
    """
    n_test = 2 * n_roots
    moments_exact = compute_rys_moments(T, n_test)
    max_error = 0.0

    if verbose:
        print(f"\nMoment matching verification for T = {T:.6f}, n_roots = {n_roots}")
        print("-" * 60)
        print(f"{'n':>4}  {'m_n (exact)':>16}  {'m_n (quad)':>16}  {'Error':>12}")
        print("-" * 60)

    for n in range(n_test):
        m_exact = moments_exact[n]
        m_quad = np.sum(weights * nodes ** n)
        error = abs(m_quad - m_exact)
        max_error = max(max_error, error)

        if verbose:
            print(f"{n:4d}  {m_exact:16.12f}  {m_quad:16.12f}  {error:12.2e}")

    if verbose:
        print("-" * 60)
        print(f"Maximum error: {max_error:.2e}")
        print(f"Passed: {max_error < 1e-10}")

    return max_error < 1e-10


def root_count_for_angular_momentum(L_total: int) -> int:
    """
    Determine number of Rys roots needed for given total angular momentum.

    For a shell quartet with total angular momentum L = l_A + l_B + l_C + l_D,
    the minimum number of Rys quadrature points is:

        n_roots = floor(L/2) + 1

    This ensures exact integration of the polynomial factors that arise
    in molecular integrals.

    Args:
        L_total: Total angular momentum (sum of four shell angular momenta)

    Returns:
        Number of Rys quadrature points needed
    """
    return L_total // 2 + 1


def main():
    """Demonstrate Rys quadrature implementation."""
    print("=" * 70)
    print("Rys Quadrature Implementation - Core Module")
    print("=" * 70)

    # Test 1: One-root case (closed form)
    print("\n[1] One-root quadrature (closed form):")
    print("-" * 50)
    for T in [0.0, 0.5, 1.0, 5.0, 10.0]:
        nodes, weights = rys_roots_weights(T, 1)
        F0_exact = boys(0, T)
        F0_quad = 0.5 * weights[0]  # F_n = (1/2) sum W_i x_i^n
        print(f"T = {T:5.1f}: x = {nodes[0]:.10f}, W = {weights[0]:.10f}, "
              f"F_0 error = {abs(F0_quad - F0_exact):.2e}")

    # Test 2: Two-root case
    print("\n[2] Two-root quadrature:")
    print("-" * 50)
    for T in [0.5, 1.0, 5.0, 10.0]:
        nodes, weights = rys_roots_weights(T, 2)
        print(f"T = {T:5.1f}:")
        print(f"  Nodes:   {nodes[0]:.10f}, {nodes[1]:.10f}")
        print(f"  Weights: {weights[0]:.10f}, {weights[1]:.10f}")
        verify_moment_matching(T, 2, nodes, weights, verbose=False)

    # Test 3: Moment matching verification
    print("\n[3] Moment matching verification (T = 2.5, n_roots = 3):")
    T = 2.5
    n_roots = 3
    nodes, weights = rys_roots_weights(T, n_roots)
    verify_moment_matching(T, n_roots, nodes, weights, verbose=True)

    # Test 4: Root count rule
    print("\n[4] Root count for angular momentum:")
    print("-" * 50)
    print(f"{'L_total':>8}  {'n_roots':>10}")
    print("-" * 50)
    for L in range(9):
        n_roots = root_count_for_angular_momentum(L)
        print(f"{L:8d}  {n_roots:10d}   (e.g., {'ssss' if L==0 else 'psss' if L==1 else 'ppss' if L==2 else '...'})")

    # Test 5: Stability across T range
    print("\n[5] Stability across T range (n_roots = 3):")
    print("-" * 50)
    T_values = [1e-10, 1e-5, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
    n_roots = 3
    print(f"{'T':>12}  {'x_1':>12}  {'x_2':>12}  {'x_3':>12}  {'Moment OK':>10}")
    print("-" * 70)
    for T in T_values:
        nodes, weights = rys_roots_weights(T, n_roots)
        passed = verify_moment_matching(T, n_roots, nodes, weights, verbose=False)
        print(f"{T:12.2e}  {nodes[0]:12.8f}  {nodes[1]:12.8f}  {nodes[2]:12.8f}  "
              f"{'Yes' if passed else 'No':>10}")

    print("\n" + "=" * 70)
    print("Rys quadrature demonstration completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
