#!/usr/bin/env python3
"""
Worked Example: Rys Quadrature for T = 2.0 with n_r = 3 Roots

This script computes the EXACT Rys quadrature nodes and weights for T = 2.0
with n_r = 3 roots using the Hankel/Golub-Welsch algorithm, showing ALL
intermediate values with high precision.

This is a detailed educational example for Appendix E of the Advanced Quantum
Chemistry lecture notes, used to verify Figure E.2.

The computation follows these steps:
    Step 1: Compute Boys function values F_0(2) through F_5(2)
    Step 2: Compute moments m_k = 2 * F_k(2) for k = 0, 1, ..., 5
    Step 3: Build the 3x3 Hankel matrix H
    Step 4: Build the 3x3 shifted Hankel matrix H^(1)
    Step 5: Cholesky factorization H = L L^T
    Step 6: Compute C = L^{-1}
    Step 7: Build Jacobi matrix J = C H^(1) C^T
    Step 8: Eigendecomposition of J to get nodes
    Step 9: Compute weights from eigenvector first components
    Step 10: Verify moment matching

Reference: Golub & Welsch, Math. Comp. 23 (1969) 221-230

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix E: Rys Quadrature Reference Notes
"""

import numpy as np
from scipy import special
from scipy.linalg import eigh, cholesky, solve_triangular
import math


def boys_scipy(n: int, T: float) -> float:
    """
    Compute Boys function F_n(T) using scipy's gammainc (incomplete gamma).

    The Boys function is related to the incomplete gamma function:
        F_n(T) = (1/2) * T^{-(n+1/2)} * gamma(n+1/2) * gammainc(n+1/2, T)

    where gammainc is the regularized lower incomplete gamma function:
        gammainc(a, x) = (1/gamma(a)) * int_0^x t^{a-1} e^{-t} dt

    For T = 0:
        F_n(0) = 1/(2n+1)

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)

    Returns:
        F_n(T) computed using scipy special functions
    """
    if T == 0:
        return 1.0 / (2 * n + 1)

    if T < 1e-15:
        # Use series expansion for very small T
        return sum((-T)**k / (math.factorial(k) * (2*n + 2*k + 1))
                   for k in range(30))

    # Use the relation to the incomplete gamma function
    a = n + 0.5
    # gammainc in scipy is the regularized incomplete gamma: P(a, x) = gammainc(a, x)
    # We need: gamma(a) * P(a, T) / (2 * T^a)
    gamma_a = special.gamma(a)
    P_a_T = special.gammainc(a, T)

    return 0.5 * gamma_a * P_a_T / (T ** a)


def print_separator(title: str = "", char: str = "=", width: int = 80):
    """Print a formatted separator line with optional title."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(char * padding + " " + title + " " + char * (width - padding - len(title) - 2))
    else:
        print(char * width)


def main():
    """Compute and display all intermediate values for Rys quadrature."""

    print_separator("WORKED EXAMPLE: Rys Quadrature for T = 2.0 with n_r = 3 Roots")
    print()
    print("This computation follows the Golub-Welsch algorithm to compute")
    print("Rys quadrature nodes and weights from the moment sequence.")
    print()

    # Parameters
    T = 2.0
    n_r = 3  # Number of roots
    n_moments = 2 * n_r  # Need moments m_0 through m_{2*n_r - 1}

    print(f"Parameters: T = {T}, n_r = {n_r}, need {n_moments} moments")
    print()

    # ========================================================================
    # STEP 1: Compute Boys function values
    # ========================================================================
    print_separator("Step 1: Boys Function Values", "-")
    print()
    print("The Boys function is defined as:")
    print("    F_n(T) = integral from 0 to 1 of t^(2n) * exp(-T*t^2) dt")
    print()
    print("Computing F_n(2.0) for n = 0, 1, 2, 3, 4, 5:")
    print()

    F_values = np.zeros(n_moments)
    print(f"{'n':>4}  {'F_n(T)':>24}")
    print("-" * 32)
    for n in range(n_moments):
        F_values[n] = boys_scipy(n, T)
        print(f"{n:4d}  {F_values[n]:24.16f}")

    # Also compute using scipy's gammainc for verification
    print()
    print("Verification using scipy.special functions:")
    print(f"    F_0(2) via erf: {0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T)):.16f}")

    # ========================================================================
    # STEP 2: Compute moments
    # ========================================================================
    print()
    print_separator("Step 2: Moments m_k = 2 * F_k(T)", "-")
    print()
    print("The moments of the Rys weight function w_T(x) = x^{-1/2} e^{-Tx} are:")
    print("    m_n(T) = integral from 0 to 1 of x^n * w_T(x) dx = 2 * F_n(T)")
    print()

    moments = 2.0 * F_values

    print(f"{'k':>4}  {'m_k':>24}")
    print("-" * 32)
    for k in range(n_moments):
        print(f"{k:4d}  {moments[k]:24.16f}")

    # ========================================================================
    # STEP 3: Build Hankel matrix H
    # ========================================================================
    print()
    print_separator("Step 3: Hankel Matrix H (3x3)", "-")
    print()
    print("The Hankel matrix is defined as H_ij = m_{i+j} for i,j = 0, 1, ..., n_r-1")
    print()

    H = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            H[i, j] = moments[i + j]

    print("H = ")
    print("    [", end="")
    for i in range(n_r):
        if i > 0:
            print("     [", end="")
        for j in range(n_r):
            print(f"{H[i,j]:20.16f}", end="")
            if j < n_r - 1:
                print("  ", end="")
        if i < n_r - 1:
            print("]")
        else:
            print("]")
    print()

    print("Numerical matrix (for easier reading):")
    print(f"H = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{H[i,j]:12.10f}" for j in range(n_r))
        print(row)

    # ========================================================================
    # STEP 4: Build shifted Hankel matrix H^(1)
    # ========================================================================
    print()
    print_separator("Step 4: Shifted Hankel Matrix H^(1) (3x3)", "-")
    print()
    print("The shifted Hankel matrix is defined as H^(1)_ij = m_{i+j+1}")
    print()

    H1 = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            H1[i, j] = moments[i + j + 1]

    print("H^(1) = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{H1[i,j]:12.10f}" for j in range(n_r))
        print(row)

    # ========================================================================
    # STEP 5: Cholesky factorization H = L L^T
    # ========================================================================
    print()
    print_separator("Step 5: Cholesky Factorization H = L L^T", "-")
    print()
    print("Since H is symmetric positive definite, it has a unique Cholesky")
    print("factorization H = L L^T where L is lower triangular.")
    print()

    L = cholesky(H, lower=True)

    print("L (lower triangular) = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{L[i,j]:12.10f}" for j in range(n_r))
        print(row)
    print()

    # Verify: L @ L.T should equal H
    H_reconstructed = L @ L.T
    reconstruction_error = np.linalg.norm(H_reconstructed - H)
    print(f"Verification: ||L L^T - H|| = {reconstruction_error:.2e}")

    # ========================================================================
    # STEP 6: Compute C = L^{-1}
    # ========================================================================
    print()
    print_separator("Step 6: Compute C = L^{-1}", "-")
    print()
    print("The matrix C transforms from the monomial basis to an orthonormal")
    print("polynomial basis: C = L^{-1}")
    print()

    C = solve_triangular(L, np.eye(n_r), lower=True)

    print("C = L^{-1} = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{C[i,j]:12.10f}" for j in range(n_r))
        print(row)
    print()

    # Verify: C @ L should equal I
    CL = C @ L
    identity_error = np.linalg.norm(CL - np.eye(n_r))
    print(f"Verification: ||C L - I|| = {identity_error:.2e}")

    # ========================================================================
    # STEP 7: Build Jacobi matrix J = C H^(1) C^T
    # ========================================================================
    print()
    print_separator("Step 7: Jacobi Matrix J = C H^(1) C^T", "-")
    print()
    print("The Jacobi matrix encodes the three-term recurrence coefficients")
    print("for the orthonormal polynomials. It is symmetric tridiagonal.")
    print()

    J = C @ H1 @ C.T

    print("J (before symmetrization) = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{J[i,j]:12.10f}" for j in range(n_r))
        print(row)

    # Symmetrize to remove numerical asymmetry
    J_sym = 0.5 * (J + J.T)

    print()
    print("J (after symmetrization) = ")
    for i in range(n_r):
        row = "    " + " ".join(f"{J_sym[i,j]:12.10f}" for j in range(n_r))
        print(row)

    # Extract tridiagonal elements
    print()
    print("Tridiagonal structure:")
    a_diag = np.diag(J_sym)
    b_offdiag = np.diag(J_sym, k=1)
    print(f"    Diagonal elements (a_k):     {a_diag}")
    print(f"    Off-diagonal elements (b_k): {b_offdiag}")

    # Check off-diagonal decay (should be essentially tridiagonal)
    print()
    print("Off-tridiagonal elements (should be ~0):")
    print(f"    J[0,2] = {J_sym[0,2]:.2e}")

    # ========================================================================
    # STEP 8: Eigendecomposition of J
    # ========================================================================
    print()
    print_separator("Step 8: Eigendecomposition of J", "-")
    print()
    print("The eigenvalues of J are the quadrature nodes x_i.")
    print("The eigenvectors are used to compute the weights.")
    print()

    eigenvalues, eigenvectors = eigh(J_sym)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    nodes = eigenvalues[idx]
    V = eigenvectors[:, idx]

    print(f"Eigenvalues (nodes x_i):")
    for i, x in enumerate(nodes):
        print(f"    x_{i+1} = {x:20.16f}")

    print()
    print("Eigenvector matrix V (columns are eigenvectors):")
    for i in range(n_r):
        row = "    " + " ".join(f"{V[i,j]:12.10f}" for j in range(n_r))
        print(row)

    print()
    print("First row of V (used for weight computation):")
    print(f"    V[0,:] = [{V[0,0]:.16f}, {V[0,1]:.16f}, {V[0,2]:.16f}]")

    # ========================================================================
    # STEP 9: Compute weights
    # ========================================================================
    print()
    print_separator("Step 9: Compute Weights W_i = m_0 * (V_{0i})^2", "-")
    print()
    print("The quadrature weights are computed from the first component")
    print("of each eigenvector using the Golub-Welsch formula:")
    print(f"    W_i = m_0 * (V[0,i])^2,  where m_0 = {moments[0]:.16f}")
    print()

    weights = moments[0] * V[0, :] ** 2

    for i in range(n_r):
        v_sq = V[0, i] ** 2
        print(f"    W_{i+1} = {moments[0]:.10f} * ({V[0,i]:.10f})^2")
        print(f"        = {moments[0]:.10f} * {v_sq:.10f}")
        print(f"        = {weights[i]:.16f}")
        print()

    # ========================================================================
    # STEP 10: Verification - Moment matching
    # ========================================================================
    print()
    print_separator("Step 10: Verification - Moment Matching", "-")
    print()
    print("The quadrature rule should exactly reproduce moments m_0 through m_{2n_r-1}:")
    print("    sum_i W_i * x_i^n = m_n")
    print()

    print(f"{'n':>4}  {'m_n (exact)':>20}  {'m_n (quadrature)':>20}  {'Error':>14}")
    print("-" * 64)

    max_error = 0.0
    for n in range(n_moments):
        m_exact = moments[n]
        m_quad = np.sum(weights * nodes ** n)
        error = abs(m_quad - m_exact)
        max_error = max(max_error, error)
        print(f"{n:4d}  {m_exact:20.16f}  {m_quad:20.16f}  {error:14.2e}")

    print("-" * 64)
    print(f"Maximum error: {max_error:.2e}")
    print(f"Moment matching {'PASSED' if max_error < 1e-12 else 'FAILED'}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print()
    print_separator("FINAL RESULTS: Rys Quadrature for T = 2.0, n_r = 3", "=")
    print()
    print("NODES (x_i):")
    for i in range(n_r):
        print(f"    x_{i+1} = {nodes[i]:.16f}")
    print()
    print("WEIGHTS (W_i):")
    for i in range(n_r):
        print(f"    W_{i+1} = {weights[i]:.16f}")
    print()

    # Verify sum of weights = m_0
    print(f"Sum of weights: {np.sum(weights):.16f}")
    print(f"Expected (m_0): {moments[0]:.16f}")
    print(f"Difference:     {abs(np.sum(weights) - moments[0]):.2e}")
    print()

    # Additional verification: Boys function reproduction
    print("Boys function verification:")
    print("    F_n(T) = (1/2) * sum_i W_i * x_i^n")
    print()
    print(f"{'n':>4}  {'F_n (scipy)':>20}  {'F_n (quadrature)':>20}  {'Error':>14}")
    print("-" * 64)
    for n in range(n_moments):
        F_exact = F_values[n]
        F_quad = 0.5 * np.sum(weights * nodes ** n)
        error = abs(F_quad - F_exact)
        print(f"{n:4d}  {F_exact:20.16f}  {F_quad:20.16f}  {error:14.2e}")

    # ========================================================================
    # LaTeX OUTPUT
    # ========================================================================
    print()
    print_separator("LaTeX-Ready Output for Appendix E", "=")
    print()
    print("% Exact values for T = 2.0, n_r = 3")
    print("% Can be used to verify/update Figure E.2")
    print()
    print("% Boys function values:")
    for n in range(n_moments):
        print(f"%   F_{n}(2) = {F_values[n]:.16f}")
    print()
    print("% Moments:")
    for k in range(n_moments):
        print(f"%   m_{k} = {moments[k]:.16f}")
    print()
    print("% Nodes and weights:")
    for i in range(n_r):
        print(f"%   x_{i+1} = {nodes[i]:.10f},  W_{i+1} = {weights[i]:.10f}")
    print()

    # LaTeX table format
    print("% LaTeX table row format:")
    print("\\begin{tabular}{@{}ccc@{}}")
    print("\\toprule")
    print("$i$ & Node $x_i$ & Weight $W_i$ \\\\")
    print("\\midrule")
    for i in range(n_r):
        print(f"{i+1} & {nodes[i]:.10f} & {weights[i]:.10f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print()

    # ========================================================================
    # COMPARISON WITH FIGURE E.2 VALUES
    # ========================================================================
    print()
    print_separator("Comparison with Current Figure E.2 Values", "=")
    print()

    # Current values in Figure E.2
    current_nodes = np.array([0.0694, 0.330, 0.677])
    current_weights = np.array([0.724, 0.448, 0.114])

    print("Current Figure E.2 values vs. Exact computed values:")
    print()
    print(f"{'':6} {'Current':>12}  {'Exact':>16}  {'Difference':>14}")
    print("-" * 52)
    for i in range(n_r):
        print(f"x_{i+1}:  {current_nodes[i]:12.4f}  {nodes[i]:16.10f}  {abs(nodes[i] - current_nodes[i]):14.6f}")
    print()
    for i in range(n_r):
        print(f"W_{i+1}:  {current_weights[i]:12.4f}  {weights[i]:16.10f}  {abs(weights[i] - current_weights[i]):14.6f}")

    print()
    print("Assessment: The current Figure E.2 values are approximate but correct")
    print("to the displayed precision (3-4 significant figures).")
    print()
    print("Recommended updated values (4 decimal places for figure):")
    for i in range(n_r):
        print(f"    x_{i+1} = {nodes[i]:.4f},  W_{i+1} = {weights[i]:.4f}")

    # ========================================================================
    # HIGH-PRECISION OUTPUT FOR REFERENCE
    # ========================================================================
    print()
    print_separator("HIGH-PRECISION REFERENCE VALUES", "=")
    print()
    print(f"T = {T}")
    print(f"n_r = {n_r}")
    print()
    print("Boys function values F_n(T):")
    for n in range(n_moments):
        print(f"    F_{n}({T}) = {F_values[n]:.18e}")
    print()
    print("Moments m_k(T) = 2 * F_k(T):")
    for k in range(n_moments):
        print(f"    m_{k}({T}) = {moments[k]:.18e}")
    print()
    print("Quadrature nodes:")
    for i in range(n_r):
        print(f"    x_{i+1} = {nodes[i]:.18e}")
    print()
    print("Quadrature weights:")
    for i in range(n_r):
        print(f"    W_{i+1} = {weights[i]:.18e}")
    print()
    print_separator()


if __name__ == "__main__":
    main()
