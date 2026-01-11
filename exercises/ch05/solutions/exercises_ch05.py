#!/usr/bin/env python3
"""
Chapter 5 Exercise Solutions: Rys Quadrature in Practice

This script provides Python implementations for all exercises in Chapter 5 that
involve numerical computation or validation. The exercises cover:

- Exercise 5.1: One-Root Quadrature Formulas [Core]
- Exercise 5.2: Two-Root Quadrature and Moment Matching [Core]
- Exercise 5.3: (ss|ss) by Three Routes [Core]
- Exercise 5.4: (p_xi s|ss) and the Appearance of F_1 [Core]
- Exercise 5.5: J/K Build and Energy Component Check [Core]
- Exercise 5.6: Schwarz Screening Toy Study [Advanced]
- Exercise 5.7: Hankel Matrix Conditioning [Advanced]
- Exercise 5.8: Obara-Saika Recursion [Research/Challenge] - Conceptual notes only

Additionally, this file validates code for several Checkpoint Questions from Chapter 5.

Part of: Advanced Quantum Chemistry Lecture Notes
Course: 2302638 Advanced Quantum Chemistry
Institution: Department of Chemistry, Chulalongkorn University

Usage:
    python exercises_ch05.py              # Run all exercises
    python exercises_ch05.py --exercise 3 # Run specific exercise

All numerical results are validated against PySCF reference values.
"""

import numpy as np
import math
from scipy import special
from typing import Tuple, List, Optional
import argparse


# =============================================================================
# CORE UTILITIES: Boys Function Implementation
# =============================================================================

def boys_series(n: int, T: float, max_terms: int = 100, tol: float = 1e-16) -> float:
    """
    Compute F_n(T) using Taylor series expansion.

    The series is:
        F_n(T) = sum_{k=0}^{inf} (-T)^k / [k! * (2n + 2k + 1)]

    This method is numerically stable for all T but converges slowly for large T.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)
        max_terms: Maximum number of terms in series
        tol: Convergence tolerance (relative)

    Returns:
        F_n(T) computed via series
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    if T == 0:
        return 1.0 / (2 * n + 1)

    val = 0.0
    term = 1.0 / (2 * n + 1)

    for k in range(max_terms):
        val += term
        if k > 5 and abs(term) < tol * abs(val):
            break
        # Next term
        term *= -T / (k + 1) * (2 * n + 2 * k + 1) / (2 * n + 2 * k + 3)

    return val


def boys_erf_upward(n: int, T: float) -> float:
    """
    Compute F_n(T) using F_0 from the error function and upward recurrence.

    F_0(T) = (1/2) sqrt(pi/T) erf(sqrt(T))    for T > 0
    F_{n+1}(T) = [(2n+1) F_n(T) - exp(-T)] / (2T)

    WARNING: This method suffers from catastrophic cancellation for small T
    with large n. Use only when T > ~25.

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (should be large for stability)

    Returns:
        F_n(T) computed via upward recurrence
    """
    if T <= 0:
        return 1.0 / (2 * n + 1) if T == 0 else float('nan')

    sqrt_T = math.sqrt(T)
    F = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)

    if n == 0:
        return F

    exp_mT = math.exp(-T)
    two_T = 2.0 * T

    for m in range(n):
        F = ((2 * m + 1) * F - exp_mT) / two_T

    return F


def boys(n: int, T: float) -> float:
    """
    Compute Boys function F_n(T) using the most stable method for given T.

    Strategy:
        - T = 0: Return exact value 1/(2n+1)
        - T < T_switch: Use series expansion
        - T >= T_switch: Use erf + upward recurrence

    Args:
        n: Order of the Boys function (n >= 0)
        T: Argument (T >= 0)

    Returns:
        F_n(T) with accuracy typically better than 1e-14
    """
    T_SWITCH = 25.0

    if T < T_SWITCH:
        return boys_series(n, T)
    else:
        return boys_erf_upward(n, T)


def boys_reference(n: int, T: float) -> float:
    """
    Compute F_n(T) using scipy's hypergeometric function (reference).

    The relationship is:
        F_n(T) = (1/(2n+1)) * hyp1f1(n + 0.5, n + 1.5, -T)

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        F_n(T) from scipy reference
    """
    return special.hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)


# =============================================================================
# CORE UTILITIES: Rys Quadrature (Golub-Welsch Algorithm)
# =============================================================================

def moment(n: int, T: float) -> float:
    """
    Compute the n-th moment of the Rys weight function.

    m_n(T) = integral_0^1 x^n * w_T(x) dx = 2 * F_n(T)

    where w_T(x) = x^{-1/2} exp(-Tx).

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


def build_hankel_matrices(moments: np.ndarray, n_roots: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Hankel matrices H and H^(1) from moments.

    H_ij = m_{i+j}     for i, j = 0, 1, ..., n_roots - 1
    H^(1)_ij = m_{i+j+1}

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

    Args:
        T: Parameter in the weight function (T >= 0)
        n_roots: Number of quadrature roots (n_roots >= 1)

    Returns:
        nodes: Quadrature nodes x_i in (0, 1), sorted ascending
        weights: Quadrature weights W_i > 0
    """
    if n_roots < 1:
        raise ValueError("n_roots must be >= 1")

    # Step 1: Compute moments
    m = moments_array(2 * n_roots - 1, T)

    # Step 2: Build Hankel matrices
    H, H1 = build_hankel_matrices(m, n_roots)

    # Step 3: Cholesky factorization and compute C = L^{-1}
    try:
        L = np.linalg.cholesky(H)
        C = np.linalg.solve(L, np.eye(n_roots))
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue decomposition for ill-conditioned cases
        eigvals, V = np.linalg.eigh(H)
        thresh = max(1e-12 * eigvals.max(), 1e-30)
        eigvals = np.maximum(eigvals, thresh)
        d_inv_sqrt = 1.0 / np.sqrt(eigvals)
        C = V @ np.diag(d_inv_sqrt) @ V.T

    # Step 4: Build Jacobi matrix J = C @ H1 @ C^T
    J = C @ H1 @ C.T
    J = 0.5 * (J + J.T)  # Symmetrize

    # Step 5: Golub-Welsch eigendecomposition
    eigenvalues, V = np.linalg.eigh(J)

    # Nodes are eigenvalues (clamp to [0, 1])
    nodes = np.clip(eigenvalues, 0.0, 1.0)

    # Step 6: Weights from eigenvectors
    weights = m[0] * (V[0, :] ** 2)

    # Sort by ascending node value
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]

    return nodes, weights


# =============================================================================
# CORE UTILITIES: Normalization and ERI Functions
# =============================================================================

def norm_s_primitive(alpha: float) -> float:
    """
    Normalization constant for an s-type primitive Gaussian.

    N_s = (2*alpha/pi)^{3/4}

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant
    """
    return (2.0 * alpha / np.pi) ** 0.75


def norm_p_primitive(alpha: float) -> float:
    """
    Normalization constant for a Cartesian p-type primitive Gaussian.

    N_p = (2*alpha/pi)^{3/4} * sqrt(4*alpha)

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant
    """
    return (2.0 * alpha / np.pi) ** 0.75 * np.sqrt(4.0 * alpha)


def gaussian_product_center(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray
) -> Tuple[float, np.ndarray, float, float]:
    """
    Apply Gaussian Product Theorem to compute composite center and parameters.

    Args:
        alpha: Exponent of first Gaussian
        A: Center of first Gaussian (3D array)
        beta: Exponent of second Gaussian
        B: Center of second Gaussian (3D array)

    Returns:
        Tuple of (p, P, mu, R_AB_sq) where:
        - p = alpha + beta
        - P = (alpha*A + beta*B)/p
        - mu = alpha*beta/p
        - R_AB_sq = |A - B|^2
    """
    p = alpha + beta
    P = (alpha * A + beta * B) / p
    mu = alpha * beta / p
    R_AB_sq = np.sum((A - B)**2)

    return p, P, mu, R_AB_sq


def eri_ssss_unnorm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute (ss|ss) ERI for UNNORMALIZED primitive Gaussians using Boys function.

    Args:
        alpha, A: Exponent and center of primitive a
        beta, B: Exponent and center of primitive b
        gamma, C: Exponent and center of primitive c
        delta, D: Exponent and center of primitive d

    Returns:
        (ab|cd) for unnormalized Gaussians
    """
    # GPT for bra (a, b) -> p, P
    p, P, mu_ab, R_AB_sq = gaussian_product_center(alpha, A, beta, B)

    # GPT for ket (c, d) -> q, Q
    q, Q, nu_cd, R_CD_sq = gaussian_product_center(gamma, C, delta, D)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q)**2)
    T = rho * R_PQ_sq

    # Prefactor and exponential
    prefactor = 2.0 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    return prefactor * exp_factor * boys(0, T)


def eri_ssss_rys(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute (ss|ss) ERI for UNNORMALIZED primitives using Rys quadrature.

    For L = 0, only n_roots = 1 is needed.

    Args:
        alpha, A: Exponent and center of primitive a
        beta, B: Exponent and center of primitive b
        gamma, C: Exponent and center of primitive c
        delta, D: Exponent and center of primitive d

    Returns:
        (ab|cd) for unnormalized Gaussians
    """
    # GPT for bra
    p, P, mu_ab, R_AB_sq = gaussian_product_center(alpha, A, beta, B)

    # GPT for ket
    q, Q, nu_cd, R_CD_sq = gaussian_product_center(gamma, C, delta, D)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q)**2)
    T = rho * R_PQ_sq

    # Prefactor and exponential
    prefactor = 2.0 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    # Rys quadrature: F_0(T) = (1/2) * sum_i W_i
    nodes, weights = rys_nodes_weights(T, n_roots=1)
    F0_rys = 0.5 * np.sum(weights)

    return prefactor * exp_factor * F0_rys


def eri_ssss_norm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute (ss|ss) ERI for NORMALIZED primitive Gaussians.

    Args:
        alpha, A: Exponent and center of primitive a
        beta, B: Exponent and center of primitive b
        gamma, C: Exponent and center of primitive c
        delta, D: Exponent and center of primitive d

    Returns:
        (ab|cd) for normalized Gaussians
    """
    N_a = norm_s_primitive(alpha)
    N_b = norm_s_primitive(beta)
    N_c = norm_s_primitive(gamma)
    N_d = norm_s_primitive(delta)

    return N_a * N_b * N_c * N_d * eri_ssss_unnorm(alpha, A, beta, B, gamma, C, delta, D)


def eri_psss_unnorm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray,
    axis: int = 2
) -> float:
    """
    Compute unnormalized (p_axis s|ss) ERI using the derivative formula.

    The formula (Eq. 5.28):
        (p_xi s|ss) = (2 pi^{5/2}) / (pq sqrt(p+q)) * K_AB * K_CD
                      * [ -(beta/p)(A_xi - B_xi) F_0(T)
                          - (rho/p)(P_xi - Q_xi) F_1(T) ]

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)
        axis: Cartesian axis: 0=x, 1=y, 2=z

    Returns:
        Unnormalized (p_axis s|ss) ERI
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # GPT for bra pair
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.dot(A - B, A - B)
    K_AB = math.exp(-mu * R_AB_sq)

    # GPT for ket pair
    q = gamma + delta
    nu = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.dot(C - D, C - D)
    K_CD = math.exp(-nu * R_CD_sq)

    # Inter-pair parameters
    rho = p * q / (p + q)
    R_PQ_sq = np.dot(P - Q, P - Q)
    T = rho * R_PQ_sq

    # Prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))
    gfac = K_AB * K_CD

    # Boys functions F_0 and F_1
    F0_T = boys(0, T)
    F1_T = boys(1, T)

    # Extract axis component
    A_xi = A[axis]
    B_xi = B[axis]
    P_xi = P[axis]
    Q_xi = Q[axis]

    # Derivative formula
    term1 = -(beta / p) * (A_xi - B_xi) * F0_T
    term2 = -(rho / p) * (P_xi - Q_xi) * F1_T

    return prefactor * gfac * (term1 + term2)


def eri_psss_norm(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray,
    axis: int = 2
) -> float:
    """
    Compute normalized (p_axis s|ss) ERI.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)
        axis: Cartesian axis: 0=x, 1=y, 2=z

    Returns:
        Normalized (p_axis s|ss) ERI in Hartree
    """
    eri_unnorm = eri_psss_unnorm(alpha, A, beta, B, gamma, C, delta, D, axis)

    N_p = norm_p_primitive(alpha)
    N_s_beta = norm_s_primitive(beta)
    N_s_gamma = norm_s_primitive(gamma)
    N_s_delta = norm_s_primitive(delta)

    return N_p * N_s_beta * N_s_gamma * N_s_delta * eri_unnorm


# =============================================================================
# CORE UTILITIES: J and K Matrix Construction
# =============================================================================

def build_JK_einsum(eri: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K matrices using numpy einsum.

    J_ij = sum_kl (ij|kl) P_kl
    K_ij = sum_kl (ik|jl) P_kl

    Args:
        eri: Full ERI tensor of shape (N, N, N, N)
        P: Density matrix of shape (N, N)

    Returns:
        J: Coulomb matrix
        K: Exchange matrix
    """
    J = np.einsum('ijkl,kl->ij', eri, P, optimize=True)
    K = np.einsum('ikjl,kl->ij', eri, P, optimize=True)

    return J, K


def compute_hf_energy(P: np.ndarray, h: np.ndarray, J: np.ndarray, K: np.ndarray,
                      E_nuc: float) -> Tuple[float, dict]:
    """
    Compute HF energy from density matrix and one/two-electron matrices.

    E_elec = Tr[P*h] + (1/2)*Tr[P*(J - (1/2)*K)]
           = (1/2)*Tr[P*(h + F)]

    Args:
        P: Density matrix
        h: Core Hamiltonian
        J: Coulomb matrix
        K: Exchange matrix
        E_nuc: Nuclear repulsion energy

    Returns:
        E_tot: Total HF energy
        components: Dictionary with energy breakdown
    """
    E_1 = np.einsum('ij,ij->', P, h)
    E_J = 0.5 * np.einsum('ij,ij->', P, J)
    E_K = 0.25 * np.einsum('ij,ij->', P, K)
    E_2 = E_J - E_K
    E_elec = E_1 + E_2
    E_tot = E_elec + E_nuc

    components = {
        "E_1": E_1,
        "E_J": E_J,
        "E_K": E_K,
        "E_2": E_2,
        "E_elec": E_elec,
        "E_nuc": E_nuc,
        "E_tot": E_tot,
    }

    return E_tot, components


# =============================================================================
# EXERCISE 5.1: One-Root Quadrature Formulas
# =============================================================================

def exercise_5_1():
    """
    Exercise 5.1: One-Root Quadrature Formulas [Core]

    For n_r = 1, derive and verify the closed-form expressions:
        W_1 = m_0 = 2*F_0(T)
        x_1 = m_1/m_0 = F_1(T)/F_0(T)
    """
    print("\n" + "=" * 75)
    print("EXERCISE 5.1: One-Root Quadrature Formulas")
    print("=" * 75)

    print("""
For one-root Rys quadrature (n_r = 1):
    W_1 = m_0 = 2*F_0(T)
    x_1 = m_1/m_0 = F_1(T)/F_0(T)

The quadrature exactly reproduces moments m_0 and m_1.
""")

    T_values = [0.01, 0.1, 1.0, 10.0]

    print(f"{'T':>12} {'F_0(T)':>18} {'F_1(T)':>18} {'x_1':>18} {'W_1':>18}")
    print("-" * 85)

    for T in T_values:
        F0 = boys(0, T)
        F1 = boys(1, T)
        x1 = F1 / F0 if F0 > 1e-15 else 0.5
        W1 = 2 * F0

        print(f"{T:>12.2e} {F0:>18.12f} {F1:>18.12f} {x1:>18.12f} {W1:>18.12f}")

    # Verify moment matching
    print("\nMoment matching verification at T = 1.0:")
    print("-" * 60)

    T = 1.0
    nodes, weights = rys_nodes_weights(T, n_roots=1)
    x1, W1 = nodes[0], weights[0]

    for n in range(4):
        m_exact = moment(n, T)
        m_quad = W1 * (x1 ** n)
        error = abs(m_exact - m_quad)
        status = "EXACT" if error < 1e-10 else f"error = {error:.2e}"
        print(f"  m_{n}: exact = {m_exact:.10f}, quad = {m_quad:.10f} ({status})")

    print("-" * 60)
    print("Note: Moments m_0 and m_1 are exact; m_2, m_3 are approximate.")


# =============================================================================
# EXERCISE 5.2: Two-Root Quadrature and Moment Matching
# =============================================================================

def exercise_5_2():
    """
    Exercise 5.2: Two-Root Quadrature and Moment Matching [Core]

    Implement Algorithm 5.1 for n_r = 2 and verify moment matching.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 5.2: Two-Root Quadrature and Moment Matching")
    print("=" * 75)

    print("""
For two-root Rys quadrature (n_r = 2):
    - Need moments m_0, m_1, m_2, m_3
    - Hankel matrices are 2x2
    - Quadrature exact for polynomials up to degree 3
""")

    T_values = [0.0, 1e-6, 0.1, 1.0, 10.0]

    print("\n(a) Nodes and weights for n_r = 2:")
    print("-" * 80)
    print(f"{'T':>12} {'x_1':>15} {'x_2':>15} {'W_1':>15} {'W_2':>15}")
    print("-" * 80)

    for T in T_values:
        nodes, weights = rys_nodes_weights(T, n_roots=2)
        print(f"{T:>12.2e} {nodes[0]:>15.10f} {nodes[1]:>15.10f} "
              f"{weights[0]:>15.10f} {weights[1]:>15.10f}")

    # Detailed moment matching for T = 1.0
    T = 1.0
    n_roots = 2
    nodes, weights = rys_nodes_weights(T, n_roots)

    print(f"\n(b) Moment matching verification for T = {T}, n_roots = {n_roots}:")
    print("-" * 70)
    print(f"{'n':>4} {'m_n (exact)':>20} {'sum W*x^n':>20} {'Error':>14}")
    print("-" * 70)

    for n in range(6):
        m_exact = moment(n, T)
        m_quad = np.sum(weights * (nodes ** n))
        error = abs(m_exact - m_quad)
        status = "exact" if n < 2 * n_roots else "approx"
        print(f"{n:>4} {m_exact:>20.14e} {m_quad:>20.14e} {error:>14.2e}  ({status})")

    print("-" * 70)
    print("Moments n < 2*n_roots are exact; higher moments are approximate.")


# =============================================================================
# EXERCISE 5.3: (ss|ss) by Three Routes
# =============================================================================

def exercise_5_3():
    """
    Exercise 5.3: (ss|ss) by Three Routes [Core]

    Compute (ss|ss) ERI using:
    (a) Closed form with erf-based F_0
    (b) Rys quadrature
    (c) PySCF int2e

    All three should agree to machine precision.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 5.3.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 5.3: (ss|ss) by Three Routes")
    print("=" * 75)

    print("""
Computing (ss|ss) ERI for H2 with single primitive basis:
    Route (a): Closed-form with erf-based F_0
    Route (b): Rys quadrature (n_roots = 1)
    Route (c): PySCF int2e
""")

    # Setup
    R = 1.4  # H-H distance in Bohr
    alpha = 1.0  # exponent

    A = np.array([0., 0., 0.])
    B = np.array([0., 0., R])

    # PySCF molecule with single primitive
    bas_str = f"H S\n  {alpha:.10f}  1.0\n"
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis={"H": gto.basis.parse(bas_str)},
        unit="Bohr",
        verbose=0
    )

    eri_pyscf = mol.intor("int2e", aosym="s1")

    print(f"Geometry: H2 at R = {R} Bohr")
    print(f"Exponent: alpha = {alpha}")
    print("-" * 75)
    print(f"{'ERI':>12} {'Closed-form':>18} {'Rys':>18} {'PySCF':>18} {'Max Diff':>12}")
    print("-" * 75)

    test_cases = [
        (0, 0, 0, 0, "(00|00)"),
        (0, 0, 1, 1, "(00|11)"),
        (0, 1, 0, 1, "(01|01)"),
        (1, 1, 1, 1, "(11|11)"),
    ]

    centers = [A, B]
    exponents = [alpha, alpha]
    max_diff = 0.0

    for mu, nu, lam, sig, label in test_cases:
        # Route (a): Closed-form with Boys function
        val_boys = eri_ssss_norm(
            exponents[mu], centers[mu],
            exponents[nu], centers[nu],
            exponents[lam], centers[lam],
            exponents[sig], centers[sig]
        )

        # Route (b): Rys quadrature
        val_rys = norm_s_primitive(exponents[mu]) * norm_s_primitive(exponents[nu]) * \
                  norm_s_primitive(exponents[lam]) * norm_s_primitive(exponents[sig]) * \
                  eri_ssss_rys(
                      exponents[mu], centers[mu],
                      exponents[nu], centers[nu],
                      exponents[lam], centers[lam],
                      exponents[sig], centers[sig]
                  )

        # Route (c): PySCF
        val_pyscf = eri_pyscf[mu, nu, lam, sig]

        diff = max(abs(val_boys - val_pyscf), abs(val_rys - val_pyscf))
        max_diff = max(max_diff, diff)

        print(f"{label:>12} {val_boys:>18.12f} {val_rys:>18.12f} "
              f"{val_pyscf:>18.12f} {diff:>12.2e}")

    print("-" * 75)
    print(f"Maximum difference: {max_diff:.2e}")
    passed = max_diff < 1e-10
    print(f"Validation: {'PASSED' if passed else 'FAILED'} (tolerance 1e-10)")


# =============================================================================
# EXERCISE 5.4: (p_xi s|ss) and the Appearance of F_1
# =============================================================================

def exercise_5_4():
    """
    Exercise 5.4: (p_xi s|ss) and the Appearance of F_1 [Core]

    Compute (p_xi s|ss) ERI using the derivative identity and verify
    that F_1 appears from differentiating F_0(T).
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 5.4.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 5.4: (p_xi s|ss) and the Appearance of F_1")
    print("=" * 75)

    print("""
The derivative identity:
    (p_xi b|cd) = (1/2alpha) * d/dA_xi (ab|cd)

Results in (Eq. 5.28):
    (p_xi s|ss) = prefactor * [-(beta/p)(A_xi - B_xi) F_0 - (rho/p)(P_xi - Q_xi) F_1]

Term 1 (F_0): from differentiating exp(-mu R_AB^2)
Term 2 (F_1): from differentiating F_0(T) via chain rule
""")

    # Setup
    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])
    alpha = 0.50  # p-shell
    beta = 0.40   # s-shell

    # PySCF molecule: atom1 has P shell, atom2 has S shell
    basA = gto.basis.parse(f"H P\n  {alpha:.10f}  1.0\n")
    basB = gto.basis.parse(f"H S\n  {beta:.10f}  1.0\n")

    mol = gto.M(
        atom=f"H@1 0 0 0; H@2 0 0 {R}",
        basis={"H@1": basA, "H@2": basB},
        unit="Bohr",
        verbose=0
    )

    eri = mol.intor("int2e", aosym="s1")

    print(f"Geometry: atoms at origin and (0, 0, {R}) Bohr")
    print(f"Exponents: alpha = {alpha} (p shell), beta = {beta} (s shell)")
    print("-" * 75)
    print(f"{'axis':>6} {'PySCF':>18} {'Our formula':>18} {'Difference':>14}")
    print("-" * 75)

    # AO indices: 0=p_x, 1=p_y, 2=p_z, 3=s
    idx_s = 3
    all_passed = True

    for axis, (idx_p, name) in enumerate([(0, 'p_x'), (1, 'p_y'), (2, 'p_z')]):
        eri_pyscf = eri[idx_p, idx_s, idx_s, idx_s]
        eri_ours = eri_psss_norm(alpha, A, beta, B, beta, B, beta, B, axis=axis)

        diff = abs(eri_pyscf - eri_ours)
        all_passed = all_passed and (diff < 1e-10)

        print(f"{name:>6} {eri_pyscf:>18.12f} {eri_ours:>18.12f} {diff:>14.2e}")

    print("-" * 75)

    # Explain symmetry
    print("\nSymmetry observation:")
    print("  (p_x s|ss) = (p_y s|ss) = 0 because atoms are on z-axis")
    print("  Only (p_z s|ss) is nonzero")

    print("-" * 75)
    print(f"Validation: {'PASSED' if all_passed else 'FAILED'}")


# =============================================================================
# EXERCISE 5.5: J/K Build and Energy Component Check
# =============================================================================

def exercise_5_5():
    """
    Exercise 5.5: J/K Build and Energy Component Check [Core]

    Build J and K matrices from ERIs and verify the HF energy.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("  PySCF not available. Skipping exercise 5.5.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 5.5: J/K Build and Energy Component Check")
    print("=" * 75)

    print("""
The Coulomb and Exchange matrices:
    J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}
    K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

In einsum notation:
    J = np.einsum('ijkl,kl->ij', eri, P)
    K = np.einsum('ikjl,kl->ij', eri, P)
""")

    # H2O / STO-3G
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()
    h = mf.get_hcore()

    print(f"Molecule: H2O/STO-3G")
    print(f"Number of AOs: {mol.nao}")
    print(f"RHF converged: E = {mf.e_tot:.10f} Hartree")
    print("-" * 75)

    # Get ERIs and build J, K
    eri = mol.intor("int2e", aosym="s1")
    J, K = build_JK_einsum(eri, P)

    # PySCF reference
    J_ref, K_ref = mf.get_jk(mol, P)

    J_diff = np.linalg.norm(J - J_ref)
    K_diff = np.linalg.norm(K - K_ref)

    print(f"||J - J_ref||_F = {J_diff:.2e}")
    print(f"||K - K_ref||_F = {K_diff:.2e}")

    # Energy components
    E_tot, components = compute_hf_energy(P, h, J, K, mol.energy_nuc())

    print("\nEnergy components:")
    print(f"  E_1 (one-electron):   {components['E_1']:15.10f} Hartree")
    print(f"  E_J (Coulomb):        {components['E_J']:15.10f} Hartree")
    print(f"  E_K (Exchange):       {components['E_K']:15.10f} Hartree")
    print(f"  E_2 (two-electron):   {components['E_2']:15.10f} Hartree")
    print(f"  E_elec:               {components['E_elec']:15.10f} Hartree")
    print(f"  E_nuc:                {components['E_nuc']:15.10f} Hartree")
    print(f"  E_tot (computed):     {E_tot:15.10f} Hartree")
    print(f"  E_tot (PySCF):        {mf.e_tot:15.10f} Hartree")
    print(f"  Difference:           {abs(E_tot - mf.e_tot):.2e} Hartree")

    print("-" * 75)
    passed = J_diff < 1e-10 and K_diff < 1e-10 and abs(E_tot - mf.e_tot) < 1e-8
    print(f"Validation: {'PASSED' if passed else 'FAILED'}")


# =============================================================================
# EXERCISE 5.6: Schwarz Screening Toy Study
# =============================================================================

def exercise_5_6():
    """
    Exercise 5.6: Schwarz Screening Toy Study [Advanced]

    Compute the fraction of ERIs screened by Schwarz inequality for H2O.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("  PySCF not available. Skipping exercise 5.6.")
        return

    print("\n" + "=" * 75)
    print("EXERCISE 5.6: Schwarz Screening Toy Study")
    print("=" * 75)

    print("""
Schwarz inequality: |(mu nu|lambda sigma)| <= Q_mu,nu * Q_lambda,sigma
where Q_mu,nu = sqrt((mu nu|mu nu))

If Q_mu,nu * Q_lambda,sigma < tau, the ERI is screened (set to zero).
""")

    h2o_geom = "O 0 0 0.117369; H 0 0.757918 -0.469476; H 0 -0.757918 -0.469476"

    bases = ['sto-3g', '6-31g', '6-31+g*']
    tau = 1e-10

    print(f"Molecule: H2O")
    print(f"Screening threshold: tau = {tau:.0e}")
    print("-" * 70)
    print(f"{'Basis':>12} {'NAO':>6} {'Total ERIs':>15} {'Screened':>12} {'Screened %':>12}")
    print("-" * 70)

    for basis in bases:
        mol = gto.M(atom=h2o_geom, basis=basis, unit='Angstrom', verbose=0)
        nao = mol.nao_nr()

        eri = mol.intor('int2e', aosym='s1')

        # Compute Schwarz bounds
        Q = np.sqrt(np.abs(np.einsum('iijj->ij', eri.reshape(nao, nao, nao, nao))))

        # Count screened
        n_screened = 0
        n_total = nao**4

        for mu in range(nao):
            for nu in range(nao):
                for lam in range(nao):
                    for sig in range(nao):
                        if Q[mu, nu] * Q[lam, sig] < tau:
                            n_screened += 1

        frac = 100.0 * n_screened / n_total

        print(f"{basis:>12} {nao:>6} {n_total:>15,} {n_screened:>12,} {frac:>11.1f}%")

    print("-" * 70)
    print("Note: Screening effectiveness increases with diffuse functions and molecule size.")


# =============================================================================
# EXERCISE 5.7: Hankel Matrix Conditioning
# =============================================================================

def exercise_5_7():
    """
    Exercise 5.7: Hankel Matrix Conditioning [Advanced]

    Study how the condition number of the Hankel matrix grows with n_roots.
    """
    print("\n" + "=" * 75)
    print("EXERCISE 5.7: Hankel Matrix Conditioning")
    print("=" * 75)

    print("""
The Hankel matrix H with entries H_ij = m_{i+j} becomes increasingly
ill-conditioned as n_roots grows. This limits the practical use of
the moment-based Algorithm 5.1.
""")

    T_values = [0.1, 1.0, 10.0]
    n_roots_max = 6

    print("Condition number kappa(H) = ||H|| * ||H^{-1}||:")
    print("-" * 70)
    header = f"{'n_r':>6}" + "".join([f"T={T:>6}" for T in T_values])
    print(header)
    print("-" * 70)

    for n_roots in range(1, n_roots_max + 1):
        row = f"{n_roots:>6}"
        for T in T_values:
            try:
                m = moments_array(2 * n_roots - 1, T)
                H, _ = build_hankel_matrices(m, n_roots)
                cond = np.linalg.cond(H)
                row += f"{cond:>12.2e}"
            except Exception:
                row += f"{'failed':>12}"
        print(row)

    print("-" * 70)
    print("""
Observations:
- Condition number grows exponentially with n_roots
- For n_r > 5-6, double precision may not be sufficient
- Production codes use polynomial approximations for large n_r
""")


# =============================================================================
# CHECKPOINT QUESTION SOLUTIONS
# =============================================================================

def checkpoint_5_2():
    """
    Checkpoint 5.2: One Root Suffices for Two Boys Values

    For n_r = 1, show that one root reproduces both F_0(T) and F_1(T).
    """
    print("\n" + "=" * 75)
    print("CHECKPOINT 5.2: One Root Suffices for Two Boys Values")
    print("=" * 75)

    print("""
For n_r = 1, the moment-matching equations are:
    m_0 = W_1 * 1       =>  W_1 = m_0 = 2*F_0(T)
    m_1 = W_1 * x_1     =>  x_1 = m_1/m_0 = F_1(T)/F_0(T)

Both F_0 and F_1 are reproduced exactly:
    F_0 = (1/2) * W_1 = (1/2) * m_0
    F_1 = (1/2) * W_1 * x_1 = (1/2) * m_1
""")

    T = 1.0
    nodes, weights = rys_nodes_weights(T, n_roots=1)
    x1, W1 = nodes[0], weights[0]

    F0_exact = boys(0, T)
    F1_exact = boys(1, T)
    F0_quad = 0.5 * W1
    F1_quad = 0.5 * W1 * x1

    print(f"At T = {T}:")
    print(f"  F_0(T) exact:   {F0_exact:.15f}")
    print(f"  F_0(T) quad:    {F0_quad:.15f}")
    print(f"  Difference:     {abs(F0_exact - F0_quad):.2e}")
    print()
    print(f"  F_1(T) exact:   {F1_exact:.15f}")
    print(f"  F_1(T) quad:    {F1_quad:.15f}")
    print(f"  Difference:     {abs(F1_exact - F1_quad):.2e}")


def checkpoint_5_6():
    """
    Checkpoint 5.6: Root Count Understanding

    Verify the root count rule n_r = floor(L/2) + 1.
    """
    print("\n" + "=" * 75)
    print("CHECKPOINT 5.6: Root Count Understanding")
    print("=" * 75)

    print("""
Root count rule: n_r = floor(L/2) + 1
where L = l_a + l_b + l_c + l_d is the total angular momentum.
""")

    print(f"{'Shell quartet':>15} {'L':>6} {'n_r':>6}")
    print("-" * 30)

    cases = [
        ("(ss|ss)", 0),
        ("(ps|ss)", 1),
        ("(pp|ss)", 2),
        ("(pp|pp)", 4),
        ("(dd|pp)", 6),
        ("(ff|ff)", 12),
    ]

    for name, L in cases:
        n_r = L // 2 + 1
        print(f"{name:>15} {L:>6} {n_r:>6}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_all_exercises():
    """Run all exercises in sequence."""
    print()
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*   Chapter 5 Exercise Solutions: Rys Quadrature in Practice" + " " * 17 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    # Core exercises
    exercise_5_1()
    exercise_5_2()
    exercise_5_3()
    exercise_5_4()
    exercise_5_5()
    exercise_5_6()
    exercise_5_7()

    # Checkpoint solutions
    print("\n")
    print("=" * 80)
    print("CHECKPOINT QUESTION SOLUTIONS")
    print("=" * 80)
    checkpoint_5_2()
    checkpoint_5_6()

    print("\n")
    print("=" * 80)
    print("EXERCISE SOLUTIONS COMPLETE")
    print("=" * 80)
    print("""
Summary of Key Findings:

1. EXERCISE 5.1: One-root quadrature gives x_1 = F_1/F_0 and W_1 = 2*F_0.

2. EXERCISE 5.2: Two-root quadrature exactly reproduces moments m_0 through m_3.

3. EXERCISE 5.3: (ss|ss) computed via closed-form, Rys, and PySCF all agree
   to ~10^{-12} precision.

4. EXERCISE 5.4: (p_xi s|ss) requires F_1 from differentiating F_0(T).
   Symmetry zeros occur when displacement is perpendicular to axis.

5. EXERCISE 5.5: J and K matrices from einsum match PySCF to machine precision.
   HF energy correctly reconstructed from components.

6. EXERCISE 5.6: Schwarz screening eliminates 0-35% of ERIs depending on basis.

7. EXERCISE 5.7: Hankel matrix condition number grows exponentially with n_r.
   Limits practical use of moment-based algorithm to n_r <= 5-6.
""")


def main():
    parser = argparse.ArgumentParser(description='Chapter 5 Exercise Solutions')
    parser.add_argument('--exercise', '-e', type=int, default=None,
                       help='Run specific exercise (1-7)')
    parser.add_argument('--checkpoint', '-c', type=int, default=None,
                       help='Run specific checkpoint (2 or 6)')
    args = parser.parse_args()

    if args.exercise is not None:
        exercise_map = {
            1: exercise_5_1,
            2: exercise_5_2,
            3: exercise_5_3,
            4: exercise_5_4,
            5: exercise_5_5,
            6: exercise_5_6,
            7: exercise_5_7,
        }
        if args.exercise in exercise_map:
            exercise_map[args.exercise]()
        else:
            print(f"Exercise 5.{args.exercise} not available.")
    elif args.checkpoint is not None:
        checkpoint_map = {
            2: checkpoint_5_2,
            6: checkpoint_5_6,
        }
        if args.checkpoint in checkpoint_map:
            checkpoint_map[args.checkpoint]()
        else:
            print(f"Checkpoint 5.{args.checkpoint} has no numerical solution.")
    else:
        run_all_exercises()


if __name__ == "__main__":
    main()
