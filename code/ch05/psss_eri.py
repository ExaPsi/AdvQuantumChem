#!/usr/bin/env python3
"""
psss_eri.py - (p_xi s|ss) ERI via Derivative Identity (Lab 5C)

This module implements the primitive (p_xi s|ss) ERI using the derivative
identity that promotes an s-type Gaussian to p-type:

    (p_xi b|cd) = (1/2alpha) * d/dA_xi (ab|cd)

where a is an s-type Gaussian centered at A with exponent alpha.

The closed-form result (Eq. 5.xx in lecture notes) is:

    (p_xi s|ss) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2)
                  * [ -(beta/p)(A_xi - B_xi) F_0(T) - (rho/p)(P_xi - Q_xi) F_1(T) ]

This formula shows explicitly that:
1. Angular momentum increases the Boys order required (F_1 appears)
2. The integral has two terms: one from differentiating the Gaussian prefactor,
   one from differentiating the Boys function

References:
    - Chapter 5, Section 6: A first angular momentum example (p_xi s|ss)
    - Eq. (5.37): Derivative identity
    - Eq. (5.42): Closed form for (p_xi s|ss)

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import Tuple

# Import from companion modules
from boys_moments import boys
from rys_quadrature import rys_nodes_weights


def normalization_s(alpha: float) -> float:
    """
    Compute the normalization constant for an s-type Gaussian.

    N_s(alpha) = (2*alpha/pi)^{3/4}

    Parameters
    ----------
    alpha : float
        Gaussian exponent

    Returns
    -------
    float
        Normalization constant
    """
    return (2.0 * alpha / math.pi) ** 0.75


def normalization_p(alpha: float) -> float:
    """
    Compute the normalization constant for a Cartesian p-type Gaussian.

    N_p(alpha) = (2*alpha/pi)^{3/4} * sqrt(4*alpha)

    This applies to p_x, p_y, or p_z individually.

    Parameters
    ----------
    alpha : float
        Gaussian exponent

    Returns
    -------
    float
        Normalization constant
    """
    return (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(4.0 * alpha)


def eri_ssss_unnormalized(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> Tuple[float, float, float, np.ndarray, np.ndarray, float, float]:
    """
    Compute unnormalized primitive (ss|ss) ERI and return intermediate quantities.

    Returns
    -------
    eri : float
        The (ss|ss) ERI value
    p : float
        Composite exponent alpha + beta
    q : float
        Composite exponent gamma + delta
    P : np.ndarray
        Composite center for bra pair
    Q : np.ndarray
        Composite center for ket pair
    rho : float
        Reduced exponent pq/(p+q)
    T : float
        Boys function argument rho * |P - Q|^2
    """
    # Composite parameters for bra pair
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB = A - B
    R_AB_sq = np.dot(R_AB, R_AB)
    K_AB = math.exp(-mu * R_AB_sq)

    # Composite parameters for ket pair
    q = gamma + delta
    nu = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD = C - D
    R_CD_sq = np.dot(R_CD, R_CD)
    K_CD = math.exp(-nu * R_CD_sq)

    # Inter-pair parameters
    rho = p * q / (p + q)
    R_PQ = P - Q
    R_PQ_sq = np.dot(R_PQ, R_PQ)
    T = rho * R_PQ_sq

    # Prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))

    # Boys function
    F0_T = boys(0, T)

    eri = prefactor * K_AB * K_CD * F0_T

    return eri, p, q, P, Q, rho, T


def eri_psss_unnormalized(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray,
    axis: int = 2
) -> float:
    """
    Compute unnormalized primitive (p_axis s|ss) ERI using the derivative formula.

    The formula is derived by differentiating (ss|ss) with respect to A_xi:

        (p_xi s|ss) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2)
                      * [ -(beta/p)(A_xi - B_xi) F_0(T) - (rho/p)(P_xi - Q_xi) F_1(T) ]

    Physical interpretation:
    - First term: from differentiating exp(-mu R_AB^2)
    - Second term: from differentiating F_0(T) via dF_0/dT = -F_1

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Gaussian exponents
    A, B, C, D : np.ndarray
        Gaussian centers (3D vectors)
    axis : int
        Cartesian axis: 0=x, 1=y, 2=z

    Returns
    -------
    float
        Unnormalized (p_axis s|ss) ERI
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Composite parameters for bra pair
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB = A - B
    R_AB_sq = np.dot(R_AB, R_AB)
    K_AB = math.exp(-mu * R_AB_sq)

    # Composite parameters for ket pair
    q = gamma + delta
    nu = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD = C - D
    R_CD_sq = np.dot(R_CD, R_CD)
    K_CD = math.exp(-nu * R_CD_sq)

    # Inter-pair parameters
    rho = p * q / (p + q)
    R_PQ = P - Q
    R_PQ_sq = np.dot(R_PQ, R_PQ)
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

    # The derivative formula (Eq. 5.42):
    # (p_xi s|ss) = prefactor * gfac * [ -(beta/p)(A_xi - B_xi) F_0 - (rho/p)(P_xi - Q_xi) F_1 ]
    term1 = -(beta / p) * (A_xi - B_xi) * F0_T
    term2 = -(rho / p) * (P_xi - Q_xi) * F1_T

    return prefactor * gfac * (term1 + term2)


def eri_psss_normalized(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray,
    axis: int = 2
) -> float:
    """
    Compute normalized primitive (p_axis s|ss) ERI.

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Gaussian exponents
    A, B, C, D : np.ndarray
        Gaussian centers (3D vectors)
    axis : int
        Cartesian axis: 0=x, 1=y, 2=z

    Returns
    -------
    float
        Normalized (p_axis s|ss) ERI in Hartree
    """
    eri_unnorm = eri_psss_unnormalized(alpha, A, beta, B, gamma, C, delta, D, axis)

    # Normalization: p on alpha, s on others
    N_p = normalization_p(alpha)
    N_s_beta = normalization_s(beta)
    N_s_gamma = normalization_s(gamma)
    N_s_delta = normalization_s(delta)

    return N_p * N_s_beta * N_s_gamma * N_s_delta * eri_unnorm


def eri_psss_via_rys(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray,
    axis: int = 2
) -> float:
    """
    Compute normalized (p_axis s|ss) ERI using Rys quadrature for F_0, F_1.

    This demonstrates using Rys quadrature nodes/weights instead of direct
    Boys function evaluation.

    For L = 1 (one p orbital), we need n_roots = 1, which gives exact
    F_0(T) and F_1(T) via:
        F_0 = (1/2) * W
        F_1 = (1/2) * W * x

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Gaussian exponents
    A, B, C, D : np.ndarray
        Gaussian centers
    axis : int
        Cartesian axis

    Returns
    -------
    float
        Normalized ERI
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Composite parameters
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.dot(A - B, A - B)
    K_AB = math.exp(-mu * R_AB_sq)

    q = gamma + delta
    nu = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.dot(C - D, C - D)
    K_CD = math.exp(-nu * R_CD_sq)

    rho = p * q / (p + q)
    R_PQ_sq = np.dot(P - Q, P - Q)
    T = rho * R_PQ_sq

    # Rys quadrature with n_roots = 1 (sufficient for L = 1)
    nodes, weights = rys_nodes_weights(T, n_roots=1)

    # Boys functions from Rys
    F0_T = 0.5 * weights[0]
    F1_T = 0.5 * weights[0] * nodes[0]

    # Prefactor and Gaussian factor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))
    gfac = K_AB * K_CD

    # Axis components
    A_xi = A[axis]
    B_xi = B[axis]
    P_xi = P[axis]
    Q_xi = Q[axis]

    # Derivative formula
    term1 = -(beta / p) * (A_xi - B_xi) * F0_T
    term2 = -(rho / p) * (P_xi - Q_xi) * F1_T

    eri_unnorm = prefactor * gfac * (term1 + term2)

    # Normalize
    return normalization_p(alpha) * normalization_s(beta) * \
           normalization_s(gamma) * normalization_s(delta) * eri_unnorm


# =============================================================================
# Validation against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate (p_xi s|ss) ERI against PySCF.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("Validation: (p_xi s|ss) ERI against PySCF")
    print("=" * 70)

    # Geometry in Bohr
    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    # Exponents
    alpha = 0.50  # p on atom 1
    beta = 0.40   # s on atom 2

    print(f"Geometry: atoms at origin and (0, 0, {R}) Bohr")
    print(f"Exponents: alpha = {alpha} (p shell), beta = {beta} (s shell)")
    print("-" * 70)

    # Build PySCF molecule: atom1 has P shell, atom2 has S shell
    basA = gto.basis.parse(f"""
H P
  {alpha:.10f}  1.0
""")
    basB = gto.basis.parse(f"""
H S
  {beta:.10f}  1.0
""")

    mol = gto.M(
        atom=f"H@1 {A[0]} {A[1]} {A[2]}; H@2 {B[0]} {B[1]} {B[2]}",
        basis={"H@1": basA, "H@2": basB},
        unit="Bohr",
        verbose=0
    )

    # Print AO labels for clarity
    labels = mol.ao_labels()
    print("AO labels:")
    for i, lab in enumerate(labels):
        print(f"  {i}: {lab}")

    # Get full ERI tensor
    eri = mol.intor("int2e", aosym="s1")
    nao = mol.nao

    # Find indices for p_x, p_y, p_z and s
    # P shell is typically ordered (x, y, z) in PySCF
    idx_px = 0  # First AO is p_x
    idx_py = 1  # Second AO is p_y
    idx_pz = 2  # Third AO is p_z
    idx_s = 3   # Fourth AO is s on atom 2

    print("\nComparing (p_axis s|ss) = (p_axis, s, s, s) ERIs:")
    print("-" * 70)
    print(f"{'axis':>6} {'PySCF':>18} {'Our formula':>18} {'Rys quad':>18} {'Diff':>12}")
    print("-" * 70)

    all_passed = True
    for axis, (idx_p, name) in enumerate([(idx_px, 'p_x'), (idx_py, 'p_y'), (idx_pz, 'p_z')]):
        # PySCF value: (p_axis, s, s, s)
        eri_pyscf = eri[idx_p, idx_s, idx_s, idx_s]

        # Our analytic formula
        eri_analytic = eri_psss_normalized(alpha, A, beta, B, beta, B, beta, B, axis=axis)

        # Via Rys quadrature
        eri_rys = eri_psss_via_rys(alpha, A, beta, B, beta, B, beta, B, axis=axis)

        diff = abs(eri_pyscf - eri_analytic)
        all_passed = all_passed and (diff < 1e-10)

        print(f"{name:>6} {eri_pyscf:>18.12f} {eri_analytic:>18.12f} {eri_rys:>18.12f} {diff:>12.2e}")

    print("-" * 70)
    if all_passed:
        print("VALIDATION PASSED: Agreement to within 1e-10 Hartree")
    else:
        print("VALIDATION FAILED: Some differences exceed threshold")

    return all_passed


def demonstrate_symmetry():
    """
    Demonstrate that (p_x s|ss) and (p_y s|ss) vanish when atoms are along z-axis.

    When A and B are both on the z-axis:
    - P is also on the z-axis
    - (A_x - B_x) = 0, (A_y - B_y) = 0
    - (P_x - Q_x) = 0, (P_y - Q_y) = 0
    Therefore (p_x s|ss) = (p_y s|ss) = 0 by symmetry.
    """
    print("\nSymmetry demonstration: atoms on z-axis")
    print("=" * 70)

    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    alpha = 0.50
    beta = 0.40

    print(f"Geometry: A = {A}, B = {B}")
    print(f"Both atoms on z-axis, so by symmetry (p_x s|ss) = (p_y s|ss) = 0")
    print("-" * 70)

    for axis, name in enumerate(['p_x', 'p_y', 'p_z']):
        eri = eri_psss_normalized(alpha, A, beta, B, beta, B, beta, B, axis=axis)
        zero_expected = "Yes" if axis < 2 else "No"
        print(f"({name} s|ss) = {eri:18.12f}   (expected zero: {zero_expected})")


def demonstrate_nonzero_case():
    """
    Demonstrate (p s|ss) for off-axis geometry where all components are nonzero.
    """
    print("\nOff-axis geometry: all (p_axis s|ss) nonzero")
    print("=" * 70)

    # Put atom 2 at a general position
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.5, 0.8])

    alpha = 0.50
    beta = 0.40

    print(f"Geometry: A = {A}, B = {B}")
    print("-" * 70)

    for axis, name in enumerate(['p_x', 'p_y', 'p_z']):
        eri = eri_psss_normalized(alpha, A, beta, B, beta, B, beta, B, axis=axis)
        print(f"({name} s|ss) = {eri:18.12f}")

    # Validate against PySCF for this geometry
    try:
        from pyscf import gto

        basA = gto.basis.parse(f"""
H P
  {alpha:.10f}  1.0
""")
        basB = gto.basis.parse(f"""
H S
  {beta:.10f}  1.0
""")

        mol = gto.M(
            atom=f"H@1 {A[0]} {A[1]} {A[2]}; H@2 {B[0]} {B[1]} {B[2]}",
            basis={"H@1": basA, "H@2": basB},
            unit="Bohr",
            verbose=0
        )

        eri_tensor = mol.intor("int2e", aosym="s1")

        print("\nComparison with PySCF:")
        for axis, name in enumerate(['p_x', 'p_y', 'p_z']):
            eri_pyscf = eri_tensor[axis, 3, 3, 3]
            eri_ours = eri_psss_normalized(alpha, A, beta, B, beta, B, beta, B, axis=axis)
            diff = abs(eri_pyscf - eri_ours)
            print(f"({name} s|ss): PySCF = {eri_pyscf:15.10f}, Ours = {eri_ours:15.10f}, Diff = {diff:.2e}")

    except ImportError:
        print("(PySCF not available for comparison)")


def analyze_term_contributions():
    """
    Analyze the two terms in the (p_xi s|ss) formula.

    Term 1: from d/dA_xi of exp(-mu R_AB^2) -> -(beta/p)(A_xi - B_xi) F_0
    Term 2: from d/dA_xi of F_0(T)          -> -(rho/p)(P_xi - Q_xi) F_1

    This shows explicitly how angular momentum introduces higher Boys orders.
    """
    print("\nAnalysis: Term contributions in (p_z s|ss)")
    print("=" * 70)

    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])
    axis = 2  # z-axis

    alpha = 0.50
    beta = 0.40

    # Compute intermediate quantities
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    K_AB = math.exp(-mu * np.dot(A - B, A - B))

    q = 2 * beta  # gamma = delta = beta
    nu = beta * beta / q
    Q = B  # C = D = B
    K_CD = 1.0  # exp(0) since C = D

    rho = p * q / (p + q)
    T = rho * np.dot(P - Q, P - Q)

    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))
    gfac = K_AB * K_CD

    F0 = boys(0, T)
    F1 = boys(1, T)

    A_z, B_z = A[axis], B[axis]
    P_z, Q_z = P[axis], Q[axis]

    term1 = -(beta / p) * (A_z - B_z) * F0
    term2 = -(rho / p) * (P_z - Q_z) * F1

    print(f"Parameters:")
    print(f"  p = {p:.6f}, q = {q:.6f}, rho = {rho:.6f}")
    print(f"  T = {T:.6f}")
    print(f"  F_0(T) = {F0:.10f}")
    print(f"  F_1(T) = {F1:.10f}")
    print(f"  prefactor * gfac = {prefactor * gfac:.10f}")
    print()
    print(f"Term 1 (from Gaussian prefactor): -(beta/p)(A_z - B_z) F_0 = {term1:.10f}")
    print(f"Term 2 (from Boys derivative):    -(rho/p)(P_z - Q_z) F_1 = {term2:.10f}")
    print(f"Sum (unnormalized):               {term1 + term2:.10f}")
    print()

    eri_unnorm = prefactor * gfac * (term1 + term2)
    eri_norm = normalization_p(alpha) * normalization_s(beta)**3 * eri_unnorm

    print(f"Unnormalized ERI: {eri_unnorm:.10f}")
    print(f"Normalized ERI:   {eri_norm:.10f}")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 5C: (p_xi s|ss) ERI via Derivative Identity")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run validation
    success = validate_against_pyscf()

    # Demonstrate symmetry
    demonstrate_symmetry()

    # Off-axis case
    demonstrate_nonzero_case()

    # Analyze term contributions
    analyze_term_contributions()

    print("\n" + "=" * 70)
    if success:
        print("All validations PASSED")
    else:
        print("Validation FAILED (or PySCF not available)")
    print("=" * 70)
