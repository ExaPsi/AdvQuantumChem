#!/usr/bin/env python3
"""
Lab 5C Solution: Compute (p_xi s|ss) ERI Using Derivative Identity

This script demonstrates computing the primitive (p_xi s|ss) two-electron
repulsion integral using the derivative identity that promotes an s-type
Gaussian to p-type.

The derivative identity:
    (p_xi b|cd) = (1/2alpha) * d/dA_xi (ab|cd)

where a is an s-type Gaussian centered at A with exponent alpha, and p_xi
denotes a p-type Gaussian in the xi direction (x, y, or z).

The closed-form result (Eq. 5.42 in lecture notes):

    (p_xi s|ss) = (2 pi^{5/2}) / (pq sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2)
                  * [ -(beta/p)(A_xi - B_xi) F_0(T) - (rho/p)(P_xi - Q_xi) F_1(T) ]

Key insights:
1. Angular momentum increases the Boys order required (F_1 appears)
2. The integral has two terms: one from differentiating the Gaussian prefactor,
   one from differentiating the Boys function
3. For L=1, we need n_roots = 1 for Rys quadrature (still sufficient!)

Learning objectives:
1. Apply the derivative identity for higher angular momentum
2. Compute Boys function derivatives (dF_0/dT = -F_1)
3. Understand why higher angular momentum requires higher Boys orders
4. Verify against PySCF (ps|ss) integrals

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 5: Rys Quadrature in Practice
"""

import math
import numpy as np
from typing import Tuple

# Import from Lab 5A solution
from lab5a_solution import boys, rys_nodes_weights

# =============================================================================
# Section 1: Normalization Constants
# =============================================================================

def normalization_s(alpha: float) -> float:
    """
    Compute the normalization constant for an s-type Gaussian.

    N_s(alpha) = (2*alpha/pi)^{3/4}

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant N_s
    """
    return (2.0 * alpha / math.pi) ** 0.75


def normalization_p(alpha: float) -> float:
    """
    Compute the normalization constant for a Cartesian p-type Gaussian.

    For a p_x orbital: chi_{p_x} = N_p * x * exp(-alpha * r^2)

    The normalization is:
        N_p(alpha) = (2*alpha/pi)^{3/4} * sqrt(4*alpha)

    This applies to p_x, p_y, or p_z individually.

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant N_p
    """
    return (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(4.0 * alpha)


# =============================================================================
# Section 2: (p_xi s|ss) ERI via Derivative Formula
# =============================================================================

def eri_psss_unnorm(alpha: float, A: np.ndarray,
                     beta: float, B: np.ndarray,
                     gamma: float, C: np.ndarray,
                     delta: float, D: np.ndarray,
                     axis: int = 2) -> float:
    """
    Compute unnormalized primitive (p_axis s|ss) ERI using the derivative formula.

    The formula is derived by differentiating (ss|ss) with respect to A_xi:

        d/dA_xi (ab|cd) = 2*alpha * (p_xi b|cd)

    Rearranging: (p_xi b|cd) = (1/2alpha) * d/dA_xi (ab|cd)

    The result is (Eq. 5.42):

        (p_xi s|ss) = (2 pi^{5/2}) / (pq sqrt(p+q)) * K_AB * K_CD
                      * [ -(beta/p)(A_xi - B_xi) F_0(T)
                          - (rho/p)(P_xi - Q_xi) F_1(T) ]

    Physical interpretation of the two terms:
    - Term 1: from d/dA_xi of exp(-mu R_AB^2) -> produces F_0
    - Term 2: from d/dA_xi of F_0(T) via chain rule -> produces F_1

    The chain rule gives: dF_0/dA_xi = (dF_0/dT) * (dT/dA_xi) = -F_1 * 2*rho*(P_xi - Q_xi)*(alpha/p)

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

    # Composite parameters for bra pair (Gaussian Product Theorem)
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


def eri_psss_norm(alpha: float, A: np.ndarray,
                   beta: float, B: np.ndarray,
                   gamma: float, C: np.ndarray,
                   delta: float, D: np.ndarray,
                   axis: int = 2) -> float:
    """
    Compute normalized primitive (p_axis s|ss) ERI.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)
        axis: Cartesian axis: 0=x, 1=y, 2=z

    Returns:
        Normalized (p_axis s|ss) ERI in Hartree
    """
    eri_unnorm = eri_psss_unnorm(alpha, A, beta, B, gamma, C, delta, D, axis)

    # Normalization: p on alpha (first function), s on all others
    N_p = normalization_p(alpha)
    N_s_beta = normalization_s(beta)
    N_s_gamma = normalization_s(gamma)
    N_s_delta = normalization_s(delta)

    return N_p * N_s_beta * N_s_gamma * N_s_delta * eri_unnorm


# =============================================================================
# Section 3: (p_xi s|ss) ERI via Rys Quadrature
# =============================================================================

def eri_psss_rys(alpha: float, A: np.ndarray,
                  beta: float, B: np.ndarray,
                  gamma: float, C: np.ndarray,
                  delta: float, D: np.ndarray,
                  axis: int = 2) -> float:
    """
    Compute normalized (p_axis s|ss) ERI using Rys quadrature for F_0, F_1.

    This demonstrates using Rys quadrature nodes/weights instead of direct
    Boys function evaluation.

    For L = 1 (one p orbital, rest s), we need n_roots = floor(1/2) + 1 = 1.
    The single Rys root provides exact F_0 and F_1 via:
        F_n = (1/2) * sum_i W_i * x_i^n

    Note: With 1 root, we get exactness for moments m_0, m_1 (i.e., F_0, F_1).

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers
        axis: Cartesian axis

    Returns:
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

    # Boys functions from Rys: F_n = (1/2) * sum W * x^n
    F0_T = 0.5 * weights[0]  # x^0 = 1
    F1_T = 0.5 * weights[0] * nodes[0]  # x^1

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
# Section 4: Boys Function Derivative
# =============================================================================

def boys_derivative(n: int, T: float) -> float:
    """
    Compute the derivative of the Boys function with respect to T.

    dF_n/dT = -F_{n+1}

    This is the key relation that introduces higher-order Boys functions
    when differentiating integrals for angular momentum promotion.

    Args:
        n: Order of the Boys function
        T: Argument

    Returns:
        dF_n/dT = -F_{n+1}
    """
    return -boys(n + 1, T)


def demonstrate_derivative_identity():
    """
    Demonstrate the derivative identity dF_n/dT = -F_{n+1} numerically.
    """
    print("=" * 75)
    print("Boys Function Derivative Identity: dF_n/dT = -F_{n+1}")
    print("=" * 75)

    T_values = [0.1, 1.0, 5.0, 10.0]
    h = 1e-6  # Finite difference step

    print(f"\n{'n':>4} {'T':>8} {'Numerical':>18} {'-F_{n+1}':>18} {'Diff':>12}")
    print("-" * 70)

    for n in [0, 1, 2]:
        for T in T_values:
            # Numerical derivative using central difference
            F_plus = boys(n, T + h)
            F_minus = boys(n, T - h)
            numerical = (F_plus - F_minus) / (2 * h)

            # Analytical derivative
            analytical = -boys(n + 1, T)

            diff = abs(numerical - analytical)
            print(f"{n:>4} {T:>8.2f} {numerical:>18.12f} {analytical:>18.12f} {diff:>12.2e}")


# =============================================================================
# Section 5: Validation Against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate (p_xi s|ss) ERI against PySCF.

    We create a molecule where atom 1 has a P shell and atom 2 has an S shell,
    then compare our computed ERIs with PySCF's int2e output.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("\n" + "=" * 75)
    print("Validation Against PySCF")
    print("=" * 75)

    # Geometry in Bohr
    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    # Exponents
    alpha = 0.50  # p on atom 1
    beta = 0.40   # s on atom 2

    print(f"\nGeometry: atoms at origin and (0, 0, {R}) Bohr")
    print(f"Exponents: alpha = {alpha} (p shell), beta = {beta} (s shell)")
    print("-" * 75)

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

    # P shell is typically ordered (x, y, z) in PySCF
    # AO indices: 0=p_x, 1=p_y, 2=p_z, 3=s
    idx_px = 0
    idx_py = 1
    idx_pz = 2
    idx_s = 3

    print("\nComparing (p_axis s|ss) = (p_axis, s, s, s) ERIs:")
    print("-" * 75)
    print(f"{'axis':>6} {'PySCF':>18} {'Our formula':>18} {'Rys quad':>18} {'Diff':>12}")
    print("-" * 75)

    all_passed = True
    for axis, (idx_p, name) in enumerate([(idx_px, 'p_x'), (idx_py, 'p_y'), (idx_pz, 'p_z')]):
        # PySCF value: (p_axis, s, s, s)
        eri_pyscf = eri[idx_p, idx_s, idx_s, idx_s]

        # Our analytic formula
        eri_analytic = eri_psss_norm(alpha, A, beta, B, beta, B, beta, B, axis=axis)

        # Via Rys quadrature
        eri_rys = eri_psss_rys(alpha, A, beta, B, beta, B, beta, B, axis=axis)

        diff = abs(eri_pyscf - eri_analytic)
        all_passed = all_passed and (diff < 1e-10)

        print(f"{name:>6} {eri_pyscf:>18.12f} {eri_analytic:>18.12f} "
              f"{eri_rys:>18.12f} {diff:>12.2e}")

    print("-" * 75)
    if all_passed:
        print("VALIDATION PASSED: Agreement to within 1e-10 Hartree")
    else:
        print("VALIDATION FAILED: Some differences exceed threshold")

    return all_passed


# =============================================================================
# Section 6: Symmetry Demonstration
# =============================================================================

def demonstrate_symmetry():
    """
    Demonstrate that (p_x s|ss) and (p_y s|ss) vanish when atoms are along z-axis.

    When A and B are both on the z-axis:
    - P is also on the z-axis
    - (A_x - B_x) = 0, (A_y - B_y) = 0
    - (P_x - Q_x) = 0, (P_y - Q_y) = 0
    Therefore (p_x s|ss) = (p_y s|ss) = 0 by symmetry.
    """
    print("\n" + "=" * 75)
    print("Symmetry Demonstration: Atoms on z-axis")
    print("=" * 75)

    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    alpha = 0.50
    beta = 0.40

    print(f"\nGeometry: A = {A}, B = {B}")
    print("Both atoms on z-axis, so by symmetry (p_x s|ss) = (p_y s|ss) = 0")
    print("-" * 60)

    for axis, name in enumerate(['p_x', 'p_y', 'p_z']):
        eri = eri_psss_norm(alpha, A, beta, B, beta, B, beta, B, axis=axis)
        zero_expected = "Yes" if axis < 2 else "No"
        print(f"({name} s|ss) = {eri:18.12f}   (expected zero: {zero_expected})")


def demonstrate_off_axis():
    """
    Demonstrate (p s|ss) for off-axis geometry where all components are nonzero.
    """
    print("\n" + "=" * 75)
    print("Off-Axis Geometry: All (p_axis s|ss) Nonzero")
    print("=" * 75)

    # Put atom 2 at a general position
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.5, 0.8])

    alpha = 0.50
    beta = 0.40

    print(f"\nGeometry: A = {A}, B = {B}")
    print("-" * 60)

    for axis, name in enumerate(['p_x', 'p_y', 'p_z']):
        eri = eri_psss_norm(alpha, A, beta, B, beta, B, beta, B, axis=axis)
        print(f"({name} s|ss) = {eri:18.12f}")


# =============================================================================
# Section 7: Term Analysis
# =============================================================================

def analyze_term_contributions():
    """
    Analyze the two terms in the (p_xi s|ss) formula.

    Term 1: from d/dA_xi of exp(-mu R_AB^2) -> -(beta/p)(A_xi - B_xi) F_0
    Term 2: from d/dA_xi of F_0(T)          -> -(rho/p)(P_xi - Q_xi) F_1

    This shows explicitly how angular momentum introduces higher Boys orders.
    """
    print("\n" + "=" * 75)
    print("Analysis: Term Contributions in (p_z s|ss)")
    print("=" * 75)

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

    print(f"\nParameters:")
    print(f"  alpha = {alpha}, beta = {beta}")
    print(f"  p = {p:.6f}, q = {q:.6f}, rho = {rho:.6f}")
    print(f"  T = {T:.6f}")
    print(f"  F_0(T) = {F0:.10f}")
    print(f"  F_1(T) = {F1:.10f}")
    print(f"  prefactor * gfac = {prefactor * gfac:.10f}")
    print()
    print("Term breakdown:")
    print(f"  Term 1 (from Gaussian prefactor):")
    print(f"    -(beta/p)(A_z - B_z) F_0 = -({beta}/{p:.4f})({A_z - B_z:.4f})({F0:.6f})")
    print(f"                            = {term1:.10f}")
    print()
    print(f"  Term 2 (from Boys derivative):")
    print(f"    -(rho/p)(P_z - Q_z) F_1 = -({rho:.4f}/{p:.4f})({P_z - Q_z:.4f})({F1:.6f})")
    print(f"                            = {term2:.10f}")
    print()
    print(f"  Sum (inside brackets): {term1 + term2:.10f}")

    eri_unnorm = prefactor * gfac * (term1 + term2)
    eri_norm = normalization_p(alpha) * normalization_s(beta)**3 * eri_unnorm

    print()
    print(f"Unnormalized ERI: {eri_unnorm:.10f}")
    print(f"Normalized ERI:   {eri_norm:.10f}")


# =============================================================================
# Section 8: Physical Interpretation
# =============================================================================

def explain_derivative_identity():
    """Explain the physical meaning of the derivative identity."""
    explanation = """
Physical Interpretation: The Derivative Identity
=================================================

The derivative identity relates s-type and p-type Gaussians:

    d/dA_x [exp(-alpha*|r-A|^2)] = 2*alpha*(x - A_x)*exp(-alpha*|r-A|^2)
                                 = 2*alpha * p_x(r; alpha, A)

where p_x is the unnormalized p_x Gaussian.

Rearranging for the ERI:
    (p_x b|cd) = (1/2alpha) * d/dA_x (ab|cd)

WHY F_1 APPEARS:
----------------
When we differentiate the (ss|ss) formula with respect to A_x, we get two
contributions:

1. From exp(-mu*R_AB^2): produces a term with F_0
   d/dA_x [exp(-mu*(A-B)^2)] = -2*mu*(A_x - B_x) * exp(...)
                              = -(2*beta/p)*(A_x - B_x) * exp(...)

2. From F_0(T) via chain rule: produces a term with F_1
   d/dA_x [F_0(T)] = (dF_0/dT) * (dT/dA_x)
                   = -F_1 * 2*rho*(P_x - Q_x)*(alpha/p)

The second term is where F_1 enters. This is a general pattern:
- Each derivative with respect to coordinates increases the maximum
  Boys function order by 1
- Higher angular momentum = more derivatives = higher Boys orders

ROOT COUNT RULE:
----------------
For shell quartet with total angular momentum L = l_a + l_b + l_c + l_d:
    n_roots = floor(L/2) + 1

For (ps|ss): L = 1 + 0 + 0 + 0 = 1, so n_roots = 1.

With 1 Rys root, we get exact F_0 and F_1:
    F_n = (1/2) * sum_i W_i * x_i^n    (exact for n = 0, 1)

GENERALIZATION:
---------------
Higher angular momentum shells (d, f, ...) require more derivatives,
introducing F_2, F_3, etc. The Rys quadrature approach scales naturally:
just increase n_roots to get exactness for higher Boys orders.
"""
    print(explanation)


# =============================================================================
# Section 9: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 5C demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 5C: (p_xi s|ss) ERI via Derivative Identity" + " " * 21 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Physical explanation
    explain_derivative_identity()

    # Demonstrate Boys derivative identity
    demonstrate_derivative_identity()

    # Analyze term contributions
    analyze_term_contributions()

    # Symmetry demonstration
    demonstrate_symmetry()

    # Off-axis case
    demonstrate_off_axis()

    # Validate against PySCF
    validation_passed = validate_against_pyscf()

    # ==========================================================================
    # Additional: Compare Boys and Rys approaches
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Boys vs Rys Quadrature for (p_z s|ss)")
    print("=" * 75)

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.4])
    alpha = 0.50
    beta = 0.40

    eri_boys = eri_psss_norm(alpha, A, beta, B, beta, B, beta, B, axis=2)
    eri_rys = eri_psss_rys(alpha, A, beta, B, beta, B, beta, B, axis=2)

    print(f"\n(p_z s|ss):")
    print(f"  Via Boys function:   {eri_boys:.15f}")
    print(f"  Via Rys quadrature:  {eri_rys:.15f}")
    print(f"  Difference:          {abs(eri_boys - eri_rys):.2e}")

    boys_rys_agree = abs(eri_boys - eri_rys) < 1e-12

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Lab 5C Summary")
    print("=" * 75)
    print(f"PySCF validation:         {'PASS' if validation_passed else 'FAIL'}")
    print(f"Boys vs Rys agreement:    {'PASS' if boys_rys_agree else 'FAIL'}")
    print("-" * 75)
    if validation_passed and boys_rys_agree:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED - Review output above")
    print("=" * 75)


if __name__ == "__main__":
    main()
