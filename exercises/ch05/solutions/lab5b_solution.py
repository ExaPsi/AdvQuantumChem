#!/usr/bin/env python3
"""
Lab 5B Solution: Compute (ss|ss) ERI Using Rys Quadrature

This script demonstrates computing the primitive (ss|ss) two-electron repulsion
integral using Rys quadrature nodes and weights, and compares it to the closed-form
Boys function approach.

The (ss|ss) ERI formula in chemist's notation:

    (ab|cd) = (2 pi^{5/2}) / (pq sqrt(p+q))
              * exp(-mu_ab R^2_AB) * exp(-nu_cd R^2_CD) * F_0(T)

where:
    - p = alpha + beta, q = gamma + delta
    - mu_ab = alpha*beta/p, nu_cd = gamma*delta/q
    - P = (alpha*A + beta*B)/p, Q = (gamma*C + delta*D)/q
    - T = rho * |P-Q|^2, rho = pq/(p+q)

The equivalence we demonstrate:
    F_0(T) = (1/2) * W_1 * 1   (single Rys root for L=0)

Since the (ss|ss) integral requires only F_0(T), a single Rys root suffices.

Learning objectives:
1. Understand the structure of the (ss|ss) ERI formula
2. See how Rys quadrature replaces direct Boys function evaluation
3. Verify numerical equivalence of both approaches
4. Validate against PySCF reference values

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 5: Rys Quadrature in Practice
"""

import math
import numpy as np
from typing import Tuple

# Import from Lab 5A solution
from lab5a_solution import boys, moment, rys_nodes_weights

# =============================================================================
# Section 1: Normalization Constants
# =============================================================================

def normalization_s(alpha: float) -> float:
    """
    Compute the normalization constant for an s-type Gaussian.

    The unnormalized Gaussian is:
        g(r; alpha, A) = exp(-alpha * |r - A|^2)

    The normalized version has N_s * g(r) with integral |N_s * g|^2 = 1.

    For s-type Gaussians:
        N_s(alpha) = (2*alpha/pi)^{3/4}

    Args:
        alpha: Gaussian exponent

    Returns:
        Normalization constant N_s
    """
    return (2.0 * alpha / math.pi) ** 0.75


# =============================================================================
# Section 2: (ss|ss) ERI via Closed-Form Boys Function
# =============================================================================

def eri_ssss_boys_unnorm(alpha: float, A: np.ndarray,
                          beta: float, B: np.ndarray,
                          gamma: float, C: np.ndarray,
                          delta: float, D: np.ndarray) -> float:
    """
    Compute unnormalized (ss|ss) ERI using the closed-form Boys function.

    The formula is (Appendix C, Eq. C.xx):

        (ab|cd) = (2 pi^{5/2}) / (pq sqrt(p+q))
                  * exp(-mu_ab R^2_AB) * exp(-nu_cd R^2_CD) * F_0(T)

    where T = rho * |P-Q|^2 and rho = pq/(p+q).

    Physical interpretation:
    - The Gaussian product theorem combines (a,b) into a single Gaussian at P
    - Similarly (c,d) combines into a Gaussian at Q
    - The 1/r_{12} operator generates the Boys function F_0(T)

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)

    Returns:
        Unnormalized (ss|ss) ERI in atomic units (Hartree)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Gaussian product theorem for bra pair (a, b) -> Gaussian at P
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.sum((A - B) ** 2)

    # Gaussian product theorem for ket pair (c, d) -> Gaussian at Q
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.sum((C - D) ** 2)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q) ** 2)
    T = rho * R_PQ_sq

    # ERI prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))

    # Gaussian decay factors
    exp_factor = math.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    # Boys function F_0(T)
    F0 = boys(0, T)

    return prefactor * exp_factor * F0


def eri_ssss_boys_norm(alpha: float, A: np.ndarray,
                        beta: float, B: np.ndarray,
                        gamma: float, C: np.ndarray,
                        delta: float, D: np.ndarray) -> float:
    """
    Compute normalized (ss|ss) ERI using the closed-form Boys function.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)

    Returns:
        Normalized (ss|ss) ERI in atomic units (Hartree)
    """
    N = normalization_s(alpha) * normalization_s(beta) * \
        normalization_s(gamma) * normalization_s(delta)

    return N * eri_ssss_boys_unnorm(alpha, A, beta, B, gamma, C, delta, D)


# =============================================================================
# Section 3: (ss|ss) ERI via Rys Quadrature
# =============================================================================

def eri_ssss_rys_unnorm(alpha: float, A: np.ndarray,
                         beta: float, B: np.ndarray,
                         gamma: float, C: np.ndarray,
                         delta: float, D: np.ndarray) -> float:
    """
    Compute unnormalized (ss|ss) ERI using Rys quadrature.

    For L = 0 (all s-shells), we need only n_roots = 1.
    The quadrature relation is:
        F_0(T) = (1/2) * sum_i W_i * x_i^0 = (1/2) * W_1

    This demonstrates how Rys quadrature replaces direct Boys evaluation.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)

    Returns:
        Unnormalized (ss|ss) ERI in atomic units (Hartree)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Gaussian product theorem for bra pair
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.sum((A - B) ** 2)

    # Gaussian product theorem for ket pair
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.sum((C - D) ** 2)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q) ** 2)
    T = rho * R_PQ_sq

    # ERI prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))

    # Gaussian decay factors
    exp_factor = math.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    # Rys quadrature: F_0(T) = (1/2) * sum_i W_i
    # For L = 0, only need n_roots = 1
    nodes, weights = rys_nodes_weights(T, n_roots=1)
    F0_rys = 0.5 * np.sum(weights)  # For n=0, x^0 = 1

    return prefactor * exp_factor * F0_rys


def eri_ssss_rys_norm(alpha: float, A: np.ndarray,
                       beta: float, B: np.ndarray,
                       gamma: float, C: np.ndarray,
                       delta: float, D: np.ndarray) -> float:
    """
    Compute normalized (ss|ss) ERI using Rys quadrature.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3D vectors)

    Returns:
        Normalized (ss|ss) ERI in atomic units (Hartree)
    """
    N = normalization_s(alpha) * normalization_s(beta) * \
        normalization_s(gamma) * normalization_s(delta)

    return N * eri_ssss_rys_unnorm(alpha, A, beta, B, gamma, C, delta, D)


# =============================================================================
# Section 4: Intermediate Quantity Analysis
# =============================================================================

def analyze_eri_components(alpha: float, A: np.ndarray,
                           beta: float, B: np.ndarray,
                           gamma: float, C: np.ndarray,
                           delta: float, D: np.ndarray) -> dict:
    """
    Compute and return all intermediate quantities in the ERI formula.

    This is useful for understanding each step of the calculation.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers

    Returns:
        Dictionary containing all intermediate values
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    # Bra pair
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB = A - B
    R_AB_sq = np.sum(R_AB ** 2)
    K_AB = math.exp(-mu_ab * R_AB_sq)

    # Ket pair
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD = C - D
    R_CD_sq = np.sum(R_CD ** 2)
    K_CD = math.exp(-nu_cd * R_CD_sq)

    # Inter-pair
    rho = p * q / (p + q)
    R_PQ = P - Q
    R_PQ_sq = np.sum(R_PQ ** 2)
    T = rho * R_PQ_sq

    # Boys function
    F0 = boys(0, T)

    # Prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))

    # Rys quadrature values
    nodes, weights = rys_nodes_weights(T, n_roots=1)
    F0_rys = 0.5 * weights[0]

    return {
        # Bra pair quantities
        "p": p,
        "mu_ab": mu_ab,
        "P": P,
        "R_AB": R_AB,
        "R_AB_sq": R_AB_sq,
        "K_AB": K_AB,
        # Ket pair quantities
        "q": q,
        "nu_cd": nu_cd,
        "Q": Q,
        "R_CD": R_CD,
        "R_CD_sq": R_CD_sq,
        "K_CD": K_CD,
        # Inter-pair quantities
        "rho": rho,
        "R_PQ": R_PQ,
        "R_PQ_sq": R_PQ_sq,
        "T": T,
        # Boys function
        "F0_direct": F0,
        "F0_rys": F0_rys,
        # Rys quadrature
        "rys_nodes": nodes,
        "rys_weights": weights,
        # Prefactor
        "prefactor": prefactor,
        # Final unnormalized ERI
        "eri_unnorm": prefactor * K_AB * K_CD * F0,
    }


# =============================================================================
# Section 5: Validation Against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate (ss|ss) ERI implementations against PySCF.

    We create a minimal molecule with only s-type basis functions and
    compare our computed ERIs with PySCF's int2e output.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("=" * 75)
    print("Validation Against PySCF")
    print("=" * 75)

    # H2 molecule with STO-3G basis
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: H2 at R = 0.74 Angstrom")
    print(f"Basis: STO-3G (each H has 1 contracted s function)")
    print(f"Number of AOs: {mol.nao}")
    print("-" * 75)

    # Get full ERI tensor from PySCF
    eri_pyscf = mol.intor("int2e", aosym="s1")

    # Selected ERIs in chemist's notation
    print("\nSelected ERIs (chemist's notation):")
    print("-" * 50)
    print(f"  (00|00) = {eri_pyscf[0, 0, 0, 0]:.12f} Hartree")
    print(f"  (00|11) = {eri_pyscf[0, 0, 1, 1]:.12f} Hartree")
    print(f"  (01|01) = {eri_pyscf[0, 1, 0, 1]:.12f} Hartree")
    print(f"  (11|11) = {eri_pyscf[1, 1, 1, 1]:.12f} Hartree")

    # Verify 8-fold symmetry
    print("\n8-fold ERI Symmetry Check:")
    print("-" * 50)
    print("(00|11) should equal (11|00), (01|10), (10|01):")
    print(f"  (00|11) = {eri_pyscf[0, 0, 1, 1]:.12f}")
    print(f"  (11|00) = {eri_pyscf[1, 1, 0, 0]:.12f}")
    print(f"  (01|10) = {eri_pyscf[0, 1, 1, 0]:.12f}")
    print(f"  (10|01) = {eri_pyscf[1, 0, 0, 1]:.12f}")

    # Physical interpretation
    print("\nPhysical Interpretation:")
    print("-" * 50)
    print("  (00|00): Self-repulsion of electron density on H1")
    print("  (00|11): Coulomb repulsion between H1 and H2 densities")
    print("  (01|01): Exchange integral (quantum mechanical, no classical analog)")

    return True


def compare_with_primitive_integrals():
    """
    Compare our primitive ERI implementations with PySCF.

    PySCF's mol.intor('int2e') computes contracted integrals, so we need
    to set up a molecule with primitive (uncontracted) basis functions.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Skipping primitive comparison.")
        return False

    print("\n" + "=" * 75)
    print("Primitive ERI Comparison")
    print("=" * 75)

    # Create molecule with uncontracted (primitive) basis
    # Single s-type Gaussian on each atom
    alpha = 0.5  # exponent
    R = 1.4      # H-H distance in Bohr

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    # Create PySCF molecule with single primitive per atom
    bas_str = f"""
H S
  {alpha:.10f}  1.0
"""

    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis={"H": gto.basis.parse(bas_str)},
        unit="Bohr",
        verbose=0
    )

    print(f"\nGeometry: H2 at R = {R} Bohr")
    print(f"Basis: single primitive s-type Gaussian, alpha = {alpha}")
    print(f"Number of AOs: {mol.nao}")
    print("-" * 75)

    # Get PySCF ERIs
    eri_pyscf = mol.intor("int2e", aosym="s1")

    # Compute our ERIs
    # (00|00) = (alpha, A | alpha, A | alpha, A | alpha, A)
    eri_0000_boys = eri_ssss_boys_norm(alpha, A, alpha, A, alpha, A, alpha, A)
    eri_0000_rys = eri_ssss_rys_norm(alpha, A, alpha, A, alpha, A, alpha, A)

    # (00|11) = (alpha, A | alpha, A | alpha, B | alpha, B)
    eri_0011_boys = eri_ssss_boys_norm(alpha, A, alpha, A, alpha, B, alpha, B)
    eri_0011_rys = eri_ssss_rys_norm(alpha, A, alpha, A, alpha, B, alpha, B)

    # (01|01) = (alpha, A | alpha, B | alpha, A | alpha, B)
    eri_0101_boys = eri_ssss_boys_norm(alpha, A, alpha, B, alpha, A, alpha, B)
    eri_0101_rys = eri_ssss_rys_norm(alpha, A, alpha, B, alpha, A, alpha, B)

    print("\nComparison: PySCF vs Boys vs Rys")
    print("-" * 75)
    print(f"{'ERI':>10} {'PySCF':>18} {'Boys':>18} {'Rys':>18} {'Diff':>12}")
    print("-" * 75)

    all_passed = True
    comparisons = [
        ("(00|00)", eri_pyscf[0, 0, 0, 0], eri_0000_boys, eri_0000_rys),
        ("(00|11)", eri_pyscf[0, 0, 1, 1], eri_0011_boys, eri_0011_rys),
        ("(01|01)", eri_pyscf[0, 1, 0, 1], eri_0101_boys, eri_0101_rys),
    ]

    for name, pyscf_val, boys_val, rys_val in comparisons:
        diff_boys = abs(pyscf_val - boys_val)
        diff_rys = abs(pyscf_val - rys_val)
        max_diff = max(diff_boys, diff_rys)

        print(f"{name:>10} {pyscf_val:>18.12f} {boys_val:>18.12f} "
              f"{rys_val:>18.12f} {max_diff:>12.2e}")

        if max_diff > 1e-10:
            all_passed = False

    print("-" * 75)
    print(f"Validation: {'PASS' if all_passed else 'FAIL'}")

    return all_passed


# =============================================================================
# Section 6: Demonstration of Boys vs Rys Equivalence
# =============================================================================

def demonstrate_boys_rys_equivalence():
    """
    Show step-by-step that Boys and Rys approaches give identical results.
    """
    print("\n" + "=" * 75)
    print("Boys vs Rys Equivalence Demonstration")
    print("=" * 75)

    # Test parameters
    alpha, beta = 0.5, 0.4
    gamma, delta = 0.6, 0.3

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.4])
    C = np.array([0.0, 0.5, 0.7])
    D = np.array([0.3, 0.0, 1.0])

    print(f"\nTest configuration:")
    print(f"  alpha = {alpha}, beta = {beta}, gamma = {gamma}, delta = {delta}")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  C = {C}")
    print(f"  D = {D}")

    # Get all intermediate quantities
    comp = analyze_eri_components(alpha, A, beta, B, gamma, C, delta, D)

    print("\nIntermediate quantities:")
    print("-" * 50)
    print(f"  Bra pair:  p = {comp['p']:.6f}, P = {comp['P']}")
    print(f"  Ket pair:  q = {comp['q']:.6f}, Q = {comp['Q']}")
    print(f"  Inter-pair: rho = {comp['rho']:.6f}, T = {comp['T']:.6f}")

    print("\nBoys function evaluation:")
    print("-" * 50)
    print(f"  F_0(T) direct:     {comp['F0_direct']:.15f}")
    print(f"  F_0(T) from Rys:   {comp['F0_rys']:.15f}")
    print(f"  Difference:        {abs(comp['F0_direct'] - comp['F0_rys']):.2e}")

    print("\nRys quadrature details (n_roots = 1):")
    print("-" * 50)
    print(f"  Node x_1:   {comp['rys_nodes'][0]:.15f}")
    print(f"  Weight W_1: {comp['rys_weights'][0]:.15f}")
    print(f"  F_0 = (1/2)*W_1 = {0.5 * comp['rys_weights'][0]:.15f}")

    # Compute full ERIs
    eri_boys = eri_ssss_boys_norm(alpha, A, beta, B, gamma, C, delta, D)
    eri_rys = eri_ssss_rys_norm(alpha, A, beta, B, gamma, C, delta, D)

    print("\nFinal normalized ERI:")
    print("-" * 50)
    print(f"  Via Boys function: {eri_boys:.15f} Hartree")
    print(f"  Via Rys quadrature: {eri_rys:.15f} Hartree")
    print(f"  Difference:         {abs(eri_boys - eri_rys):.2e}")


# =============================================================================
# Section 7: Physical Interpretation
# =============================================================================

def explain_ssss_eri():
    """Explain the physical meaning of the (ss|ss) ERI."""
    explanation = """
Physical Interpretation of (ss|ss) ERI
=======================================

The two-electron repulsion integral (ss|ss) represents the Coulomb interaction
between two electron distributions, each described by s-type Gaussians:

    (ab|cd) = integral integral chi_a(r1) chi_b(r1) (1/r12) chi_c(r2) chi_d(r2) dr1 dr2

For s-type Gaussians:
    chi_a(r) = N_a * exp(-alpha * |r - A|^2)

INTERPRETATION OF TERMS:
------------------------
1. Prefactor (2*pi^{5/2}) / (pq*sqrt(p+q)):
   - Comes from the 6D Gaussian integration over r1 and r2
   - Contains the angular factors from spherical symmetry

2. Gaussian decay exp(-mu*R^2_AB) * exp(-nu*R^2_CD):
   - Measures how much the basis functions overlap
   - Decays exponentially with separation of centers

3. Boys function F_0(T):
   - Contains the 1/r12 physics
   - T = rho*|P-Q|^2 measures the "charge separation"
   - F_0(0) = 1: point charges at same location
   - F_0(T) ~ 1/sqrt(T) for large T: separated charges

ROOT COUNT FOR (ss|ss):
-----------------------
Total angular momentum L = l_a + l_b + l_c + l_d = 0 + 0 + 0 + 0 = 0
n_roots = floor(L/2) + 1 = 1

Only F_0(T) is needed, so a single Rys root suffices!

WHY RYS QUADRATURE?
-------------------
For higher angular momentum shells (p, d, f, ...), the integral involves
a polynomial in the integration variable, which requires multiple Rys roots.
The (ss|ss) case is the simplest, but the Rys framework generalizes seamlessly.
"""
    print(explanation)


# =============================================================================
# Section 8: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 5B demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 5B: (ss|ss) ERI via Rys Quadrature" + " " * 31 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Physical explanation
    explain_ssss_eri()

    # Demonstrate equivalence
    demonstrate_boys_rys_equivalence()

    # Validate against PySCF
    validate_against_pyscf()

    # Compare primitive integrals
    prim_passed = compare_with_primitive_integrals()

    # ==========================================================================
    # Additional test: Sweep over T values
    # ==========================================================================

    print("\n" + "=" * 75)
    print("F_0(T) via Boys vs Rys: Sweep over T values")
    print("=" * 75)

    T_values = [0.0, 1e-6, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0]

    print(f"\n{'T':>12} {'F_0 (Boys)':>20} {'F_0 (Rys)':>20} {'Difference':>14}")
    print("-" * 75)

    all_match = True
    for T in T_values:
        F0_boys = boys(0, T)
        nodes, weights = rys_nodes_weights(T, n_roots=1)
        F0_rys = 0.5 * weights[0]
        diff = abs(F0_boys - F0_rys)

        print(f"{T:>12.2e} {F0_boys:>20.15f} {F0_rys:>20.15f} {diff:>14.2e}")

        if diff > 1e-12:
            all_match = False

    print("-" * 75)
    print(f"F_0(T) agreement: {'PASS' if all_match else 'FAIL'}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Lab 5B Summary")
    print("=" * 75)
    print(f"F_0(T) Boys vs Rys agreement: {'PASS' if all_match else 'FAIL'}")
    print(f"Primitive ERI validation:     {'PASS' if prim_passed else 'FAIL'}")
    print("-" * 75)
    if all_match and prim_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED - Review output above")
    print("=" * 75)


if __name__ == "__main__":
    main()
