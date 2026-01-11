#!/usr/bin/env python3
"""
Lab 3B Solution: Analytic vs PySCF for Custom s-Basis

This solution script implements the analytic formulas for one-electron
integrals using the Gaussian Product Theorem (GPT) and validates them
against PySCF to machine precision (~10^-15).

KEY EQUATIONS IMPLEMENTED:
==========================

1. Gaussian Product Theorem (GPT):
   exp(-alpha*|r-A|^2) * exp(-beta*|r-B|^2) = K_AB * exp(-p*|r-P|^2)

   where:
   - p = alpha + beta           (composite exponent)
   - mu = alpha*beta/p          (reduced exponent)
   - P = (alpha*A + beta*B)/p   (composite center)
   - K_AB = exp(-mu*R_AB^2)     (pre-exponential factor)

2. Overlap Integral (normalized s-type):
   S_ab = N(alpha) * N(beta) * (pi/p)^(3/2) * exp(-mu*R_AB^2)

   where N(alpha) = (2*alpha/pi)^(3/4)

3. Kinetic Energy Integral (s-type):
   T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab

   This formula follows from:
   <a|-1/2 nabla^2|b> = mu*(3 - 2*mu*R^2) * <a|b>

4. Nuclear Attraction Integral (s-type):
   V_ab = -Z * N(alpha) * N(beta) * (2*pi/p) * exp(-mu*R_AB^2) * F_0(T)

   where:
   - T = p * |P - C|^2     (Boys function argument)
   - F_0(T) = erf(sqrt(T)) * sqrt(pi/T) / 2  (Boys function)
   - C = nuclear position

Physical Insight:
-----------------
The Gaussian Product Theorem is the foundation of molecular integral
evaluation. It states that the product of two Gaussians centered at
different points is itself a Gaussian (at a new center). This allows
4-center integrals to be reduced to products of simpler terms.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
from typing import Tuple, List
import math
from pyscf import gto


# =============================================================================
# CONSTANTS AND NORMALIZATION
# =============================================================================

def norm_s(alpha: float) -> float:
    """
    Normalization constant for s-type primitive Gaussian.

    N(alpha) = (2*alpha/pi)^(3/4)

    This ensures <phi|phi> = 1 for phi(r) = N * exp(-alpha*r^2)

    Parameters
    ----------
    alpha : float
        Gaussian exponent (Bohr^-2)

    Returns
    -------
    float
        Normalization constant
    """
    return (2.0 * alpha / math.pi) ** 0.75


# =============================================================================
# GAUSSIAN PRODUCT THEOREM
# =============================================================================

def gaussian_product_params(alpha: float, A: np.ndarray,
                             beta: float, B: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
    """
    Compute Gaussian Product Theorem parameters.

    Given two Gaussians:
        G_a(r) = exp(-alpha*|r-A|^2)
        G_b(r) = exp(-beta*|r-B|^2)

    Their product is:
        G_a * G_b = K_AB * exp(-p*|r-P|^2)

    Parameters
    ----------
    alpha : float
        Exponent of first Gaussian
    A : np.ndarray
        Center of first Gaussian (3D coordinates in Bohr)
    beta : float
        Exponent of second Gaussian
    B : np.ndarray
        Center of second Gaussian (3D coordinates in Bohr)

    Returns
    -------
    p : float
        Composite exponent: p = alpha + beta
    mu : float
        Reduced exponent: mu = alpha*beta/(alpha+beta)
    P : np.ndarray
        Composite center: P = (alpha*A + beta*B)/(alpha+beta)
    K_AB : float
        Pre-exponential factor: K_AB = exp(-mu*R_AB^2)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Composite exponent
    p = alpha + beta

    # Reduced exponent (analogous to reduced mass in two-body problem)
    mu = alpha * beta / p

    # Composite center (weighted average)
    P = (alpha * A + beta * B) / p

    # Pre-exponential factor
    R_AB = np.linalg.norm(A - B)
    K_AB = math.exp(-mu * R_AB * R_AB)

    return p, mu, P, K_AB


# =============================================================================
# BOYS FUNCTION
# =============================================================================

def boys_F0(T: float) -> float:
    """
    Boys function F_0(T).

    F_0(T) = integral from 0 to 1 of exp(-T*t^2) dt

    For T > 0:
        F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))

    For T = 0:
        F_0(0) = 1

    The Boys function appears in all nuclear attraction and electron
    repulsion integrals involving Gaussian basis functions. It arises
    from the integral over the Coulomb operator 1/r.

    Parameters
    ----------
    T : float
        Argument of Boys function (non-negative)

    Returns
    -------
    float
        Value of F_0(T)
    """
    if T < 1e-12:
        # Taylor series for small T: F_0(T) = 1 - T/3 + T^2/10 - ...
        return 1.0 - T / 3.0 + T * T / 10.0 - T * T * T / 42.0

    sqrt_T = math.sqrt(T)
    return 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)


# =============================================================================
# OVERLAP INTEGRAL
# =============================================================================

def overlap_ss(alpha: float, A: np.ndarray,
               beta: float, B: np.ndarray) -> float:
    """
    Normalized s-s overlap integral.

    S_ab = N(alpha) * N(beta) * (pi/p)^(3/2) * exp(-mu*R_AB^2)

    This is the fundamental integral that measures how much two
    basis functions "overlap" in space. It defines the metric
    tensor for the non-orthogonal AO basis.

    Derivation:
    -----------
    <a|b> = integral of G_a(r) * G_b(r) d^3r
          = K_AB * integral of exp(-p*|r-P|^2) d^3r
          = K_AB * (pi/p)^(3/2)
          = (pi/p)^(3/2) * exp(-mu*R_AB^2)

    Parameters
    ----------
    alpha : float
        Exponent of first Gaussian (Bohr^-2)
    A : np.ndarray
        Center of first Gaussian (Bohr)
    beta : float
        Exponent of second Gaussian (Bohr^-2)
    B : np.ndarray
        Center of second Gaussian (Bohr)

    Returns
    -------
    float
        Overlap integral value
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Get GPT parameters
    p, mu, P, K_AB = gaussian_product_params(alpha, A, beta, B)

    # Normalization constants
    N_a = norm_s(alpha)
    N_b = norm_s(beta)

    # Unnormalized overlap (from Gaussian integral formula)
    S_unnorm = (math.pi / p) ** 1.5 * K_AB

    # Normalized overlap
    return N_a * N_b * S_unnorm


# =============================================================================
# KINETIC ENERGY INTEGRAL
# =============================================================================

def kinetic_ss(alpha: float, A: np.ndarray,
               beta: float, B: np.ndarray) -> float:
    """
    Normalized s-s kinetic energy integral.

    T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab

    This elegant relation arises from applying the kinetic operator
    -1/2 nabla^2 to the ket:

    <a|-1/2 nabla^2|b> = <a| -1/2 nabla^2 [N_b * exp(-beta*|r-B|^2)] |>

    After differentiation and integration using GPT:
    T_ab / S_ab = mu * (3 - 2*mu*R_AB^2)

    Physical interpretation:
    - mu = reduced exponent = "effective curvature"
    - Factor of 3: from three spatial dimensions
    - Term -2*mu*R_AB^2: correction for displaced centers

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss

    Returns
    -------
    float
        Kinetic energy integral value (Hartree)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Get GPT parameters
    p = alpha + beta
    mu = alpha * beta / p
    R_AB_sq = np.sum((A - B) ** 2)

    # Get overlap
    S = overlap_ss(alpha, A, beta, B)

    # Kinetic from the T/S relation
    T_over_S = mu * (3.0 - 2.0 * mu * R_AB_sq)

    return T_over_S * S


# =============================================================================
# NUCLEAR ATTRACTION INTEGRAL
# =============================================================================

def nuclear_attraction_ss_single(alpha: float, A: np.ndarray,
                                  beta: float, B: np.ndarray,
                                  Z: float, C: np.ndarray) -> float:
    """
    s-s nuclear attraction integral for a single nucleus.

    V_ab(C) = -Z * N(alpha) * N(beta) * (2*pi/p) * exp(-mu*R_AB^2) * F_0(T)

    where T = p * |P - C|^2

    The Boys function F_0(T) arises from integrating the Coulomb
    operator 1/|r-C| over the product Gaussian distribution.

    Physical interpretation:
    - Z: nuclear charge (protons)
    - The minus sign indicates attraction (V < 0)
    - F_0(T) decreases as T increases (distant nuclei contribute less)
    - F_0(0) = 1 when P = C (electron on top of nucleus)

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss
    Z : float
        Nuclear charge
    C : np.ndarray
        Nuclear position (Bohr)

    Returns
    -------
    float
        Nuclear attraction integral (Hartree, negative)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    # Get GPT parameters
    p, mu, P, K_AB = gaussian_product_params(alpha, A, beta, B)

    # Normalization constants
    N_a = norm_s(alpha)
    N_b = norm_s(beta)

    # Boys function argument: T = p * |P - C|^2
    PC_sq = np.sum((P - C) ** 2)
    T = p * PC_sq

    # Prefactor for nuclear attraction
    prefactor = N_a * N_b * (2.0 * math.pi / p) * K_AB

    # Nuclear attraction (negative for attraction)
    return -Z * prefactor * boys_F0(T)


def nuclear_attraction_ss(alpha: float, A: np.ndarray,
                           beta: float, B: np.ndarray,
                           nuclei: List[Tuple[float, np.ndarray]]) -> float:
    """
    Total s-s nuclear attraction integral summed over all nuclei.

    V_ab = sum_C V_ab(C) = sum_C <a| -Z_C/|r-C| |b>

    This is the full nuclear attraction matrix element including
    contributions from all nuclei in the molecule.

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss
    nuclei : list of (Z, C) tuples
        List of (charge, position) for each nucleus

    Returns
    -------
    float
        Total nuclear attraction integral (Hartree, negative)
    """
    V_total = 0.0
    for Z, C in nuclei:
        V_total += nuclear_attraction_ss_single(alpha, A, beta, B, Z, np.asarray(C))
    return V_total


# =============================================================================
# VALIDATION AGAINST PYSCF
# =============================================================================

def compare_with_pyscf(alpha: float, A: np.ndarray,
                        beta: float, B: np.ndarray,
                        nuclei: List[Tuple[float, np.ndarray]],
                        verbose: bool = True) -> dict:
    """
    Compare analytic integrals against PySCF.

    Creates a custom PySCF molecule with single-primitive s-type
    basis functions and compares our analytic formulas.

    Parameters
    ----------
    alpha, A : exponent and center of first Gaussian
    beta, B : exponent and center of second Gaussian
    nuclei : list of (Z, C) tuples for nuclear charges
    verbose : bool
        Print detailed comparison

    Returns
    -------
    dict
        Comparison results including differences
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Create custom basis for PySCF
    # Format: element, angular momentum, [(exponent, coefficient), ...]
    basA = gto.basis.parse(f"""
H S
  {alpha:.15f}  1.0
""")
    basB = gto.basis.parse(f"""
H S
  {beta:.15f}  1.0
""")

    # Build atom string
    atoms_str = f"H@1 {A[0]:.10f} {A[1]:.10f} {A[2]:.10f}; "
    atoms_str += f"H@2 {B[0]:.10f} {B[1]:.10f} {B[2]:.10f}"

    # Create molecule
    mol = gto.M(
        atom=atoms_str,
        basis={"H@1": basA, "H@2": basB},
        unit="Bohr",
        verbose=0
    )

    # Extract PySCF integrals
    S_pyscf = mol.intor("int1e_ovlp")
    T_pyscf = mol.intor("int1e_kin")
    V_pyscf = mol.intor("int1e_nuc")

    # Compute analytic integrals
    S11_analytic = overlap_ss(alpha, A, alpha, A)
    S22_analytic = overlap_ss(beta, B, beta, B)
    S12_analytic = overlap_ss(alpha, A, beta, B)

    T11_analytic = kinetic_ss(alpha, A, alpha, A)
    T22_analytic = kinetic_ss(beta, B, beta, B)
    T12_analytic = kinetic_ss(alpha, A, beta, B)

    V12_analytic = nuclear_attraction_ss(alpha, A, beta, B, nuclei)

    # Compute differences
    results = {
        'S11': {'pyscf': S_pyscf[0, 0], 'analytic': S11_analytic,
                'diff': abs(S_pyscf[0, 0] - S11_analytic)},
        'S22': {'pyscf': S_pyscf[1, 1], 'analytic': S22_analytic,
                'diff': abs(S_pyscf[1, 1] - S22_analytic)},
        'S12': {'pyscf': S_pyscf[0, 1], 'analytic': S12_analytic,
                'diff': abs(S_pyscf[0, 1] - S12_analytic)},
        'T11': {'pyscf': T_pyscf[0, 0], 'analytic': T11_analytic,
                'diff': abs(T_pyscf[0, 0] - T11_analytic)},
        'T22': {'pyscf': T_pyscf[1, 1], 'analytic': T22_analytic,
                'diff': abs(T_pyscf[1, 1] - T22_analytic)},
        'T12': {'pyscf': T_pyscf[0, 1], 'analytic': T12_analytic,
                'diff': abs(T_pyscf[0, 1] - T12_analytic)},
        'V12': {'pyscf': V_pyscf[0, 1], 'analytic': V12_analytic,
                'diff': abs(V_pyscf[0, 1] - V12_analytic)},
    }

    if verbose:
        print("\n   Comparison of analytic vs PySCF integrals:")
        print("   " + "-" * 55)
        print(f"   {'Integral':8s}  {'PySCF':>20s}  {'Analytic':>20s}  {'Diff':>10s}")
        print("   " + "-" * 55)
        for name, vals in results.items():
            print(f"   {name:8s}  {vals['pyscf']:20.15f}  {vals['analytic']:20.15f}  {vals['diff']:.2e}")

    return results


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("Lab 3B Solution: Analytic vs PySCF for Custom s-Basis")
    print("=" * 70)

    # =========================================================================
    # SECTION 1: Define the test system
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. TEST SYSTEM SETUP")
    print("-" * 70)

    # H2 molecule geometry (in Bohr)
    R_bond = 1.4  # Bohr (approximately 0.74 Angstrom)
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R_bond])

    # Use different exponents for asymmetric test
    alpha = 0.50  # Bohr^-2
    beta = 0.40   # Bohr^-2

    print(f"\n   Molecule: H2")
    print(f"\n   Geometry (Bohr):")
    print(f"      H_1 at A = {A}")
    print(f"      H_2 at B = {B}")
    print(f"      Bond length R = {R_bond:.4f} Bohr = {R_bond * 0.529177:.4f} Angstrom")

    print(f"\n   Basis parameters:")
    print(f"      alpha = {alpha:.2f} Bohr^-2  (tighter)")
    print(f"      beta  = {beta:.2f} Bohr^-2  (more diffuse)")

    print(f"\n   Normalization constants:")
    print(f"      N(alpha) = {norm_s(alpha):.10f}")
    print(f"      N(beta)  = {norm_s(beta):.10f}")

    # =========================================================================
    # SECTION 2: Gaussian Product Theorem parameters
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. GAUSSIAN PRODUCT THEOREM PARAMETERS")
    print("-" * 70)

    p, mu, P, K_AB = gaussian_product_params(alpha, A, beta, B)

    print(f"\n   For alpha={alpha}, beta={beta}, R={R_bond}:")
    print(f"\n   Composite exponent:")
    print(f"      p = alpha + beta = {p:.6f} Bohr^-2")

    print(f"\n   Reduced exponent (like reduced mass):")
    print(f"      mu = alpha*beta/p = {mu:.6f} Bohr^-2")

    print(f"\n   Composite center (weighted average):")
    print(f"      P = (alpha*A + beta*B)/p")
    print(f"        = ({alpha:.2f}*{A} + {beta:.2f}*{B}) / {p:.2f}")
    print(f"        = {P}")

    print(f"\n   Pre-exponential factor:")
    print(f"      K_AB = exp(-mu*R^2) = exp(-{mu:.4f}*{R_bond**2:.4f})")
    print(f"           = {K_AB:.10f}")

    # =========================================================================
    # SECTION 3: Overlap integral derivation
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. OVERLAP INTEGRAL")
    print("-" * 70)

    S_12 = overlap_ss(alpha, A, beta, B)
    S_11 = overlap_ss(alpha, A, alpha, A)
    S_22 = overlap_ss(beta, B, beta, B)

    print("\n   Formula: S_ab = N(a)*N(b) * (pi/p)^(3/2) * exp(-mu*R^2)")
    print(f"\n   Step-by-step for S_12:")
    print(f"      N(alpha)*N(beta) = {norm_s(alpha)*norm_s(beta):.10f}")
    print(f"      (pi/p)^(3/2) = {(math.pi/p)**1.5:.10f}")
    print(f"      exp(-mu*R^2) = {K_AB:.10f}")
    print(f"      S_12 = {S_12:.15f}")

    print(f"\n   Results:")
    print(f"      S_11 = {S_11:.15f}  (should be 1.0: self-overlap)")
    print(f"      S_22 = {S_22:.15f}  (should be 1.0: self-overlap)")
    print(f"      S_12 = {S_12:.15f}  (off-diagonal overlap)")

    # =========================================================================
    # SECTION 4: Kinetic energy integral
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. KINETIC ENERGY INTEGRAL")
    print("-" * 70)

    T_12 = kinetic_ss(alpha, A, beta, B)
    T_11 = kinetic_ss(alpha, A, alpha, A)
    T_22 = kinetic_ss(beta, B, beta, B)

    print("\n   Formula: T_ab = mu * (3 - 2*mu*R^2) * S_ab")

    R_sq = R_bond ** 2
    factor = mu * (3.0 - 2.0 * mu * R_sq)

    print(f"\n   Step-by-step for T_12:")
    print(f"      mu = {mu:.6f}")
    print(f"      R^2 = {R_sq:.6f}")
    print(f"      2*mu*R^2 = {2*mu*R_sq:.6f}")
    print(f"      (3 - 2*mu*R^2) = {3 - 2*mu*R_sq:.6f}")
    print(f"      mu*(3 - 2*mu*R^2) = {factor:.6f}")
    print(f"      T_12 = factor * S_12 = {T_12:.15f}")

    print(f"\n   Results:")
    print(f"      T_11 = {T_11:.15f}  (larger exponent -> higher kinetic)")
    print(f"      T_22 = {T_22:.15f}  (smaller exponent -> lower kinetic)")
    print(f"      T_12 = {T_12:.15f}")

    print(f"\n   Physical insight:")
    print(f"      T_11 / T_22 = {T_11/T_22:.3f}")
    print(f"      Tighter orbitals (larger alpha) have more curvature")
    print(f"      and thus higher kinetic energy (uncertainty principle)")

    # =========================================================================
    # SECTION 5: Nuclear attraction integral
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. NUCLEAR ATTRACTION INTEGRAL")
    print("-" * 70)

    # Define nuclei (H2: two protons)
    nuclei = [(1.0, A), (1.0, B)]

    V_12 = nuclear_attraction_ss(alpha, A, beta, B, nuclei)
    V_12_A = nuclear_attraction_ss_single(alpha, A, beta, B, 1.0, A)
    V_12_B = nuclear_attraction_ss_single(alpha, A, beta, B, 1.0, B)

    print("\n   Formula: V_ab = -Z * N(a)*N(b) * (2*pi/p) * exp(-mu*R^2) * F_0(T)")
    print("            where T = p * |P - C|^2")

    # Boys function arguments for each nucleus
    T_A = p * np.sum((P - A) ** 2)
    T_B = p * np.sum((P - B) ** 2)

    print(f"\n   For nucleus at A:")
    print(f"      |P - A|^2 = {np.sum((P-A)**2):.6f}")
    print(f"      T = p*|P-A|^2 = {T_A:.6f}")
    print(f"      F_0(T) = {boys_F0(T_A):.10f}")
    print(f"      V_12(A) = {V_12_A:.15f}")

    print(f"\n   For nucleus at B:")
    print(f"      |P - B|^2 = {np.sum((P-B)**2):.6f}")
    print(f"      T = p*|P-B|^2 = {T_B:.6f}")
    print(f"      F_0(T) = {boys_F0(T_B):.10f}")
    print(f"      V_12(B) = {V_12_B:.15f}")

    print(f"\n   Total: V_12 = V_12(A) + V_12(B) = {V_12:.15f}")
    print(f"\n   Sign check: V_12 < 0? {V_12 < 0} (should be negative for attraction)")

    # =========================================================================
    # SECTION 6: Validation against PySCF
    # =========================================================================
    print("\n" + "-" * 70)
    print("6. VALIDATION AGAINST PYSCF")
    print("-" * 70)

    results = compare_with_pyscf(alpha, A, beta, B, nuclei, verbose=True)

    # Check all differences are at machine precision
    all_pass = all(r['diff'] < 1e-13 for r in results.values())

    print("\n   Validation summary:")
    max_diff = max(r['diff'] for r in results.values())
    print(f"      Maximum difference: {max_diff:.2e}")
    print(f"      All integrals match to ~10^-15: {all_pass}")

    if all_pass:
        print("\n   VALIDATION PASSED: Analytic formulas agree with PySCF!")
    else:
        print("\n   VALIDATION FAILED: Check implementation!")

    # =========================================================================
    # SECTION 7: Physical insights
    # =========================================================================
    print("\n" + "-" * 70)
    print("7. PHYSICAL INSIGHTS")
    print("-" * 70)

    print("""
   KEY OBSERVATIONS:
   -----------------

   1. Overlap decay:
      - S_12 = {:.6f} < 1 due to spatial separation
      - Decays as exp(-mu*R^2): tighter orbitals (larger mu) decay faster

   2. Kinetic-overlap relation:
      - T_ab = mu*(3 - 2*mu*R^2) * S_ab
      - The factor (3 - 2*mu*R^2) can become negative for large R!
      - This doesn't violate physics: T is always positive semidefinite
        as a matrix (sum of squared terms), but individual matrix
        elements can be negative.

   3. Nuclear attraction:
      - V < 0 always (attractive potential)
      - The Boys function F_0(T) modulates the Coulomb interaction:
        * F_0(0) = 1: maximum attraction when electron on nucleus
        * F_0(T) -> 0 as T -> infinity: distant nucleus contributes less

   4. Why Gaussians?
      - The Gaussian Product Theorem makes integrals analytically tractable
      - Product of two Gaussians is a Gaussian at a new center
      - This property does NOT hold for Slater orbitals (exp(-zeta*r))

   5. Numerical precision:
      - Our simple formulas achieve machine precision (~10^-15)
      - This validates both the formulas and our implementation
      - PySCF uses the same fundamental formulas (libcint library)
""".format(S_12))

    print("=" * 70)
    print("Lab 3B Solution Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
