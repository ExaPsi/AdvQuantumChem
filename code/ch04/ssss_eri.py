#!/usr/bin/env python3
"""
ssss_eri.py - Primitive (ss|ss) ERI Calculation (Lab 4B)

This module implements the closed-form formula for the (ss|ss) primitive ERI:

    (ab|cd) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2) * F_0(T)

where:
    - p = alpha + beta, q = gamma + delta (composite exponents)
    - mu = alpha*beta/p, nu = gamma*delta/q (reduced exponents)
    - P = (alpha*A + beta*B)/p, Q = (gamma*C + delta*D)/q (composite centers)
    - rho = p*q/(p+q)
    - T = rho * |P - Q|^2

The formula is validated against PySCF.

References:
    - Chapter 4, Section 5: The Fundamental Primitive ERI
    - Szabo & Ostlund, Appendix A
    - Helgaker, Jorgensen, Olsen, Section 9.2

Course: 2302638 Advanced Quantum Chemistry
"""

import math
import numpy as np
from typing import Tuple

# Import Boys function from companion module
from boys_function import boys


def normalization_s(alpha: float) -> float:
    """
    Compute the normalization constant for an s-type Gaussian.

    N_s(alpha) = (2*alpha/pi)^{3/4}

    For a normalized primitive: phi(r) = N_s * exp(-alpha * |r - A|^2)

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


def gaussian_product_center(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray
) -> Tuple[float, np.ndarray, float]:
    """
    Apply the Gaussian Product Theorem.

    exp(-alpha|r-A|^2) * exp(-beta|r-B|^2) = K_AB * exp(-p|r-P|^2)

    where:
        p = alpha + beta
        P = (alpha*A + beta*B) / p
        K_AB = exp(-mu * |A-B|^2), mu = alpha*beta/p

    Parameters
    ----------
    alpha : float
        First exponent
    A : np.ndarray
        First center (3D vector)
    beta : float
        Second exponent
    B : np.ndarray
        Second center (3D vector)

    Returns
    -------
    p : float
        Composite exponent
    P : np.ndarray
        Composite center
    K_AB : float
        Exponential prefactor
    """
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.dot(A - B, A - B)
    K_AB = math.exp(-mu * R_AB_sq)

    return p, P, K_AB


def eri_ssss_unnormalized(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute an unnormalized primitive (ss|ss) ERI.

    (ab|cd) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu R_AB^2) * exp(-nu R_CD^2) * F_0(T)

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Gaussian exponents
    A, B, C, D : np.ndarray
        Gaussian centers (3D vectors)

    Returns
    -------
    float
        Unnormalized ERI value
    """
    # Apply GPT to bra pair
    p, P, K_AB = gaussian_product_center(alpha, A, beta, B)

    # Apply GPT to ket pair
    q, Q, K_CD = gaussian_product_center(gamma, C, delta, D)

    # Compute inter-pair parameters
    rho = p * q / (p + q)
    R_PQ_sq = np.dot(P - Q, P - Q)
    T = rho * R_PQ_sq

    # Prefactor
    prefactor = 2.0 * (math.pi ** 2.5) / (p * q * math.sqrt(p + q))

    # Boys function
    F0_T = boys(0, T)

    return prefactor * K_AB * K_CD * F0_T


def eri_ssss_normalized(
    alpha: float, A: np.ndarray,
    beta: float, B: np.ndarray,
    gamma: float, C: np.ndarray,
    delta: float, D: np.ndarray
) -> float:
    """
    Compute a normalized primitive (ss|ss) ERI.

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Gaussian exponents
    A, B, C, D : np.ndarray
        Gaussian centers (3D vectors)

    Returns
    -------
    float
        Normalized ERI value in Hartree
    """
    # Compute unnormalized ERI
    eri_unnorm = eri_ssss_unnormalized(alpha, A, beta, B, gamma, C, delta, D)

    # Apply normalization constants
    N_a = normalization_s(alpha)
    N_b = normalization_s(beta)
    N_c = normalization_s(gamma)
    N_d = normalization_s(delta)

    return N_a * N_b * N_c * N_d * eri_unnorm


def compute_eri_tensor(
    exponents: list,
    centers: list
) -> np.ndarray:
    """
    Compute the full ERI tensor for a set of s-type primitives.

    Parameters
    ----------
    exponents : list of float
        Gaussian exponents for each primitive
    centers : list of np.ndarray
        Centers for each primitive (3D vectors)

    Returns
    -------
    np.ndarray
        4D ERI tensor of shape (n, n, n, n)
    """
    n = len(exponents)
    eri = np.zeros((n, n, n, n))

    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    eri[mu, nu, lam, sig] = eri_ssss_normalized(
                        exponents[mu], centers[mu],
                        exponents[nu], centers[nu],
                        exponents[lam], centers[lam],
                        exponents[sig], centers[sig]
                    )

    return eri


# =============================================================================
# Validation against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate the analytic (ss|ss) ERI formula against PySCF.

    Creates a diatomic with one primitive s function per atom and
    compares all ERI tensor elements.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("Validation: Analytic (ss|ss) ERI vs PySCF")
    print("=" * 70)

    # Geometry in Bohr
    R = 1.4  # bond length
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    # Exponents
    alpha = 0.50  # exponent on atom 1
    beta = 0.40   # exponent on atom 2

    print(f"Geometry: H2 at R = {R} Bohr")
    print(f"Exponents: alpha = {alpha}, beta = {beta}")
    print("-" * 70)

    # Define one-primitive s basis on each atom in PySCF
    basA = gto.basis.parse(f"""
H S
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

    # Get PySCF ERIs
    eri_pyscf = mol.intor("int2e", aosym="s1")

    # Compute analytic ERIs
    exponents = [alpha, beta]
    centers = [A, B]
    eri_analytic = compute_eri_tensor(exponents, centers)

    # Compare all elements
    print(f"{'(i,j,k,l)':>12} {'PySCF':>18} {'Analytic':>18} {'Difference':>15}")
    print("-" * 70)

    max_diff = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    pyscf_val = eri_pyscf[i, j, k, l]
                    analytic_val = eri_analytic[i, j, k, l]
                    diff = abs(pyscf_val - analytic_val)
                    max_diff = max(max_diff, diff)

                    print(f"({i},{j},{k},{l})     {pyscf_val:>18.12f} {analytic_val:>18.12f} {diff:>15.2e}")

    print("-" * 70)
    print(f"Maximum difference: {max_diff:.2e}")

    success = max_diff < 1e-10
    if success:
        print("VALIDATION PASSED: Agreement to within 1e-10 Hartree")
    else:
        print("VALIDATION FAILED: Difference exceeds threshold")

    return success


def explore_distance_dependence():
    """
    Explore how ERIs depend on internuclear separation.

    Demonstrates that ERIs decrease as atoms separate, consistent with
    the 1/R decay from F_0(T) for large T.
    """
    print("\nERI dependence on internuclear distance")
    print("=" * 70)

    alpha = 0.5
    beta = 0.5
    A = np.array([0.0, 0.0, 0.0])

    print(f"Computing (00|11) for H2 with alpha = beta = {alpha}")
    print(f"{'R (Bohr)':>10} {'ERI (Hartree)':>18} {'T parameter':>15}")
    print("-" * 70)

    for R in [0.5, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0]:
        B = np.array([0.0, 0.0, R])

        # (00|11) means both electrons on atom 0, measuring interaction
        # with charge distribution on atoms 1
        eri_00_11 = eri_ssss_normalized(alpha, A, alpha, A, beta, B, beta, B)

        # Compute T parameter
        p = 2 * alpha
        q = 2 * beta
        P = A  # When both on same center
        Q = B
        rho = p * q / (p + q)
        T = rho * R**2

        print(f"{R:>10.2f} {eri_00_11:>18.12f} {T:>15.6f}")


def explore_exponent_dependence():
    """
    Explore how ERIs depend on Gaussian exponents.

    More diffuse functions (smaller exponents) give smaller ERIs
    because the charge distributions are more spread out.
    """
    print("\nERI dependence on exponents")
    print("=" * 70)

    R = 1.4
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    print(f"Computing (00|11) for H2 at R = {R} Bohr")
    print(f"{'alpha=beta':>12} {'ERI (Hartree)':>18}")
    print("-" * 70)

    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        eri_00_11 = eri_ssss_normalized(alpha, A, alpha, A, alpha, B, alpha, B)
        print(f"{alpha:>12.2f} {eri_00_11:>18.12f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Lab 4B: Primitive (ss|ss) ERI Calculation")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run validation against PySCF
    success = validate_against_pyscf()

    # Explore parameter dependence
    explore_distance_dependence()
    explore_exponent_dependence()

    print("\n" + "=" * 70)
    if success:
        print("All validations PASSED")
    else:
        print("Some validations FAILED (or PySCF not available)")
    print("=" * 70)
