#!/usr/bin/env python3
"""
Rys Quadrature ERI Example: Complete (ss|ss) Integral Evaluation

This module demonstrates the use of Rys quadrature to compute two-electron
repulsion integrals (ERIs). We focus on the (ss|ss) case, which is the
simplest ERI involving four s-type Gaussian primitives.

The (ss|ss) integral in chemists' notation is:
    (ab|cd) = integral integral chi_a(r1) chi_b(r1) (1/r12) chi_c(r2) chi_d(r2) dr1 dr2

For s-type Gaussians centered at A, B, C, D with exponents alpha, beta, gamma, delta:

    (ab|cd) = (2*pi^(5/2)) / (p*q*sqrt(p+q)) * exp(-mu*R_AB^2) * exp(-nu*R_CD^2) * F_0(T)

where:
    p = alpha + beta,  mu = alpha*beta / p,  P = (alpha*A + beta*B) / p
    q = gamma + delta, nu = gamma*delta / q, Q = (gamma*C + delta*D) / q
    rho = p*q / (p+q)
    T = rho * |P - Q|^2

Key insight: F_0(T) = (1/2) * sum_i W_i provides the Boys function via Rys quadrature.

This module compares three approaches:
    1. Direct Boys function evaluation
    2. Rys quadrature with one root
    3. PySCF reference calculation

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix E: Rys Quadrature Reference Notes
"""

import numpy as np
import sys
import os
from typing import Tuple

# Import from local modules
sys.path.insert(0, os.path.dirname(__file__))
from rys_quadrature import rys_roots_weights, root_count_for_angular_momentum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'appendix_d'))
from boys_function import boys

# Try to import PySCF for reference calculations
try:
    from pyscf import gto
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
    print("Warning: PySCF not available. Reference calculations will be skipped.")


def gaussian_product_theorem(alpha: float, A: np.ndarray,
                              beta: float, B: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Apply Gaussian Product Theorem to two s-type primitives.

    exp(-alpha*|r-A|^2) * exp(-beta*|r-B|^2)
        = exp(-p*|r-P|^2) * exp(-mu*R_AB^2)

    Args:
        alpha: Exponent of first Gaussian
        A: Center of first Gaussian (3-vector)
        beta: Exponent of second Gaussian
        B: Center of second Gaussian (3-vector)

    Returns:
        Tuple (p, mu, P) where:
            p: Combined exponent (alpha + beta)
            mu: Reduced exponent (alpha*beta/p)
            P: Combined center ((alpha*A + beta*B)/p)
    """
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p
    return p, mu, P


def eri_ssss_direct(alpha: float, A: np.ndarray,
                    beta: float, B: np.ndarray,
                    gamma: float, C: np.ndarray,
                    delta: float, D: np.ndarray,
                    normalized: bool = True) -> float:
    """
    Compute (ss|ss) ERI using direct Boys function evaluation.

    The unnormalized primitive ERI is:
        (ab|cd) = (2*pi^(5/2)) / (p*q*sqrt(p+q)) * exp(-mu*R_AB^2) * exp(-nu*R_CD^2) * F_0(T)

    For normalized primitives, multiply by the four normalization constants:
        N(alpha) = (2*alpha/pi)^(3/4)

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3-vectors)
        normalized: If True, use normalized primitives

    Returns:
        The (ss|ss) electron repulsion integral
    """
    # Apply GPT to bra (ab) and ket (cd)
    p, mu, P = gaussian_product_theorem(alpha, A, beta, B)
    q, nu, Q = gaussian_product_theorem(gamma, C, delta, D)

    # Compute intermediate quantities
    R_AB_sq = np.sum((A - B) ** 2)
    R_CD_sq = np.sum((C - D) ** 2)
    R_PQ_sq = np.sum((P - Q) ** 2)

    rho = p * q / (p + q)
    T = rho * R_PQ_sq

    # Prefactor
    prefactor = 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))

    # Exponential factors from GPT
    exp_AB = np.exp(-mu * R_AB_sq)
    exp_CD = np.exp(-nu * R_CD_sq)

    # Boys function
    F0 = boys(0, T)

    # Unnormalized primitive ERI
    eri = prefactor * exp_AB * exp_CD * F0

    # Apply normalization if requested
    if normalized:
        # N_s(alpha) = (2*alpha/pi)^(3/4)
        N_a = (2 * alpha / np.pi) ** 0.75
        N_b = (2 * beta / np.pi) ** 0.75
        N_c = (2 * gamma / np.pi) ** 0.75
        N_d = (2 * delta / np.pi) ** 0.75
        eri *= N_a * N_b * N_c * N_d

    return eri


def eri_ssss_rys(alpha: float, A: np.ndarray,
                 beta: float, B: np.ndarray,
                 gamma: float, C: np.ndarray,
                 delta: float, D: np.ndarray,
                 normalized: bool = True) -> float:
    """
    Compute (ss|ss) ERI using Rys quadrature.

    For (ss|ss) integrals, L_total = 0, so n_roots = 1.
    The Rys quadrature gives:
        F_0(T) = (1/2) * sum_i W_i * x_i^0 = (1/2) * sum_i W_i = (1/2) * W_1

    For one root: W_1 = m_0 = 2*F_0(T), so F_0 = W_1/2 (consistent!)

    This demonstrates that Rys quadrature reproduces the direct Boys result.
    For higher angular momentum, Rys quadrature enables efficient evaluation
    of multiple F_n values simultaneously.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3-vectors)
        normalized: If True, use normalized primitives

    Returns:
        The (ss|ss) electron repulsion integral
    """
    # Apply GPT to bra (ab) and ket (cd)
    p, mu, P = gaussian_product_theorem(alpha, A, beta, B)
    q, nu, Q = gaussian_product_theorem(gamma, C, delta, D)

    # Compute intermediate quantities
    R_AB_sq = np.sum((A - B) ** 2)
    R_CD_sq = np.sum((C - D) ** 2)
    R_PQ_sq = np.sum((P - Q) ** 2)

    rho = p * q / (p + q)
    T = rho * R_PQ_sq

    # Prefactor
    prefactor = 2 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))

    # Exponential factors from GPT
    exp_AB = np.exp(-mu * R_AB_sq)
    exp_CD = np.exp(-nu * R_CD_sq)

    # Rys quadrature for F_0(T)
    # For (ss|ss), L = 0, so n_roots = 1
    n_roots = root_count_for_angular_momentum(0)  # = 1
    _nodes, weights = rys_roots_weights(T, n_roots)  # nodes unused for n=0

    # F_0(T) = (1/2) * sum_i W_i * x_i^0 = (1/2) * sum_i W_i
    F0_rys = 0.5 * np.sum(weights)

    # Unnormalized primitive ERI
    eri = prefactor * exp_AB * exp_CD * F0_rys

    # Apply normalization if requested
    if normalized:
        N_a = (2 * alpha / np.pi) ** 0.75
        N_b = (2 * beta / np.pi) ** 0.75
        N_c = (2 * gamma / np.pi) ** 0.75
        N_d = (2 * delta / np.pi) ** 0.75
        eri *= N_a * N_b * N_c * N_d

    return eri


def eri_pyscf_reference(alpha: float, A: np.ndarray,
                        beta: float, B: np.ndarray,
                        gamma: float, C: np.ndarray,
                        delta: float, D: np.ndarray) -> float:
    """
    Compute (ss|ss) ERI using PySCF as reference.

    Creates a molecule with four hydrogen atoms at the Gaussian centers,
    each with a single s-type primitive having the specified exponent.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers (3-vectors) in Bohr

    Returns:
        The (ss|ss) electron repulsion integral from PySCF
    """
    if not HAS_PYSCF:
        return np.nan

    # Build custom basis with single primitives
    # Each "atom" has one s function with specified exponent
    # PySCF expects coefficients to be normalized, so we use 1.0
    custom_basis = {
        'H1': [[0, [alpha, 1.0]]],  # s function on center A
        'H2': [[0, [beta, 1.0]]],   # s function on center B
        'H3': [[0, [gamma, 1.0]]],  # s function on center C
        'H4': [[0, [delta, 1.0]]],  # s function on center D
    }

    # Create molecule with four "atoms" at specified centers
    atom_str = f"H1 {A[0]} {A[1]} {A[2]}; H2 {B[0]} {B[1]} {B[2]}; "
    atom_str += f"H3 {C[0]} {C[1]} {C[2]}; H4 {D[0]} {D[1]} {D[2]}"

    mol = gto.M(
        atom=atom_str,
        basis=custom_basis,
        unit="Bohr",
        verbose=0
    )

    # Get the full ERI tensor (4x4x4x4 for 4 basis functions)
    eri = mol.intor("int2e", aosym="s1")

    # We want (0 0 | 0 0) in chemist's notation, which is eri[0,0,0,0]
    # But our functions are on different atoms, so we need (0,1|2,3)
    # In PySCF ordering: functions are 0=A, 1=B, 2=C, 3=D
    # Chemist's (ab|cd) = eri[a,b,c,d]
    return eri[0, 1, 2, 3]


def compare_methods_single(alpha: float, A: np.ndarray,
                            beta: float, B: np.ndarray,
                            gamma: float, C: np.ndarray,
                            delta: float, D: np.ndarray,
                            label: str = ""):
    """
    Compare all three ERI evaluation methods for a single integral.

    Args:
        alpha, beta, gamma, delta: Gaussian exponents
        A, B, C, D: Gaussian centers
        label: Optional label for output
    """
    print(f"\nTest case: {label}" if label else "\nTest case:")
    print("-" * 60)

    # Method 1: Direct Boys
    eri_direct = eri_ssss_direct(alpha, A, beta, B, gamma, C, delta, D)
    print(f"Direct Boys:  {eri_direct:20.14f}")

    # Method 2: Rys quadrature
    eri_rys = eri_ssss_rys(alpha, A, beta, B, gamma, C, delta, D)
    print(f"Rys (1 root): {eri_rys:20.14f}")

    # Method 3: PySCF reference
    if HAS_PYSCF:
        eri_pyscf = eri_pyscf_reference(alpha, A, beta, B, gamma, C, delta, D)
        print(f"PySCF:        {eri_pyscf:20.14f}")
    else:
        eri_pyscf = np.nan

    # Comparisons
    print("\nComparisons:")
    print(f"  |Direct - Rys|:   {abs(eri_direct - eri_rys):.2e}")
    if HAS_PYSCF:
        print(f"  |Direct - PySCF|: {abs(eri_direct - eri_pyscf):.2e}")
        print(f"  |Rys - PySCF|:    {abs(eri_rys - eri_pyscf):.2e}")


def test_eri_symmetry():
    """
    Test ERI symmetry properties.

    The (ss|ss) integral has 8-fold symmetry:
        (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc)
               = (cd|ab) = (dc|ab) = (cd|ba) = (dc|ba)

    For s-type Gaussians with different centers, some of these become
    distinct expressions, but all should give the same value.
    """
    print("\n" + "=" * 70)
    print("ERI Symmetry Tests")
    print("=" * 70)

    # Test parameters
    alpha, beta, gamma, delta = 0.5, 0.8, 0.6, 0.7
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    C = np.array([0.0, 1.0, 0.0])
    D = np.array([0.5, 0.5, 0.5])

    # Compute various permutations
    eri_abcd = eri_ssss_direct(alpha, A, beta, B, gamma, C, delta, D)
    eri_bacd = eri_ssss_direct(beta, B, alpha, A, gamma, C, delta, D)
    eri_abdc = eri_ssss_direct(alpha, A, beta, B, delta, D, gamma, C)
    eri_cdab = eri_ssss_direct(gamma, C, delta, D, alpha, A, beta, B)
    eri_dcba = eri_ssss_direct(delta, D, gamma, C, beta, B, alpha, A)

    print("\nSymmetry verification:")
    print("-" * 60)
    print(f"(ab|cd) = {eri_abcd:.14f}")
    print(f"(ba|cd) = {eri_bacd:.14f}  diff = {abs(eri_abcd - eri_bacd):.2e}")
    print(f"(ab|dc) = {eri_abdc:.14f}  diff = {abs(eri_abcd - eri_abdc):.2e}")
    print(f"(cd|ab) = {eri_cdab:.14f}  diff = {abs(eri_abcd - eri_cdab):.2e}")
    print(f"(dc|ba) = {eri_dcba:.14f}  diff = {abs(eri_abcd - eri_dcba):.2e}")

    # All should be equal within numerical precision
    all_equal = (
        np.isclose(eri_abcd, eri_bacd, atol=1e-14) and
        np.isclose(eri_abcd, eri_abdc, atol=1e-14) and
        np.isclose(eri_abcd, eri_cdab, atol=1e-14) and
        np.isclose(eri_abcd, eri_dcba, atol=1e-14)
    )
    print(f"\nAll permutations equal: {all_equal}")


def test_eri_limiting_cases():
    """
    Test ERI behavior in limiting cases.

    1. Same center for all Gaussians: T = 0, uses F_0(0) = 1
    2. Large separation: ERI -> 0 exponentially
    3. Very diffuse Gaussians: integral approaches 1/R
    """
    print("\n" + "=" * 70)
    print("ERI Limiting Cases")
    print("=" * 70)

    # Case 1: All Gaussians at same center
    print("\n[1] All Gaussians at same center (T = 0):")
    print("-" * 50)
    alpha = beta = gamma = delta = 1.0
    A = B = C = D = np.array([0.0, 0.0, 0.0])

    eri = eri_ssss_direct(alpha, A, beta, B, gamma, C, delta, D)
    print(f"(aa|aa) with alpha=1.0: {eri:.14f}")

    # Analytical for this case (all same exponent, same center):
    # (aa|aa) = N^4 * (2*pi^2.5)/(p*q*sqrt(p+q)) * F_0(0)
    # where p = q = 2*alpha, F_0(0) = 1
    N = (2 * alpha / np.pi) ** 0.75
    p = 2 * alpha
    analytical = N**4 * 2 * np.pi**2.5 / (p * p * np.sqrt(2 * p)) * 1.0
    print(f"Analytical:             {analytical:.14f}")
    print(f"Difference:             {abs(eri - analytical):.2e}")

    # Case 2: Large separation
    print("\n[2] Large separation between bra and ket:")
    print("-" * 50)
    alpha = beta = gamma = delta = 1.0
    A = B = np.array([0.0, 0.0, 0.0])
    separations = [1.0, 2.0, 5.0, 10.0]

    print(f"{'R':>8}  {'(ab|cd)':>18}  {'Decay':>12}")
    print("-" * 45)
    prev_eri = None
    for R in separations:
        C = D = np.array([R, 0.0, 0.0])
        eri = eri_ssss_direct(alpha, A, beta, B, gamma, C, delta, D)
        if prev_eri is not None and prev_eri > 0:
            decay = np.log(prev_eri / eri) if eri > 0 else np.inf
        else:
            decay = 0
        print(f"{R:8.1f}  {eri:18.12f}  {decay:12.4f}")
        prev_eri = eri

    # Case 3: Limit as exponents -> 0 (classical point charges)
    print("\n[3] Diffuse limit (small exponents, fixed R):")
    print("-" * 50)
    A = B = np.array([0.0, 0.0, 0.0])
    C = D = np.array([2.0, 0.0, 0.0])
    R = 2.0  # |P - Q| for this geometry

    print("As exponents decrease, (ab|cd)*sqrt(prod(exponents)) -> 1/R")
    print(f"1/R = {1/R:.10f}")
    print()
    print(f"{'alpha':>10}  {'ERI':>18}  {'ERI*sqrt(prod)':>18}")
    print("-" * 55)

    for exp_val in [1.0, 0.1, 0.01, 0.001]:
        eri = eri_ssss_direct(exp_val, A, exp_val, B, exp_val, C, exp_val, D,
                              normalized=False)
        # For unnormalized Gaussians in the diffuse limit
        # the ERI times (16*alpha^4)^(1/2) should approach 1/R
        scale = (16 * exp_val**4) ** 0.5
        scaled = eri * scale
        print(f"{exp_val:10.4f}  {eri:18.10f}  {scaled:18.10f}")


def compare_pyscf_h2_eri():
    """
    Compare our ERI implementation with PySCF for H2 in STO-3G basis.

    STO-3G for hydrogen uses 3 primitives contracted into 1 function.
    We compare individual primitive integrals.
    """
    if not HAS_PYSCF:
        print("\nPySCF not available - skipping H2 comparison")
        return

    print("\n" + "=" * 70)
    print("Comparison with PySCF H2 STO-3G")
    print("=" * 70)

    # H2 geometry
    R_HH = 1.4  # Bohr
    A = np.array([0.0, 0.0, 0.0])
    _B = np.array([0.0, 0.0, R_HH])  # Second H position (used in mol definition)

    # STO-3G exponents for H (from standard basis sets)
    # Each s orbital is a contraction of 3 primitives
    sto3g_exponents = [3.42525091, 0.62391373, 0.16885540]
    sto3g_coeffs = [0.15432897, 0.53532814, 0.44463454]

    print("\nSTO-3G primitive exponents:", sto3g_exponents)
    print("Contraction coefficients:", sto3g_coeffs)

    # Build molecule in PySCF
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R_HH}",
        basis="sto-3g",
        unit="Bohr",
        verbose=0
    )

    # Get full ERI from PySCF
    eri_pyscf = mol.intor("int2e", aosym="s1")
    print(f"\nPySCF ERI tensor shape: {eri_pyscf.shape}")
    print(f"PySCF (00|00) = {eri_pyscf[0,0,0,0]:.14f}")
    print(f"PySCF (00|11) = {eri_pyscf[0,0,1,1]:.14f}")
    print(f"PySCF (01|01) = {eri_pyscf[0,1,0,1]:.14f}")

    # Compute contracted (00|00) integral manually
    # (00|00) = sum_{ijkl} c_i c_j c_k c_l (prim_i prim_j | prim_k prim_l)
    print("\nManual contracted (00|00):")
    eri_manual = 0.0
    for ai, ci in zip(sto3g_exponents, sto3g_coeffs):
        for aj, cj in zip(sto3g_exponents, sto3g_coeffs):
            for ak, ck in zip(sto3g_exponents, sto3g_coeffs):
                for al, cl in zip(sto3g_exponents, sto3g_coeffs):
                    prim_eri = eri_ssss_direct(ai, A, aj, A, ak, A, al, A)
                    eri_manual += ci * cj * ck * cl * prim_eri

    print(f"Manual (00|00) = {eri_manual:.14f}")
    print(f"PySCF  (00|00) = {eri_pyscf[0,0,0,0]:.14f}")
    print(f"Difference:     {abs(eri_manual - eri_pyscf[0,0,0,0]):.2e}")


def main():
    """Run comprehensive ERI examples."""
    print("=" * 70)
    print("Rys Quadrature ERI Examples")
    print("=" * 70)

    # Basic test cases
    print("\n" + "=" * 70)
    print("Basic ERI Calculations")
    print("=" * 70)

    # Test 1: Two atoms, symmetric case
    compare_methods_single(
        1.0, np.array([0.0, 0.0, 0.0]),
        1.0, np.array([0.0, 0.0, 0.0]),
        1.0, np.array([2.0, 0.0, 0.0]),
        1.0, np.array([2.0, 0.0, 0.0]),
        label="Two centers, R = 2 Bohr, all exponents = 1"
    )

    # Test 2: Four different centers
    compare_methods_single(
        0.5, np.array([0.0, 0.0, 0.0]),
        0.8, np.array([1.0, 0.0, 0.0]),
        0.6, np.array([0.0, 1.0, 0.0]),
        0.7, np.array([0.0, 0.0, 1.0]),
        label="Four different centers, different exponents"
    )

    # Test 3: Same center, different exponents
    compare_methods_single(
        0.3, np.array([0.0, 0.0, 0.0]),
        0.5, np.array([0.0, 0.0, 0.0]),
        0.7, np.array([0.0, 0.0, 0.0]),
        0.9, np.array([0.0, 0.0, 0.0]),
        label="Same center, different exponents (T = 0)"
    )

    # Symmetry tests
    test_eri_symmetry()

    # Limiting cases
    test_eri_limiting_cases()

    # PySCF comparison
    compare_pyscf_h2_eri()

    print("\n" + "=" * 70)
    print("ERI examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
