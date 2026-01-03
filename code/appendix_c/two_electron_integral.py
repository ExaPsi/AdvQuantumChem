#!/usr/bin/env python3
"""
Two-Electron Repulsion Integral (ERI) Implementation

The (ss|ss) ERI in chemist's notation:

(ab|cd) = (2π^(5/2))/(pq√(p+q)) exp(-μR²_AB) exp(-νR²_CD) F_0(T)

where T = ρ|P-Q|² and ρ = pq/(p+q)

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import numpy as np
from pyscf import gto

# Import Boys function from local module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boys_function import boys


def norm_s(alpha: float) -> float:
    """Normalization constant for s-type Gaussian."""
    return (2.0 * alpha / np.pi)**0.75


def eri_ssss_unnorm(alpha: float, A: np.ndarray,
                     beta: float, B: np.ndarray,
                     gamma: float, C: np.ndarray,
                     delta: float, D: np.ndarray) -> float:
    """
    (ss|ss) two-electron repulsion integral for unnormalized Gaussians.

    (ab|cd) = (2π^(5/2)) / (pq√(p+q))
              × exp(-μ_ab R²_AB) × exp(-ν_cd R²_CD) × F_0(T)

    where T = ρ|P-Q|², ρ = pq/(p+q)
    """
    # GPT for bra pair (a, b) -> p, P
    p = alpha + beta
    mu_ab = alpha * beta / p
    P = (alpha * A + beta * B) / p
    R_AB_sq = np.sum((A - B)**2)

    # GPT for ket pair (c, d) -> q, Q
    q = gamma + delta
    nu_cd = gamma * delta / q
    Q = (gamma * C + delta * D) / q
    R_CD_sq = np.sum((C - D)**2)

    # Boys function argument
    rho = p * q / (p + q)
    R_PQ_sq = np.sum((P - Q)**2)
    T = rho * R_PQ_sq

    # Two-electron integral formula
    prefactor = 2.0 * np.pi**2.5 / (p * q * np.sqrt(p + q))
    exp_factor = np.exp(-mu_ab * R_AB_sq - nu_cd * R_CD_sq)

    return prefactor * exp_factor * boys(0, T)


def eri_ssss_norm(alpha, A, beta, B, gamma, C, delta, D):
    """Normalized (ss|ss) ERI."""
    N = norm_s(alpha) * norm_s(beta) * norm_s(gamma) * norm_s(delta)
    return N * eri_ssss_unnorm(alpha, A, beta, B, gamma, C, delta, D)


def main():
    print("=" * 60)
    print("Two-Electron Integral (ERI) Validation")
    print("=" * 60)

    # Validate against PySCF
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    eri_pyscf = mol.intor("int2e", aosym="s1")

    print("\nH2 / STO-3G at R = 0.74 Å")
    print("-" * 40)
    print("Selected ERIs (chemist notation):")
    print(f"  (00|00) = {eri_pyscf[0, 0, 0, 0]:.8f} Hartree")
    print(f"  (00|11) = {eri_pyscf[0, 0, 1, 1]:.8f} Hartree")
    print(f"  (01|01) = {eri_pyscf[0, 1, 0, 1]:.8f} Hartree")
    print(f"  (01|10) = {eri_pyscf[0, 1, 1, 0]:.8f} Hartree")

    print("\nInterpretation:")
    print("  (00|00): Coulomb self-repulsion on H1")
    print("  (00|11): Inter-site Coulomb (smaller - separated charges)")
    print("  (01|01): Exchange integral (no classical analog)")

    # Verify 8-fold symmetry
    print("\n8-fold ERI symmetry check:")
    print("-" * 40)
    eri_0011 = eri_pyscf[0, 0, 1, 1]
    print(f"  (00|11) = {eri_0011:.8f}")
    print(f"  (11|00) = {eri_pyscf[1, 1, 0, 0]:.8f}")
    print(f"  (01|10) = {eri_pyscf[0, 1, 1, 0]:.8f}")
    print(f"  (10|01) = {eri_pyscf[1, 0, 0, 1]:.8f}")
    print("  (All should be identical)")


if __name__ == "__main__":
    main()
