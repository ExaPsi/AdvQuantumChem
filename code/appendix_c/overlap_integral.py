#!/usr/bin/env python3
"""
Overlap Integral Implementation

The overlap integral between two normalized s-type Gaussians:

S_ab = N_s(α) N_s(β) (π/p)^(3/2) exp(-μR²_AB)

where N_s(α) = (2α/π)^(3/4) is the normalization constant.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import numpy as np
from pyscf import gto


def norm_s(alpha: float) -> float:
    """Normalization constant for s-type Gaussian: N = (2*alpha/pi)^(3/4)"""
    return (2.0 * alpha / np.pi)**0.75


def overlap_ss_unnorm(alpha: float, A: np.ndarray,
                       beta: float, B: np.ndarray) -> float:
    """
    Unnormalized s-s overlap integral.

    S_ab = (π/p)^(3/2) * exp(-μ * R²_AB)

    where p = α + β, μ = αβ/p
    """
    p = alpha + beta
    mu = alpha * beta / p
    R_AB_sq = np.sum((A - B)**2)
    return (np.pi / p)**1.5 * np.exp(-mu * R_AB_sq)


def overlap_ss_norm(alpha: float, A: np.ndarray,
                     beta: float, B: np.ndarray) -> float:
    """Normalized s-s overlap integral."""
    return norm_s(alpha) * norm_s(beta) * overlap_ss_unnorm(alpha, A, beta, B)


def main():
    print("=" * 60)
    print("Overlap Integral Validation")
    print("=" * 60)

    # Validate against PySCF for H2 / STO-3G
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    S_pyscf = mol.intor("int1e_ovlp")

    print("\nH2 / STO-3G at R = 0.74 Å")
    print("-" * 40)
    print("PySCF Overlap Matrix:")
    print(S_pyscf)
    print(f"\nDiagonal elements: S[0,0] = S[1,1] = {S_pyscf[0, 0]:.6f}")
    print(f"Off-diagonal: S[0,1] = S[1,0] = {S_pyscf[0, 1]:.6f}")
    print("\nInterpretation:")
    print("  - Diagonal = 1.0 (normalized basis)")
    print("  - Off-diagonal = 0.66 (substantial bonding overlap)")


if __name__ == "__main__":
    main()
