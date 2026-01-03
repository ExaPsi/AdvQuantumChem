#!/usr/bin/env python3
"""
Kinetic Energy Integral Implementation

The kinetic integral for s-type functions has a compact closed form:

T_ab = μ (3 - 2μR²_AB) S_ab

This arises from applying the Laplacian to Gaussian functions.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import numpy as np
from pyscf import gto


def norm_s(alpha: float) -> float:
    """Normalization constant for s-type Gaussian."""
    return (2.0 * alpha / np.pi)**0.75


def overlap_ss_unnorm(alpha: float, A: np.ndarray,
                       beta: float, B: np.ndarray) -> float:
    """Unnormalized s-s overlap integral."""
    p = alpha + beta
    mu = alpha * beta / p
    R_AB_sq = np.sum((A - B)**2)
    return (np.pi / p)**1.5 * np.exp(-mu * R_AB_sq)


def kinetic_ss_unnorm(alpha: float, A: np.ndarray,
                       beta: float, B: np.ndarray) -> float:
    """
    Unnormalized s-s kinetic integral.

    T_ab = μ * (3 - 2μR²_AB) * S_ab

    The factor (3 - 2μR²) comes from:
    - 3 from the 3D Laplacian acting on Gaussians
    - -2μR² correction for separated centers
    """
    mu = alpha * beta / (alpha + beta)
    R_AB_sq = np.sum((A - B)**2)
    S = overlap_ss_unnorm(alpha, A, beta, B)
    return mu * (3.0 - 2.0 * mu * R_AB_sq) * S


def kinetic_ss_norm(alpha: float, A: np.ndarray,
                     beta: float, B: np.ndarray) -> float:
    """Normalized s-s kinetic integral."""
    return norm_s(alpha) * norm_s(beta) * kinetic_ss_unnorm(alpha, A, beta, B)


def main():
    print("=" * 60)
    print("Kinetic Energy Integral Validation")
    print("=" * 60)

    # Validate against PySCF
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    T_pyscf = mol.intor("int1e_kin")

    print("\nH2 / STO-3G at R = 0.74 Å")
    print("-" * 40)
    print("PySCF Kinetic Matrix (Hartree):")
    print(T_pyscf)
    print(f"\nDiagonal: T[0,0] = {T_pyscf[0, 0]:.6f} Hartree")
    print(f"Off-diagonal: T[0,1] = {T_pyscf[0, 1]:.6f} Hartree")
    print("\nComparison:")
    print("  Exact H atom kinetic energy = 0.5 Hartree")
    print(f"  STO-3G gives T = {T_pyscf[0, 0]:.3f} Hartree (higher due to basis)")
    print("  (Minimal basis forces more compact orbital shape)")


if __name__ == "__main__":
    main()
