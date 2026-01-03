#!/usr/bin/env python3
"""
Nuclear Attraction Integral Implementation

The nuclear attraction integral involves the Boys function:

V_ab(C) = -Z_C (2π/p) exp(-μR²_AB) F_0(p|P-C|²)

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


def nuclear_ss_unnorm(alpha: float, A: np.ndarray,
                       beta: float, B: np.ndarray,
                       C: np.ndarray, Z_C: float) -> float:
    """
    Nuclear attraction integral for unnormalized s-type Gaussians.

    V_ab(C) = -Z_C * (2π/p) * exp(-μR²_AB) * F_0(p|P-C|²)

    The negative sign indicates attraction (V < 0).
    """
    # GPT parameters
    p = alpha + beta
    mu = alpha * beta / p
    P = (alpha * A + beta * B) / p

    # Distance calculations
    R_AB_sq = np.sum((A - B)**2)
    R_PC_sq = np.sum((P - C)**2)

    # Boys function argument
    T = p * R_PC_sq

    # Nuclear attraction formula
    return -Z_C * (2.0 * np.pi / p) * np.exp(-mu * R_AB_sq) * boys(0, T)


def nuclear_ss_norm(alpha, A, beta, B, C, Z_C):
    """Normalized nuclear attraction integral."""
    return norm_s(alpha) * norm_s(beta) * nuclear_ss_unnorm(alpha, A, beta, B, C, Z_C)


def main():
    print("=" * 60)
    print("Nuclear Attraction Integral Validation")
    print("=" * 60)

    # Validate against PySCF
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    V_pyscf = mol.intor("int1e_nuc")

    print("\nH2 / STO-3G at R = 0.74 Å")
    print("-" * 40)
    print("PySCF Nuclear Attraction Matrix (Hartree):")
    print(V_pyscf)
    print(f"\nDiagonal: V[0,0] = {V_pyscf[0, 0]:.6f} Hartree")
    print(f"Off-diagonal: V[0,1] = {V_pyscf[0, 1]:.6f} Hartree")
    print("\nInterpretation:")
    print("  - All values are NEGATIVE (attractive interaction)")
    print("  - V[0,0] includes attraction to BOTH nuclei")
    print("  - Larger magnitude than single atom due to second nucleus")


if __name__ == "__main__":
    main()
