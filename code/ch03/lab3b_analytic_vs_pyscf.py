#!/usr/bin/env python3
"""
Lab 3B: Analytic vs PySCF for a Custom s-Basis

Compare hand-coded analytic formulas for overlap, kinetic, and nuclear
attraction integrals against PySCF results using a minimal custom basis.

Key equations validated:
- Overlap: S_ab = N_a * N_b * (pi/p)^(3/2) * exp(-mu * R_AB^2)
- Kinetic: T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab
- Nuclear: V_ab = -Z * N_a * N_b * (2*pi/p) * exp(-mu*R_AB^2) * F_0(T)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
import math
from pyscf import gto


def Ns(alpha: float) -> float:
    """Normalization constant for s-type primitive Gaussian."""
    return (2 * alpha / math.pi) ** 0.75


def boys0(T: float) -> float:
    """
    Boys function F_0(T).

    F_0(T) = (1/2) * sqrt(pi/T) * erf(sqrt(T))  for T > 0
    F_0(0) = 1

    Uses series expansion for small T to avoid numerical issues.
    """
    if T < 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))


def overlap_ss(alpha: float, A: np.ndarray,
               beta: float, B: np.ndarray) -> float:
    """
    Normalized s-s overlap integral.

    S_ab = N(alpha) * N(beta) * (pi/p)^(3/2) * exp(-mu * R_AB^2)

    Parameters
    ----------
    alpha : float
        Exponent of Gaussian centered at A
    A : np.ndarray
        Center of first Gaussian (3D coordinates)
    beta : float
        Exponent of Gaussian centered at B
    B : np.ndarray
        Center of second Gaussian (3D coordinates)

    Returns
    -------
    float
        Overlap integral value
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    p = alpha + beta
    mu = alpha * beta / p
    R2 = float(np.dot(A - B, A - B))

    return Ns(alpha) * Ns(beta) * (math.pi / p) ** 1.5 * math.exp(-mu * R2)


def kinetic_ss(alpha: float, A: np.ndarray,
               beta: float, B: np.ndarray) -> float:
    """
    Normalized s-s kinetic energy integral.

    T_ab = mu * (3 - 2*mu*R_AB^2) * S_ab

    This relation follows from applying -1/2 nabla^2 to the ket
    and using the Gaussian product theorem.

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss

    Returns
    -------
    float
        Kinetic energy integral value
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    p = alpha + beta
    mu = alpha * beta / p
    R2 = float(np.dot(A - B, A - B))
    S = overlap_ss(alpha, A, beta, B)

    return mu * (3.0 - 2.0 * mu * R2) * S


def nucattr_ss_single(alpha: float, A: np.ndarray,
                      beta: float, B: np.ndarray,
                      Z: float, C: np.ndarray) -> float:
    """
    s-s nuclear attraction integral for a single nucleus.

    V_ab(C) = -Z * N_a * N_b * (2*pi/p) * exp(-mu*R_AB^2) * F_0(p*|P-C|^2)

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss
    Z : float
        Nuclear charge
    C : np.ndarray
        Nuclear position (3D coordinates)

    Returns
    -------
    float
        Nuclear attraction integral (negative for attraction)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    p = alpha + beta
    mu = alpha * beta / p
    R2 = float(np.dot(A - B, A - B))

    # Composite center P = (alpha*A + beta*B) / p
    P = (alpha * A + beta * B) / p

    # Argument of Boys function: T = p * |P - C|^2
    PC2 = float(np.dot(P - C, P - C))
    T = p * PC2

    # Prefactor
    pref = Ns(alpha) * Ns(beta) * (2 * math.pi / p) * math.exp(-mu * R2)

    return -Z * pref * boys0(T)


def nucattr_ss_total(alpha: float, A: np.ndarray,
                     beta: float, B: np.ndarray,
                     nuclei: list) -> float:
    """
    Total nuclear attraction integral summed over all nuclei.

    V_ab = sum_C V_ab(C)

    Parameters
    ----------
    alpha, A, beta, B : see overlap_ss
    nuclei : list of (Z, C) tuples
        Nuclear charges and positions

    Returns
    -------
    float
        Total nuclear attraction integral
    """
    V = 0.0
    for Z, C in nuclei:
        V += nucattr_ss_single(alpha, A, beta, B, Z, C)
    return V


def main():
    print("=" * 70)
    print("Lab 3B: Analytic vs PySCF for Custom s-Basis")
    print("=" * 70)

    # Define geometry in Bohr
    R = 1.4  # H-H bond length in Bohr
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, R])

    # Gaussian exponents (different for asymmetric test)
    alpha = 0.50
    beta = 0.40

    print(f"\nGeometry:")
    print(f"   H at A = {A} Bohr")
    print(f"   H at B = {B} Bohr")
    print(f"   Bond length R = {R} Bohr")

    print(f"\nBasis parameters:")
    print(f"   alpha = {alpha} Bohr^-2")
    print(f"   beta  = {beta} Bohr^-2")

    # Build PySCF molecule with custom single-primitive s bases
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

    # Extract PySCF integrals
    S_pyscf = mol.intor("int1e_ovlp")
    T_pyscf = mol.intor("int1e_kin")
    V_pyscf = mol.intor("int1e_nuc")

    # Off-diagonal elements
    S12_pyscf = S_pyscf[0, 1]
    T12_pyscf = T_pyscf[0, 1]
    V12_pyscf = V_pyscf[0, 1]

    # Diagonal elements
    S11_pyscf = S_pyscf[0, 0]
    T11_pyscf = T_pyscf[0, 0]

    # Analytic calculations
    S12 = overlap_ss(alpha, A, beta, B)
    T12 = kinetic_ss(alpha, A, beta, B)

    # Nuclear attraction includes both nuclei (Z=1 at A and Z=1 at B)
    nuclei = [(1.0, A), (1.0, B)]
    V12 = nucattr_ss_total(alpha, A, beta, B, nuclei)

    # Diagonal overlap and kinetic
    S11 = overlap_ss(alpha, A, alpha, A)
    T11 = kinetic_ss(alpha, A, alpha, A)

    # Print results
    print("\n" + "-" * 50)
    print("OVERLAP INTEGRALS")
    print("-" * 50)
    print(f"   S[0,0] (diagonal):")
    print(f"      PySCF:    {S11_pyscf:.15f}")
    print(f"      Analytic: {S11:.15f}")
    print(f"      Diff:     {abs(S11_pyscf - S11):.2e}")

    print(f"\n   S[0,1] (off-diagonal):")
    print(f"      PySCF:    {S12_pyscf:.15f}")
    print(f"      Analytic: {S12:.15f}")
    print(f"      Diff:     {abs(S12_pyscf - S12):.2e}")

    print("\n" + "-" * 50)
    print("KINETIC INTEGRALS")
    print("-" * 50)
    print(f"   T[0,0] (diagonal):")
    print(f"      PySCF:    {T11_pyscf:.15f}")
    print(f"      Analytic: {T11:.15f}")
    print(f"      Diff:     {abs(T11_pyscf - T11):.2e}")

    print(f"\n   T[0,1] (off-diagonal):")
    print(f"      PySCF:    {T12_pyscf:.15f}")
    print(f"      Analytic: {T12:.15f}")
    print(f"      Diff:     {abs(T12_pyscf - T12):.2e}")

    print("\n" + "-" * 50)
    print("NUCLEAR ATTRACTION INTEGRALS")
    print("-" * 50)
    print(f"   V[0,1] (off-diagonal):")
    print(f"      PySCF:    {V12_pyscf:.15f}")
    print(f"      Analytic: {V12:.15f}")
    print(f"      Diff:     {abs(V12_pyscf - V12):.2e}")

    # Verify kinetic-overlap relation
    print("\n" + "-" * 50)
    print("KINETIC-OVERLAP RELATION VERIFICATION")
    print("-" * 50)

    p = alpha + beta
    mu = alpha * beta / p
    R2 = np.dot(A - B, A - B)
    T_from_S = mu * (3.0 - 2.0 * mu * R2) * S12

    print(f"   T = mu * (3 - 2*mu*R^2) * S")
    print(f"   mu = alpha*beta/(alpha+beta) = {mu:.6f}")
    print(f"   R^2 = {R2:.6f} Bohr^2")
    print(f"   Factor = mu*(3 - 2*mu*R^2) = {mu * (3 - 2*mu*R2):.6f}")
    print(f"\n   T from S: {T_from_S:.15f}")
    print(f"   T direct: {T12:.15f}")
    print(f"   Match:    {np.isclose(T_from_S, T12, atol=1e-14)}")

    # Physical insights
    print("\n" + "-" * 50)
    print("PHYSICAL INSIGHTS")
    print("-" * 50)

    print(f"""
   1. Diagonal overlap S[i,i] = 1: Normalized basis functions

   2. Off-diagonal overlap decays exponentially:
      S_ab ~ exp(-mu * R^2) = exp(-{mu:.3f} * {R2:.3f})
                            = exp(-{mu * R2:.3f})
                            = {math.exp(-mu * R2):.6f}

   3. Kinetic energy is always positive:
      - Measures curvature of wavefunction
      - Larger exponents -> tighter orbitals -> more curvature -> larger T

   4. Nuclear attraction is negative (attractive):
      V < 0 always (electron-nucleus attraction)
      V = {V12:.6f} Hartree

   5. The Boys function F_0(T) modulates the Coulomb integral:
      - F_0(0) = 1 (coincident centers)
      - F_0(T) -> 0 as T -> infinity (distant centers)
    """)

    print("\n" + "=" * 70)
    print("Lab 3B Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
