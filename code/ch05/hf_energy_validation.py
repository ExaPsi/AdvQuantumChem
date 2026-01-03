#!/usr/bin/env python3
"""
hf_energy_validation.py - Complete HF Energy from Integrals (Exercise 5.5)

This module demonstrates the central course theme:

    Hartree-Fock = Integrals + Linear Algebra

We compute the HF electronic energy entirely from integrals:
    E_elec = Tr[P*h] + (1/2) Tr[P*(J - (1/2)K)]
           = (1/2) Tr[P*(h + F)]

where:
    - h = T + V (core Hamiltonian: kinetic + nuclear attraction)
    - J = Coulomb matrix from ERIs
    - K = Exchange matrix from ERIs
    - F = h + J - (1/2)K (Fock matrix)
    - P = density matrix from converged MO coefficients

This validates that our understanding of integrals and contractions
correctly reproduces the PySCF SCF energy.

References:
    - Chapter 5, Section 7: Building J and K
    - Exercise 5.5: J/K build and energy component check
    - Chapter 6 preview: HF energy expressions

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import Tuple


def compute_hf_energy_from_integrals(
    h: np.ndarray,
    eri: np.ndarray,
    P: np.ndarray,
    E_nuc: float
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Compute HF energy from integrals using explicit formulas.

    E_elec = Tr[P*h] + (1/2) Tr[P*(J - (1/2)K)]

    Parameters
    ----------
    h : np.ndarray
        Core Hamiltonian (kinetic + nuclear attraction)
    eri : np.ndarray
        Full ERI tensor (N, N, N, N)
    P : np.ndarray
        Density matrix (N, N)
    E_nuc : float
        Nuclear repulsion energy

    Returns
    -------
    E_total : float
        Total HF energy
    E_elec : float
        Electronic energy
    E_1e : float
        One-electron energy Tr[P*h]
    E_2e : float
        Two-electron energy (1/2) Tr[P*(J - K/2)]
    J : np.ndarray
        Coulomb matrix
    K : np.ndarray
        Exchange matrix
    """
    # Build J and K matrices
    # J_mn = sum_ls (mn|ls) P_ls
    J = np.einsum('ijkl,lk->ij', eri, P, optimize=True)

    # K_mn = sum_ls (ml|ns) P_ls
    K = np.einsum('ikjl,lk->ij', eri, P, optimize=True)

    # One-electron energy: Tr[P*h]
    E_1e = np.einsum('ij,ij->', P, h)

    # Two-electron energy: (1/2) Tr[P*(J - K/2)]
    # = (1/2) Tr[P*J] - (1/4) Tr[P*K]
    E_J = np.einsum('ij,ij->', P, J)
    E_K = np.einsum('ij,ij->', P, K)
    E_2e = 0.5 * E_J - 0.25 * E_K

    # Total electronic energy
    E_elec = E_1e + E_2e

    # Total energy = electronic + nuclear
    E_total = E_elec + E_nuc

    return E_total, E_elec, E_1e, E_2e, J, K


def verify_energy_formula(h: np.ndarray, J: np.ndarray, K: np.ndarray,
                          P: np.ndarray, E_nuc: float) -> Tuple[float, float]:
    """
    Verify the alternative energy formula using Fock matrix.

    E_elec = (1/2) Tr[P*(h + F)]    where F = h + J - K/2

    This should give the same result as the explicit formula.

    Parameters
    ----------
    h : np.ndarray
        Core Hamiltonian
    J, K : np.ndarray
        Coulomb and Exchange matrices
    P : np.ndarray
        Density matrix
    E_nuc : float
        Nuclear repulsion energy

    Returns
    -------
    E_total : float
        Total energy from Fock matrix formula
    E_elec : float
        Electronic energy
    """
    # Build Fock matrix
    F = h + J - 0.5 * K

    # E_elec = (1/2) Tr[P*(h + F)]
    E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)

    E_total = E_elec + E_nuc

    return E_total, E_elec


def verify_electron_count(P: np.ndarray, S: np.ndarray, n_expected: int) -> float:
    """
    Verify electron count from density matrix.

    N_elec = Tr[P*S]

    Parameters
    ----------
    P : np.ndarray
        Density matrix
    S : np.ndarray
        Overlap matrix
    n_expected : int
        Expected number of electrons

    Returns
    -------
    n_elec : float
        Computed electron count
    """
    n_elec = np.trace(P @ S)
    return n_elec


# =============================================================================
# Main validation
# =============================================================================

def validate_h2o_sto3g():
    """
    Complete HF energy validation for H2O/STO-3G.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Cannot run validation.")
        return False

    print("Complete HF Energy Validation: H2O/STO-3G")
    print("=" * 70)
    print("Theme: Hartree-Fock = Integrals + Linear Algebra")
    print("=" * 70)

    # Build molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: H2O")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {mol.nao}")
    print(f"Number of electrons: {mol.nelectron}")

    # Step 1: Extract all integrals
    print("\n" + "-" * 70)
    print("Step 1: Extract integrals from PySCF")
    print("-" * 70)

    S = mol.intor("int1e_ovlp")          # Overlap
    T = mol.intor("int1e_kin")           # Kinetic energy
    V = mol.intor("int1e_nuc")           # Nuclear attraction
    h = T + V                             # Core Hamiltonian
    eri = mol.intor("int2e", aosym="s1") # ERIs (full tensor)
    E_nuc = mol.energy_nuc()             # Nuclear repulsion

    print(f"  S (overlap) shape: {S.shape}")
    print(f"  T (kinetic) shape: {T.shape}")
    print(f"  V (nuclear) shape: {V.shape}")
    print(f"  h (core Ham) shape: {h.shape}")
    print(f"  ERI shape: {eri.shape}")
    print(f"  ERI memory: {eri.nbytes / 1e6:.2f} MB")
    print(f"  E_nuc = {E_nuc:.10f} Hartree")

    # Step 2: Run PySCF HF for reference
    print("\n" + "-" * 70)
    print("Step 2: Run PySCF RHF for reference")
    print("-" * 70)

    mf = scf.RHF(mol)
    E_pyscf = mf.kernel()
    P = mf.make_rdm1()  # Converged density matrix

    print(f"  RHF converged in {mf.cycles} iterations")
    print(f"  RHF total energy: {E_pyscf:.10f} Hartree")

    # Step 3: Compute energy from integrals
    print("\n" + "-" * 70)
    print("Step 3: Compute HF energy from integrals")
    print("-" * 70)

    E_total, E_elec, E_1e, E_2e, J, K = compute_hf_energy_from_integrals(h, eri, P, E_nuc)

    print("\nEnergy breakdown:")
    print(f"  Tr[P*h]     = {E_1e:15.10f} Hartree (one-electron)")
    print(f"  Tr[P*J]/2   = {0.5*np.einsum('ij,ij->', P, J):15.10f} Hartree (Coulomb)")
    print(f"  -Tr[P*K]/4  = {-0.25*np.einsum('ij,ij->', P, K):15.10f} Hartree (Exchange)")
    print(f"  E_2e        = {E_2e:15.10f} Hartree (two-electron)")
    print(f"  E_elec      = {E_elec:15.10f} Hartree (electronic)")
    print(f"  E_nuc       = {E_nuc:15.10f} Hartree (nuclear)")
    print(f"  E_total     = {E_total:15.10f} Hartree")

    # Step 4: Verify with Fock matrix formula
    print("\n" + "-" * 70)
    print("Step 4: Verify with Fock matrix formula")
    print("-" * 70)

    E_total_F, E_elec_F = verify_energy_formula(h, J, K, P, E_nuc)

    print(f"  E_elec = (1/2) Tr[P*(h+F)] = {E_elec_F:.10f} Hartree")
    print(f"  E_total = {E_total_F:.10f} Hartree")
    print(f"  Consistency check: |E_elec - E_elec_F| = {abs(E_elec - E_elec_F):.2e}")

    # Step 5: Verify electron count
    print("\n" + "-" * 70)
    print("Step 5: Verify electron count")
    print("-" * 70)

    n_elec = verify_electron_count(P, S, mol.nelectron)
    print(f"  Tr[P*S] = {n_elec:.10f}")
    print(f"  Expected: {mol.nelectron}")
    print(f"  Difference: {abs(n_elec - mol.nelectron):.2e}")

    # Step 6: Final validation
    print("\n" + "-" * 70)
    print("Step 6: Final validation against PySCF")
    print("-" * 70)

    diff = abs(E_total - E_pyscf)
    print(f"\n  Our E_total:   {E_total:.10f} Hartree")
    print(f"  PySCF E_total: {E_pyscf:.10f} Hartree")
    print(f"  |Difference|:  {diff:.2e} Hartree")

    # Verify J and K against PySCF
    J_ref, K_ref = mf.get_jk(mol, P)
    J_diff = np.linalg.norm(J - J_ref)
    K_diff = np.linalg.norm(K - K_ref)

    print(f"\n  ||J - J_ref||_F = {J_diff:.2e}")
    print(f"  ||K - K_ref||_F = {K_diff:.2e}")

    # Success criteria
    success = (diff < 1e-10 and
               abs(n_elec - mol.nelectron) < 1e-10 and
               J_diff < 1e-10 and
               K_diff < 1e-10)

    print("\n" + "=" * 70)
    if success:
        print("VALIDATION PASSED: All checks within tolerance")
        print("\nThis confirms: HF energy = Integrals + Linear Algebra!")
    else:
        print("VALIDATION FAILED: Some checks exceeded tolerance")
    print("=" * 70)

    return success


def validate_h2_minimal():
    """
    Minimal validation for H2/STO-3G (smallest system).
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return False

    print("\n" + "=" * 70)
    print("Minimal validation: H2/STO-3G")
    print("=" * 70)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Extract integrals
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    # Run HF
    mf = scf.RHF(mol)
    E_pyscf = mf.kernel()
    P = mf.make_rdm1()

    # Compute from integrals
    E_total, E_elec, E_1e, E_2e, J, K = compute_hf_energy_from_integrals(h, eri, P, E_nuc)

    print(f"Number of AOs: {mol.nao}")
    print(f"E_total (computed): {E_total:.10f}")
    print(f"E_total (PySCF):    {E_pyscf:.10f}")
    print(f"|Difference|:       {abs(E_total - E_pyscf):.2e}")

    success = abs(E_total - E_pyscf) < 1e-10
    print(f"Status: {'PASS' if success else 'FAIL'}")

    return success


def print_energy_formulas():
    """
    Print the key energy formulas for reference.
    """
    print("\n" + "=" * 70)
    print("Key Hartree-Fock Energy Formulas")
    print("=" * 70)

    print("""
INTEGRALS:
    S = overlap:          S_mn = <m|n>
    T = kinetic:          T_mn = <m|-1/2 nabla^2|n>
    V = nuclear:          V_mn = <m|sum_A -Z_A/|r-R_A||n>
    h = core Hamiltonian: h = T + V
    ERI:                  (mn|ls) = <mn|r12^{-1}|ls>

DENSITY MATRIX (RHF closed-shell):
    P_mn = 2 * sum_i^{n_occ} C_mi C_ni    (factor 2 for double occupancy)

COULOMB AND EXCHANGE:
    J_mn = sum_ls (mn|ls) P_ls            (classical Coulomb)
    K_mn = sum_ls (ml|ns) P_ls            (quantum Exchange)

FOCK MATRIX:
    F = h + J - (1/2) K                   (factor 1/2 for RHF)

ENERGY:
    E_1e = Tr[P*h]                        (one-electron energy)
    E_2e = (1/2) Tr[P*J] - (1/4) Tr[P*K]  (two-electron energy)
         = (1/2) Tr[P*(J - K/2)]
    E_elec = E_1e + E_2e
           = Tr[P*h] + (1/2) Tr[P*(J - K/2)]
           = (1/2) Tr[P*(h + F)]          (alternative form)
    E_total = E_elec + E_nuc

ELECTRON COUNT:
    N_elec = Tr[P*S]
""")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 5.5: Complete HF Energy from Integrals")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Print formulas for reference
    print_energy_formulas()

    # Run validations
    success1 = validate_h2_minimal()
    success2 = validate_h2o_sto3g()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"H2/STO-3G validation:  {'PASS' if success1 else 'FAIL'}")
    print(f"H2O/STO-3G validation: {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print("\nAll validations PASSED")
        print("\nKey takeaway: Hartree-Fock is entirely determined by:")
        print("  1. One-electron integrals (S, T, V)")
        print("  2. Two-electron integrals (ERIs)")
        print("  3. Linear algebra (contractions, diagonalization)")
    else:
        print("\nSome validations FAILED")
    print("=" * 70)
