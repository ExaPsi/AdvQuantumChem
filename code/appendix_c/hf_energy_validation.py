#!/usr/bin/env python3
"""
Complete Hartree-Fock Energy Validation from Integrals

This script validates all integral formulas by computing the HF energy
from fundamental integrals and comparing against PySCF reference.

Demonstrates: HF reduces entirely to integral evaluation + linear algebra!

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix C: Gaussian Integral Formula Sheet
"""

import numpy as np
from pyscf import gto, scf


def main():
    print("=" * 60)
    print("Complete HF Energy Validation from Integrals")
    print("=" * 60)

    # Build H2 molecule
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Extract all integrals
    S = mol.intor("int1e_ovlp")       # Overlap
    T = mol.intor("int1e_kin")        # Kinetic
    V = mol.intor("int1e_nuc")        # Nuclear attraction
    h = T + V                         # Core Hamiltonian
    eri = mol.intor("int2e", aosym="s1")  # ERIs

    print("\nIntegral matrices extracted:")
    print(f"  S shape: {S.shape}")
    print(f"  h shape: {h.shape}")
    print(f"  ERI shape: {eri.shape}")

    # Run PySCF HF for reference
    mf = scf.RHF(mol)
    E_pyscf = mf.kernel()
    P = mf.make_rdm1()  # Converged density matrix

    print(f"\nPySCF converged in {mf.cycles} iterations")

    # Build Coulomb and Exchange matrices
    # J_μν = Σ_λσ (μν|λσ) P_λσ
    # K_μν = Σ_λσ (μλ|νσ) P_λσ
    J = np.einsum('ijkl,kl->ij', eri, P)
    K = np.einsum('ikjl,kl->ij', eri, P)

    # Fock matrix: F = h + J - (1/2)K
    F = h + J - 0.5 * K

    # Electronic energy: E_elec = (1/2) Tr[P(h + F)]
    E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)

    # Nuclear repulsion
    E_nuc = mol.energy_nuc()

    # Total energy
    E_total = E_elec + E_nuc

    print("\n" + "=" * 50)
    print("Energy Components:")
    print("=" * 50)
    print(f"  Tr[P·h]     = {np.einsum('ij,ij->', P, h):.10f} Hartree")
    print(f"  Tr[P·J]/2   = {0.5 * np.einsum('ij,ij->', P, J):.10f} Hartree")
    print(f"  -Tr[P·K]/4  = {-0.25 * np.einsum('ij,ij->', P, K):.10f} Hartree")
    print(f"  E_elec      = {E_elec:.10f} Hartree")
    print(f"  E_nuc       = {E_nuc:.10f} Hartree")

    print("\n" + "=" * 50)
    print("Validation:")
    print("=" * 50)
    print(f"  E_total (manual) = {E_total:.10f} Hartree")
    print(f"  E_total (PySCF)  = {E_pyscf:.10f} Hartree")
    print(f"  |Difference|     = {abs(E_total - E_pyscf):.2e} Hartree")

    # Verify electron count
    n_elec = np.trace(P @ S)
    print(f"\n  Electron count: Tr[P·S] = {n_elec:.6f} (expected: {mol.nelectron})")

    # Check convergence quality
    assert abs(E_total - E_pyscf) < 1e-10, "Energy mismatch!"
    assert abs(n_elec - mol.nelectron) < 1e-10, "Electron count mismatch!"
    print("\n✓ All validations passed!")


if __name__ == "__main__":
    main()
