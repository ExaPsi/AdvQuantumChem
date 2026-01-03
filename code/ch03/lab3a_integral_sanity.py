#!/usr/bin/env python3
"""
Lab 3A: One-Electron Integral Sanity Checks

Extract S, T, and V matrices from PySCF and verify their basic properties:
- Symmetry (Hermitian)
- Expected signs (T > 0, V < 0)
- Matrix norms (scale understanding)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
from pyscf import gto


def main():
    print("=" * 70)
    print("Lab 3A: One-Electron Integral Sanity Checks")
    print("=" * 70)

    # Build H2O molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Extract one-electron integrals
    S = mol.intor("int1e_ovlp")  # Overlap
    T = mol.intor("int1e_kin")   # Kinetic energy
    V = mol.intor("int1e_nuc")   # Nuclear attraction
    h = T + V                     # Core Hamiltonian

    nao = mol.nao_nr()

    print(f"\nMolecule: H2O")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {nao}")

    # Section 1: Symmetry checks
    print("\n" + "-" * 50)
    print("1. SYMMETRY CHECKS (Hermitian property)")
    print("-" * 50)

    S_sym = np.allclose(S, S.T, atol=1e-12)
    T_sym = np.allclose(T, T.T, atol=1e-12)
    V_sym = np.allclose(V, V.T, atol=1e-12)

    print(f"   S symmetric: {S_sym}")
    print(f"   T symmetric: {T_sym}")
    print(f"   V symmetric: {V_sym}")
    print(f"   h symmetric: {np.allclose(h, h.T, atol=1e-12)}")

    # Section 2: Diagonal elements
    print("\n" + "-" * 50)
    print("2. DIAGONAL ELEMENTS")
    print("-" * 50)

    print("\n   Overlap matrix diagonals (should be ~1 for normalized AOs):")
    print(f"   S_diag = {np.diag(S)}")

    print("\n   Kinetic energy diagonals (should be positive):")
    print(f"   T_diag = {np.diag(T)}")
    T_diag_positive = np.all(np.diag(T) > 0)
    print(f"   All positive: {T_diag_positive}")

    print("\n   Nuclear attraction diagonals (should be negative):")
    print(f"   V_diag = {np.diag(V)}")
    V_diag_negative = np.all(np.diag(V) < 0)
    print(f"   All negative: {V_diag_negative}")

    # Section 3: Matrix norms
    print("\n" + "-" * 50)
    print("3. MATRIX NORMS (Frobenius)")
    print("-" * 50)

    print(f"   ||S||_F = {np.linalg.norm(S):12.6f}")
    print(f"   ||T||_F = {np.linalg.norm(T):12.6f}")
    print(f"   ||V||_F = {np.linalg.norm(V):12.6f}")
    print(f"   ||h||_F = {np.linalg.norm(h):12.6f}")

    print("\n   Physical insight:")
    print(f"   - |V|/|T| ratio: {np.linalg.norm(V)/np.linalg.norm(T):.3f}")
    print("     (Nuclear attraction dominates for atoms with large Z)")

    # Section 4: Eigenvalue analysis
    print("\n" + "-" * 50)
    print("4. EIGENVALUE ANALYSIS")
    print("-" * 50)

    eig_S = np.linalg.eigvalsh(S)
    eig_T = np.linalg.eigvalsh(T)
    eig_V = np.linalg.eigvalsh(V)

    print("\n   Overlap eigenvalues (all should be positive for valid metric):")
    print(f"   min(eig_S) = {eig_S.min():.6e}")
    print(f"   max(eig_S) = {eig_S.max():.6e}")
    print(f"   Condition number: {eig_S.max()/eig_S.min():.2f}")

    print("\n   Kinetic eigenvalues (all positive, curvature of wavefunction):")
    print(f"   min(eig_T) = {eig_T.min():.6e}")
    print(f"   max(eig_T) = {eig_T.max():.6e}")

    print("\n   Nuclear attraction eigenvalues (all negative, bound system):")
    print(f"   min(eig_V) = {eig_V.min():.6e}")
    print(f"   max(eig_V) = {eig_V.max():.6e}")

    # Section 5: Electron count verification
    print("\n" + "-" * 50)
    print("5. ELECTRON COUNT VERIFICATION")
    print("-" * 50)

    # For this check, we need a density matrix from SCF
    from pyscf import scf
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    P = mf.make_rdm1()

    n_elec_computed = np.trace(P @ S)
    n_elec_expected = mol.nelectron

    print(f"   Expected electrons: {n_elec_expected}")
    print(f"   Tr[P*S] = {n_elec_computed:.10f}")
    print(f"   Agreement: {np.isclose(n_elec_computed, n_elec_expected, atol=1e-10)}")

    # Section 6: Physical interpretation
    print("\n" + "-" * 50)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 50)

    print("""
   Overlap (S):
   - Diagonal = 1: AOs are normalized
   - Off-diagonal: measures spatial overlap between AOs
   - Large |S_ij|: AOs i,j share significant space

   Kinetic (T):
   - Always positive (curvature of wavefunction)
   - Large T: rapid spatial variation (tight orbitals)
   - Small T: slowly varying (diffuse orbitals)

   Nuclear Attraction (V):
   - Always negative (attractive interaction)
   - Large |V|: electron close to nuclei
   - Core orbitals have larger |V| than valence
    """)

    print("\n" + "=" * 70)
    print("Lab 3A Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
