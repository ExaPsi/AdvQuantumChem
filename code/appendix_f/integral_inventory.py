#!/usr/bin/env python3
"""
Comprehensive Integral Extraction from PySCF

Demonstrates extraction of all one-electron and two-electron integrals,
including verification of matrix properties (symmetry, trace, etc.).

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference
"""

import numpy as np
from pyscf import gto, scf


def main():
    print("=" * 70)
    print("Comprehensive Integral Inventory from PySCF")
    print("=" * 70)

    # Build water molecule
    mol = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nTest system: H2O / cc-pVDZ")
    print(f"  Number of AOs: {mol.nao}")
    print(f"  Number of electrons: {mol.nelectron}")

    # =========================================================================
    # Section 1: One-Electron Integrals
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. ONE-ELECTRON INTEGRALS")
    print("=" * 50)

    # -----------------------------------------
    # Overlap matrix S
    # -----------------------------------------
    S = mol.intor("int1e_ovlp")
    print("\n--- Overlap Matrix S = <mu|nu> ---")
    print(f"  Shape: {S.shape}")
    print(f"  Symmetry check: ||S - S^T|| = {np.linalg.norm(S - S.T):.2e}")
    print(f"  Diagonal elements (self-overlap, should be 1 for normalized):")
    print(f"    S[0,0] = {S[0,0]:.10f}")
    print(f"    S[5,5] = {S[5,5]:.10f}")
    print(f"  Trace(S) = {np.trace(S):.6f} (equals nao for orthonormal)")
    print(f"  Eigenvalues of S (all positive if well-conditioned):")
    S_eig = np.linalg.eigvalsh(S)
    print(f"    min = {S_eig.min():.6e}, max = {S_eig.max():.6e}")
    print(f"    Condition number = {S_eig.max()/S_eig.min():.2e}")

    # -----------------------------------------
    # Kinetic energy matrix T
    # -----------------------------------------
    T = mol.intor("int1e_kin")
    print("\n--- Kinetic Energy Matrix T = <mu|-0.5*nabla^2|nu> ---")
    print(f"  Shape: {T.shape}")
    print(f"  Symmetry check: ||T - T^T|| = {np.linalg.norm(T - T.T):.2e}")
    print(f"  All diagonal elements positive? {np.all(np.diag(T) > 0)}")
    print(f"  Diagonal range: [{np.diag(T).min():.4f}, {np.diag(T).max():.4f}]")
    print(f"  Trace(T) = {np.trace(T):.6f} Hartree")

    # -----------------------------------------
    # Nuclear attraction matrix V
    # -----------------------------------------
    V = mol.intor("int1e_nuc")
    print("\n--- Nuclear Attraction Matrix V = <mu|V_nuc|nu> ---")
    print(f"  Shape: {V.shape}")
    print(f"  Symmetry check: ||V - V^T|| = {np.linalg.norm(V - V.T):.2e}")
    print(f"  All diagonal elements negative? {np.all(np.diag(V) < 0)}")
    print(f"  Diagonal range: [{np.diag(V).min():.4f}, {np.diag(V).max():.4f}]")
    print(f"  Trace(V) = {np.trace(V):.6f} Hartree")

    # -----------------------------------------
    # Core Hamiltonian h = T + V
    # -----------------------------------------
    h = T + V
    print("\n--- Core Hamiltonian h = T + V ---")
    print(f"  Trace(h) = {np.trace(h):.6f} Hartree")
    print(f"  Eigenvalue range: [{np.linalg.eigvalsh(h).min():.4f}, {np.linalg.eigvalsh(h).max():.4f}]")

    # -----------------------------------------
    # Dipole integrals
    # -----------------------------------------
    print("\n--- Dipole Integrals <mu|r|nu> ---")
    mol.set_common_origin([0, 0, 0])
    r_ints = mol.intor("int1e_r")  # Shape: (3, nao, nao)
    print(f"  Shape: {r_ints.shape} (components: x, y, z)")
    print(f"  Each component is symmetric:")
    for i, axis in enumerate(['x', 'y', 'z']):
        print(f"    ||r_{axis} - r_{axis}^T|| = {np.linalg.norm(r_ints[i] - r_ints[i].T):.2e}")

    # Verify dipole with density
    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    # Electronic dipole (negative of electron position expectation value)
    mu_elec = -np.einsum('xij,ji->x', r_ints, dm)

    # Nuclear dipole
    mu_nuc = np.zeros(3)
    for i in range(mol.natm):
        Z = mol.atom_charge(i)
        coords = mol.atom_coord(i)
        mu_nuc += Z * coords

    mu_total = mu_elec + mu_nuc
    print(f"\n  Dipole moment calculation:")
    print(f"    Electronic: {mu_elec}")
    print(f"    Nuclear:    {mu_nuc}")
    print(f"    Total (a.u.): {mu_total}")
    print(f"    |mu| (Debye): {np.linalg.norm(mu_total) * 2.541746:.4f}")  # 1 a.u. = 2.541746 Debye

    # =========================================================================
    # Section 2: Two-Electron Integrals
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. TWO-ELECTRON INTEGRALS (ERIs)")
    print("=" * 50)

    # Use smaller molecule for full ERI demonstration
    mol_h2 = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nUsing H2/STO-3G for ERI demonstration (nao={mol_h2.nao})")

    # -----------------------------------------
    # Full 4-index tensor (aosym='s1')
    # -----------------------------------------
    eri_s1 = mol_h2.intor("int2e", aosym="s1")
    print(f"\n--- ERI with aosym='s1' (no symmetry) ---")
    print(f"  Shape: {eri_s1.shape}")
    print(f"  Storage: {eri_s1.nbytes / 1024:.2f} KB")
    print(f"  Total elements: {eri_s1.size}")

    # -----------------------------------------
    # With 4-fold symmetry (aosym='s4')
    # -----------------------------------------
    eri_s4 = mol_h2.intor("int2e", aosym="s4")
    print(f"\n--- ERI with aosym='s4' (4-fold symmetry) ---")
    print(f"  Shape: {eri_s4.shape}")
    print(f"  Packed as triangular in (ij) and (kl) pairs")

    # -----------------------------------------
    # With 8-fold symmetry (aosym='s8')
    # -----------------------------------------
    eri_s8 = mol_h2.intor("int2e", aosym="s8")
    print(f"\n--- ERI with aosym='s8' (8-fold symmetry) ---")
    print(f"  Shape: {eri_s8.shape}")
    print(f"  Storage: {eri_s8.nbytes / 1024:.2f} KB")

    # Verify symmetries numerically
    print("\n--- Numerical verification of ERI symmetries ---")
    nao = mol_h2.nao
    eri = eri_s1  # Use full tensor

    # (ij|kl) = (ji|kl)
    check1 = np.allclose(eri, eri.transpose(1, 0, 2, 3))
    print(f"  (ij|kl) = (ji|kl): {check1}")

    # (ij|kl) = (ij|lk)
    check2 = np.allclose(eri, eri.transpose(0, 1, 3, 2))
    print(f"  (ij|kl) = (ij|lk): {check2}")

    # (ij|kl) = (kl|ij)
    check3 = np.allclose(eri, eri.transpose(2, 3, 0, 1))
    print(f"  (ij|kl) = (kl|ij): {check3}")

    # =========================================================================
    # Section 3: Integral Properties and Sanity Checks
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. INTEGRAL SANITY CHECKS")
    print("=" * 50)

    # Use water for these checks
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")

    # Run HF to get density
    dm = mf.make_rdm1()

    # Check 1: Electron count
    n_elec = np.trace(dm @ S)
    print(f"\n  Electron count: Tr[P*S] = {n_elec:.6f}")
    print(f"    Expected: {mol.nelectron}")
    print(f"    Match: {np.isclose(n_elec, mol.nelectron)}")

    # Check 2: One-electron energy
    E_1e = np.einsum('ij,ji->', h, dm)
    print(f"\n  One-electron energy: Tr[h*P] = {E_1e:.10f} Hartree")

    # Check 3: Kinetic energy expectation value
    T_expect = np.einsum('ij,ji->', T, dm)
    print(f"\n  Kinetic energy: <T> = {T_expect:.10f} Hartree")
    print(f"    Should be positive: {T_expect > 0}")

    # Check 4: Nuclear attraction expectation value
    V_expect = np.einsum('ij,ji->', V, dm)
    print(f"\n  Nuclear attraction: <V_en> = {V_expect:.10f} Hartree")
    print(f"    Should be negative: {V_expect < 0}")

    # Check 5: Virial theorem diagnostic (approximate for finite basis)
    E_nuc = mol.energy_nuc()
    E_total = mf.e_tot
    E_elec = E_total - E_nuc
    virial = -T_expect / (E_total - T_expect)  # Approximate virial ratio
    print(f"\n  Virial ratio diagnostic: -<T>/<V_total> = {virial:.6f}")
    print(f"    (Should be ~1 for variational minimum)")

    # =========================================================================
    # Section 4: Additional One-Electron Integrals
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. ADDITIONAL ONE-ELECTRON INTEGRALS")
    print("=" * 50)

    # Quadrupole integrals
    mol.set_common_origin([0, 0, 0])
    rr_ints = mol.intor("int1e_rr")  # <mu|r*r|nu>
    print(f"\n--- Quadrupole-type integrals <mu|r_i r_j|nu> ---")
    print(f"  Shape: {rr_ints.shape} (xx, xy, xz, yy, yz, zz components)")

    # Kinetic energy gradient (useful for force calculations)
    print("\n--- Integral gradient availability ---")
    print("  Available gradient integrals in PySCF:")
    print("    int1e_ipovlp: Overlap gradient")
    print("    int1e_ipkin:  Kinetic gradient")
    print("    int1e_ipnuc:  Nuclear attraction gradient")
    print("    int2e_ip1:    ERI gradient on first center")

    # =========================================================================
    # Section 5: Memory Scaling
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. MEMORY SCALING WITH BASIS SIZE")
    print("=" * 50)

    print("\n  ERI storage scaling (4-index tensor):")
    print("  n_ao   |  s1 elements  |  s8 elements  |  s1 (MB)  |  s8 (MB)")
    print("  " + "-" * 60)

    for n in [10, 50, 100, 200]:
        n4 = n**4
        n8 = n*(n+1)//2
        n8 = n8*(n8+1)//2
        mb_s1 = n4 * 8 / (1024**2)
        mb_s8 = n8 * 8 / (1024**2)
        print(f"  {n:4d}   | {n4:12d}  | {n8:12d}  | {mb_s1:8.1f}  | {mb_s8:8.1f}")

    print("\n  Note: For n=200, s8 saves factor of ~8 in storage")

    print("\n" + "=" * 70)
    print("Integral inventory demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
