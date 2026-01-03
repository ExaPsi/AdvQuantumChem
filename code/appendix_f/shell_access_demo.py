#!/usr/bin/env python3
"""
Shell-by-Shell Integral Access in PySCF

Demonstrates shell-by-shell integral extraction using intor_by_shell,
shell range mapping with ao_loc_nr(), and debugging techniques.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

This is useful for:
- Understanding how integrals are organized
- Debugging integral evaluation issues
- Implementing direct SCF algorithms
"""

import numpy as np
from pyscf import gto


def main():
    print("=" * 70)
    print("Shell-by-Shell Integral Access in PySCF")
    print("=" * 70)

    # =========================================================================
    # Section 1: Shell Structure Overview
    # =========================================================================

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
    print(f"  Number of atoms: {mol.natm}")
    print(f"  Number of shells: {mol.nbas}")
    print(f"  Number of AOs: {mol.nao}")

    # =========================================================================
    # Section 2: ao_loc_nr() - Shell to AO Index Mapping
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. Shell to AO Index Mapping (ao_loc_nr)")
    print("=" * 50)

    ao_loc = mol.ao_loc_nr()
    print(f"\n  ao_loc array: {ao_loc}")
    print(f"  Length: {len(ao_loc)} (n_shells + 1)")

    print("\n  Interpretation:")
    print("    ao_loc[i] = starting AO index for shell i")
    print("    ao_loc[i+1] - ao_loc[i] = number of AOs in shell i")

    print("\n  Shell-by-shell breakdown:")
    print("  " + "-" * 65)
    print(f"  {'Shell':>5} | {'Atom':>6} | {'L':>3} | {'Type':>6} | {'AO range':>12} | {'nAO':>4}")
    print("  " + "-" * 65)

    L_names = ['s', 'p', 'd', 'f', 'g']
    for i in range(mol.nbas):
        atom_id = mol.bas_atom(i)
        atom_sym = mol.atom_symbol(atom_id)
        L = mol.bas_angular(i)
        L_name = L_names[L]
        ao_start = ao_loc[i]
        ao_end = ao_loc[i + 1]
        n_ao = ao_end - ao_start
        print(f"  {i:5d} | {atom_id:2d} ({atom_sym:2s}) | {L:3d} | {L_name:>6} | [{ao_start:3d}, {ao_end:3d}) | {n_ao:4d}")

    # =========================================================================
    # Section 3: One-Electron Shell Pair Integrals
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. One-Electron Shell Pair Integrals")
    print("=" * 50)

    # Get full overlap matrix for comparison
    S_full = mol.intor("int1e_ovlp")

    print("\n  Extracting overlap integrals shell-by-shell...")

    # Example: Get overlap between shells 0 and 1
    shell_i, shell_j = 0, 1
    S_ij = mol.intor_by_shell("int1e_ovlp", [shell_i, shell_j])

    print(f"\n  Shell pair ({shell_i}, {shell_j}):")
    print(f"    Shell {shell_i}: L={mol.bas_angular(shell_i)}, "
          f"AOs [{ao_loc[shell_i]}, {ao_loc[shell_i+1]})")
    print(f"    Shell {shell_j}: L={mol.bas_angular(shell_j)}, "
          f"AOs [{ao_loc[shell_j]}, {ao_loc[shell_j+1]})")
    print(f"    Integral block shape: {S_ij.shape}")
    print(f"    Values:\n{S_ij}")

    # Verify against full matrix
    i_start, i_end = ao_loc[shell_i], ao_loc[shell_i + 1]
    j_start, j_end = ao_loc[shell_j], ao_loc[shell_j + 1]
    S_ij_ref = S_full[i_start:i_end, j_start:j_end]

    print(f"\n    Verification: ||S_shell - S_full_block|| = "
          f"{np.linalg.norm(S_ij - S_ij_ref):.2e}")

    # Build full matrix from shell pairs
    print("\n  Building full overlap matrix from shell pairs...")
    S_rebuilt = np.zeros((mol.nao, mol.nao))

    for i in range(mol.nbas):
        for j in range(mol.nbas):
            S_block = mol.intor_by_shell("int1e_ovlp", [i, j])
            i_s, i_e = ao_loc[i], ao_loc[i + 1]
            j_s, j_e = ao_loc[j], ao_loc[j + 1]
            S_rebuilt[i_s:i_e, j_s:j_e] = S_block

    print(f"    ||S_rebuilt - S_full|| = {np.linalg.norm(S_rebuilt - S_full):.2e}")

    # =========================================================================
    # Section 4: Two-Electron Shell Quartet Integrals
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Two-Electron Shell Quartet Integrals")
    print("=" * 50)

    # Use smaller system for clarity
    mol_small = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    ao_loc_small = mol_small.ao_loc_nr()
    eri_full = mol_small.intor("int2e", aosym="s1")

    print(f"\n  Using H2/STO-3G (2 shells, 2 AOs)")
    print(f"  ao_loc: {ao_loc_small}")

    # Get ERI for shell quartet (0,0,0,0)
    shells = [0, 0, 0, 0]
    eri_0000 = mol_small.intor_by_shell("int2e", shells)

    print(f"\n  Shell quartet {tuple(shells)}:")
    print(f"    Shape: {eri_0000.shape}")
    print(f"    Value: {eri_0000.flatten()}")

    # Verify
    eri_0000_ref = eri_full[0:1, 0:1, 0:1, 0:1]
    print(f"    Matches full tensor: {np.allclose(eri_0000, eri_0000_ref)}")

    # All shell quartets for H2/STO-3G
    print("\n  All shell quartets for H2/STO-3G:")
    print("  " + "-" * 50)

    for i in range(mol_small.nbas):
        for j in range(mol_small.nbas):
            for k in range(mol_small.nbas):
                for l in range(mol_small.nbas):
                    eri_ijkl = mol_small.intor_by_shell("int2e", [i, j, k, l])
                    i_s = ao_loc_small[i]
                    j_s = ao_loc_small[j]
                    k_s = ao_loc_small[k]
                    l_s = ao_loc_small[l]
                    ref_val = eri_full[i_s, j_s, k_s, l_s]
                    print(f"    ({i}{j}|{k}{l}): {eri_ijkl.flatten()[0]:.10f} "
                          f"(ref: {ref_val:.10f})")

    # =========================================================================
    # Section 5: Shell Information Functions
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. Shell Information Functions")
    print("=" * 50)

    print("\n  Available shell information functions:")

    mol_demo = mol  # Use H2O

    print(f"\n  For H2O/cc-pVDZ, shell 3:")
    shell_id = 3

    print(f"    bas_atom({shell_id})     = {mol_demo.bas_atom(shell_id)} "
          f"(atom index)")
    print(f"    bas_angular({shell_id})  = {mol_demo.bas_angular(shell_id)} "
          f"(angular momentum L)")
    print(f"    bas_nprim({shell_id})    = {mol_demo.bas_nprim(shell_id)} "
          f"(number of primitives)")
    print(f"    bas_nctr({shell_id})     = {mol_demo.bas_nctr(shell_id)} "
          f"(number of contractions)")

    # Get exponents and coefficients
    print(f"\n    Exponents (bas_exp):")
    exps = mol_demo.bas_exp(shell_id)
    for idx, exp in enumerate(exps):
        print(f"      [{idx}]: {exp:.10f}")

    print(f"\n    Contraction coefficients (bas_ctr_coeff):")
    coeffs = mol_demo.bas_ctr_coeff(shell_id)
    print(f"      Shape: {coeffs.shape}")
    print(f"      Values:\n{coeffs}")

    # =========================================================================
    # Section 6: Debugging Example
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. Debugging Example: Verifying Specific Integrals")
    print("=" * 50)

    print("\n  Scenario: Verify a specific ERI value")
    print("  Question: What is (1s_O 1s_O | 1s_H1 1s_H1) for H2O?")

    # Identify shells
    # Shell 0 is 1s on O, shells containing H 1s
    print("\n  Step 1: Identify relevant shells")
    for i in range(mol.nbas):
        atom = mol.bas_atom(i)
        L = mol.bas_angular(i)
        if L == 0:  # s-type
            print(f"    Shell {i}: {mol.atom_symbol(atom)} 1s")

    print("\n  Step 2: Extract shell quartet integral")
    # O 1s is shell 0, H1 1s is shell 3, H2 1s is shell 5 (approximate)
    # Let's find them properly
    O_1s_shell = 0  # First shell on O
    H1_1s_shell = None
    H2_1s_shell = None

    for i in range(mol.nbas):
        atom = mol.bas_atom(i)
        L = mol.bas_angular(i)
        if L == 0:  # s-type
            if mol.atom_symbol(atom) == 'H' and H1_1s_shell is None:
                H1_1s_shell = i
            elif mol.atom_symbol(atom) == 'H' and H1_1s_shell is not None:
                H2_1s_shell = i
                break

    print(f"    O 1s: shell {O_1s_shell}")
    print(f"    H1 1s: shell {H1_1s_shell}")
    print(f"    H2 1s: shell {H2_1s_shell}")

    if H1_1s_shell is not None:
        eri_O_H1 = mol.intor_by_shell("int2e", [O_1s_shell, O_1s_shell,
                                                 H1_1s_shell, H1_1s_shell])
        print(f"\n  Step 3: Result")
        print(f"    (1s_O 1s_O | 1s_H1 1s_H1) = {eri_O_H1[0,0,0,0]:.10f} Hartree")

    # =========================================================================
    # Section 7: AO Labels
    # =========================================================================
    print("\n" + "=" * 50)
    print("6. AO Labels for Debugging")
    print("=" * 50)

    print("\n  AO labels for H2O/cc-pVDZ:")
    labels = mol.ao_labels()
    for i, label in enumerate(labels):
        print(f"    AO {i:2d}: {label}")

    print("\n" + "=" * 70)
    print("Shell access demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
