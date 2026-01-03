#!/usr/bin/env python3
"""
ERI Symmetry Demonstration

Numerically verifies all 8-fold ERI symmetries and compares storage
requirements for different aosym options.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

ERI symmetries (chemist's notation):
  (1) (ij|kl) = (ji|kl)    swap i <-> j
  (2) (ij|kl) = (ij|lk)    swap k <-> l
  (3) (ij|kl) = (ji|lk)    combination of (1) and (2)
  (4) (ij|kl) = (kl|ij)    swap bra <-> ket
  (5-8) combinations of above
"""

import numpy as np
from pyscf import gto, scf
import time


def main():
    print("=" * 70)
    print("ERI Symmetry Demonstration")
    print("=" * 70)

    # =========================================================================
    # Section 1: Build Test Systems
    # =========================================================================

    # Small system for detailed verification
    mol_small = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Medium system for storage comparison
    mol_medium = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nTest systems:")
    print(f"  H2/STO-3G:   nao = {mol_small.nao}")
    print(f"  H2O/cc-pVDZ: nao = {mol_medium.nao}")

    # =========================================================================
    # Section 2: Numerical Verification of 8-fold Symmetry
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. Numerical Verification of 8-fold ERI Symmetry")
    print("=" * 50)

    # Get full ERI tensor
    eri = mol_small.intor("int2e", aosym="s1")
    nao = mol_small.nao

    print(f"\nUsing H2/STO-3G (nao = {nao})")
    print(f"ERI shape: {eri.shape}")

    print("\nVerifying all 8 symmetry operations:")
    print("-" * 50)

    # Symmetry 1: (ij|kl) = (ji|kl)
    eri_sym1 = eri.transpose(1, 0, 2, 3)
    diff1 = np.max(np.abs(eri - eri_sym1))
    print(f"(1) (ij|kl) = (ji|kl):  max diff = {diff1:.2e}")

    # Symmetry 2: (ij|kl) = (ij|lk)
    eri_sym2 = eri.transpose(0, 1, 3, 2)
    diff2 = np.max(np.abs(eri - eri_sym2))
    print(f"(2) (ij|kl) = (ij|lk):  max diff = {diff2:.2e}")

    # Symmetry 3: (ij|kl) = (ji|lk)
    eri_sym3 = eri.transpose(1, 0, 3, 2)
    diff3 = np.max(np.abs(eri - eri_sym3))
    print(f"(3) (ij|kl) = (ji|lk):  max diff = {diff3:.2e}")

    # Symmetry 4: (ij|kl) = (kl|ij)
    eri_sym4 = eri.transpose(2, 3, 0, 1)
    diff4 = np.max(np.abs(eri - eri_sym4))
    print(f"(4) (ij|kl) = (kl|ij):  max diff = {diff4:.2e}")

    # Symmetry 5: (ij|kl) = (lk|ij)
    eri_sym5 = eri.transpose(3, 2, 0, 1)
    diff5 = np.max(np.abs(eri - eri_sym5))
    print(f"(5) (ij|kl) = (lk|ij):  max diff = {diff5:.2e}")

    # Symmetry 6: (ij|kl) = (kl|ji)
    eri_sym6 = eri.transpose(2, 3, 1, 0)
    diff6 = np.max(np.abs(eri - eri_sym6))
    print(f"(6) (ij|kl) = (kl|ji):  max diff = {diff6:.2e}")

    # Symmetry 7: (ij|kl) = (lk|ji)
    eri_sym7 = eri.transpose(3, 2, 1, 0)
    diff7 = np.max(np.abs(eri - eri_sym7))
    print(f"(7) (ij|kl) = (lk|ji):  max diff = {diff7:.2e}")

    # Symmetry 8: Identity (included for completeness)
    diff8 = np.max(np.abs(eri - eri))
    print(f"(8) (ij|kl) = (ij|kl):  max diff = {diff8:.2e} (identity)")

    print("\nAll symmetries verified!" if max(diff1, diff2, diff3, diff4, diff5, diff6, diff7) < 1e-14 else "Symmetry violation detected!")

    # =========================================================================
    # Section 3: Storage Comparison
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. Storage Comparison: s1 vs s4 vs s8")
    print("=" * 50)

    # Compute ERIs with different symmetry options
    print(f"\nUsing H2O/cc-pVDZ (nao = {mol_medium.nao})")

    t0 = time.time()
    eri_s1 = mol_medium.intor("int2e", aosym="s1")
    t_s1 = time.time() - t0

    t0 = time.time()
    eri_s4 = mol_medium.intor("int2e", aosym="s4")
    t_s4 = time.time() - t0

    t0 = time.time()
    eri_s8 = mol_medium.intor("int2e", aosym="s8")
    t_s8 = time.time() - t0

    nao = mol_medium.nao

    print("\n  aosym | Shape" + " " * 25 + "| Elements    | Size (KB)  | Time (ms)")
    print("  " + "-" * 75)
    print(f"  s1    | {str(eri_s1.shape):30s} | {eri_s1.size:10d} | {eri_s1.nbytes/1024:9.1f} | {t_s1*1000:8.2f}")
    print(f"  s4    | {str(eri_s4.shape):30s} | {eri_s4.size:10d} | {eri_s4.nbytes/1024:9.1f} | {t_s4*1000:8.2f}")
    print(f"  s8    | {str(eri_s8.shape):30s} | {eri_s8.size:10d} | {eri_s8.nbytes/1024:9.1f} | {t_s8*1000:8.2f}")

    print(f"\n  Storage reduction factors:")
    print(f"    s1 -> s4: {eri_s1.size / eri_s4.size:.1f}x")
    print(f"    s1 -> s8: {eri_s1.size / eri_s8.size:.1f}x")

    # Theoretical storage
    n = nao
    n_s1 = n**4
    n_s4 = (n*(n+1)//2)**2
    n_s8 = (n*(n+1)//2) * ((n*(n+1)//2)+1) // 2

    print(f"\n  Theoretical storage (nao = {nao}):")
    print(f"    s1: n^4              = {n_s1:,d}")
    print(f"    s4: [n(n+1)/2]^2     = {n_s4:,d}")
    print(f"    s8: n_s4*(n_s4+1)/2  = {n_s8:,d}")

    # =========================================================================
    # Section 4: Unpacking s8 to Full Tensor
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Unpacking Symmetric ERIs to Full Tensor")
    print("=" * 50)

    # Use small system
    eri_s1_small = mol_small.intor("int2e", aosym="s1")
    eri_s8_small = mol_small.intor("int2e", aosym="s8")
    nao_small = mol_small.nao

    print(f"\nH2/STO-3G: Unpacking s8 -> s1")
    print(f"  s8 shape: {eri_s8_small.shape}")
    print(f"  s1 shape: {eri_s1_small.shape}")

    # Unpack s8 to full tensor using ao2mo utilities
    from pyscf import ao2mo
    eri_unpacked = ao2mo.restore(1, eri_s8_small, nao_small)

    print(f"\n  Unpacked shape: {eri_unpacked.shape}")
    print(f"  Max difference: {np.max(np.abs(eri_unpacked - eri_s1_small)):.2e}")
    print(f"  Arrays match: {np.allclose(eri_unpacked, eri_s1_small)}")

    # =========================================================================
    # Section 5: Shell Quartet Structure
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. Shell Quartet Structure")
    print("=" * 50)

    print(f"\nH2O/cc-pVDZ shell structure:")
    print(f"  Number of shells: {mol_medium.nbas}")

    # Get shell ranges
    ao_loc = mol_medium.ao_loc_nr()
    print(f"\n  Shell AO ranges (ao_loc):")
    print(f"    {ao_loc}")

    print(f"\n  Shell details:")
    for i in range(mol_medium.nbas):
        atom_id = mol_medium.bas_atom(i)
        L = mol_medium.bas_angular(i)
        nprim = mol_medium.bas_nprim(i)
        nctr = mol_medium.bas_nctr(i)
        ao_start = ao_loc[i]
        ao_end = ao_loc[i+1]
        L_name = ['s', 'p', 'd', 'f', 'g'][L]
        print(f"    Shell {i:2d}: Atom {atom_id} ({mol_medium.atom_symbol(atom_id)}), "
              f"L={L} ({L_name}), {nprim} prim, {nctr} ctr, AOs [{ao_start}:{ao_end})")

    # Number of shell quartets
    n_shells = mol_medium.nbas
    n_quartets = n_shells**4
    n_quartets_sym = n_shells * (n_shells+1) // 2
    n_quartets_sym = n_quartets_sym * (n_quartets_sym + 1) // 2

    print(f"\n  Shell quartet count:")
    print(f"    Without symmetry: {n_quartets:,d}")
    print(f"    With 8-fold sym:  {n_quartets_sym:,d}")
    print(f"    Reduction factor: {n_quartets / n_quartets_sym:.1f}x")

    # =========================================================================
    # Section 6: Specific ERI Values
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. Specific ERI Values")
    print("=" * 50)

    eri_small = mol_small.intor("int2e", aosym="s1")
    nao_small = mol_small.nao

    print(f"\nH2/STO-3G ERI values (all 2x2x2x2 = 16 elements):")
    print("\n  (ij|kl)    Value")
    print("  " + "-" * 30)

    for i in range(nao_small):
        for j in range(nao_small):
            for k in range(nao_small):
                for l in range(nao_small):
                    print(f"  ({i}{j}|{k}{l})    {eri_small[i,j,k,l]:.10f}")

    print("\n  Note: Many values are equal due to 8-fold symmetry")
    print(f"  Unique values: {len(set(eri_small.flatten()))}")

    # =========================================================================
    # Section 7: J and K Construction from ERIs
    # =========================================================================
    print("\n" + "=" * 50)
    print("6. J and K Matrix Construction")
    print("=" * 50)

    # Get converged density matrix
    mf = scf.RHF(mol_small)
    mf.kernel()
    P = mf.make_rdm1()

    print(f"\nUsing H2/STO-3G with converged density matrix")
    print(f"  Density matrix P:\n{P}")

    # Coulomb matrix: J_ij = sum_kl (ij|kl) P_kl
    J = np.einsum('ijkl,kl->ij', eri_small, P)

    # Exchange matrix: K_ij = sum_kl (ik|jl) P_kl
    K = np.einsum('ikjl,kl->ij', eri_small, P)

    print(f"\n  Coulomb matrix J:\n{J}")
    print(f"\n  Exchange matrix K:\n{K}")

    # Verify against PySCF
    J_ref, K_ref = mf.get_jk()
    print(f"\n  Verification against PySCF:")
    print(f"    ||J - J_ref|| = {np.linalg.norm(J - J_ref):.2e}")
    print(f"    ||K - K_ref|| = {np.linalg.norm(K - K_ref):.2e}")

    print("\n" + "=" * 70)
    print("ERI symmetry demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
