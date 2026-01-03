#!/usr/bin/env python3
"""
J/K Matrix Construction Comparison

Compares manual J/K construction using einsum against PySCF's optimized
implementation. Includes timing comparison and verification.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

Key equations (chemist's notation):
  Coulomb:  J_ij = sum_kl (ij|kl) P_kl
  Exchange: K_ij = sum_kl (ik|jl) P_kl
  Fock:     F = h + J - 0.5*K (RHF)
"""

import numpy as np
from pyscf import gto, scf
import time


def build_J_einsum(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build Coulomb matrix using einsum.

    J_ij = sum_kl (ij|kl) P_kl

    Physical meaning: Coulomb repulsion between electron at (i,j)
    and total electron density P_kl.
    """
    return np.einsum('ijkl,kl->ij', eri, P)


def build_K_einsum(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build Exchange matrix using einsum.

    K_ij = sum_kl (ik|jl) P_kl

    Physical meaning: Exchange interaction arising from
    antisymmetry of fermionic wavefunction.
    """
    return np.einsum('ikjl,kl->ij', eri, P)


def build_J_loops(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build Coulomb matrix using explicit loops (pedagogical).

    Slower but clearer for understanding the contraction pattern.
    """
    nao = P.shape[0]
    J = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):
                    J[i, j] += eri[i, j, k, l] * P[k, l]
    return J


def build_K_loops(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build Exchange matrix using explicit loops (pedagogical).

    Note the different index pattern compared to Coulomb.
    """
    nao = P.shape[0]
    K = np.zeros((nao, nao))
    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):
                    K[i, j] += eri[i, k, j, l] * P[k, l]
    return K


def main():
    print("=" * 70)
    print("J/K Matrix Construction Comparison")
    print("=" * 70)

    # =========================================================================
    # Setup Test System
    # =========================================================================

    mol = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nTest system: H2O / STO-3G")
    print(f"  nao = {mol.nao}")

    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")

    # Run HF to get density matrix
    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    print(f"  Converged HF energy: {mf.e_tot:.10f} Hartree")

    # =========================================================================
    # Section 1: Compare Different J/K Construction Methods
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. Comparing J/K Construction Methods")
    print("=" * 50)

    # Method 1: einsum
    t0 = time.time()
    J_einsum = build_J_einsum(eri, P)
    K_einsum = build_K_einsum(eri, P)
    t_einsum = time.time() - t0

    # Method 2: explicit loops (for small systems only)
    t0 = time.time()
    J_loops = build_J_loops(eri, P)
    K_loops = build_K_loops(eri, P)
    t_loops = time.time() - t0

    # Method 3: PySCF get_jk
    t0 = time.time()
    J_pyscf, K_pyscf = mf.get_jk(mol, P)
    t_pyscf = time.time() - t0

    print(f"\n  Method         | J error      | K error      | Time (ms)")
    print("  " + "-" * 60)
    print(f"  einsum         | {np.linalg.norm(J_einsum - J_pyscf):.2e}   | "
          f"{np.linalg.norm(K_einsum - K_pyscf):.2e}   | {t_einsum*1000:.3f}")
    print(f"  explicit loops | {np.linalg.norm(J_loops - J_pyscf):.2e}   | "
          f"{np.linalg.norm(K_loops - K_pyscf):.2e}   | {t_loops*1000:.3f}")
    print(f"  PySCF get_jk   | 0.00e+00     | 0.00e+00     | {t_pyscf*1000:.3f}")

    # =========================================================================
    # Section 2: Verify J and K Properties
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. Verifying J and K Matrix Properties")
    print("=" * 50)

    J = J_einsum
    K = K_einsum

    print(f"\n  Coulomb matrix J:")
    print(f"    Shape: {J.shape}")
    print(f"    Symmetric: ||J - J^T|| = {np.linalg.norm(J - J.T):.2e}")
    print(f"    All positive diagonal: {np.all(np.diag(J) > 0)}")
    print(f"    Diagonal range: [{np.diag(J).min():.4f}, {np.diag(J).max():.4f}]")
    print(f"    Trace: {np.trace(J):.6f}")

    print(f"\n  Exchange matrix K:")
    print(f"    Shape: {K.shape}")
    print(f"    Symmetric: ||K - K^T|| = {np.linalg.norm(K - K.T):.2e}")
    print(f"    All positive diagonal: {np.all(np.diag(K) > 0)}")
    print(f"    Diagonal range: [{np.diag(K).min():.4f}, {np.diag(K).max():.4f}]")
    print(f"    Trace: {np.trace(K):.6f}")

    # Physical insight: J > K (Coulomb larger than exchange)
    print(f"\n  Physical check: J > K (elementwise)?")
    print(f"    Mean(J) = {np.mean(J):.6f}")
    print(f"    Mean(K) = {np.mean(K):.6f}")
    print(f"    Tr[PJ] = {np.einsum('ij,ji->', P, J):.6f}")
    print(f"    Tr[PK] = {np.einsum('ij,ji->', P, K):.6f}")

    # =========================================================================
    # Section 3: Fock Matrix Verification
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Fock Matrix Verification")
    print("=" * 50)

    # Build Fock matrix
    F_manual = h + J - 0.5 * K
    F_pyscf = mf.get_fock(dm=P)

    print(f"\n  Fock matrix F = h + J - 0.5*K")
    print(f"    ||F_manual - F_pyscf|| = {np.linalg.norm(F_manual - F_pyscf):.2e}")

    # Energy contributions
    E_nuc = mol.energy_nuc()
    E_1e = np.einsum('ij,ji->', P, h)
    E_J = 0.5 * np.einsum('ij,ji->', P, J)
    E_K = -0.25 * np.einsum('ij,ji->', P, K)
    E_elec = E_1e + E_J + E_K
    E_tot = E_elec + E_nuc

    print(f"\n  Energy breakdown:")
    print(f"    E_1e (one-electron):  {E_1e:.10f} Hartree")
    print(f"    E_J  (Coulomb/2):     {E_J:.10f} Hartree")
    print(f"    E_K  (-Exchange/4):   {E_K:.10f} Hartree")
    print(f"    E_elec:               {E_elec:.10f} Hartree")
    print(f"    E_nuc:                {E_nuc:.10f} Hartree")
    print(f"    E_tot:                {E_tot:.10f} Hartree")
    print(f"    PySCF E_tot:          {mf.e_tot:.10f} Hartree")
    print(f"    Difference:           {abs(E_tot - mf.e_tot):.2e} Hartree")

    # Alternative energy formula
    E_alt = 0.5 * np.einsum('ij,ij->', P, h + F_manual) + E_nuc
    print(f"\n  Alternative: E = 0.5*Tr[P(h+F)] + E_nuc")
    print(f"    E_tot (alternative): {E_alt:.10f} Hartree")

    # =========================================================================
    # Section 4: Timing on Larger System
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. Timing Comparison on Larger Basis")
    print("=" * 50)

    mol_large = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\n  Test system: H2O / cc-pVDZ (nao = {mol_large.nao})")

    mf_large = scf.RHF(mol_large)
    mf_large.kernel()
    P_large = mf_large.make_rdm1()

    # Full ERI approach
    t0 = time.time()
    eri_large = mol_large.intor("int2e", aosym="s1")
    t_eri = time.time() - t0

    t0 = time.time()
    J_large = np.einsum('ijkl,kl->ij', eri_large, P_large)
    K_large = np.einsum('ikjl,kl->ij', eri_large, P_large)
    t_jk_einsum = time.time() - t0

    # PySCF approach (more efficient)
    t0 = time.time()
    J_ref, K_ref = mf_large.get_jk(mol_large, P_large)
    t_jk_pyscf = time.time() - t0

    print(f"\n  Method           | Time (ms)")
    print("  " + "-" * 35)
    print(f"  ERI computation  | {t_eri*1000:.2f}")
    print(f"  J/K (einsum)     | {t_jk_einsum*1000:.2f}")
    print(f"  J/K (PySCF)      | {t_jk_pyscf*1000:.2f}")
    print(f"  Total (manual)   | {(t_eri + t_jk_einsum)*1000:.2f}")

    print(f"\n  Memory for ERI: {eri_large.nbytes / 1024**2:.1f} MB")

    # Verify results match
    print(f"\n  Verification:")
    print(f"    ||J_einsum - J_pyscf|| = {np.linalg.norm(J_large - J_ref):.2e}")
    print(f"    ||K_einsum - K_pyscf|| = {np.linalg.norm(K_large - K_ref):.2e}")

    # =========================================================================
    # Section 5: Understanding Index Patterns
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. Understanding Index Patterns")
    print("=" * 50)

    # Small system for clarity
    mol_tiny = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                     unit="Angstrom", verbose=0)
    eri_tiny = mol_tiny.intor("int2e", aosym="s1")
    mf_tiny = scf.RHF(mol_tiny)
    mf_tiny.kernel()
    P_tiny = mf_tiny.make_rdm1()

    print(f"\n  H2 / STO-3G (2 AOs)")

    print(f"\n  Density matrix P:")
    print(f"    {P_tiny}")

    print(f"\n  Coulomb J_00 = sum_kl (00|kl) P_kl:")
    J00_manual = 0
    print(f"    ", end="")
    terms = []
    for k in range(2):
        for l in range(2):
            term = eri_tiny[0, 0, k, l] * P_tiny[k, l]
            terms.append(f"(00|{k}{l})*P_{k}{l}")
            J00_manual += term
    print(" + ".join(terms))
    print(f"    = {J00_manual:.6f}")

    J_tiny = build_J_einsum(eri_tiny, P_tiny)
    print(f"    J_00 (einsum) = {J_tiny[0,0]:.6f}")

    print(f"\n  Exchange K_00 = sum_kl (0k|0l) P_kl:")
    K00_manual = 0
    print(f"    ", end="")
    terms = []
    for k in range(2):
        for l in range(2):
            term = eri_tiny[0, k, 0, l] * P_tiny[k, l]
            terms.append(f"(0{k}|0{l})*P_{k}{l}")
            K00_manual += term
    print(" + ".join(terms))
    print(f"    = {K00_manual:.6f}")

    K_tiny = build_K_einsum(eri_tiny, P_tiny)
    print(f"    K_00 (einsum) = {K_tiny[0,0]:.6f}")

    print("\n" + "=" * 70)
    print("J/K construction comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
