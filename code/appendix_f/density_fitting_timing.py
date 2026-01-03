#!/usr/bin/env python3
"""
Density Fitting (DF) vs Conventional HF Timing Comparison

Compares conventional HF with density-fitted HF (DF-HF, also called RI-HF)
in terms of accuracy and computational cost.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

Density Fitting factorizes ERIs:
  (ij|kl) ~ sum_Q B_ij^Q B_kl^Q

where B_ij^Q = sum_P (ij|P) (P|Q)^(-1/2)

This reduces ERI storage from O(N^4) to O(N^2 * N_aux).
"""

import numpy as np
from pyscf import gto, scf, df
import time


def run_conventional_hf(mol, verbose: bool = True) -> dict:
    """Run conventional (in-core) RHF calculation."""
    mf = scf.RHF(mol)
    mf.verbose = 0

    t0 = time.time()
    E = mf.kernel()
    t_total = time.time() - t0

    return {
        'energy': E,
        'time': t_total,
        'converged': mf.converged,
        'iterations': mf.cycles
    }


def run_df_hf(mol, auxbasis: str = None, verbose: bool = True) -> dict:
    """Run density-fitted RHF calculation."""
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = 0

    t0 = time.time()
    E = mf.kernel()
    t_total = time.time() - t0

    # Get auxiliary basis info
    auxmol = mf.with_df.auxmol
    naux = auxmol.nao if auxmol is not None else 0

    return {
        'energy': E,
        'time': t_total,
        'converged': mf.converged,
        'iterations': mf.cycles,
        'naux': naux,
        'auxbasis': auxbasis or 'auto'
    }


def main():
    print("=" * 70)
    print("Density Fitting vs Conventional HF Comparison")
    print("=" * 70)

    # =========================================================================
    # Test 1: Small molecule (H2O)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: H2O - Small System")
    print("=" * 50)

    mol_h2o = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\n  Molecule: H2O")
    print(f"  Basis: cc-pVDZ")
    print(f"  Number of AOs: {mol_h2o.nao}")

    # Conventional HF
    result_conv = run_conventional_hf(mol_h2o)

    # DF-HF with default auxiliary basis
    result_df = run_df_hf(mol_h2o)

    # DF-HF with explicit auxiliary basis
    result_df_jkfit = run_df_hf(mol_h2o, auxbasis='cc-pvdz-jkfit')

    print(f"\n  Results:")
    print(f"  {'Method':<25} | {'Energy (Hartree)':<18} | {'Error (mHa)':<12} | {'Time (s)':<10}")
    print("  " + "-" * 75)
    print(f"  {'Conventional HF':<25} | {result_conv['energy']:<18.10f} | {'---':<12} | {result_conv['time']:<10.4f}")
    print(f"  {'DF-HF (auto aux)':<25} | {result_df['energy']:<18.10f} | "
          f"{(result_df['energy'] - result_conv['energy'])*1000:<12.6f} | {result_df['time']:<10.4f}")
    print(f"  {'DF-HF (cc-pvdz-jkfit)':<25} | {result_df_jkfit['energy']:<18.10f} | "
          f"{(result_df_jkfit['energy'] - result_conv['energy'])*1000:<12.6f} | {result_df_jkfit['time']:<10.4f}")

    print(f"\n  Auxiliary basis size: {result_df_jkfit['naux']} (vs {mol_h2o.nao} AOs)")

    # =========================================================================
    # Test 2: Medium molecule (benzene approximation using 6 water molecules)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: (H2O)_2 - Larger System")
    print("=" * 50)

    mol_2h2o = gto.M(
        atom="""
            O   0.0000   0.0000   0.0000
            H   0.7572   0.5869   0.0000
            H  -0.7572   0.5869   0.0000
            O   3.0000   0.0000   0.0000
            H   3.7572   0.5869   0.0000
            H   2.2428   0.5869   0.0000
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\n  Molecule: (H2O)_2 (water dimer)")
    print(f"  Basis: cc-pVDZ")
    print(f"  Number of AOs: {mol_2h2o.nao}")

    result_conv_2 = run_conventional_hf(mol_2h2o)
    result_df_2 = run_df_hf(mol_2h2o, auxbasis='cc-pvdz-jkfit')

    print(f"\n  Results:")
    print(f"  {'Method':<20} | {'Energy (Hartree)':<18} | {'Error (mHa)':<12} | {'Time (s)':<10}")
    print("  " + "-" * 70)
    print(f"  {'Conventional HF':<20} | {result_conv_2['energy']:<18.10f} | {'---':<12} | {result_conv_2['time']:<10.4f}")
    print(f"  {'DF-HF':<20} | {result_df_2['energy']:<18.10f} | "
          f"{(result_df_2['energy'] - result_conv_2['energy'])*1000:<12.6f} | {result_df_2['time']:<10.4f}")

    # =========================================================================
    # Test 3: Basis set dependence of DF error
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: DF Error vs Basis Set")
    print("=" * 50)

    print(f"\n  Molecule: H2O")
    print(f"  {'Basis':<12} | {'nAO':<5} | {'E_conv (Ha)':<16} | {'E_DF (Ha)':<16} | {'Error (uHa)':<12}")
    print("  " + "-" * 75)

    for basis in ['sto-3g', 'cc-pvdz', 'cc-pvtz']:
        mol_test = gto.M(
            atom="""
                O   0.0000   0.0000   0.1173
                H   0.0000   0.7572  -0.4692
                H   0.0000  -0.7572  -0.4692
            """,
            basis=basis,
            unit="Angstrom",
            verbose=0
        )

        mf_conv = scf.RHF(mol_test)
        mf_conv.verbose = 0
        E_conv = mf_conv.kernel()

        mf_df = scf.RHF(mol_test).density_fit()
        mf_df.verbose = 0
        E_df = mf_df.kernel()

        error_uha = (E_df - E_conv) * 1e6

        print(f"  {basis:<12} | {mol_test.nao:<5} | {E_conv:<16.10f} | {E_df:<16.10f} | {error_uha:<12.2f}")

    # =========================================================================
    # Test 4: Memory and Scaling Analysis
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 4: Memory Scaling Analysis")
    print("=" * 50)

    print(f"\n  Theoretical memory scaling (8 bytes per float):")
    print(f"  {'nAO':<6} | {'ERI (s1)':<15} | {'ERI (s8)':<15} | {'DF (3-idx)':<15}")
    print("  " + "-" * 55)

    for nao in [10, 50, 100, 200, 500]:
        # Conventional: n^4 (s1) or n^4/8 (s8)
        mem_s1 = nao**4 * 8 / (1024**3)  # GB
        n_s8 = (nao * (nao + 1) // 2)
        mem_s8 = (n_s8 * (n_s8 + 1) // 2) * 8 / (1024**3)

        # DF: nao^2 * naux (typically naux ~ 3*nao for cc-pVDZ-jkfit)
        naux = 3 * nao
        mem_df = nao * nao * naux * 8 / (1024**3)

        print(f"  {nao:<6} | {mem_s1:<15.3f} | {mem_s8:<15.3f} | {mem_df:<15.3f}")

    print(f"\n  Units: GB")
    print(f"  Note: DF memory is O(N^2 * N_aux) vs O(N^4) for conventional")

    # =========================================================================
    # Test 5: Accessing DF Intermediates
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 5: Accessing DF Intermediates")
    print("=" * 50)

    mol_demo = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    mf_df_demo = scf.RHF(mol_demo).density_fit(auxbasis='cc-pvdz-jkfit')
    mf_df_demo.verbose = 0
    mf_df_demo.kernel()

    print(f"\n  H2 / cc-pVDZ with cc-pVDZ-jkfit auxiliary basis")
    print(f"  nAO: {mol_demo.nao}")

    # Access the density fitting object
    mydf = mf_df_demo.with_df

    print(f"\n  DF object attributes:")
    print(f"    Auxiliary basis: {mydf.auxbasis}")

    if hasattr(mydf, 'auxmol') and mydf.auxmol is not None:
        print(f"    Auxiliary nAO: {mydf.auxmol.nao}")

    # The 3-center integrals are stored in _cderi
    if hasattr(mydf, '_cderi') and mydf._cderi is not None:
        cderi = mydf._cderi
        print(f"    3-index tensor shape: {cderi.shape}")
        print(f"    3-index tensor size: {cderi.nbytes / 1024:.1f} KB")

    # =========================================================================
    # Test 6: DF-HF Energy Decomposition
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 6: Verifying DF Approximation Quality")
    print("=" * 50)

    # Use HF molecule (hydrogen fluoride) - well-supported auxiliary basis
    mol_hf_test = gto.M(
        atom="H 0 0 0; F 0 0 0.92",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    # Conventional
    mf_conv_hf = scf.RHF(mol_hf_test)
    mf_conv_hf.verbose = 0
    E_conv_hf = mf_conv_hf.kernel()
    dm_conv = mf_conv_hf.make_rdm1()

    # DF (automatic auxiliary basis selection)
    mf_df_hf = scf.RHF(mol_hf_test).density_fit()
    mf_df_hf.verbose = 0
    E_df_hf = mf_df_hf.kernel()
    dm_df = mf_df_hf.make_rdm1()

    print(f"\n  HF / cc-pVDZ")
    print(f"  Conventional energy: {E_conv_hf:.10f} Hartree")
    print(f"  DF energy:           {E_df_hf:.10f} Hartree")
    print(f"  Difference:          {(E_df_hf - E_conv_hf)*1e6:.2f} microHartree")

    # Compare density matrices
    dm_diff = np.linalg.norm(dm_df - dm_conv)
    print(f"\n  Density matrix difference:")
    print(f"    ||P_DF - P_conv|| = {dm_diff:.2e}")

    # Compare J and K matrices
    J_conv, K_conv = mf_conv_hf.get_jk(mol_hf_test, dm_conv)
    J_df, K_df = mf_df_hf.get_jk(mol_hf_test, dm_conv)  # Use same density for fair comparison

    print(f"\n  J/K matrix differences (same density):")
    print(f"    ||J_DF - J_conv|| = {np.linalg.norm(J_df - J_conv):.2e}")
    print(f"    ||K_DF - K_conv|| = {np.linalg.norm(K_df - K_conv):.2e}")

    print("\n" + "=" * 70)
    print("Density fitting comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
