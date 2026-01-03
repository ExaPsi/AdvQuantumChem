#!/usr/bin/env python3
"""
Scaling Analysis for Hartree-Fock Calculations
===============================================

This script analyzes the computational scaling of different HF components
and demonstrates how density fitting changes the scaling behavior.

Key concepts from Chapter 7:
    - ERI computation: O(N^4) storage and computation
    - J/K build: O(N^4) for conventional, O(N^2 N_aux) ~ O(N^3) for DF
    - Diagonalization: O(N^3)
    - Memory bottleneck: 4-index ERI tensor

Reference: Section 7.2 (Scaling Anatomy of HF)
"""

import time
import numpy as np
from pyscf import gto, scf


def estimate_integral_counts(n_ao):
    """
    Estimate the number of integrals for a given basis size.

    Parameters
    ----------
    n_ao : int
        Number of atomic orbitals

    Returns
    -------
    dict with counts for different integral types
    """
    # One-electron integrals: O(N^2)
    n_1e = n_ao * n_ao

    # Two-electron integrals with 8-fold symmetry: O(N^4/8)
    # Exact count: N*(N+1)/2 * (N*(N+1)/2 + 1) / 2
    n_ao_pairs = n_ao * (n_ao + 1) // 2
    n_2e_unique = n_ao_pairs * (n_ao_pairs + 1) // 2

    # Full 4-index tensor (no symmetry): N^4
    n_2e_full = n_ao ** 4

    # DF: 3-index tensor with N_aux ~ 3N
    n_aux = 3 * n_ao
    n_3index = n_ao * n_ao * n_aux

    return {
        'n_ao': n_ao,
        'n_1e': n_1e,
        'n_2e_unique': n_2e_unique,
        'n_2e_full': n_2e_full,
        'n_3index': n_3index,
        'ratio_4to3': n_2e_full / n_3index if n_3index > 0 else 0,
    }


def measure_timing_components(mol, use_df=False):
    """
    Measure timing of different HF components.

    Returns
    -------
    dict with timing for integrals, J/K build, diagonalization, total
    """
    # Integral computation
    t0 = time.time()
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    t_1e = time.time() - t0

    # Two-electron integrals (for conventional) or 3-index (for DF)
    t0 = time.time()
    if use_df:
        mf = scf.RHF(mol).density_fit()
        # Force building of 3-index tensor
        mf.with_df.build()
        t_2e = time.time() - t0
    else:
        if mol.nao <= 100:  # Only compute full ERIs for small systems
            eri = mol.intor('int2e')
            t_2e = time.time() - t0
        else:
            t_2e = np.nan  # Skip for large systems

    # Full SCF timing
    t0 = time.time()
    if use_df:
        mf = scf.RHF(mol).density_fit()
    else:
        mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    t_total = time.time() - t0

    return {
        't_1e': t_1e,
        't_2e': t_2e,
        't_total': t_total,
        'converged': mf.converged,
        'energy': mf.e_tot,
        'nao': mol.nao,
    }


def demo_scaling_estimates():
    """
    Demonstrate scaling estimates for different system sizes.
    """
    print("=" * 70)
    print("Scaling Analysis: Integral Counts")
    print("=" * 70)

    n_ao_values = [10, 25, 50, 100, 200, 500]

    print("\n" + "-" * 80)
    print(f"{'N_AO':>8} {'1e integrals':>14} {'2e (unique)':>14} "
          f"{'2e (full)':>14} {'3-index (DF)':>14} {'4/3 ratio':>10}")
    print("-" * 80)

    for n in n_ao_values:
        counts = estimate_integral_counts(n)
        print(f"{n:>8} {counts['n_1e']:>14,} {counts['n_2e_unique']:>14,} "
              f"{counts['n_2e_full']:>14,} {counts['n_3index']:>14,} "
              f"{counts['ratio_4to3']:>10.1f}")

    print("\n" + "=" * 70)
    print("Memory Estimates (assuming 8 bytes per double)")
    print("=" * 70)

    print("\n" + "-" * 60)
    print(f"{'N_AO':>8} {'Full ERI':>15} {'DF 3-index':>15} {'Reduction':>12}")
    print("-" * 60)

    for n in n_ao_values:
        counts = estimate_integral_counts(n)
        mem_full = counts['n_2e_full'] * 8 / (1024**3)  # GB
        mem_df = counts['n_3index'] * 8 / (1024**3)      # GB

        if mem_df > 0:
            reduction = mem_full / mem_df
        else:
            reduction = np.inf

        print(f"{n:>8} {mem_full:>12.2f} GB {mem_df:>12.2f} GB {reduction:>11.1f}x")


def demo_timing_comparison():
    """
    Compare timing of conventional vs DF-HF for increasing system size.
    """
    print("\n" + "=" * 70)
    print("Timing Comparison: Conventional vs DF-HF")
    print("=" * 70)

    # Series of molecules with increasing size
    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74'),
        ('H2O', 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'),
        ('CH4', '''C 0 0 0; H 0.6276 0.6276 0.6276; H 0.6276 -0.6276 -0.6276;
                   H -0.6276 0.6276 -0.6276; H -0.6276 -0.6276 0.6276'''),
        ('C2H6', '''C -0.756 0 0; C 0.756 0 0;
                    H -1.14 0.59 0.83; H -1.14 0.48 -0.89; H -1.14 -1.07 0.05;
                    H 1.14 -0.59 -0.83; H 1.14 -0.48 0.89; H 1.14 1.07 -0.05'''),
    ]

    basis = 'cc-pVDZ'

    print(f"\nBasis: {basis}")
    print("-" * 70)
    print(f"{'Molecule':<10} {'N_AO':>6} {'t_conv (s)':>12} {'t_DF (s)':>12} "
          f"{'Speedup':>10} {'ΔE (Eh)':>12}")
    print("-" * 70)

    for name, atoms in molecules:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)

        # Conventional HF
        t0 = time.time()
        mf_conv = scf.RHF(mol)
        mf_conv.verbose = 0
        mf_conv.kernel()
        t_conv = time.time() - t0

        # DF-HF
        t0 = time.time()
        mf_df = scf.RHF(mol).density_fit()
        mf_df.verbose = 0
        mf_df.kernel()
        t_df = time.time() - t0

        speedup = t_conv / t_df if t_df > 0 else np.inf
        delta_E = mf_df.e_tot - mf_conv.e_tot

        print(f"{name:<10} {mol.nao:>6} {t_conv:>12.4f} {t_df:>12.4f} "
              f"{speedup:>9.2f}x {delta_E:>12.2e}")

    print("\nNote: Speedup increases with system size due to better DF scaling.")


def demo_scaling_exponent():
    """
    Estimate the scaling exponent from timing data.
    """
    print("\n" + "=" * 70)
    print("Scaling Exponent Estimation")
    print("=" * 70)

    # Use water with different basis sizes
    atom_str = 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'
    basis_sets = ['STO-3G', '6-31G', 'cc-pVDZ', 'cc-pVTZ']

    print("\nMolecule: H2O with different basis sets")
    print("-" * 60)

    nao_list = []
    t_conv_list = []
    t_df_list = []

    for basis in basis_sets:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)

        # Conventional
        t0 = time.time()
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()
        t_conv = time.time() - t0

        # DF
        t0 = time.time()
        mf_df = scf.RHF(mol).density_fit()
        mf_df.verbose = 0
        mf_df.kernel()
        t_df = time.time() - t0

        nao_list.append(mol.nao)
        t_conv_list.append(t_conv)
        t_df_list.append(t_df)

        print(f"{basis:<12} N_AO={mol.nao:>4}  t_conv={t_conv:.4f}s  t_DF={t_df:.4f}s")

    # Estimate scaling exponent via log-log fit
    if len(nao_list) >= 2:
        log_n = np.log(nao_list)
        log_t_conv = np.log(t_conv_list)
        log_t_df = np.log(t_df_list)

        # Linear fit
        slope_conv = np.polyfit(log_n, log_t_conv, 1)[0]
        slope_df = np.polyfit(log_n, log_t_df, 1)[0]

        print("\n" + "-" * 60)
        print("Estimated Scaling Exponents:")
        print(f"  Conventional HF: O(N^{slope_conv:.1f})")
        print(f"  DF-HF:           O(N^{slope_df:.1f})")
        print("\nExpected: Conventional ~O(N^4), DF ~O(N^3)")
        print("(Note: Small system sizes may not show asymptotic behavior)")


def demo_bottleneck_identification():
    """
    Identify the computational bottleneck in HF calculations.
    """
    print("\n" + "=" * 70)
    print("Bottleneck Identification")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    # Measure individual components
    t0 = time.time()
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    t_1e = time.time() - t0

    t0 = time.time()
    eri = mol.intor('int2e')
    t_2e = time.time() - t0

    t0 = time.time()
    eig = np.linalg.eigh(S)
    t_diag = time.time() - t0

    t0 = time.time()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    t_scf = time.time() - t0

    print(f"\nMolecule: H2O/cc-pVDZ (N_AO = {mol.nao})")
    print("-" * 50)
    print(f"{'Component':<25} {'Time (s)':>12} {'Fraction':>12}")
    print("-" * 50)
    print(f"{'One-electron integrals':<25} {t_1e:>12.4f} {t_1e/t_scf:>11.1%}")
    print(f"{'Two-electron integrals':<25} {t_2e:>12.4f} {t_2e/t_scf:>11.1%}")
    print(f"{'Matrix diagonalization':<25} {t_diag:>12.4f} {t_diag/t_scf:>11.1%}")
    print(f"{'Total SCF (incl. J/K)':<25} {t_scf:>12.4f} {'100.0%':>12}")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print("""
For small systems, the overhead of Python and PySCF setup may dominate.
For larger systems, the ERI computation and J/K build become the bottleneck.

Theoretical scaling:
  - One-electron integrals: O(N^2)
  - Two-electron integrals: O(N^4)
  - Fock matrix build:      O(N^4) conventional, O(N^3) DF
  - Diagonalization:        O(N^3)

The O(N^4) ERI step is the asymptotic bottleneck, motivating density fitting.
""")


def validate_scaling_analysis():
    """
    Validate scaling analysis calculations.
    """
    print("\n" + "=" * 70)
    print("Validation: Scaling Analysis")
    print("=" * 70)

    # Check integral count formulas
    n = 10
    counts = estimate_integral_counts(n)

    assert counts['n_1e'] == n * n, "1e integral count formula error"
    assert counts['n_2e_full'] == n ** 4, "Full 2e integral count error"
    assert counts['ratio_4to3'] > 1, "DF should reduce integral count"

    # Check that timing measurements work
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='STO-3G', verbose=0)
    timing = measure_timing_components(mol, use_df=False)

    assert timing['t_1e'] >= 0, "Negative timing"
    assert timing['converged'], "SCF should converge for H2/STO-3G"

    print("[PASSED] All validations successful!")


if __name__ == '__main__':
    demo_scaling_estimates()
    demo_timing_comparison()
    demo_scaling_exponent()
    demo_bottleneck_identification()
    validate_scaling_analysis()
