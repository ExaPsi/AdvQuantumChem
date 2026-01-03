#!/usr/bin/env python3
"""
Lab 7A: Conventional HF vs Density-Fitted HF Comparison
========================================================

This script demonstrates the density fitting (DF) / resolution-of-the-identity (RI)
approximation for Hartree-Fock calculations. It compares:
  - Wall-clock timing between conventional and DF-HF
  - Energy differences (DF error)
  - Scaling behavior with system size

Key concepts from Chapter 7:
  - DF factorization: (μν|λσ) ≈ Σ_Q B_μν^Q B_λσ^Q
  - Storage reduction: O(N^4) → O(N^2 N_aux) ~ O(N^3)
  - DF error is typically 10^-5 to 10^-6 Eh, much smaller than basis set error

Reference: Section 7.3 (Density Fitting / RI)
"""

import time
import numpy as np
from pyscf import gto, scf


def run_conventional_hf(mol, verbose=False):
    """Run conventional RHF and return energy and timing."""
    t0 = time.time()
    mf = scf.RHF(mol)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.time() - t0
    return energy, elapsed, mf


def run_df_hf(mol, auxbasis=None, verbose=False):
    """Run density-fitted RHF and return energy and timing."""
    t0 = time.time()
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.time() - t0
    return energy, elapsed, mf


def compare_conventional_vs_df(mol, auxbasis=None):
    """
    Compare conventional and DF-HF for a given molecule.

    Returns
    -------
    dict with keys: E_conv, E_df, t_conv, t_df, delta_E, speedup
    """
    E_conv, t_conv, mf_conv = run_conventional_hf(mol)
    E_df, t_df, mf_df = run_df_hf(mol, auxbasis=auxbasis)

    return {
        'E_conv': E_conv,
        'E_df': E_df,
        't_conv': t_conv,
        't_df': t_df,
        'delta_E': E_df - E_conv,
        'speedup': t_conv / t_df if t_df > 0 else np.inf,
        'nao': mol.nao,
    }


def demo_water_comparison():
    """
    Demonstrate DF-HF on water with different basis sets.

    This corresponds to Lab 7A in the lecture notes.
    """
    print("=" * 70)
    print("Lab 7A: Conventional HF vs Density-Fitted HF")
    print("=" * 70)

    # Water molecule geometry
    atom_str = """
    O  0.0000  0.0000  0.0000
    H  0.7586  0.0000  0.5043
    H -0.7586  0.0000  0.5043
    """

    basis_sets = ['cc-pVDZ', 'cc-pVTZ']

    print("\nMolecule: H2O")
    print("-" * 70)
    print(f"{'Basis':<12} {'N_AO':>6} {'E_conv (Eh)':>16} {'E_DF (Eh)':>16} "
          f"{'ΔE (Eh)':>12} {'t_conv':>8} {'t_DF':>8} {'Speedup':>8}")
    print("-" * 70)

    results = []
    for basis in basis_sets:
        mol = gto.M(
            atom=atom_str,
            basis=basis,
            unit='Angstrom',
            verbose=0
        )

        result = compare_conventional_vs_df(mol)
        results.append(result)

        print(f"{basis:<12} {result['nao']:>6} {result['E_conv']:>16.10f} "
              f"{result['E_df']:>16.10f} {result['delta_E']:>12.2e} "
              f"{result['t_conv']:>7.2f}s {result['t_df']:>7.2f}s "
              f"{result['speedup']:>7.2f}x")

    # Compute basis set error vs DF error
    if len(results) >= 2:
        basis_set_error = abs(results[1]['E_conv'] - results[0]['E_conv'])
        df_error = abs(results[0]['delta_E'])

        print("\n" + "=" * 70)
        print("Analysis: DF Error vs Basis Set Error")
        print("=" * 70)
        print(f"Basis set error (TZ - DZ): {basis_set_error:.6e} Eh")
        print(f"DF error (DZ):             {df_error:.6e} Eh")
        print(f"Ratio (basis/DF):          {basis_set_error/df_error:.1f}x")
        print("\nConclusion: DF error is typically much smaller than basis set error,")
        print("making density fitting 'safe' for most applications.")


def demo_scaling_study():
    """
    Demonstrate scaling behavior of DF-HF vs conventional HF.

    Uses a series of alkanes to show how timing scales with system size.
    """
    print("\n" + "=" * 70)
    print("Scaling Study: DF-HF vs Conventional HF")
    print("=" * 70)

    # Series of small molecules with increasing size
    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74'),
        ('CH4', '''
            C  0.0000  0.0000  0.0000
            H  0.6276  0.6276  0.6276
            H  0.6276 -0.6276 -0.6276
            H -0.6276  0.6276 -0.6276
            H -0.6276 -0.6276  0.6276
        '''),
        ('C2H6', '''
            C -0.7560  0.0000  0.0000
            C  0.7560  0.0000  0.0000
            H -1.1404  0.5916  0.8327
            H -1.1404  0.4757 -0.8863
            H -1.1404 -1.0673  0.0537
            H  1.1404 -0.5916 -0.8327
            H  1.1404 -0.4757  0.8863
            H  1.1404  1.0673 -0.0537
        '''),
    ]

    basis = 'cc-pVDZ'

    print(f"\nBasis: {basis}")
    print("-" * 70)
    print(f"{'Molecule':<10} {'N_AO':>6} {'t_conv (s)':>12} {'t_DF (s)':>12} "
          f"{'Speedup':>10} {'ΔE (Eh)':>12}")
    print("-" * 70)

    for name, atom_str in molecules:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
        result = compare_conventional_vs_df(mol)

        print(f"{name:<10} {result['nao']:>6} {result['t_conv']:>12.3f} "
              f"{result['t_df']:>12.3f} {result['speedup']:>9.2f}x "
              f"{result['delta_E']:>12.2e}")

    print("\nNote: Speedup increases with system size due to better scaling of DF.")
    print("Conventional HF: O(N^4), DF-HF: O(N^2 N_aux) ~ O(N^3)")


def demo_auxiliary_basis_comparison():
    """
    Compare different auxiliary basis sets for density fitting.

    This corresponds to Exercise 7.11 concepts.
    """
    print("\n" + "=" * 70)
    print("Auxiliary Basis Comparison")
    print("=" * 70)

    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.0000
        H  0.7586  0.0000  0.5043
        H -0.7586  0.0000  0.5043
        ''',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    # Reference: conventional HF
    E_conv, t_conv, _ = run_conventional_hf(mol)

    aux_bases = [
        ('def2-universal-jkfit', 'def2-universal-jkfit'),
        ('cc-pVDZ-jkfit', 'cc-pVDZ-jkfit'),
        ('cc-pVTZ-jkfit', 'cc-pVTZ-jkfit'),
    ]

    print(f"\nOrbital basis: cc-pVDZ (N_AO = {mol.nao})")
    print(f"Reference (conventional HF): E = {E_conv:.10f} Eh")
    print("-" * 70)
    print(f"{'Aux Basis':<25} {'N_aux':>8} {'N_aux/N':>10} "
          f"{'ΔE (Eh)':>14} {'Time (s)':>10}")
    print("-" * 70)

    for name, auxbasis in aux_bases:
        try:
            E_df, t_df, mf_df = run_df_hf(mol, auxbasis=auxbasis)

            # Get auxiliary basis size
            auxmol = mf_df.with_df.auxmol
            n_aux = auxmol.nao if auxmol is not None else 0

            delta_E = E_df - E_conv
            ratio = n_aux / mol.nao if mol.nao > 0 else 0

            print(f"{name:<25} {n_aux:>8} {ratio:>10.1f} "
                  f"{delta_E:>14.2e} {t_df:>10.3f}")
        except Exception as e:
            print(f"{name:<25} {'Error':>8} {str(e)[:30]}")


def validate_against_pyscf():
    """
    Validate our understanding against PySCF internal values.
    """
    print("\n" + "=" * 70)
    print("Validation: Checking DF-HF Implementation")
    print("=" * 70)

    mol = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    # Conventional HF
    mf_conv = scf.RHF(mol)
    mf_conv.kernel()

    # DF-HF
    mf_df = scf.RHF(mol).density_fit()
    mf_df.kernel()

    print(f"\nMolecule: HF")
    print(f"Basis: cc-pVDZ (N_AO = {mol.nao})")
    print("-" * 50)
    print(f"Conventional HF energy: {mf_conv.e_tot:.10f} Eh")
    print(f"DF-HF energy:           {mf_df.e_tot:.10f} Eh")
    print(f"Energy difference:      {mf_df.e_tot - mf_conv.e_tot:.2e} Eh")

    # Verify convergence
    assert mf_conv.converged, "Conventional HF did not converge!"
    assert mf_df.converged, "DF-HF did not converge!"

    # Check DF error is small
    df_error = abs(mf_df.e_tot - mf_conv.e_tot)
    assert df_error < 1e-4, f"DF error {df_error} exceeds threshold!"

    print("\n[PASSED] All validations successful!")
    print(f"  - Both methods converged")
    print(f"  - DF error ({df_error:.2e} Eh) is within acceptable range")


if __name__ == '__main__':
    demo_water_comparison()
    demo_scaling_study()
    demo_auxiliary_basis_comparison()
    validate_against_pyscf()
