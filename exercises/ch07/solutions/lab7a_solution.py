#!/usr/bin/env python3
"""
Lab 7A Solution: Conventional HF vs Density-Fitted HF
======================================================

This script demonstrates the density fitting (DF) / resolution-of-the-identity (RI)
approximation for Hartree-Fock calculations and provides a comprehensive comparison
of computational cost and accuracy.

Learning objectives:
1. Understand the DF approximation: (mu nu|la si) ~ Sum_Q B_mu_nu^Q B_la_si^Q
2. Compare wall-clock times between conventional and DF-HF
3. Analyze DF error relative to basis set error
4. Understand memory scaling: O(N^4) vs O(N^2 N_aux)

Test molecule: H2O with cc-pVTZ basis (large enough to show timing differences)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 7: Scaling and Properties
"""

import time
import numpy as np
from pyscf import gto, scf

# =============================================================================
# Section 1: Core Timing and Comparison Functions
# =============================================================================


def run_conventional_hf(mol: gto.Mole, verbose: bool = False) -> tuple:
    """
    Run conventional RHF calculation and measure wall-clock time.

    Args:
        mol: PySCF molecule object
        verbose: If True, print PySCF output

    Returns:
        energy: Converged HF energy (Hartree)
        elapsed: Wall-clock time (seconds)
        mf: PySCF SCF object (for further analysis)
    """
    t0 = time.perf_counter()
    mf = scf.RHF(mol)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.perf_counter() - t0
    return energy, elapsed, mf


def run_df_hf(mol: gto.Mole, auxbasis: str = None, verbose: bool = False) -> tuple:
    """
    Run density-fitted RHF calculation and measure wall-clock time.

    The DF approximation factorizes two-electron integrals:
        (mu nu|la si) ~ Sum_Q B_mu_nu^Q B_la_si^Q

    where B_mu_nu^Q = Sum_P (mu nu|P) (P|Q)^(-1/2)

    Args:
        mol: PySCF molecule object
        auxbasis: Auxiliary basis set (default: auto-generated)
        verbose: If True, print PySCF output

    Returns:
        energy: Converged DF-HF energy (Hartree)
        elapsed: Wall-clock time (seconds)
        mf: PySCF DF-SCF object
    """
    t0 = time.perf_counter()
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.verbose = 4 if verbose else 0
    energy = mf.kernel()
    elapsed = time.perf_counter() - t0
    return energy, elapsed, mf


def get_memory_estimate(mol: gto.Mole, mf_df=None) -> dict:
    """
    Estimate memory requirements for conventional vs DF-HF.

    Conventional: O(N^4) for full ERI tensor
    DF: O(N^2 * N_aux) for 3-index tensor

    Args:
        mol: PySCF molecule object
        mf_df: DF-HF object (to get auxiliary basis size)

    Returns:
        dict with memory estimates in MB
    """
    nao = mol.nao
    n_aux = 0

    # Estimate auxiliary basis size if not provided
    if mf_df is not None and hasattr(mf_df, 'with_df'):
        try:
            auxmol = mf_df.with_df.auxmol
            if auxmol is not None:
                n_aux = auxmol.nao
        except Exception:
            n_aux = 3 * nao  # Typical ratio

    if n_aux == 0:
        n_aux = 3 * nao  # Typical N_aux/N ratio is ~3

    # Memory in bytes (8 bytes per float64)
    bytes_per_element = 8

    # Conventional: full (nao, nao, nao, nao) tensor
    conv_elements = nao ** 4
    conv_memory = conv_elements * bytes_per_element / (1024 ** 2)  # MB

    # With 8-fold symmetry: roughly N^4/8
    conv_sym_elements = nao ** 4 / 8
    conv_sym_memory = conv_sym_elements * bytes_per_element / (1024 ** 2)

    # DF: (n_aux, nao, nao) tensor (or packed lower triangle)
    df_elements = n_aux * nao * (nao + 1) // 2  # Lower triangle
    df_memory = df_elements * bytes_per_element / (1024 ** 2)

    return {
        'nao': nao,
        'n_aux': n_aux,
        'conv_full_MB': conv_memory,
        'conv_sym_MB': conv_sym_memory,
        'df_MB': df_memory,
        'ratio': conv_sym_memory / df_memory if df_memory > 0 else np.inf,
    }


# =============================================================================
# Section 2: Main Demonstration - Water Comparison
# =============================================================================


def demo_water_comparison() -> None:
    """
    Demonstrate conventional vs DF-HF for water with multiple basis sets.

    This is the core of Lab 7A, showing:
    1. Energy comparison (DF error is small)
    2. Timing comparison (DF is faster for large bases)
    3. Memory scaling analysis
    """
    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 7A: Conventional HF vs Density-Fitted HF" + " " * 24 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Water molecule at equilibrium geometry
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    print("Test Molecule: H2O (Water)")
    print("-" * 40)
    print("Geometry (Angstrom):")
    print(h2o_geometry)

    # Basis sets: increasing size to show scaling
    basis_sets = ['cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ']

    print()
    print("=" * 85)
    print("Conventional HF vs Density-Fitted HF: Energy and Timing Comparison")
    print("=" * 85)
    print()
    print(f"{'Basis':<12} {'N_AO':>6} {'E_conv (Hartree)':>18} {'E_DF (Hartree)':>18} "
          f"{'Delta_E':>12} {'t_conv':>8} {'t_DF':>8} {'Speedup':>8}")
    print("-" * 95)

    results = []
    for basis in basis_sets:
        mol = gto.M(
            atom=h2o_geometry,
            basis=basis,
            unit='Angstrom',
            verbose=0
        )

        # Run both methods
        E_conv, t_conv, mf_conv = run_conventional_hf(mol)
        E_df, t_df, mf_df = run_df_hf(mol)

        # Calculate DF error
        delta_E = E_df - E_conv
        speedup = t_conv / t_df if t_df > 0 else np.inf

        # Memory estimate
        mem = get_memory_estimate(mol, mf_df)

        result = {
            'basis': basis,
            'nao': mol.nao,
            'E_conv': E_conv,
            'E_df': E_df,
            'delta_E': delta_E,
            't_conv': t_conv,
            't_df': t_df,
            'speedup': speedup,
            'memory': mem,
        }
        results.append(result)

        print(f"{basis:<12} {mol.nao:>6} {E_conv:>18.10f} {E_df:>18.10f} "
              f"{delta_E:>12.2e} {t_conv:>7.2f}s {t_df:>7.2f}s {speedup:>7.2f}x")

    print("-" * 95)

    # ==========================================================================
    # Section 3: Error Analysis
    # ==========================================================================

    print()
    print("=" * 75)
    print("Analysis: DF Error vs Basis Set Error")
    print("=" * 75)

    # Compute basis set errors (relative to largest basis)
    E_best = results[-1]['E_conv']  # cc-pVQZ as reference

    print()
    print(f"Reference: cc-pVQZ conventional HF = {E_best:.10f} Hartree")
    print()
    print(f"{'Basis':<12} {'Basis Set Error':>16} {'DF Error':>16} {'Ratio':>12}")
    print("-" * 60)

    for r in results:
        basis_error = abs(r['E_conv'] - E_best)
        df_error = abs(r['delta_E'])
        ratio = basis_error / df_error if df_error > 1e-16 else np.inf

        print(f"{r['basis']:<12} {basis_error:>16.2e} {df_error:>16.2e} {ratio:>12.0f}x")

    print("-" * 60)
    print()
    print("Key observation: DF error is 4-6 orders of magnitude smaller than")
    print("basis set incompleteness error, making DF a 'safe' approximation.")

    # ==========================================================================
    # Section 4: Memory Scaling Analysis
    # ==========================================================================

    print()
    print("=" * 75)
    print("Memory Scaling Analysis")
    print("=" * 75)
    print()
    print(f"{'Basis':<12} {'N_AO':>6} {'N_aux':>6} {'Conv (sym)':>14} "
          f"{'DF':>14} {'Reduction':>12}")
    print("-" * 70)

    for r in results:
        m = r['memory']
        print(f"{r['basis']:<12} {m['nao']:>6} {m['n_aux']:>6} "
              f"{m['conv_sym_MB']:>12.1f} MB {m['df_MB']:>12.1f} MB "
              f"{m['ratio']:>10.1f}x")

    print("-" * 70)
    print()
    print("Scaling summary:")
    print("  Conventional HF: O(N^4) storage for ERIs")
    print("  DF-HF:           O(N^2 * N_aux) ~ O(N^3) storage for 3-index tensor")
    print("  The memory reduction becomes more significant as N grows.")


# =============================================================================
# Section 5: Scaling Study with System Size
# =============================================================================


def demo_scaling_study() -> None:
    """
    Demonstrate how DF speedup increases with system size.

    Uses a series of molecules with increasing size to show that
    DF advantage grows with system size due to better scaling.
    """
    print()
    print("=" * 75)
    print("Scaling Study: DF Speedup vs System Size")
    print("=" * 75)

    # Series of molecules with increasing N_AO
    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74'),
        ('H2O', '''
            O  0.0000  0.0000  0.1174
            H  0.7570  0.0000 -0.4696
            H -0.7570  0.0000 -0.4696
        '''),
        ('CH4', '''
            C  0.0000  0.0000  0.0000
            H  0.6276  0.6276  0.6276
            H  0.6276 -0.6276 -0.6276
            H -0.6276  0.6276 -0.6276
            H -0.6276 -0.6276  0.6276
        '''),
        ('C2H4', '''
            C  0.0000  0.0000  0.6695
            C  0.0000  0.0000 -0.6695
            H  0.0000  0.9289  1.2321
            H  0.0000 -0.9289  1.2321
            H  0.0000  0.9289 -1.2321
            H  0.0000 -0.9289 -1.2321
        '''),
    ]

    basis = 'cc-pVTZ'

    print()
    print(f"Basis: {basis}")
    print()
    print(f"{'Molecule':<12} {'N_AO':>6} {'t_conv (s)':>12} {'t_DF (s)':>12} "
          f"{'Speedup':>10} {'DF Error':>14}")
    print("-" * 75)

    for name, atoms in molecules:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)

        E_conv, t_conv, _ = run_conventional_hf(mol)
        E_df, t_df, _ = run_df_hf(mol)

        delta_E = E_df - E_conv
        speedup = t_conv / t_df if t_df > 0 else np.inf

        print(f"{name:<12} {mol.nao:>6} {t_conv:>12.3f} {t_df:>12.3f} "
              f"{speedup:>9.2f}x {delta_E:>14.2e}")

    print("-" * 75)
    print()
    print("Observation: Speedup increases with system size because:")
    print("  - Conventional HF scales as O(N^4) for ERI computation")
    print("  - DF-HF scales as O(N^2 * N_aux) ~ O(N^3) for 3-index contractions")
    print("  - For large systems, DF becomes essential for tractability")


# =============================================================================
# Section 6: Auxiliary Basis Comparison
# =============================================================================


def demo_auxiliary_basis_comparison() -> None:
    """
    Compare different auxiliary basis sets for density fitting.

    The auxiliary basis determines the accuracy of the DF approximation.
    Larger auxiliary bases give smaller DF errors but increase cost.
    """
    print()
    print("=" * 75)
    print("Auxiliary Basis Comparison")
    print("=" * 75)

    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.1174
        H  0.7570  0.0000 -0.4696
        H -0.7570  0.0000 -0.4696
        ''',
        basis='cc-pVTZ',
        unit='Angstrom',
        verbose=0
    )

    # Reference: conventional HF
    E_conv, t_conv, _ = run_conventional_hf(mol)

    # Various auxiliary basis sets
    aux_bases = [
        ('def2-universal-jkfit', 'Universal JKFIT'),
        ('cc-pVTZ-jkfit', 'Matched JKFIT'),
        ('cc-pVQZ-jkfit', 'Larger JKFIT'),
    ]

    print()
    print(f"Orbital basis: cc-pVTZ (N_AO = {mol.nao})")
    print(f"Reference (conventional HF): E = {E_conv:.10f} Hartree")
    print()
    print(f"{'Aux Basis':<25} {'N_aux':>8} {'N_aux/N':>10} "
          f"{'DF Error':>14} {'Time (s)':>10}")
    print("-" * 75)

    for aux_name, description in aux_bases:
        try:
            E_df, t_df, mf_df = run_df_hf(mol, auxbasis=aux_name)

            # Get auxiliary basis size
            n_aux = 0
            if hasattr(mf_df, 'with_df') and mf_df.with_df.auxmol is not None:
                n_aux = mf_df.with_df.auxmol.nao

            delta_E = E_df - E_conv
            ratio = n_aux / mol.nao if mol.nao > 0 else 0

            print(f"{aux_name:<25} {n_aux:>8} {ratio:>10.1f} "
                  f"{delta_E:>14.2e} {t_df:>10.3f}")
        except Exception as e:
            print(f"{aux_name:<25} {'Error':>8} {str(e)[:30]}")

    print("-" * 75)
    print()
    print("Notes on auxiliary basis selection:")
    print("  - N_aux/N ratio is typically 2-4 for good accuracy")
    print("  - Matched JKFIT bases (same cardinal number) are standard choice")
    print("  - Larger auxiliary bases improve accuracy but increase cost")


# =============================================================================
# Section 7: Physical Interpretation
# =============================================================================


def explain_df_approximation() -> None:
    """Explain the physical and mathematical basis of density fitting."""

    explanation = """
Physical and Mathematical Basis of Density Fitting
===================================================

THE APPROXIMATION:
------------------
Density fitting (DF), also called Resolution of the Identity (RI),
approximates 4-center ERIs using an auxiliary basis:

    (mu nu | la si) = Sum_Q B_mu_nu^Q B_la_si^Q

where the 3-index quantities are:

    B_mu_nu^Q = Sum_P (mu nu | P) (P | Q)^(-1/2)

Here {P, Q} are auxiliary basis functions, typically atom-centered
Gaussians optimized for this purpose.

WHY IT WORKS:
-------------
The product of two AOs |mu nu) lives in a space spanned by products.
For Gaussian AOs, this product space can be represented by a relatively
small auxiliary basis because:

1. Gaussian products are Gaussians (GPT)
2. The auxiliary functions span the density-like space efficiently
3. Empirically, ~3N auxiliary functions suffice for high accuracy

COMPUTATIONAL SAVINGS:
----------------------
                        Storage           Cost (JK build)
Conventional:           O(N^4)            O(N^4)
Density Fitted:         O(N^2 N_aux)      O(N^2 N_aux)

Since N_aux ~ 3N, DF reduces both storage and computation by ~N.

ERROR CHARACTERISTICS:
----------------------
- DF error is variational: E_DF >= E_exact (for Coulomb fitting)
- Typical errors: 10^-5 to 10^-6 Hartree per atom
- Much smaller than basis set incompleteness error
- Error decreases with larger auxiliary basis

WHEN TO USE DF:
---------------
- Almost always for production calculations with >100 AOs
- Essential for large molecules (N > 500 AOs)
- Not recommended only for very small systems or when
  exact ERIs are specifically needed
"""
    print(explanation)


# =============================================================================
# Section 8: Validation
# =============================================================================


def validate_df_implementation() -> None:
    """
    Validate DF-HF implementation against conventional HF.

    This checks that:
    1. Both methods converge
    2. DF error is within expected range
    3. Results are reproducible
    """
    print()
    print("=" * 75)
    print("Validation: DF-HF Implementation Check")
    print("=" * 75)

    # Test molecule
    mol = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    print()
    print(f"Test molecule: HF")
    print(f"Basis: cc-pVDZ (N_AO = {mol.nao})")
    print("-" * 50)

    # Run both methods
    E_conv, t_conv, mf_conv = run_conventional_hf(mol)
    E_df, t_df, mf_df = run_df_hf(mol)

    print(f"Conventional HF energy:  {E_conv:.10f} Hartree")
    print(f"DF-HF energy:            {E_df:.10f} Hartree")
    print(f"Energy difference:       {E_df - E_conv:.2e} Hartree")

    # Validation checks
    all_passed = True

    # Check 1: Both converged
    print()
    if mf_conv.converged:
        print("[PASS] Conventional HF converged")
    else:
        print("[FAIL] Conventional HF did not converge!")
        all_passed = False

    if mf_df.converged:
        print("[PASS] DF-HF converged")
    else:
        print("[FAIL] DF-HF did not converge!")
        all_passed = False

    # Check 2: DF error is small
    df_error = abs(E_df - E_conv)
    if df_error < 1e-4:
        print(f"[PASS] DF error ({df_error:.2e}) < 1e-4 Hartree")
    else:
        print(f"[FAIL] DF error ({df_error:.2e}) exceeds threshold!")
        all_passed = False

    # Check 3: Energies are reasonable
    if E_conv < 0 and E_df < 0:
        print("[PASS] Both energies are negative (physical)")
    else:
        print("[FAIL] Non-physical positive energy!")
        all_passed = False

    print()
    if all_passed:
        print("=" * 50)
        print("All validation checks passed!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("Some validation checks failed!")
        print("=" * 50)


# =============================================================================
# Section 9: What You Should Observe
# =============================================================================


def print_observations() -> None:
    """Print summary of key observations from this lab."""

    observations = """
================================================================================
What You Should Observe (Lab 7A)
================================================================================

1. DF ERROR IS SMALL:
   - Typical DF error: 10^-5 to 10^-6 Hartree
   - This is 4-6 orders of magnitude smaller than basis set error
   - DF is a "safe" approximation that doesn't limit accuracy

2. DF SPEEDUP INCREASES WITH SYSTEM SIZE:
   - Small molecules (N < 50): DF may be slower due to overhead
   - Medium molecules (N ~ 100): DF speedup of 2-5x
   - Large molecules (N > 200): DF speedup of 10x or more

3. MEMORY REDUCTION:
   - Conventional: O(N^4) for full ERI tensor
   - DF: O(N^2 * N_aux) ~ O(N^3)
   - For N = 100, reduction is ~30x; for N = 500, ~150x

4. AUXILIARY BASIS MATTERS:
   - Matched JKFIT bases give ~3N auxiliary functions
   - Larger auxiliary bases reduce error but cost more
   - N_aux/N ~ 3 is a good balance

5. PRACTICAL IMPLICATIONS:
   - Use DF for any system with N_AO > 100
   - DF is the default in most modern QC codes
   - The ERI bottleneck is replaced by 3-index operations

================================================================================
"""
    print(observations)


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Run the complete Lab 7A demonstration."""

    # Core demonstration
    demo_water_comparison()

    # Scaling study
    demo_scaling_study()

    # Auxiliary basis comparison
    demo_auxiliary_basis_comparison()

    # Physical interpretation
    explain_df_approximation()

    # Validation
    validate_df_implementation()

    # Summary
    print_observations()

    print()
    print("=" * 75)
    print("Lab 7A Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
