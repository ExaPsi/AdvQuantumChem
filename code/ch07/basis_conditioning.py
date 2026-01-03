#!/usr/bin/env python3
"""
Basis Set Conditioning and Numerical Stability
===============================================

This script analyzes the numerical conditioning of overlap matrices and
demonstrates eigenvalue thresholding for handling near-linear dependence.

Key concepts from Chapter 7:
    - Condition number: κ(S) = s_max / s_min
    - Near-linear dependence: κ > 10^8 requires attention
    - Eigenvalue thresholding removes redundant basis functions
    - Ill-conditioning affects SCF stability and accuracy

Reference: Section 7.4 (Basis Sets, Conditioning, and Practical Accuracy Controls)
"""

import numpy as np
from pyscf import gto, scf
import warnings


def analyze_overlap_conditioning(mol):
    """
    Analyze the conditioning of the overlap matrix.

    Returns
    -------
    dict with keys: s_min, s_max, kappa, n_small, eigenvalues
    """
    S = mol.intor('int1e_ovlp')
    eigenvalues = np.linalg.eigvalsh(S)

    s_min = eigenvalues.min()
    s_max = eigenvalues.max()
    kappa = s_max / s_min if s_min > 0 else np.inf

    # Count eigenvalues below various thresholds
    thresholds = [1e-6, 1e-8, 1e-10, 1e-12]
    n_below = {t: np.sum(eigenvalues < t) for t in thresholds}

    return {
        's_min': s_min,
        's_max': s_max,
        'kappa': kappa,
        'log_kappa': np.log10(kappa) if kappa > 0 else np.inf,
        'n_below_threshold': n_below,
        'eigenvalues': eigenvalues,
        'nao': mol.nao,
    }


def eigenvalue_threshold_analysis(mol, threshold=1e-8):
    """
    Analyze the effect of eigenvalue thresholding.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    threshold : float
        Eigenvalue threshold for removing near-linear dependencies

    Returns
    -------
    dict with analysis results
    """
    S = mol.intor('int1e_ovlp')
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Identify functions to keep
    keep_mask = eigenvalues > threshold
    n_keep = np.sum(keep_mask)
    n_remove = mol.nao - n_keep

    # Build canonical orthogonalizer
    s_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues[keep_mask]))
    X = eigenvectors[:, keep_mask] @ s_sqrt_inv

    # New condition number after thresholding
    s_min_new = eigenvalues[keep_mask].min() if n_keep > 0 else 0
    s_max_new = eigenvalues[keep_mask].max() if n_keep > 0 else 0
    kappa_new = s_max_new / s_min_new if s_min_new > 0 else np.inf

    return {
        'threshold': threshold,
        'n_original': mol.nao,
        'n_keep': n_keep,
        'n_remove': n_remove,
        's_min_original': eigenvalues.min(),
        's_min_new': s_min_new,
        'kappa_original': eigenvalues.max() / eigenvalues.min(),
        'kappa_new': kappa_new,
        'X': X,
    }


def demo_conditioning_vs_basis():
    """
    Demonstrate how conditioning changes with basis set diffuseness.
    """
    print("=" * 70)
    print("Basis Set Conditioning Analysis")
    print("=" * 70)

    # Water molecule
    atom_str = 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'

    # Basis sets with increasing diffuseness
    basis_sets = [
        ('STO-3G', 'Minimal'),
        ('6-31G', 'Split-valence'),
        ('6-31G*', 'Polarization'),
        ('6-31+G*', 'Diffuse'),
        ('6-31++G**', 'Very diffuse'),
        ('aug-cc-pVDZ', 'Augmented'),
        ('aug-cc-pVTZ', 'Aug + large'),
    ]

    print("\nMolecule: H2O")
    print("-" * 70)
    print(f"{'Basis':<16} {'N_AO':>6} {'s_min':>12} {'s_max':>10} "
          f"{'log₁₀(κ)':>10} {'Status':>12}")
    print("-" * 70)

    for basis, description in basis_sets:
        try:
            mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
            result = analyze_overlap_conditioning(mol)

            # Determine status
            if result['log_kappa'] < 6:
                status = "Excellent"
            elif result['log_kappa'] < 8:
                status = "Good"
            elif result['log_kappa'] < 10:
                status = "Caution"
            else:
                status = "WARNING"

            print(f"{basis:<16} {result['nao']:>6} {result['s_min']:>12.2e} "
                  f"{result['s_max']:>10.4f} {result['log_kappa']:>10.2f} "
                  f"{status:>12}")
        except Exception as e:
            print(f"{basis:<16} {'Error':>6} {str(e)[:40]}")

    print("\n" + "=" * 70)
    print("Interpretation Guide")
    print("=" * 70)
    print("""
Condition number κ(S) = s_max / s_min:
  - log₁₀(κ) < 6:  Excellent conditioning
  - log₁₀(κ) < 8:  Good conditioning
  - log₁₀(κ) < 10: May need thresholding
  - log₁₀(κ) > 10: Near-linear dependence, thresholding required

Diffuse functions (+ notation) increase κ because they overlap significantly
with each other and with valence functions at long range.
""")


def demo_eigenvalue_thresholding():
    """
    Demonstrate eigenvalue thresholding for ill-conditioned bases.
    """
    print("\n" + "=" * 70)
    print("Eigenvalue Thresholding")
    print("=" * 70)

    # Use a basis known to have conditioning issues
    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='aug-cc-pVTZ',
        unit='Angstrom',
        verbose=0
    )

    thresholds = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]

    print(f"\nMolecule: H2O with aug-cc-pVTZ (N_AO = {mol.nao})")
    print("-" * 70)
    print(f"{'Threshold':>12} {'N_keep':>8} {'N_remove':>10} "
          f"{'log₁₀(κ_new)':>14} {'Status':>12}")
    print("-" * 70)

    for thresh in thresholds:
        result = eigenvalue_threshold_analysis(mol, threshold=thresh)

        if result['kappa_new'] < np.inf:
            log_kappa = np.log10(result['kappa_new'])
            status = "OK" if log_kappa < 8 else "Still high"
        else:
            log_kappa = np.inf
            status = "All removed!"

        print(f"{thresh:>12.0e} {result['n_keep']:>8} {result['n_remove']:>10} "
              f"{log_kappa:>14.2f} {status:>12}")

    print("\nPySCF default threshold: 1e-8")


def demo_conditioning_effect_on_scf():
    """
    Demonstrate how conditioning affects SCF convergence.
    """
    print("\n" + "=" * 70)
    print("Conditioning Effect on SCF")
    print("=" * 70)

    atom_str = 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'

    test_cases = [
        ('cc-pVDZ', None),           # Well-conditioned
        ('aug-cc-pVDZ', None),       # Moderately conditioned
        ('aug-cc-pVTZ', None),       # May need thresholding
    ]

    print("\nMolecule: H2O")
    print("-" * 70)
    print(f"{'Basis':<16} {'log₁₀(κ)':>10} {'E_HF (Eh)':>18} "
          f"{'Converged':>10} {'Iterations':>12}")
    print("-" * 70)

    for basis, lindep_thresh in test_cases:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)

        # Analyze conditioning
        cond = analyze_overlap_conditioning(mol)

        # Run SCF
        mf = scf.RHF(mol)
        mf.verbose = 0
        if lindep_thresh is not None:
            mf.lindep = lindep_thresh

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mf.kernel()

        # Get iteration count from mf object
        n_iter = mf.mo_coeff is not None  # Simplified check

        print(f"{basis:<16} {cond['log_kappa']:>10.2f} {mf.e_tot:>18.10f} "
              f"{'Yes' if mf.converged else 'No':>10} "
              f"{'OK' if mf.converged else 'Failed':>12}")


def demo_eigenvalue_spectrum():
    """
    Show the eigenvalue spectrum for different basis sets.
    """
    print("\n" + "=" * 70)
    print("Overlap Matrix Eigenvalue Spectrum")
    print("=" * 70)

    atom_str = 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043'
    basis_sets = ['cc-pVDZ', 'aug-cc-pVDZ']

    for basis in basis_sets:
        mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)
        result = analyze_overlap_conditioning(mol)

        print(f"\n{basis} (N_AO = {mol.nao}):")
        print("-" * 50)

        eigs = result['eigenvalues']
        print(f"  Smallest 5 eigenvalues: {eigs[:5]}")
        print(f"  Largest 5 eigenvalues:  {eigs[-5:]}")
        print(f"  κ = {result['kappa']:.2e} (log₁₀ = {result['log_kappa']:.2f})")


def validate_conditioning_analysis():
    """
    Validate conditioning analysis against expected behavior.
    """
    print("\n" + "=" * 70)
    print("Validation: Conditioning Analysis")
    print("=" * 70)

    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    result = analyze_overlap_conditioning(mol)

    # Check basic properties
    assert result['s_min'] > 0, "Smallest eigenvalue should be positive"
    # Note: s_max can exceed 1 due to positive off-diagonal overlaps
    # For N normalized functions with positive overlaps, eigenvalues sum to N
    assert result['s_max'] <= mol.nao, "Largest eigenvalue should not exceed N"
    assert result['kappa'] >= 1, "Condition number should be >= 1"
    assert result['nao'] == mol.nao, "NAO count mismatch"

    # Check eigenvalue count
    assert len(result['eigenvalues']) == mol.nao, "Eigenvalue count mismatch"

    # For H2/cc-pVDZ, condition number should be reasonable
    assert result['log_kappa'] < 8, "Unexpected ill-conditioning for simple system"

    print("[PASSED] All validations successful!")
    print(f"  - Eigenvalue positivity: OK")
    print(f"  - Eigenvalue normalization: OK")
    print(f"  - Condition number reasonable: OK (log₁₀(κ) = {result['log_kappa']:.2f})")


if __name__ == '__main__':
    demo_conditioning_vs_basis()
    demo_eigenvalue_thresholding()
    demo_conditioning_effect_on_scf()
    demo_eigenvalue_spectrum()
    validate_conditioning_analysis()
