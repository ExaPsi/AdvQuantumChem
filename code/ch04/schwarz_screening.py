#!/usr/bin/env python3
"""
schwarz_screening.py - Schwarz Inequality for ERI Screening (Exercise 4.3)

This module demonstrates the Schwarz inequality for screening negligible ERIs:

    |(μν|λσ)| ≤ √(μν|μν) × √(λσ|λσ)

The "pair norms" √(μν|μν) can be precomputed once, allowing rapid identification
of shell quartets that will yield negligible ERIs without computing them.

References:
    - Chapter 4, Section 3: Schwarz Screening
    - Häser & Ahlrichs, JCP 91, 360 (1989)

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import Tuple, Dict
import time


def compute_pair_norms(mol) -> np.ndarray:
    """
    Compute the Schwarz pair norms √(μν|μν) for all AO pairs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object

    Returns
    -------
    np.ndarray
        2D array of shape (nao, nao) with pair norms
    """
    nao = mol.nao_nr()
    eri_diag = np.zeros((nao, nao))

    # Compute (μν|μν) for all pairs
    eri = mol.intor("int2e", aosym="s1")
    for mu in range(nao):
        for nu in range(nao):
            eri_diag[mu, nu] = eri[mu, nu, mu, nu]

    return np.sqrt(eri_diag)


def compute_schwarz_bounds(pair_norms: np.ndarray) -> np.ndarray:
    """
    Compute Schwarz upper bounds for all ERIs from pair norms.

    |(μν|λσ)| ≤ Q_μν × Q_λσ

    where Q_μν = √(μν|μν)

    Parameters
    ----------
    pair_norms : np.ndarray
        2D array of pair norms from compute_pair_norms

    Returns
    -------
    np.ndarray
        4D array of Schwarz bounds
    """
    nao = pair_norms.shape[0]
    bounds = np.zeros((nao, nao, nao, nao))

    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    bounds[mu, nu, lam, sig] = pair_norms[mu, nu] * pair_norms[lam, sig]

    return bounds


def count_screened_eris(
    pair_norms: np.ndarray,
    threshold: float = 1e-10
) -> Tuple[int, int, float]:
    """
    Count how many ERIs are screened out by the Schwarz inequality.

    Parameters
    ----------
    pair_norms : np.ndarray
        2D array of pair norms
    threshold : float
        Screening threshold

    Returns
    -------
    n_total : int
        Total number of unique ERIs (with 8-fold symmetry)
    n_screened : int
        Number of ERIs guaranteed below threshold
    fraction : float
        Fraction of ERIs screened
    """
    nao = pair_norms.shape[0]
    n_total = 0
    n_screened = 0

    # Count unique ERIs with 8-fold symmetry
    for mu in range(nao):
        for nu in range(mu + 1):
            for lam in range(mu + 1):
                sig_max = nu if lam == mu else lam
                for sig in range(sig_max + 1):
                    n_total += 1
                    bound = pair_norms[mu, nu] * pair_norms[lam, sig]
                    if bound < threshold:
                        n_screened += 1

    fraction = n_screened / n_total if n_total > 0 else 0.0
    return n_total, n_screened, fraction


def verify_bounds(mol, n_samples: int = 100) -> Tuple[bool, float]:
    """
    Verify that Schwarz bounds are never violated for a random sample of ERIs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object
    n_samples : int
        Number of random ERIs to check

    Returns
    -------
    all_valid : bool
        True if no bounds were violated
    max_ratio : float
        Maximum ratio of |ERI|/bound (should be ≤ 1)
    """
    nao = mol.nao_nr()
    pair_norms = compute_pair_norms(mol)
    eri = mol.intor("int2e", aosym="s1")

    np.random.seed(42)  # Reproducibility
    max_ratio = 0.0
    all_valid = True

    for _ in range(n_samples):
        mu = np.random.randint(0, nao)
        nu = np.random.randint(0, nao)
        lam = np.random.randint(0, nao)
        sig = np.random.randint(0, nao)

        eri_val = abs(eri[mu, nu, lam, sig])
        bound = pair_norms[mu, nu] * pair_norms[lam, sig]

        if bound > 1e-15:  # Avoid division by zero
            ratio = eri_val / bound
            max_ratio = max(max_ratio, ratio)
            if ratio > 1.0 + 1e-10:  # Allow small numerical error
                all_valid = False

    return all_valid, max_ratio


def analyze_screening_vs_basis(molecule_str: str, bases: list) -> Dict:
    """
    Analyze how screening efficiency changes with basis set.

    Parameters
    ----------
    molecule_str : str
        Molecule specification (e.g., "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59")
    bases : list
        List of basis set names

    Returns
    -------
    dict
        Results for each basis set
    """
    from pyscf import gto

    results = {}
    threshold = 1e-10

    for basis in bases:
        mol = gto.M(atom=molecule_str, basis=basis, verbose=0)
        nao = mol.nao_nr()
        pair_norms = compute_pair_norms(mol)
        n_total, n_screened, fraction = count_screened_eris(pair_norms, threshold)

        results[basis] = {
            'nao': nao,
            'n_total': n_total,
            'n_screened': n_screened,
            'fraction': fraction,
            'n_computed': n_total - n_screened
        }

    return results


# =============================================================================
# Main demonstration
# =============================================================================

def main():
    """Run Schwarz screening demonstration for Exercise 4.3."""
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Cannot run demonstration.")
        return

    print("=" * 70)
    print("Exercise 4.3: Schwarz Screening Experiment")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Water molecule geometry
    water = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"

    # Part (a): Compute pair norms for STO-3G
    print("\n(a) Pair norms for H2O/STO-3G")
    print("-" * 70)

    mol = gto.M(atom=water, basis="sto-3g", verbose=0)
    nao = mol.nao_nr()
    pair_norms = compute_pair_norms(mol)

    print(f"Number of AO basis functions: {nao}")
    print(f"Pair norm matrix shape: {pair_norms.shape}")
    print("\nPair norms (first 5x5 block):")
    for i in range(min(5, nao)):
        row = " ".join(f"{pair_norms[i, j]:8.5f}" for j in range(min(5, nao)))
        print(f"  {row}")

    # Part (b) & (c): Schwarz bounds and screening counts
    print("\n(b)-(c) Schwarz screening analysis")
    print("-" * 70)

    threshold = 1e-10
    n_total, n_screened, fraction = count_screened_eris(pair_norms, threshold)

    print(f"Screening threshold: {threshold:.0e}")
    print(f"Total unique ERIs (8-fold symmetry): {n_total}")
    print(f"ERIs screened (bound < threshold): {n_screened}")
    print(f"Fraction screened: {fraction:.2%}")
    print(f"ERIs to compute: {n_total - n_screened}")

    # Part (d): Compare across basis sets
    print("\n(d) Screening efficiency across basis sets")
    print("-" * 70)

    bases = ["sto-3g", "6-31g", "6-31+g*"]
    results = analyze_screening_vs_basis(water, bases)

    print(f"{'Basis':>12} {'N_AO':>6} {'N_ERI':>10} {'N_screen':>10} "
          f"{'%screen':>10} {'N_compute':>10}")
    print("-" * 70)

    for basis, data in results.items():
        print(f"{basis:>12} {data['nao']:>6} {data['n_total']:>10} "
              f"{data['n_screened']:>10} {data['fraction']:>10.2%} "
              f"{data['n_computed']:>10}")

    # Validation: Verify bounds are never violated
    print("\nValidation: Checking 100 random ERIs")
    print("-" * 70)

    mol = gto.M(atom=water, basis="6-31g", verbose=0)
    all_valid, max_ratio = verify_bounds(mol, n_samples=100)

    print(f"Basis: 6-31G")
    print(f"All bounds satisfied: {all_valid}")
    print(f"Maximum |ERI|/bound ratio: {max_ratio:.6f}")

    if all_valid and max_ratio <= 1.0:
        print("VALIDATION PASSED: Schwarz inequality never violated")
    else:
        print("VALIDATION FAILED: Bound violation detected!")

    # Additional insight: Show tightness of bounds
    print("\n" + "=" * 70)
    print("Insight: Tightness of Schwarz bounds")
    print("=" * 70)

    eri = mol.intor("int2e", aosym="s1")
    nao = mol.nao_nr()
    pair_norms = compute_pair_norms(mol)

    ratios = []
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    bound = pair_norms[mu, nu] * pair_norms[lam, sig]
                    if bound > 1e-12:
                        ratio = abs(eri[mu, nu, lam, sig]) / bound
                        ratios.append(ratio)

    ratios = np.array(ratios)
    print(f"Distribution of |ERI|/bound ratios:")
    print(f"  Min:    {ratios.min():.6f}")
    print(f"  Mean:   {ratios.mean():.6f}")
    print(f"  Median: {np.median(ratios):.6f}")
    print(f"  Max:    {ratios.max():.6f}")
    print(f"  Std:    {ratios.std():.6f}")

    print("\nThe Schwarz bound is an upper bound, so ratios should be ≤ 1.")
    print("Smaller ratios indicate the bound is loose (conservative).")
    print("=" * 70)


if __name__ == "__main__":
    main()
