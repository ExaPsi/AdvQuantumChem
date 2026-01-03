#!/usr/bin/env python3
"""
eri_symmetry.py - ERI Symmetry Verification (Exercise 4.6)

This module verifies the 8-fold permutation symmetry of electron repulsion integrals:

    (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ)     [4-fold from exchange within pairs]
            = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)  [× 2 from bra-ket exchange]

For real orbitals, all eight permutations yield identical values.

References:
    - Chapter 4, Section 2: ERI Symmetries
    - Szabo & Ostlund, Section 3.1.2

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import List, Tuple


def get_symmetry_equivalent_indices(mu: int, nu: int, lam: int, sig: int) -> List[Tuple]:
    """
    Generate all 8 symmetry-equivalent index tuples for an ERI.

    Parameters
    ----------
    mu, nu, lam, sig : int
        AO indices

    Returns
    -------
    list of tuple
        All 8 permutations (duplicates possible if indices coincide)
    """
    return [
        (mu, nu, lam, sig),  # Original
        (nu, mu, lam, sig),  # Exchange 1-2
        (mu, nu, sig, lam),  # Exchange 3-4
        (nu, mu, sig, lam),  # Exchange 1-2 and 3-4
        (lam, sig, mu, nu),  # Exchange bra-ket
        (sig, lam, mu, nu),  # Exchange bra-ket + 1-2
        (lam, sig, nu, mu),  # Exchange bra-ket + 3-4
        (sig, lam, nu, mu),  # Exchange bra-ket + 1-2 + 3-4
    ]


def verify_eri_symmetry(eri: np.ndarray, tol: float = 1e-12) -> Tuple[bool, float, dict]:
    """
    Verify all 8-fold symmetry relations for an ERI tensor.

    Parameters
    ----------
    eri : np.ndarray
        4D ERI tensor of shape (nao, nao, nao, nao)
    tol : float
        Tolerance for symmetry violation

    Returns
    -------
    all_passed : bool
        True if all symmetries are satisfied within tolerance
    max_deviation : float
        Maximum deviation from exact symmetry
    stats : dict
        Statistics about symmetry violations
    """
    nao = eri.shape[0]
    max_deviation = 0.0
    n_checked = 0
    n_violations = 0
    violations = []

    # Check all unique quartets
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    ref_val = eri[mu, nu, lam, sig]
                    equiv_indices = get_symmetry_equivalent_indices(mu, nu, lam, sig)

                    for indices in equiv_indices:
                        sym_val = eri[indices]
                        deviation = abs(ref_val - sym_val)
                        max_deviation = max(max_deviation, deviation)
                        n_checked += 1

                        if deviation > tol:
                            n_violations += 1
                            if len(violations) < 10:  # Store first 10 violations
                                violations.append({
                                    'original': (mu, nu, lam, sig),
                                    'permuted': indices,
                                    'deviation': deviation
                                })

    stats = {
        'n_checked': n_checked,
        'n_violations': n_violations,
        'violations': violations,
        'max_deviation': max_deviation
    }

    all_passed = (n_violations == 0)
    return all_passed, max_deviation, stats


def verify_individual_symmetries(eri: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Verify each symmetry relation individually.

    Symmetries tested:
    1. Exchange of indices 1,2: (μν|λσ) = (νμ|λσ)
    2. Exchange of indices 3,4: (μν|λσ) = (μν|σλ)
    3. Exchange of bra-ket: (μν|λσ) = (λσ|μν)

    Parameters
    ----------
    eri : np.ndarray
        4D ERI tensor
    tol : float
        Tolerance

    Returns
    -------
    dict
        Results for each symmetry type
    """
    nao = eri.shape[0]
    results = {}

    # Symmetry 1: (μν|λσ) = (νμ|λσ)
    max_dev_1 = 0.0
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    dev = abs(eri[mu, nu, lam, sig] - eri[nu, mu, lam, sig])
                    max_dev_1 = max(max_dev_1, dev)
    results['exchange_12'] = {
        'description': '(μν|λσ) = (νμ|λσ)',
        'max_deviation': max_dev_1,
        'passed': max_dev_1 < tol
    }

    # Symmetry 2: (μν|λσ) = (μν|σλ)
    max_dev_2 = 0.0
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    dev = abs(eri[mu, nu, lam, sig] - eri[mu, nu, sig, lam])
                    max_dev_2 = max(max_dev_2, dev)
    results['exchange_34'] = {
        'description': '(μν|λσ) = (μν|σλ)',
        'max_deviation': max_dev_2,
        'passed': max_dev_2 < tol
    }

    # Symmetry 3: (μν|λσ) = (λσ|μν)
    max_dev_3 = 0.0
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    dev = abs(eri[mu, nu, lam, sig] - eri[lam, sig, mu, nu])
                    max_dev_3 = max(max_dev_3, dev)
    results['exchange_braket'] = {
        'description': '(μν|λσ) = (λσ|μν)',
        'max_deviation': max_dev_3,
        'passed': max_dev_3 < tol
    }

    return results


def count_unique_eris(nao: int) -> dict:
    """
    Count unique ERIs under various symmetry assumptions.

    Parameters
    ----------
    nao : int
        Number of AO basis functions

    Returns
    -------
    dict
        Counts under different symmetry assumptions
    """
    # No symmetry
    n_full = nao ** 4

    # 2-fold symmetry (exchange 1-2 or 3-4)
    n_2fold = nao * (nao + 1) // 2 * nao * nao

    # 4-fold symmetry (exchange 1-2 AND 3-4)
    n_pairs = nao * (nao + 1) // 2
    n_4fold = n_pairs * n_pairs

    # 8-fold symmetry (including bra-ket exchange)
    # Count unique quartets (μν|λσ) with μ≥ν, λ≥σ, and μν≥λσ
    n_8fold = 0
    for mu in range(nao):
        for nu in range(mu + 1):
            munu = mu * (mu + 1) // 2 + nu
            for lam in range(nao):
                for sig in range(lam + 1):
                    lamsig = lam * (lam + 1) // 2 + sig
                    if munu >= lamsig:
                        n_8fold += 1

    return {
        'no_symmetry': n_full,
        '2_fold': n_2fold,
        '4_fold': n_4fold,
        '8_fold': n_8fold,
        'reduction_factor': n_full / n_8fold if n_8fold > 0 else 0
    }


# =============================================================================
# Main demonstration
# =============================================================================

def main():
    """Run ERI symmetry verification for Exercise 4.6."""
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Cannot run demonstration.")
        return

    print("=" * 70)
    print("Exercise 4.6: ERI Symmetry Verification")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Water molecule
    water = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
    mol = gto.M(atom=water, basis="sto-3g", verbose=0)
    nao = mol.nao_nr()

    print(f"\nMolecule: H2O")
    print(f"Basis: STO-3G")
    print(f"Number of AO basis functions: {nao}")

    # Part (a): Compute all N^4 ERIs
    print("\n(a) Computing full ERI tensor")
    print("-" * 70)

    eri = mol.intor("int2e", aosym="s1")
    print(f"ERI tensor shape: {eri.shape}")
    print(f"Total elements: {eri.size}")

    # Part (b): Verify all 8-fold symmetry relations
    print("\n(b) Verifying 8-fold symmetry relations")
    print("-" * 70)

    all_passed, max_dev, stats = verify_eri_symmetry(eri, tol=1e-12)

    print(f"Comparisons performed: {stats['n_checked']}")
    print(f"Violations (> 1e-12): {stats['n_violations']}")
    print(f"Maximum deviation: {max_dev:.2e}")

    # Part (c): Report maximum deviation
    print("\n(c) Individual symmetry analysis")
    print("-" * 70)

    sym_results = verify_individual_symmetries(eri)
    print(f"{'Symmetry':>30} {'Max Dev':>15} {'Status':>10}")
    print("-" * 70)

    for sym_name, data in sym_results.items():
        status = "PASS" if data['passed'] else "FAIL"
        print(f"{data['description']:>30} {data['max_deviation']:>15.2e} {status:>10}")

    # Validation summary
    print("\n" + "-" * 70)
    if all_passed:
        print("VALIDATION PASSED: All 8-fold symmetries satisfied to 1e-12")
    else:
        print("VALIDATION FAILED: Symmetry violations detected")
        print("First few violations:")
        for v in stats['violations'][:5]:
            print(f"  {v['original']} vs {v['permuted']}: {v['deviation']:.2e}")

    # Additional: Count unique ERIs
    print("\n" + "=" * 70)
    print("Unique ERI counts under symmetry")
    print("=" * 70)

    counts = count_unique_eris(nao)
    print(f"No symmetry (N^4):     {counts['no_symmetry']:>10}")
    print(f"2-fold symmetry:       {counts['2_fold']:>10}")
    print(f"4-fold symmetry:       {counts['4_fold']:>10}")
    print(f"8-fold symmetry:       {counts['8_fold']:>10}")
    print(f"Reduction factor:      {counts['reduction_factor']:>10.2f}x")

    # Show a specific example
    print("\n" + "=" * 70)
    print("Example: All 8 equivalent ERIs for (0,1,2,3)")
    print("=" * 70)

    if nao >= 4:
        equiv = get_symmetry_equivalent_indices(0, 1, 2, 3)
        print(f"{'Indices':>20} {'Value':>20}")
        print("-" * 45)
        for idx in equiv:
            val = eri[idx]
            print(f"{str(idx):>20} {val:>20.12f}")

        # Show unique value
        values = [eri[idx] for idx in equiv]
        print(f"\nRange of values: {min(values):.12f} to {max(values):.12f}")
        print(f"Spread: {max(values) - min(values):.2e}")

    print("\n" + "=" * 70)
    print("The maximum deviation should be at floating-point roundoff level")
    print("(~1e-14 to 1e-15 for well-conditioned ERIs)")
    print("=" * 70)


if __name__ == "__main__":
    main()
