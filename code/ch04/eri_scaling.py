#!/usr/bin/env python3
"""
eri_scaling.py - ERI Storage and Computation Scaling (Exercise 4.7)

This module investigates how ERI storage and computation scale with basis set size:
- Full tensor: O(N^4) storage
- 8-fold symmetry: ~N^4/8 unique elements
- Memory requirements in double precision (8 bytes/element)

References:
    - Chapter 4, Section 2: ERI Scaling
    - Helgaker, Jorgensen, Olsen, Section 9.4

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import List, Dict, Tuple


def count_unique_eris_8fold(nao: int) -> int:
    """
    Count unique ERIs exploiting full 8-fold symmetry.

    With symmetries (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν) and combinations,
    we need only store quartets with μ≥ν, λ≥σ, and compound index μν ≥ λσ.

    Parameters
    ----------
    nao : int
        Number of AO basis functions

    Returns
    -------
    int
        Number of unique ERIs
    """
    n_pairs = nao * (nao + 1) // 2
    # Using s8 symmetry: n_pairs*(n_pairs+1)/2
    return n_pairs * (n_pairs + 1) // 2


def memory_bytes(n_elements: int, dtype: str = 'float64') -> int:
    """
    Calculate memory in bytes for storing n_elements.

    Parameters
    ----------
    n_elements : int
        Number of elements
    dtype : str
        Data type ('float64' = 8 bytes, 'float32' = 4 bytes)

    Returns
    -------
    int
        Memory in bytes
    """
    bytes_per_element = {'float64': 8, 'float32': 4, 'float16': 2}
    return n_elements * bytes_per_element.get(dtype, 8)


def format_bytes(n_bytes: int) -> str:
    """Format bytes in human-readable form."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes/1024:.2f} KB"
    elif n_bytes < 1024**3:
        return f"{n_bytes/1024**2:.2f} MB"
    else:
        return f"{n_bytes/1024**3:.2f} GB"


def analyze_system(mol) -> Dict:
    """
    Analyze ERI storage requirements for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object

    Returns
    -------
    dict
        Analysis results
    """
    nao = mol.nao_nr()

    # Full tensor (s1)
    n_full = nao ** 4
    mem_s1 = memory_bytes(n_full)

    # 8-fold symmetry (s8)
    n_unique = count_unique_eris_8fold(nao)
    mem_s8 = memory_bytes(n_unique)

    return {
        'nao': nao,
        'n_full': n_full,
        'n_unique': n_unique,
        'mem_s1_bytes': mem_s1,
        'mem_s8_bytes': mem_s8,
        'mem_s1': format_bytes(mem_s1),
        'mem_s8': format_bytes(mem_s8),
        'reduction': n_full / n_unique if n_unique > 0 else 0
    }


def find_memory_limit(basis: str, max_memory_gb: float = 1.0) -> Tuple[int, int]:
    """
    Find the approximate system size where memory becomes prohibitive.

    Parameters
    ----------
    basis : str
        Basis set name (for estimating functions per atom)
    max_memory_gb : float
        Maximum allowed memory in GB

    Returns
    -------
    n_atoms_s1 : int
        Max atoms for full tensor storage
    n_atoms_s8 : int
        Max atoms for s8 storage
    """
    # Approximate functions per atom for common bases
    funcs_per_atom = {
        'sto-3g': 7,      # O: 5, H: 1 -> average ~2.3 for H2O
        '6-31g': 13,      # O: 9, H: 2 -> average ~4.3 for H2O
        '6-31+g*': 19,    # More for diffuse + polarization
        'cc-pvdz': 24,
        'cc-pvtz': 58,
    }

    funcs = funcs_per_atom.get(basis, 10)
    max_bytes = max_memory_gb * 1024**3

    # For s1: N^4 * 8 bytes < max_bytes -> N < (max_bytes/8)^0.25
    max_nao_s1 = int((max_bytes / 8) ** 0.25)
    n_atoms_s1 = max_nao_s1 // funcs

    # For s8: N^2(N+1)^2/8 * 8 bytes ~ N^4/8 * 8 < max_bytes
    max_nao_s8 = int((max_bytes) ** 0.25)
    n_atoms_s8 = max_nao_s8 // funcs

    return n_atoms_s1, n_atoms_s8


def generate_water_clusters(n_waters: int) -> str:
    """
    Generate geometry for a linear chain of water molecules.

    Parameters
    ----------
    n_waters : int
        Number of water molecules

    Returns
    -------
    str
        PySCF geometry string
    """
    atoms = []
    spacing = 3.0  # Bohr between water molecules

    for i in range(n_waters):
        z_offset = i * spacing
        atoms.append(f"O 0 0 {z_offset}")
        atoms.append(f"H 0 0.76 {z_offset + 0.59}")
        atoms.append(f"H 0 -0.76 {z_offset + 0.59}")

    return "; ".join(atoms)


# =============================================================================
# Main demonstration
# =============================================================================

def main():
    """Run ERI scaling analysis for Exercise 4.7."""
    try:
        from pyscf import gto
    except ImportError:
        print("PySCF not installed. Cannot run demonstration.")
        return

    print("=" * 70)
    print("Exercise 4.7: ERI Scaling with Basis Size")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Part (a) & (b): Water clusters with 6-31G
    print("\n(a)-(b) ERI counts and memory for water clusters (6-31G)")
    print("-" * 70)

    basis = "6-31g"
    results = []

    for n_waters in [1, 2, 3]:
        geom = generate_water_clusters(n_waters)
        mol = gto.M(atom=geom, basis=basis, unit="Bohr", verbose=0)
        data = analyze_system(mol)
        data['n_waters'] = n_waters
        results.append(data)

    print(f"{'System':>10} {'N_AO':>6} {'N^4':>12} {'N_unique':>12} "
          f"{'Mem(s1)':>10} {'Mem(s8)':>10} {'Factor':>8}")
    print("-" * 70)

    for r in results:
        system = f"(H2O)_{r['n_waters']}"
        print(f"{system:>10} {r['nao']:>6} {r['n_full']:>12} {r['n_unique']:>12} "
              f"{r['mem_s1']:>10} {r['mem_s8']:>10} {r['reduction']:>8.2f}x")

    # Verify O(N^4) scaling
    print("\n(b) Verifying O(N^4) scaling")
    print("-" * 70)

    if len(results) >= 2:
        for i in range(1, len(results)):
            nao_ratio = results[i]['nao'] / results[0]['nao']
            eri_ratio = results[i]['n_full'] / results[0]['n_full']
            expected_ratio = nao_ratio ** 4
            print(f"N_AO ratio: {nao_ratio:.2f}, ERI ratio: {eri_ratio:.2f}, "
                  f"Expected (N^4): {expected_ratio:.2f}")

    # Part (c): Find prohibitive system size
    print("\n(c) Memory threshold analysis (1 GB limit)")
    print("-" * 70)

    max_mem_gb = 1.0
    max_bytes = max_mem_gb * 1024**3

    print("Approximate max N_AO before exceeding 1 GB:")
    print(f"  Full tensor (s1): N_AO = {int((max_bytes/8)**0.25)}")
    print(f"  8-fold sym (s8):  N_AO = {int((8*max_bytes/8)**0.25)}")

    # Detailed table
    print("\nDetailed memory for increasing N_AO:")
    print(f"{'N_AO':>6} {'N^4':>15} {'Mem(s1)':>12} {'N_unique':>15} {'Mem(s8)':>12}")
    print("-" * 65)

    for nao in [10, 20, 30, 50, 75, 100, 150, 200]:
        n_full = nao ** 4
        n_unique = count_unique_eris_8fold(nao)
        mem_s1 = memory_bytes(n_full)
        mem_s8 = memory_bytes(n_unique)

        flag_s1 = " *" if mem_s1 > max_bytes else ""
        flag_s8 = " *" if mem_s8 > max_bytes else ""

        print(f"{nao:>6} {n_full:>15} {format_bytes(mem_s1):>10}{flag_s1:>2} "
              f"{n_unique:>15} {format_bytes(mem_s8):>10}{flag_s8:>2}")

    print("\n  * = exceeds 1 GB threshold")

    # Part (d): Compare s1 vs s8 memory savings
    print("\n(d) Memory savings from 8-fold symmetry")
    print("-" * 70)

    print(f"{'System':>12} {'s1 Memory':>12} {'s8 Memory':>12} {'Savings':>10}")
    print("-" * 50)

    for r in results:
        system = f"(H2O)_{r['n_waters']}"
        savings = (1 - r['mem_s8_bytes'] / r['mem_s1_bytes']) * 100
        print(f"{system:>12} {r['mem_s1']:>12} {r['mem_s8']:>12} {savings:>9.1f}%")

    # Summary table as requested
    print("\n" + "=" * 70)
    print("Summary Table: N, Number of ERIs, Memory (s1 and s8)")
    print("=" * 70)

    print(f"{'System':>12} {'N':>6} {'ERIs (s1)':>14} {'ERIs (s8)':>14} "
          f"{'Mem(s1)':>10} {'Mem(s8)':>10}")
    print("-" * 70)

    for r in results:
        system = f"(H2O)_{r['n_waters']}"
        print(f"{system:>12} {r['nao']:>6} {r['n_full']:>14} {r['n_unique']:>14} "
              f"{r['mem_s1']:>10} {r['mem_s8']:>10}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  - ERI count scales as O(N^4), dominating HF computational cost")
    print("  - 8-fold symmetry reduces storage by factor of ~8")
    print("  - For N_AO > ~100, full tensor exceeds 1 GB")
    print("  - Direct SCF avoids storage by recomputing ERIs each iteration")
    print("=" * 70)


if __name__ == "__main__":
    main()
