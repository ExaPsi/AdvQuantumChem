#!/usr/bin/env python3
"""
lab6c_jk_comparison.py - In-Core vs Direct J/K Building (Lab 6C)

This module compares two strategies for building the Coulomb (J) and
Exchange (K) matrices:

1. In-core: Store full ERI tensor, contract with einsum
2. Direct: PySCF's integral-driven approach (compute ERIs on-the-fly)

For small molecules, in-core is simpler and may even be faster due to
cache-efficient tensor operations. For large molecules, direct SCF is
essential because storing O(N^4) ERIs is impractical.

Key concepts:
    - ERI storage: O(N^4) for in-core, O(1) for direct
    - In-core: one-time integral computation, fast contractions
    - Direct: on-the-fly integrals with Schwarz screening
    - The crossover point depends on memory and basis size

References:
    - Chapter 6, Section 8: Integral-driven viewpoint
    - Listing 6.5: Compare in-core vs direct J/K building
    - Chapter 7: Scaling analysis and density fitting

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
import time
from typing import Tuple


def build_jk_incore(eri: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K using full in-core ERIs and einsum.

    This approach stores the full ERI tensor and uses optimized
    tensor contractions:
        J_pq = sum_rs (pq|rs) P_rs
        K_pq = sum_rs (pr|qs) P_rs

    Parameters
    ----------
    eri : np.ndarray
        Full ERI tensor (pq|rs) with shape (N, N, N, N)
    P : np.ndarray
        Density matrix

    Returns
    -------
    J, K : np.ndarray
        Coulomb and Exchange matrices
    """
    J = np.einsum("pqrs,rs->pq", eri, P, optimize=True)
    K = np.einsum("prqs,rs->pq", eri, P, optimize=True)
    return J, K


def compare_jk_methods(mol, dm):
    """
    Compare in-core vs direct J/K building.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object
    dm : np.ndarray
        Density matrix

    Returns
    -------
    results : dict
        Timing and accuracy comparison
    """
    from pyscf import scf

    mf = scf.RHF(mol)

    results = {}

    # Direct J/K (PySCF reference builder)
    t0 = time.time()
    J_dir, K_dir = mf.get_jk(mol, dm)
    t_direct = time.time() - t0
    results['time_direct'] = t_direct

    # In-core J/K (only if ERI tensor fits in memory)
    t0 = time.time()
    eri = mol.intor("int2e", aosym="s1")
    t_eri = time.time() - t0
    results['time_eri'] = t_eri

    t0 = time.time()
    J_inc = np.einsum("pqrs,rs->pq", eri, dm, optimize=True)
    K_inc = np.einsum("prqs,rs->pq", eri, dm, optimize=True)
    t_einsum = time.time() - t0
    results['time_einsum'] = t_einsum

    # Total in-core time (integral + contraction)
    results['time_incore_total'] = t_eri + t_einsum

    # Compare results
    results['J_diff'] = np.linalg.norm(J_inc - J_dir)
    results['K_diff'] = np.linalg.norm(K_inc - K_dir)

    # Memory usage
    results['eri_memory_mb'] = eri.nbytes / 1e6
    results['nao'] = mol.nao

    return results, J_dir, K_dir, J_inc, K_inc, eri


# =============================================================================
# Demonstrations
# =============================================================================

def demonstrate_small_molecule():
    """
    Demonstrate J/K comparison for a small molecule (N2/6-31G).
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return

    print("=" * 70)
    print("Lab 6C: In-Core vs Direct J/K Building")
    print("=" * 70)

    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.1",
        basis="6-31g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: N2")
    print(f"Basis: 6-31G")
    print(f"Number of AOs: {mol.nao}")

    mf = scf.RHF(mol)
    dm = mf.get_init_guess()

    results, J_dir, K_dir, J_inc, K_inc, eri = compare_jk_methods(mol, dm)

    print(f"\nERI tensor shape: {eri.shape}")
    print(f"ERI tensor memory: {results['eri_memory_mb']:.2f} MB")

    print("\n" + "-" * 70)
    print("Timing Comparison")
    print("-" * 70)
    print(f"  Direct get_jk time:    {results['time_direct']*1000:.2f} ms")
    print(f"  In-core ERI time:      {results['time_eri']*1000:.2f} ms")
    print(f"  In-core einsum time:   {results['time_einsum']*1000:.2f} ms")
    print(f"  In-core total time:    {results['time_incore_total']*1000:.2f} ms")

    print("\n" + "-" * 70)
    print("Accuracy Comparison")
    print("-" * 70)
    print(f"  ||J_inc - J_dir||_F = {results['J_diff']:.2e}")
    print(f"  ||K_inc - K_dir||_F = {results['K_diff']:.2e}")


def demonstrate_basis_size_scaling():
    """
    Demonstrate how timing scales with basis size.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return

    print("\n" + "=" * 70)
    print("Basis Size Scaling Study")
    print("=" * 70)

    basis_sets = ["sto-3g", "6-31g", "cc-pvdz"]

    print("\nMolecule: H2O")
    print(f"{'Basis':<12} {'NAO':>6} {'ERI (MB)':>10} {'Direct (ms)':>12} {'In-core (ms)':>14}")
    print("-" * 60)

    for basis in basis_sets:
        mol = gto.M(
            atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
            basis=basis,
            unit="Angstrom",
            verbose=0
        )

        mf = scf.RHF(mol)
        dm = mf.get_init_guess()

        try:
            results, *_ = compare_jk_methods(mol, dm)
            print(f"{basis:<12} {results['nao']:>6} {results['eri_memory_mb']:>10.2f} "
                  f"{results['time_direct']*1000:>12.2f} {results['time_incore_total']*1000:>14.2f}")
        except MemoryError:
            print(f"{basis:<12} {mol.nao:>6} {'N/A (OOM)':<10}")


def explain_direct_scf_advantage():
    """
    Explain why direct SCF can be faster even for small systems.
    """
    print("\n" + "=" * 70)
    print("Why Direct SCF Can Be Faster")
    print("=" * 70)

    print("""
Direct SCF (integral-driven) can be faster than in-core for several reasons:

1. CACHE LOCALITY
   - Direct algorithms process shell quartets contiguously
   - Better CPU cache utilization
   - Avoids random access patterns of large tensors

2. SCHWARZ SCREENING
   - Skip negligible integrals: |(mn|ls)| <= sqrt(mn|mn) * sqrt(ls|ls)
   - For large systems, 90%+ of integrals are screened out
   - Screening overhead is minimal compared to savings

3. SYMMETRY EXPLOITATION
   - 8-fold ERI symmetry reduces unique integrals by ~8x
   - Direct algorithms naturally exploit this
   - In-core with full tensor doesn't benefit

4. MEMORY BANDWIDTH
   - ERI tensor is O(N^4), often exceeds CPU cache
   - Memory-bound operations are slow
   - Direct avoids storing the full tensor

5. OPTIMIZED INTEGRAL BATCHING
   - Production codes (libcint) batch similar integrals
   - Vectorized evaluation of primitive integrals
   - Rys quadrature is highly optimized

For small systems (< 100 AOs), in-core may still win due to:
   - Simple einsum contractions are very fast
   - No screening overhead
   - Full tensor fits in cache

Crossover point: typically 100-300 AOs, depending on:
   - Available memory
   - CPU cache size
   - Basis set (diffuse functions need more screening)
""")


def checkpoint_direct_vs_incore():
    """
    Checkpoint question about direct vs in-core.
    """
    print("\n" + "=" * 70)
    print("Checkpoint: Why Might mf.get_jk() Be Faster?")
    print("=" * 70)

    print("""
Question: For a small system like H2O/STO-3G, PySCF's mf.get_jk() might
be faster than explicit einsum with stored ERIs. Why?

Consider:
1. Cache locality - how does memory access pattern affect speed?
2. Screening - does PySCF skip negligible integrals?
3. Symmetry - does PySCF exploit ERI symmetry?
4. Optimized batching - what overhead does einsum have?

Answer:
Even for small systems, PySCF's get_jk() is highly optimized:
- Written in C with careful memory management
- Exploits 8-fold ERI symmetry (computes ~1/8 of integrals)
- Uses Schwarz screening even for small systems
- Batches shell quartets for cache efficiency

The einsum approach:
- Works on full N^4 tensor (no symmetry exploitation)
- Python/NumPy overhead for memory allocation
- Computes all elements, including symmetric partners

For educational purposes, einsum is clearer and shows the formula directly.
For production, use get_jk() which is 10-100x faster for large systems.
""")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 6C: Compare In-Core vs Direct J/K Building")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Basic comparison
    demonstrate_small_molecule()

    # Scaling study
    demonstrate_basis_size_scaling()

    # Theory explanation
    explain_direct_scf_advantage()

    # Checkpoint
    checkpoint_direct_vs_incore()

    print("\n" + "=" * 70)
    print("Lab 6C Complete")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. In-core: store O(N^4) ERIs, fast contractions")
    print("  2. Direct: O(1) storage, on-the-fly integrals with screening")
    print("  3. For large systems (N > 100-300 AOs), direct is essential")
    print("  4. PySCF's get_jk() exploits symmetry, screening, and batching")
    print("=" * 70)
