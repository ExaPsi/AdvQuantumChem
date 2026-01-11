#!/usr/bin/env python3
"""
Lab 6C Solution: In-Core vs Direct J/K Building

This script compares two approaches for building the Coulomb (J) and Exchange (K)
matrices in Hartree-Fock calculations:

1. IN-CORE: Store the full ERI tensor, then contract with density via einsum
2. DIRECT: Build J/K on-the-fly by looping over shell quartets

Learning Objectives:
1. Understand memory scaling of in-core ERIs: O(N^4) storage
2. Learn the shell-quartet structure of integral evaluation
3. Compare timing and memory usage between approaches
4. Understand when each approach is appropriate

Physical Insight:
-----------------
The two-electron integrals (ERIs) are the computational bottleneck of HF:
- Storage: O(N^4) elements (N = number of AO basis functions)
- Computation: O(N^4) integrals to evaluate

For small molecules (N < 100), storing all ERIs is feasible and allows
fast repeated contractions during SCF. For large molecules, the direct
approach computes integrals on-the-fly and discards them after use.

In practice, efficient codes use:
- Schwarz screening to skip negligible shell quartets
- 8-fold symmetry to reduce computation by ~8x
- Integral batching for cache efficiency

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 6: Hartree-Fock Self-Consistent Field as an Integral-Driven Algorithm
"""

import numpy as np
from pyscf import gto, scf
from pyscf.scf import _vhf
import time
import sys
from typing import Tuple, Optional


# =============================================================================
# Section 1: In-Core J/K Building
# =============================================================================

def build_jk_incore(P: np.ndarray,
                    eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K matrices from pre-computed full ERI tensor.

    This is the simplest approach: store all ERIs in memory, then
    contract with the density matrix using tensor operations.

    Memory requirement: O(N^4) for the ERI tensor
    Time per SCF iteration: O(N^4) for contractions

    Args:
        P: Density matrix (nao x nao), includes factor of 2 for RHF
        eri: Full ERI tensor in chemist's notation (nao, nao, nao, nao)
             eri[u,v,l,s] = (uv|ls)

    Returns:
        J: Coulomb matrix
        K: Exchange matrix
    """
    # J_uv = sum_{ls} (uv|ls) P_ls
    # Contract over the last two indices
    J = np.einsum('uvls,ls->uv', eri, P, optimize=True)

    # K_uv = sum_{ls} (ul|vs) P_ls
    # Note the index pattern: (ul|vs) requires transposition of middle indices
    K = np.einsum('ulvs,ls->uv', eri, P, optimize=True)

    return J, K


def build_jk_incore_explicit(P: np.ndarray,
                              eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K matrices with explicit loops (for educational clarity).

    This version shows exactly how the contractions work, element by element.
    Much slower than einsum but crystal clear mathematically.

    J_uv = sum_{l,s} (uv|ls) P_ls    [same indices on bra: uv]
    K_uv = sum_{l,s} (ul|vs) P_ls    [crossed indices: u-l, v-s]
    """
    nao = P.shape[0]
    J = np.zeros((nao, nao))
    K = np.zeros((nao, nao))

    for u in range(nao):
        for v in range(nao):
            # Coulomb: sum over all l,s
            for l in range(nao):
                for s in range(nao):
                    J[u, v] += eri[u, v, l, s] * P[l, s]

            # Exchange: different index pattern
            for l in range(nao):
                for s in range(nao):
                    K[u, v] += eri[u, l, v, s] * P[l, s]

    return J, K


# =============================================================================
# Section 2: Direct J/K Building (Shell-by-Shell)
# =============================================================================

def get_shell_info(mol: gto.Mole) -> list:
    """
    Extract shell information from a PySCF molecule.

    Each shell has:
    - atom_id: which atom it's centered on
    - angular_momentum: L value (0=s, 1=p, 2=d, ...)
    - n_functions: number of AO functions in this shell
    - ao_start: starting index in AO list

    Returns:
        List of shell info dictionaries
    """
    shells = []
    ao_idx = 0

    for shell_id in range(mol.nbas):
        atom_id = mol.bas_atom(shell_id)
        L = mol.bas_angular(shell_id)
        # Number of spherical or Cartesian functions
        if mol.cart:
            n_func = (L + 1) * (L + 2) // 2
        else:
            n_func = 2 * L + 1

        shells.append({
            'shell_id': shell_id,
            'atom_id': atom_id,
            'L': L,
            'n_func': n_func,
            'ao_start': ao_idx
        })

        ao_idx += n_func

    return shells


def build_jk_direct_concept(mol: gto.Mole,
                            P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K by looping over shell quartets (conceptual implementation).

    This demonstrates the DIRECT SCF approach where integrals are computed
    on-the-fly and immediately contracted, avoiding storage of the full tensor.

    In real codes, this would include:
    - Schwarz screening to skip negligible quartets
    - Symmetry exploitation (8-fold ERI symmetry)
    - Batching for cache efficiency

    Memory requirement: O(N^2) for J, K matrices only
    Time per iteration: O(N^4) integral evaluations

    Args:
        mol: PySCF Mole object
        P: Density matrix

    Returns:
        J, K: Coulomb and Exchange matrices
    """
    nao = mol.nao_nr()
    J = np.zeros((nao, nao))
    K = np.zeros((nao, nao))

    shells = get_shell_info(mol)
    n_shells = len(shells)

    # Loop over all shell quartets (A, B | C, D)
    # This is O(n_shells^4) but each quartet has variable size
    for iA, shellA in enumerate(shells):
        for iB, shellB in enumerate(shells):
            for iC, shellC in enumerate(shells):
                for iD, shellD in enumerate(shells):
                    # Get the integral block for this shell quartet
                    # Using PySCF's intor_by_shell for shell-level access
                    eri_block = mol.intor_by_shell('int2e_sph', [iA, iB, iC, iD])

                    # Get AO index ranges for each shell
                    startA = shellA['ao_start']
                    endA = startA + shellA['n_func']
                    startB = shellB['ao_start']
                    endB = startB + shellB['n_func']
                    startC = shellC['ao_start']
                    endC = startC + shellC['n_func']
                    startD = shellD['ao_start']
                    endD = startD + shellD['n_func']

                    # Extract the density block for this quartet
                    P_block = P[startC:endC, startD:endD]

                    # Contract into J: (AB|CD) * P_CD -> J_AB
                    # J[A,B] += sum_{C,D} (AB|CD) P_CD
                    J_contrib = np.einsum('abcd,cd->ab', eri_block, P_block)
                    J[startA:endA, startB:endB] += J_contrib

                    # Contract into K: (AC|BD) * P_CD -> K_AB
                    # K[A,B] += sum_{C,D} (AC|BD) P_CD
                    # For K, we need a different index mapping
                    # (AB|CD) -> need (AC|BD), so swap B<->C in the contraction
                    P_block_K = P[startB:endB, startD:endD]
                    K_contrib = np.einsum('abcd,bd->ac', eri_block, P_block_K)
                    K[startA:endA, startC:endC] += K_contrib

    return J, K


def build_jk_pyscf_direct(mol: gto.Mole,
                          P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K using PySCF's optimized direct J/K builder.

    This uses the production-quality implementation in PySCF which includes:
    - Schwarz screening
    - 8-fold symmetry exploitation
    - Optimized integral evaluation
    - Cache-efficient batching

    This is what real quantum chemistry codes use.
    """
    # PySCF's get_jk automatically handles all optimizations
    mf = scf.RHF(mol)
    J, K = mf.get_jk(mol, P)
    return J, K


# =============================================================================
# Section 3: Memory Analysis
# =============================================================================

def estimate_memory(n: int, dtype: np.dtype = np.float64) -> dict:
    """
    Estimate memory requirements for different J/K building approaches.

    Args:
        n: Number of AO basis functions
        dtype: Data type (default float64)

    Returns:
        Dictionary with memory estimates
    """
    bytes_per_element = np.dtype(dtype).itemsize

    # Full ERI tensor: (n, n, n, n)
    eri_elements = n ** 4
    eri_bytes = eri_elements * bytes_per_element

    # With 8-fold symmetry: approximately n^4 / 8
    eri_sym_elements = n * (n + 1) // 2
    eri_sym_elements = eri_sym_elements * (eri_sym_elements + 1) // 2
    eri_sym_bytes = eri_sym_elements * bytes_per_element

    # Direct approach: only J, K, P matrices
    direct_elements = 3 * n ** 2  # J, K, P
    direct_bytes = direct_elements * bytes_per_element

    # One shell quartet (maximum size: g-functions, L=4)
    max_shell_size = 9  # 2*4+1 spherical
    max_quartet = max_shell_size ** 4
    quartet_bytes = max_quartet * bytes_per_element

    return {
        'n_ao': n,
        'eri_full_gb': eri_bytes / 1e9,
        'eri_sym_gb': eri_sym_bytes / 1e9,
        'direct_mb': direct_bytes / 1e6,
        'quartet_kb': quartet_bytes / 1e3,
        'eri_elements': eri_elements,
    }


def print_memory_analysis(molecules: list):
    """Print memory analysis for a list of molecules."""
    print("\n" + "=" * 70)
    print("Memory Requirements: In-Core vs Direct")
    print("=" * 70)

    print("\n" + "-" * 70)
    print(f"{'System':<20} {'NAO':>6} {'ERI Full':>12} {'ERI Sym':>12} {'Direct':>12}")
    print(f"{'':<20} {'':>6} {'(GB)':>12} {'(GB)':>12} {'(MB)':>12}")
    print("-" * 70)

    for name, mol in molecules:
        n = mol.nao_nr()
        mem = estimate_memory(n)
        print(f"{name:<20} {n:>6} {mem['eri_full_gb']:>12.3f} "
              f"{mem['eri_sym_gb']:>12.3f} {mem['direct_mb']:>12.3f}")

    print("-" * 70)


# =============================================================================
# Section 4: Timing Benchmarks
# =============================================================================

def benchmark_jk_building(mol: gto.Mole,
                          n_repeats: int = 5) -> dict:
    """
    Benchmark different J/K building approaches.

    Args:
        mol: PySCF Mole object
        n_repeats: Number of timing repeats

    Returns:
        Dictionary with timing results
    """
    nao = mol.nao_nr()
    print(f"\nBenchmarking J/K building for {nao} AO functions")
    print("-" * 50)

    # Get a density matrix to use
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    P = mf.make_rdm1()

    results = {}

    # Method 1: In-core with full ERI (if memory permits)
    mem = estimate_memory(nao)
    if mem['eri_full_gb'] < 1.0:  # Only if < 1 GB
        print("Timing: In-core (full ERI + einsum)...", end=" ", flush=True)

        # Time ERI computation
        t0 = time.perf_counter()
        eri = mol.intor("int2e", aosym="s1")
        t_eri = time.perf_counter() - t0

        # Time J/K contraction
        times_contract = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            J, K = build_jk_incore(P, eri)
            times_contract.append(time.perf_counter() - t0)

        t_contract = np.mean(times_contract)
        results['incore'] = {
            't_eri': t_eri,
            't_contract': t_contract,
            't_total_first': t_eri + t_contract,
            'J': J.copy(),
            'K': K.copy()
        }
        print(f"Done. ERI: {t_eri:.4f}s, Contract: {t_contract:.4f}s")
    else:
        print(f"Skipping in-core: would require {mem['eri_full_gb']:.1f} GB")
        results['incore'] = None

    # Method 2: PySCF direct J/K builder
    print("Timing: PySCF direct get_jk()...", end=" ", flush=True)
    times_direct = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        J_dir, K_dir = build_jk_pyscf_direct(mol, P)
        times_direct.append(time.perf_counter() - t0)

    t_direct = np.mean(times_direct)
    results['direct'] = {
        't_total': t_direct,
        'J': J_dir.copy(),
        'K': K_dir.copy()
    }
    print(f"Done. {t_direct:.4f}s")

    return results


def validate_jk_agreement(results: dict, tol: float = 1e-10):
    """Validate that different methods give the same J and K."""
    print("\nValidating J/K agreement between methods:")

    if results['incore'] is not None:
        J_inc = results['incore']['J']
        K_inc = results['incore']['K']
        J_dir = results['direct']['J']
        K_dir = results['direct']['K']

        J_diff = np.linalg.norm(J_inc - J_dir)
        K_diff = np.linalg.norm(K_inc - K_dir)

        print(f"  ||J_incore - J_direct|| = {J_diff:.2e}")
        print(f"  ||K_incore - K_direct|| = {K_diff:.2e}")

        if J_diff < tol and K_diff < tol:
            print("  PASS: Methods agree within tolerance")
        else:
            print("  WARNING: Methods differ!")
    else:
        print("  (In-core method skipped due to memory)")


# =============================================================================
# Section 5: Scaling Study
# =============================================================================

def study_scaling():
    """Study how timing scales with system size."""
    print("\n" + "=" * 70)
    print("Scaling Study: J/K Build Time vs System Size")
    print("=" * 70)

    # Create a series of water clusters of increasing size
    systems = []

    # Single water
    systems.append(("H2O", gto.M(
        atom="O 0 0 0; H 0.76 0 0.50; H -0.76 0 0.50",
        basis="sto-3g", unit="Angstrom", verbose=0)))

    # Two waters
    systems.append(("(H2O)2", gto.M(
        atom="""
            O  0.00  0.00  0.00
            H  0.76  0.00  0.50
            H -0.76  0.00  0.50
            O  3.00  0.00  0.00
            H  3.76  0.00  0.50
            H  2.24  0.00  0.50
        """,
        basis="sto-3g", unit="Angstrom", verbose=0)))

    # Methane
    systems.append(("CH4", gto.M(
        atom="""
            C  0.000  0.000  0.000
            H  0.629  0.629  0.629
            H -0.629 -0.629  0.629
            H -0.629  0.629 -0.629
            H  0.629 -0.629 -0.629
        """,
        basis="sto-3g", unit="Angstrom", verbose=0)))

    # N2 with larger basis
    systems.append(("N2/6-31G", gto.M(
        atom="N 0 0 0; N 0 0 1.1",
        basis="6-31g", unit="Angstrom", verbose=0)))

    # H2O with larger basis
    systems.append(("H2O/6-31G*", gto.M(
        atom="O 0 0 0; H 0.76 0 0.50; H -0.76 0 0.50",
        basis="6-31g*", unit="Angstrom", verbose=0)))

    # Benzene STO-3G
    systems.append(("C6H6/STO-3G", gto.M(
        atom="""
            C  1.3862  0.0000  0.0000
            C  0.6931  1.2004  0.0000
            C -0.6931  1.2004  0.0000
            C -1.3862  0.0000  0.0000
            C -0.6931 -1.2004  0.0000
            C  0.6931 -1.2004  0.0000
            H  2.4618  0.0000  0.0000
            H  1.2309  2.1320  0.0000
            H -1.2309  2.1320  0.0000
            H -2.4618  0.0000  0.0000
            H -1.2309 -2.1320  0.0000
            H  1.2309 -2.1320  0.0000
        """,
        basis="sto-3g", unit="Angstrom", verbose=0)))

    print("\n" + "-" * 80)
    print(f"{'System':<15} {'NAO':>5} {'N^4':>12} {'In-Core (s)':>14} {'Direct (s)':>12} {'Ratio':>8}")
    print("-" * 80)

    for name, mol in systems:
        n = mol.nao_nr()
        n4 = n ** 4
        mem = estimate_memory(n)

        results = benchmark_jk_building(mol, n_repeats=3)

        if results['incore'] is not None:
            t_inc = results['incore']['t_eri'] + results['incore']['t_contract']
            t_dir = results['direct']['t_total']
            ratio = t_inc / t_dir if t_dir > 0 else 0
            print(f"{name:<15} {n:>5} {n4:>12,} {t_inc:>14.4f} {t_dir:>12.4f} {ratio:>8.2f}")
        else:
            t_dir = results['direct']['t_total']
            print(f"{name:<15} {n:>5} {n4:>12,} {'N/A':>14} {t_dir:>12.4f} {'N/A':>8}")

    print("-" * 80)


# =============================================================================
# Section 6: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 6C demonstration."""

    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*   Lab 6C: In-Core vs Direct J/K Building                         *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # =========================================================================
    # Part 1: Memory Analysis
    # =========================================================================
    molecules = [
        ("H2/STO-3G", gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                           unit="Angstrom", verbose=0)),
        ("H2O/STO-3G", gto.M(atom="O 0 0 0; H 0.76 0 0.50; H -0.76 0 0.50",
                            basis="sto-3g", unit="Angstrom", verbose=0)),
        ("H2O/6-31G*", gto.M(atom="O 0 0 0; H 0.76 0 0.50; H -0.76 0 0.50",
                            basis="6-31g*", unit="Angstrom", verbose=0)),
        ("C6H6/STO-3G", gto.M(atom="""
            C  1.3862  0.0000  0.0000; C  0.6931  1.2004  0.0000
            C -0.6931  1.2004  0.0000; C -1.3862  0.0000  0.0000
            C -0.6931 -1.2004  0.0000; C  0.6931 -1.2004  0.0000
            H  2.4618  0.0000  0.0000; H  1.2309  2.1320  0.0000
            H -1.2309  2.1320  0.0000; H -2.4618  0.0000  0.0000
            H -1.2309 -2.1320  0.0000; H  1.2309 -2.1320  0.0000
        """, basis="sto-3g", unit="Angstrom", verbose=0)),
        ("C6H6/6-31G*", gto.M(atom="""
            C  1.3862  0.0000  0.0000; C  0.6931  1.2004  0.0000
            C -0.6931  1.2004  0.0000; C -1.3862  0.0000  0.0000
            C -0.6931 -1.2004  0.0000; C  0.6931 -1.2004  0.0000
            H  2.4618  0.0000  0.0000; H  1.2309  2.1320  0.0000
            H -1.2309  2.1320  0.0000; H -2.4618  0.0000  0.0000
            H -1.2309 -2.1320  0.0000; H  1.2309 -2.1320  0.0000
        """, basis="6-31g*", unit="Angstrom", verbose=0)),
    ]

    print_memory_analysis(molecules)

    # =========================================================================
    # Part 2: Detailed Benchmark on Single System
    # =========================================================================
    print("\n" + "=" * 70)
    print("Detailed Benchmark: N2 / 6-31G")
    print("=" * 70)

    mol_n2 = gto.M(
        atom="N 0 0 0; N 0 0 1.1",
        basis="6-31g",
        unit="Angstrom",
        verbose=0
    )

    results = benchmark_jk_building(mol_n2, n_repeats=5)
    validate_jk_agreement(results)

    # =========================================================================
    # Part 3: Scaling Study
    # =========================================================================
    study_scaling()

    # =========================================================================
    # Part 4: What You Should Observe
    # =========================================================================
    print("\n" + "=" * 70)
    print("What You Should Observe")
    print("=" * 70)

    observations = """
1. MEMORY SCALING (O(N^4)):
   - ERI tensor size grows as N^4 where N = number of AO functions
   - H2O/STO-3G (7 AOs): ~19 KB for full ERI
   - H2O/6-31G* (24 AOs): ~2.6 MB for full ERI
   - C6H6/6-31G* (102 AOs): ~870 MB for full ERI
   - Large molecules quickly exceed available memory!

2. DIRECT APPROACH ADVANTAGE:
   - Memory requirement is only O(N^2) for J, K matrices
   - Integrals computed on-the-fly and discarded
   - Enables calculations on much larger systems

3. TIMING COMPARISON:
   - In-core has one-time ERI computation cost
   - In-core contractions are fast (dense matrix operations)
   - Direct recomputes integrals every SCF iteration
   - For many SCF iterations, in-core can be faster IF memory permits

4. WHEN TO USE EACH APPROACH:
   IN-CORE:
   - Small molecules (N < 100-200)
   - Many SCF iterations expected
   - Plenty of RAM available

   DIRECT:
   - Large molecules (N > 200-500)
   - Memory-constrained systems
   - Modern codes default to direct

5. PYSCF OPTIMIZATIONS:
   - Schwarz screening: skip negligible integrals
   - 8-fold symmetry: reduce work by ~8x
   - Integral batching: better cache utilization
   - These make direct SCF competitive even for small systems

6. PRACTICAL CROSSOVER POINT:
   - Typically around N ~ 100-300 AOs
   - Depends on available RAM and SCF iteration count
   - Modern codes often use direct even for small systems
     (the overhead is acceptable and memory is saved)

7. ADVANCED TECHNIQUES (beyond this lab):
   - Density Fitting (DF/RI): O(N^3) storage with 3-index tensors
   - Integral screening with rigorous bounds
   - Domain-based local approaches for O(N) scaling
"""
    print(observations)

    # =========================================================================
    # Part 5: Code Structure Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Code Structure for Direct SCF (Pseudocode)")
    print("=" * 70)

    pseudocode = """
for each SCF iteration:
    J = zeros(nao, nao)
    K = zeros(nao, nao)

    for shell_A in shells:
        for shell_B in shells:
            for shell_C in shells:
                for shell_D in shells:

                    # Schwarz screening
                    bound = sqrt((AB|AB)) * sqrt((CD|CD))
                    if bound * max(P_CD) < threshold:
                        continue  # Skip this quartet

                    # Compute integral block
                    eri_block = compute_shell_quartet(A, B, C, D)
                    # This is where Rys quadrature is called!

                    # Contract into J
                    J[A,B] += sum_{C,D} eri[A,B,C,D] * P[C,D]

                    # Contract into K
                    K[A,C] += sum_{B,D} eri[A,B,C,D] * P[B,D]

    # Apply 8-fold symmetry to avoid redundant computation
    # (handled by looping structure and symmetry flags)

    F = h + J - 0.5 * K
    # Continue with Roothaan-Hall solve...

Key insight: The Rys quadrature (Chapter 5) enters at "compute_shell_quartet"
where primitive Gaussian integrals are evaluated using Boys functions.
"""
    print(pseudocode)

    print("\n" + "=" * 70)
    print("Lab 6C Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
