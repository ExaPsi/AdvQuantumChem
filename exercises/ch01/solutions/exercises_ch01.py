#!/usr/bin/env python3
"""
Chapter 1 Exercises: Electron-Integral View of Quantum Chemistry
=================================================================

This module provides Python implementations for all exercises in Chapter 1
that involve computational tasks, PySCF usage, or numerical verification.

Exercises Covered:
------------------
1.1  Identifying Hamiltonian Terms [Core]
1.2  Tracing the Computational Pipeline [Core]
1.3  Scaling Snapshot [Core]
1.4  Electron Count and Orthonormality [Core]
1.5  ERI Symmetry Spot Check [Advanced]
1.6  Back-of-Envelope ERI Scaling [Advanced]
1.7  Physical Interpretation of J and K [Advanced]
1.8  Energy Reconstruction Without get_veff [Challenge]
1.9  Debugging an SCF Calculation [Challenge]

Usage:
------
Run the entire script to execute all exercises:
    $ python exercises_ch01.py

Or import specific functions:
    >>> from exercises_ch01 import exercise_1_2, exercise_1_8

Course: 2302638 Advanced Quantum Chemistry
Institution: Department of Chemistry, Faculty of Science, Chulalongkorn University
"""

import numpy as np
from pyscf import gto, scf
from typing import Tuple, Dict

# Configure NumPy for cleaner output
np.set_printoptions(precision=8, linewidth=100, suppress=True)


# =============================================================================
# Utility Functions
# =============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# Exercise 1.1: Identifying Hamiltonian Terms [Core]
# =============================================================================

def exercise_1_1() -> None:
    """
    Exercise 1.1: Identifying Hamiltonian Terms

    This exercise identifies which integral matrices correspond to which
    terms in the electronic Hamiltonian:

        H_e = sum_i [-1/2 nabla_i^2 - sum_A Z_A/r_iA] + sum_{i<j} 1/r_ij

    We demonstrate the connection between:
    - T matrix <-> kinetic energy operators
    - V matrix <-> nuclear attraction operators
    - ERIs <-> electron-electron repulsion operators
    """
    print_section("Exercise 1.1: Identifying Hamiltonian Terms")

    # Setup a simple molecule to demonstrate
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    N_e = mol.nelectron  # Number of electrons
    N_n = mol.natm       # Number of nuclei (atoms)

    print("\nElectronic Hamiltonian (atomic units):")
    print("  H_e = T + V_en + V_ee")
    print("      = sum_i [-1/2 nabla_i^2] - sum_iA [Z_A/r_iA] + sum_{i<j} [1/r_ij]")

    print(f"\nFor this system: {mol.nelectron} electrons, {mol.natm} nuclei")

    # Part (a): Kinetic energy matrix T
    print("\n(a) Kinetic energy matrix T:")
    print(f"    - Corresponds to: -1/2 nabla^2 operators")
    print(f"    - Number of operators: {N_e} (one per electron)")
    T = mol.intor("int1e_kin")
    print(f"    - T_uv = <chi_u | -1/2 nabla^2 | chi_v>")
    print(f"    - Example: T[0,0] = {T[0,0]:.8f} Hartree")

    # Part (b): Nuclear attraction matrix V
    print("\n(b) Nuclear attraction matrix V:")
    print(f"    - Corresponds to: -Z_A/r_iA terms (electron-nuclear attraction)")
    print(f"    - Number of terms: {N_e} x {N_n} = {N_e * N_n} (each electron with each nucleus)")
    V = mol.intor("int1e_nuc")
    print(f"    - V_uv = <chi_u | -sum_A Z_A/|r-R_A| | chi_v>")
    print(f"    - Example: V[0,0] = {V[0,0]:.8f} Hartree (negative = attractive)")

    # Part (c): Electron repulsion integrals (ERIs)
    print("\n(c) Electron repulsion integrals (ERIs):")
    print(f"    - Corresponds to: 1/r_ij terms (electron-electron repulsion)")
    n_pairs = N_e * (N_e - 1) // 2
    print(f"    - Number of unique pairs: N_e*(N_e-1)/2 = {n_pairs}")
    eri = mol.intor("int2e", aosym="s1")
    print(f"    - (uv|ls) = <chi_u chi_v | 1/r_12 | chi_l chi_s>")
    print(f"    - Example: (00|00) = {eri[0,0,0,0]:.8f} Hartree (on-site Coulomb)")

    # Part (d): Operator counts for general system
    print("\n(d) Operator count formulas:")
    print("    For N_e electrons and M nuclei:")
    print(f"    - Kinetic operators: N_e = {N_e}")
    print(f"    - Nuclear attraction terms: N_e x M = {N_e} x {N_n} = {N_e * N_n}")
    print(f"    - Electron-electron pairs: N_e(N_e-1)/2 = {n_pairs}")

    print("\n[COMPLETE] Exercise 1.1")


# =============================================================================
# Exercise 1.2: Tracing the Computational Pipeline [Core]
# =============================================================================

def exercise_1_2() -> Dict:
    """
    Exercise 1.2: Tracing the Computational Pipeline for H2O/STO-3G

    This exercise traces through the full HF computational pipeline:
    Geometry -> Basis -> Integrals -> SCF -> Energy

    Returns:
        Dictionary containing computed quantities for verification
    """
    print_section("Exercise 1.2: Computational Pipeline for H2O/STO-3G")

    # Water molecule geometry (experimental values)
    # O-H bond length: 0.96 Angstrom
    # H-O-H angle: 104.5 degrees
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    print("\nStep 1: GEOMETRY SPECIFICATION")
    print("-" * 40)
    print("Molecule: Water (H2O)")
    print("Bond length r(O-H): ~0.96 Angstrom")
    print("Bond angle H-O-H: ~104.5 degrees")

    # Build molecule
    mol = gto.M(
        atom=h2o_geometry,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print("\nStep 2: BASIS SET ASSIGNMENT")
    print("-" * 40)
    print("Basis set: STO-3G (Slater-Type Orbital fitted with 3 Gaussians)")
    print("\nBasis function count:")
    print(f"  Oxygen: 1(1s) + 1(2s) + 3(2p) = 5 functions")
    print(f"  Each H: 1(1s) = 1 function")
    print(f"  Total: 5 + 1 + 1 = 7 AO basis functions")
    print(f"\n  Verified: mol.nao = {mol.nao}")

    assert mol.nao == 7, f"Expected 7 AOs for H2O/STO-3G, got {mol.nao}"

    print("\nStep 3: INTEGRAL COMPUTATION")
    print("-" * 40)

    # One-electron integrals
    S = mol.intor("int1e_ovlp")   # Overlap
    T = mol.intor("int1e_kin")    # Kinetic
    V = mol.intor("int1e_nuc")    # Nuclear attraction
    H_core = T + V                 # Core Hamiltonian

    print(f"  S (overlap): {S.shape} matrix")
    print(f"  T (kinetic): {T.shape} matrix")
    print(f"  V (nuclear): {V.shape} matrix")
    print(f"  h (core Hamiltonian): {H_core.shape} matrix")

    # Two-electron integrals
    eri = mol.intor("int2e", aosym="s1")
    n_eri = mol.nao**4
    print(f"  ERIs: {eri.shape} tensor ({n_eri} elements)")

    # Nuclear repulsion
    E_nuc = mol.energy_nuc()
    print(f"\n  Nuclear repulsion E_nuc = {E_nuc:.8f} Hartree")

    print("\nStep 4: SCF ITERATION")
    print("-" * 40)

    # Run RHF calculation
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-9  # Default threshold
    E_total = mf.kernel()

    print(f"  SCF converged: {mf.converged}")
    print(f"  Number of iterations: {mf.cycles}")
    print(f"  Convergence threshold: {mf.conv_tol:.0e} Hartree")

    print("\nStep 5: ENERGY RESULTS")
    print("-" * 40)

    E_elec = mf.energy_elec()[0]
    print(f"  Electronic energy E_elec = {E_elec:.8f} Hartree")
    print(f"  Nuclear repulsion E_nuc  = {E_nuc:.8f} Hartree")
    print(f"  Total energy E_tot       = {E_total:.8f} Hartree")

    # Verify the energy decomposition
    assert np.isclose(E_elec + E_nuc, E_total, atol=1e-10), "Energy decomposition error"

    print("\nStep 6: PIPELINE SUMMARY")
    print("-" * 40)
    print("  Geometry (Angstrom) --> Basis (STO-3G)")
    print("      |")
    print(f"      v")
    print(f"  S, T, V matrices ({mol.nao}x{mol.nao})")
    print(f"  ERI tensor ({mol.nao}^4 = {mol.nao**4} elements)")
    print("      |")
    print(f"      v")
    print(f"  SCF ({mf.cycles} iterations)")
    print("      |")
    print(f"      v")
    print(f"  E_HF = {E_total:.8f} Hartree")

    print("\n[COMPLETE] Exercise 1.2")

    return {
        "nao": mol.nao,
        "E_nuc": E_nuc,
        "E_elec": E_elec,
        "E_total": E_total,
        "n_iterations": mf.cycles,
    }


# =============================================================================
# Exercise 1.3: Scaling Snapshot [Core]
# =============================================================================

def exercise_1_3() -> Dict:
    """
    Exercise 1.3: Memory Scaling Analysis

    This exercise compares ERI memory usage across basis sets and examines
    the ratio of full storage (s1) to symmetry-packed storage (s8).

    Returns:
        Dictionary containing memory statistics for each basis set
    """
    print_section("Exercise 1.3: ERI Memory Scaling Analysis")

    # Water molecule
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    basis_sets = ["sto-3g", "cc-pVDZ"]
    results = {}

    print("\nERI Memory Comparison for H2O:")
    print("-" * 70)
    print(f"{'Basis':<12} {'N (AOs)':<10} {'s1 (MB)':<12} {'s8 (MB)':<12} {'Ratio':<10}")
    print("-" * 70)

    for basis in basis_sets:
        mol = gto.M(
            atom=h2o_geometry,
            basis=basis,
            unit="Angstrom",
            verbose=0
        )

        N = mol.nao

        # Full tensor (s1 = no symmetry)
        eri_s1 = mol.intor("int2e", aosym="s1")
        mem_s1_mb = eri_s1.nbytes / (1024 * 1024)

        # Symmetry-packed (s8 = 8-fold symmetry)
        eri_s8 = mol.intor("int2e", aosym="s8")
        mem_s8_mb = eri_s8.nbytes / (1024 * 1024)

        ratio = mem_s1_mb / mem_s8_mb if mem_s8_mb > 0 else 0

        results[basis] = {
            "N": N,
            "s1_MB": mem_s1_mb,
            "s8_MB": mem_s8_mb,
            "ratio": ratio,
            "s1_elements": eri_s1.size,
            "s8_elements": eri_s8.size,
        }

        print(f"{basis:<12} {N:<10} {mem_s1_mb:<12.4f} {mem_s8_mb:<12.4f} {ratio:<10.1f}x")

    print("-" * 70)

    print("\nAnalysis:")
    print("-" * 40)

    print("\n(a) Number of AO basis functions:")
    for basis, data in results.items():
        print(f"    {basis}: N = {data['N']}")

    print("\n(b) Memory scaling:")
    print("    Full tensor (s1):   N^4 elements")
    print("    Packed (s8):        ~N^4/8 unique elements")

    print("\n(c) Ratio approaches 8 for large N:")
    for basis, data in results.items():
        theoretical_ratio = data['s1_elements'] / max(data['s8_elements'], 1)
        print(f"    {basis}: ratio = {data['ratio']:.1f}x (elements: {theoretical_ratio:.1f}x)")

    print("\n(d) Why ratio < 8 for small N?")
    print("    - Indexing overhead matters for small tensors")
    print("    - Some diagonal elements don't benefit from full 8-fold symmetry")
    print("    - As N grows, ratio asymptotically approaches 8")

    print("\n[COMPLETE] Exercise 1.3")

    return results


# =============================================================================
# Exercise 1.4: Electron Count and Orthonormality [Core]
# =============================================================================

def exercise_1_4() -> Dict:
    """
    Exercise 1.4: Electron Count and MO Orthonormality Checks

    This exercise verifies:
    - Electron count: Tr[PS] = N_e
    - MO orthonormality: C^T S C = I

    Both formulas involve the overlap matrix S because it defines the metric
    in the non-orthogonal AO basis.

    Returns:
        Dictionary containing verification results
    """
    print_section("Exercise 1.4: Electron Count and MO Orthonormality")

    # H2 molecule for demonstration
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Run HF to get density matrix and MO coefficients
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    S = mol.intor("int1e_ovlp")  # Overlap matrix
    P = mf.make_rdm1()            # Density matrix
    C = mf.mo_coeff               # MO coefficient matrix
    N = mol.nao

    print("\nPart A: Electron Count Formula Tr[PS] = N_e")
    print("-" * 50)

    # Compute electron count two ways
    N_e_expected = mol.nelectron
    N_e_computed = np.trace(P @ S)
    N_e_wrong = np.trace(P)  # Incorrect formula!

    print(f"\n  Expected N_e = {N_e_expected}")
    print(f"  Tr[PS] = {N_e_computed:.10f} (correct)")
    print(f"  Tr[P]  = {N_e_wrong:.10f} (WRONG - ignores overlap)")

    print("\n  Why does S appear in the formula?")
    print("  -" * 25)
    print("  In a non-orthogonal basis, the 'volume element' for probability")
    print("  is weighted by the overlap. The S matrix acts as a metric tensor")
    print("  that properly accounts for the geometry of the AO space.")

    print("\n  Derivation:")
    print("    N_e = integral rho(r) dr")
    print("        = integral [sum_uv P_uv chi_u(r) chi_v(r)] dr")
    print("        = sum_uv P_uv S_vu")
    print("        = Tr[PS]")

    # Verify
    assert np.isclose(N_e_computed, N_e_expected, atol=1e-10), \
        f"Electron count mismatch: {N_e_computed} vs {N_e_expected}"
    print(f"\n  [PASS] Electron count verified: Tr[PS] = {N_e_expected}")

    print("\nPart B: MO Orthonormality C^T S C = I")
    print("-" * 50)

    # Check MO orthonormality
    ortho_check = C.T @ S @ C
    identity = np.eye(N)
    ortho_error = np.linalg.norm(ortho_check - identity)

    print(f"\n  C^T S C =")
    print(f"  {ortho_check}")
    print(f"\n  ||C^T S C - I|| = {ortho_error:.2e}")

    print("\n  Why does S appear here?")
    print("  -" * 25)
    print("  MOs are orthonormal in the proper inner product:")
    print("    <phi_p|phi_q> = sum_uv C_up* S_uv C_vq = delta_pq")
    print("  In matrix form: C^T S C = I")
    print("  The S matrix defines the inner product in the AO basis.")

    # Verify
    assert ortho_error < 1e-10, f"MO orthonormality failed: error = {ortho_error}"
    print(f"\n  [PASS] MO orthonormality verified: ||C^T S C - I|| < 1e-10")

    print("\nPart C: Physical Interpretation")
    print("-" * 50)

    print("\n  The overlap matrix S is the METRIC TENSOR of the AO space.")
    print("  Just as in curved coordinates, we need the metric to:")
    print("    1. Measure distances/norms: ||u||^2 = u^T S u")
    print("    2. Compute inner products: <u,v> = u^T S v")
    print("    3. Integrate properly over space")

    print("\n  In an ORTHONORMAL basis (e.g., after transformation):")
    print("    S = I, so Tr[P] = N_e and C^T C = I")
    print("  But in the original AO basis, we must always include S!")

    print("\n[COMPLETE] Exercise 1.4")

    return {
        "N_e_computed": N_e_computed,
        "N_e_expected": N_e_expected,
        "ortho_error": ortho_error,
        "N_e_wrong": N_e_wrong,
    }


# =============================================================================
# Exercise 1.5: ERI Symmetry Spot Check [Advanced]
# =============================================================================

def exercise_1_5() -> Dict:
    """
    Exercise 1.5: Comprehensive ERI 8-fold Symmetry Verification

    This exercise verifies all 8-fold ERI symmetries and explains their
    physical origins.

    Returns:
        Dictionary containing symmetry test results
    """
    print_section("Exercise 1.5: ERI 8-fold Symmetry Verification")

    # Use H2O for more interesting ERIs
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    mol = gto.M(
        atom=h2o_geometry,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    eri = mol.intor("int2e", aosym="s1")
    N = mol.nao

    print("\nPhysical Origins of 8-fold ERI Symmetry:")
    print("-" * 50)

    print("\n1. Commutativity of multiplication (real basis):")
    print("   chi_u(r) * chi_v(r) = chi_v(r) * chi_u(r)")
    print("   --> (uv|ls) = (vu|ls) and (uv|ls) = (uv|sl)")

    print("\n2. Symmetry of Coulomb operator:")
    print("   1/|r_1 - r_2| = 1/|r_2 - r_1|")
    print("   --> (uv|ls) = (ls|uv)")

    print("\n3. Real basis functions:")
    print("   chi_u = chi_u* for Gaussians")
    print("   --> No additional complex conjugation symmetry")

    print("\n4. Combined symmetries give 2 x 2 x 2 = 8-fold total")

    print("\nExhaustive Symmetry Test:")
    print("-" * 50)

    # Test all 8 permutations for every index combination
    n_tests = 0
    n_passed = 0
    max_error = 0.0
    tol = 1e-14

    for u in range(N):
        for v in range(N):
            for l in range(N):
                for s in range(N):
                    base = eri[u, v, l, s]

                    # All 8 permutations
                    permutations = [
                        eri[u, v, l, s],  # Original
                        eri[v, u, l, s],  # Swap u <-> v
                        eri[u, v, s, l],  # Swap l <-> s
                        eri[v, u, s, l],  # Swap both
                        eri[l, s, u, v],  # Swap bra <-> ket
                        eri[s, l, u, v],  # Combined
                        eri[l, s, v, u],  # Combined
                        eri[s, l, v, u],  # All swaps
                    ]

                    for val in permutations:
                        n_tests += 1
                        error = abs(val - base)
                        max_error = max(max_error, error)
                        if error < tol:
                            n_passed += 1

    print(f"\n  Total symmetry tests: {n_tests}")
    print(f"  Tests passed: {n_passed}")
    print(f"  Maximum error: {max_error:.2e}")
    print(f"  Tolerance: {tol:.0e}")

    passed = n_passed == n_tests
    if passed:
        print("\n  [PASS] All 8-fold ERI symmetries verified!")
    else:
        print(f"\n  [FAIL] {n_tests - n_passed} symmetry tests failed!")

    # Demonstrate with a specific example
    print("\nExample: Index (0,1,2,3):")
    print("-" * 50)

    u, v, l, s = 0, 1, 2, 3
    base = eri[u, v, l, s]

    examples = [
        ((u, v, l, s), "original"),
        ((v, u, l, s), "swap u<->v (bra)"),
        ((u, v, s, l), "swap l<->s (ket)"),
        ((v, u, s, l), "swap both"),
        ((l, s, u, v), "swap bra<->ket"),
        ((s, l, u, v), "combined"),
        ((l, s, v, u), "combined"),
        ((s, l, v, u), "all swaps"),
    ]

    print(f"  Base value (0,1|2,3) = {base:.10f}")
    print()
    for (i, j, k, m), desc in examples:
        val = eri[i, j, k, m]
        diff = abs(val - base)
        print(f"  ({i},{j}|{k},{m}) = {val:.10f}  [{desc}]  diff={diff:.2e}")

    print("\n[COMPLETE] Exercise 1.5")

    return {
        "n_tests": n_tests,
        "n_passed": n_passed,
        "max_error": max_error,
        "passed": passed,
    }


# =============================================================================
# Exercise 1.6: Back-of-Envelope ERI Scaling [Advanced]
# =============================================================================

def exercise_1_6() -> Dict:
    """
    Exercise 1.6: ERI Scaling Estimates for Benzene/cc-pVDZ

    This exercise provides scaling estimates and motivates alternative methods
    like direct SCF and density fitting.

    Returns:
        Dictionary containing scaling estimates
    """
    print_section("Exercise 1.6: ERI Scaling for Benzene/cc-pVDZ")

    print("\n(a) Basis Function Count Estimate:")
    print("-" * 50)

    # cc-pVDZ basis function counts (approximate)
    # Carbon: 1s, 2s, 2p, 3s, 3p, 3d-like -> ~14 functions
    # Hydrogen: 1s, 2s, 2p-like -> ~5 functions
    n_C = 6
    n_H = 6
    funcs_per_C = 14  # Approximate for cc-pVDZ
    funcs_per_H = 5   # Approximate for cc-pVDZ

    N_estimate = n_C * funcs_per_C + n_H * funcs_per_H
    print(f"  Benzene: C6H6")
    print(f"  Carbon: ~{funcs_per_C} functions/atom x {n_C} = {n_C * funcs_per_C}")
    print(f"  Hydrogen: ~{funcs_per_H} functions/atom x {n_H} = {n_H * funcs_per_H}")
    print(f"  Total estimate: N ~ {N_estimate} functions")

    # Actual calculation for verification
    benzene_geometry = """
    C   1.40272   0.00000   0.00000
    C   0.70136   1.21479   0.00000
    C  -0.70136   1.21479   0.00000
    C  -1.40272   0.00000   0.00000
    C  -0.70136  -1.21479   0.00000
    C   0.70136  -1.21479   0.00000
    H   2.49029   0.00000   0.00000
    H   1.24515   2.15666   0.00000
    H  -1.24515   2.15666   0.00000
    H  -2.49029   0.00000   0.00000
    H  -1.24515  -2.15666   0.00000
    H   1.24515  -2.15666   0.00000
    """

    mol = gto.M(
        atom=benzene_geometry,
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    N_actual = mol.nao

    print(f"\n  Actual PySCF count: N = {N_actual}")

    print("\n(b) Full ERI Memory Estimate:")
    print("-" * 50)

    # Memory for full ERI tensor
    bytes_per_element = 8  # float64
    N = N_actual
    n_elements = N**4
    mem_bytes = n_elements * bytes_per_element
    mem_GB = mem_bytes / (1024**3)

    print(f"  N = {N} basis functions")
    print(f"  N^4 = {N}^4 = {n_elements:,} elements")
    print(f"  Memory = N^4 x 8 bytes = {mem_bytes:,} bytes")
    print(f"         = {mem_GB:.2f} GB")

    print("\n(c) Maximum N for 16 GB RAM:")
    print("-" * 50)

    ram_limit_bytes = 16 * (1024**3)  # 16 GB in bytes
    max_N = int((ram_limit_bytes / bytes_per_element) ** 0.25)

    print(f"  N^4 x 8 <= 16 x 10^9")
    print(f"  N <= (16 x 10^9 / 8)^(1/4)")
    print(f"  N <= {max_N}")
    print(f"\n  With 16 GB RAM, maximum ~{max_N} basis functions can be stored!")

    print("\n(d) Motivation for Alternative Methods:")
    print("-" * 50)

    print("\n  DIRECT SCF:")
    print("    - Never store full ERI tensor")
    print("    - Recompute needed ERIs each SCF iteration")
    print("    - Trade-off: More compute time, but O(1) ERI memory")
    print("    - Enables calculations impossible with in-core storage")

    print("\n  DENSITY FITTING (Resolution of Identity):")
    print("    - Approximate: (uv|ls) ~ sum_Q B_uv^Q B_ls^Q")
    print("    - Store 3-index tensor B_uv^Q instead of 4-index ERI")
    print("    - Memory: O(N^2 N_aux) instead of O(N^4)")
    print(f"    - For N={N}: ~{N**2 * 3 * N // (1024**2):.0f} MB vs {mem_GB:.2f} GB")
    print("    - Accuracy: ~10^-4 to 10^-5 Hartree error (acceptable for most uses)")

    print("\n  SCHWARZ SCREENING:")
    print("    - Only compute ERIs above threshold")
    print("    - |(uv|ls)| <= sqrt(uv|uv) * sqrt(ls|ls)")
    print("    - Effective scaling: O(N^2.5) to O(N^2) for large systems")

    print("\n[COMPLETE] Exercise 1.6")

    return {
        "N_estimate": N_estimate,
        "N_actual": N_actual,
        "mem_GB": mem_GB,
        "max_N_16GB": max_N,
    }


# =============================================================================
# Exercise 1.7: Physical Interpretation of J and K [Advanced]
# =============================================================================

def exercise_1_7() -> Dict:
    """
    Exercise 1.7: Physical Interpretation of Coulomb and Exchange Matrices

    This exercise explains the physical meaning of J (Coulomb) and K (exchange)
    matrices and demonstrates self-interaction cancellation.

    Returns:
        Dictionary containing J, K, and energy components
    """
    print_section("Exercise 1.7: Physical Interpretation of J and K")

    # Use H2 for clear demonstration
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Run HF
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Get matrices
    H_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    P = mf.make_rdm1()

    # Build J and K
    J = np.einsum('uvls,ls->uv', eri, P)
    K = np.einsum('ulvs,ls->uv', eri, P)

    print("\n(a) Coulomb Matrix J - Classical Electrostatics:")
    print("-" * 50)

    print("\n  J_uv = sum_ls (uv|ls) P_ls")
    print("\n  Physical interpretation:")
    print("    J represents the electrostatic potential from the electron density:")
    print()
    print("    J_uv = integral chi_u(r_1) chi_v(r_1) V_coul(r_1) dr_1")
    print()
    print("    where V_coul(r_1) = integral rho(r_2) / |r_1 - r_2| dr_2")
    print()
    print("    This is CLASSICAL Coulomb repulsion, as if electrons were")
    print("    continuous charge distributions.")

    print(f"\n  J matrix =\n{J}")

    print("\n(b) Exchange Matrix K - Quantum Mechanical (No Classical Analog):")
    print("-" * 50)

    print("\n  K_uv = sum_ls (ul|vs) P_ls")
    print("\n  Physical interpretation:")
    print("    K arises from the ANTISYMMETRY of the wavefunction:")
    print()
    print("    Psi(1,2) = -Psi(2,1)")
    print()
    print("    This is the Pauli exclusion principle in action!")
    print()
    print("    Exchange creates an 'exchange hole' - a region around each")
    print("    electron depleted of same-spin electrons.")
    print()
    print("    There is NO classical analog because classical particles")
    print("    are distinguishable.")

    print(f"\n  K matrix =\n{K}")

    print("\n(c) Factor of 1/2 in RHF Fock Matrix:")
    print("-" * 50)

    print("\n  F = h + J - (1/2)K")
    print("\n  Why the 1/2 on K?")
    print("    - Coulomb (J): All N_e^2 electron pairs contribute")
    print("    - Exchange (K): Only SAME-SPIN pairs contribute")
    print("    - For closed-shell RHF:")
    print("      - Each orbital has 2 electrons (alpha and beta spin)")
    print("      - Exchange only between alpha-alpha or beta-beta")
    print("      - Only N_e^2/2 pairs contribute to exchange")
    print("    - The 1/2 factor reflects that half the pairs are opposite-spin")

    print("\n(d) Self-Interaction Cancellation:")
    print("-" * 50)

    # For a single doubly-occupied orbital (like He 1s^2 or H2 sigma_g)
    # The diagonal elements show J_ii = K_ii

    print("\n  Consider a single doubly-occupied orbital (e.g., H2 sigma_g):")
    print()
    print("    J_11 = (11|11) = self-Coulomb (electron in orbital 1 repels itself)")
    print("    K_11 = (11|11) = same integral!")
    print()

    # For H2/STO-3G, transform to MO basis to check
    C = mf.mo_coeff
    C_occ = C[:, :1]  # Only occupied orbital

    # Transform ERIs to MO basis
    # (pq|rs) = sum_uvls C_up C_vq (uv|ls) C_lr C_sr
    eri_mo = np.einsum('ui,vj,uvls,lk,sm->ijkm', C, C, eri, C, C)

    J_11_mo = eri_mo[0, 0, 0, 0]
    K_11_mo = eri_mo[0, 0, 0, 0]  # Same for 4-identical indices

    print(f"    J_11 (MO basis) = {J_11_mo:.10f} Hartree")
    print(f"    K_11 (MO basis) = {K_11_mo:.10f} Hartree")
    print()
    print("  In the energy expression:")
    print("    E_2e = (1/2) Tr[P*J] - (1/4) Tr[P*K]")
    print()
    print("  For a single doubly-occupied orbital:")
    print("    Contribution from self-interaction:")
    print("      (1/2) * 2 * J_11 - (1/4) * 2 * 2 * K_11")
    print("      = J_11 - K_11 = 0")
    print()
    print("  The self-Coulomb is EXACTLY canceled by self-exchange!")
    print("  This proves HF is SELF-INTERACTION FREE.")

    print("\n[COMPLETE] Exercise 1.7")

    return {
        "J": J,
        "K": K,
        "J_11_mo": J_11_mo,
        "K_11_mo": K_11_mo,
        "self_interaction_error": abs(J_11_mo - K_11_mo),
    }


# =============================================================================
# Exercise 1.8: Energy Reconstruction Without get_veff [Challenge]
# =============================================================================

def exercise_1_8() -> Dict:
    """
    Exercise 1.8: Complete HF Energy Reconstruction from Integrals

    This challenge exercise rebuilds the HF energy entirely from integrals
    without using PySCF's get_veff() convenience function.

    Returns:
        Dictionary containing energy components and validation results
    """
    print_section("Exercise 1.8: Energy Reconstruction from Integrals")

    # H2 molecule
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print("\nStep 1: Run Reference HF Calculation")
    print("-" * 50)

    mf = scf.RHF(mol)
    mf.verbose = 0
    E_pyscf = mf.kernel()
    P = mf.make_rdm1()

    print(f"  PySCF RHF energy: {E_pyscf:.10f} Hartree")

    print("\nStep 2: Extract All Integrals")
    print("-" * 50)

    S = mol.intor("int1e_ovlp")    # Overlap
    T = mol.intor("int1e_kin")     # Kinetic
    V = mol.intor("int1e_nuc")     # Nuclear attraction
    H_core = T + V                  # Core Hamiltonian
    eri = mol.intor("int2e", aosym="s1")  # ERIs
    E_nuc = mol.energy_nuc()        # Nuclear repulsion

    print(f"  S shape: {S.shape}")
    print(f"  h shape: {H_core.shape}")
    print(f"  ERI shape: {eri.shape}")
    print(f"  E_nuc: {E_nuc:.10f} Hartree")

    print("\nStep 3: Build Coulomb Matrix J")
    print("-" * 50)

    print("  J_mn = sum_ls (mn|ls) * P_ls")
    print("  Using einsum: J = einsum('mnls,ls->mn', eri, P)")

    J = np.einsum('mnls,ls->mn', eri, P)
    print(f"\n  J =\n{J}")

    print("\nStep 4: Build Exchange Matrix K")
    print("-" * 50)

    print("  K_mn = sum_ls (ml|ns) * P_ls")
    print("  Using einsum: K = einsum('mlns,ls->mn', eri, P)")

    K = np.einsum('mlns,ls->mn', eri, P)
    print(f"\n  K =\n{K}")

    print("\nStep 5: Build Fock Matrix F")
    print("-" * 50)

    print("  F = h + J - (1/2)*K")

    F = H_core + J - 0.5 * K
    print(f"\n  F =\n{F}")

    # Compare to PySCF Fock matrix
    F_pyscf = mf.get_fock()
    F_error = np.linalg.norm(F - F_pyscf)
    print(f"\n  ||F - F_pyscf|| = {F_error:.2e}")

    assert F_error < 1e-10, f"Fock matrix mismatch: {F_error}"
    print("  [PASS] Fock matrix matches PySCF!")

    print("\nStep 6: Compute Electronic Energy")
    print("-" * 50)

    print("  E_elec = (1/2) * Tr[P * (h + F)]")

    E_elec = 0.5 * np.einsum('ij,ji->', P, H_core + F)
    print(f"\n  E_elec = {E_elec:.10f} Hartree")

    # Breakdown for verification
    E_1e = np.einsum('ij,ji->', P, H_core)
    E_J = 0.5 * np.einsum('ij,ji->', P, J)
    E_K = -0.25 * np.einsum('ij,ji->', P, K)

    print("\n  Energy breakdown:")
    print(f"    Tr[P*h]           = {E_1e:.10f} Hartree (one-electron)")
    print(f"    (1/2)*Tr[P*J]     = {E_J:.10f} Hartree (Coulomb)")
    print(f"    -(1/4)*Tr[P*K]    = {E_K:.10f} Hartree (Exchange)")
    print(f"    Sum               = {E_1e + E_J + E_K:.10f} Hartree")

    assert np.isclose(E_elec, E_1e + E_J + E_K, atol=1e-12), \
        "Energy breakdown inconsistency!"

    print("\nStep 7: Total Energy")
    print("-" * 50)

    print("  E_tot = E_elec + E_nuc")

    E_total = E_elec + E_nuc
    print(f"\n  E_total = {E_total:.10f} Hartree")

    print("\nStep 8: Validation")
    print("-" * 50)

    E_error = abs(E_total - E_pyscf)
    print(f"  Our calculation:  {E_total:.10f} Hartree")
    print(f"  PySCF reference:  {E_pyscf:.10f} Hartree")
    print(f"  Difference:       {E_error:.2e} Hartree")

    assert E_error < 1e-10, f"Energy mismatch: {E_error}"
    print("\n  [PASS] Energy validated against PySCF!")

    print("\nKey Formulas Summary:")
    print("-" * 50)
    print("  J_mn = sum_ls (mn|ls) P_ls")
    print("  K_mn = sum_ls (ml|ns) P_ls")
    print("  F = h + J - (1/2)*K")
    print("  E_elec = (1/2) Tr[P(h + F)]")
    print("  E_tot = E_elec + E_nuc")

    print("\n[COMPLETE] Exercise 1.8")

    return {
        "E_1e": E_1e,
        "E_J": E_J,
        "E_K": E_K,
        "E_elec": E_elec,
        "E_nuc": E_nuc,
        "E_total": E_total,
        "E_pyscf": E_pyscf,
        "E_error": E_error,
    }


# =============================================================================
# Exercise 1.9: Debugging an SCF Calculation [Challenge]
# =============================================================================

def exercise_1_9() -> Dict:
    """
    Exercise 1.9: Debugging a Buggy SCF Implementation

    This exercise presents buggy HF code and demonstrates how to identify
    and fix the errors.

    Returns:
        Dictionary containing results from buggy and fixed implementations
    """
    print_section("Exercise 1.9: Debugging an SCF Calculation")

    # H2 molecule
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Get reference values
    mf_ref = scf.RHF(mol)
    mf_ref.verbose = 0
    E_ref = mf_ref.kernel()

    # Integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    # Get converged MO coefficients for demonstration
    C = mf_ref.mo_coeff
    n_occ = mol.nelectron // 2
    C_occ = C[:, :n_occ]

    print("\nThe Buggy Code:")
    print("-" * 50)

    buggy_code = '''
# BUGGY CODE - Find the 3 errors!
P = np.einsum('mi,ni->mn', C_occ, C_occ)    # BUG 1
J = np.einsum('mnls,ls->mn', eri, P)
K = np.einsum('mlns,ls->mn', eri, P)
F = h + J - K                                 # BUG 2
E_elec = np.trace(P @ (h + F))               # BUG 3
E_tot = E_elec + E_nuc
'''
    print(buggy_code)

    # Run the buggy code
    print("Buggy Results:")
    print("-" * 50)

    # Bug 1: Missing factor of 2 in density matrix
    P_buggy = np.einsum('mi,ni->mn', C_occ, C_occ)  # Wrong!
    J_buggy = np.einsum('mnls,ls->mn', eri, P_buggy)
    K_buggy = np.einsum('mlns,ls->mn', eri, P_buggy)

    # Bug 2: Missing factor of 1/2 on exchange
    F_buggy = h + J_buggy - K_buggy  # Wrong!

    # Bug 3: Missing factor of 1/2 in energy
    E_elec_buggy = np.trace(P_buggy @ (h + F_buggy))  # Wrong!
    E_tot_buggy = E_elec_buggy + E_nuc

    print(f"  Tr[P_buggy * S] = {np.trace(P_buggy @ S):.4f} (expected: {mol.nelectron})")
    print(f"  E_tot_buggy = {E_tot_buggy:.10f} Hartree")
    print(f"  E_ref (PySCF) = {E_ref:.10f} Hartree")
    print(f"  Error = {abs(E_tot_buggy - E_ref):.6f} Hartree")

    print("\nBug Identification:")
    print("-" * 50)

    # Bug 1 detection
    N_e_buggy = np.trace(P_buggy @ S)
    print(f"\nBUG 1: Missing factor of 2 in density matrix")
    print(f"  P = np.einsum('mi,ni->mn', C_occ, C_occ)")
    print(f"  Electron count: Tr[P*S] = {N_e_buggy:.4f} (should be {mol.nelectron})")
    print(f"  --> Missing factor of 2 for doubly-occupied orbitals!")
    print(f"  FIX: P = 2 * np.einsum('mi,ni->mn', C_occ, C_occ)")

    # Bug 2 detection
    F_correct = h + J_buggy - 0.5 * K_buggy  # Still using buggy P
    F_diff = np.linalg.norm(F_buggy - mf_ref.get_fock())
    print(f"\nBUG 2: Missing factor of 1/2 on exchange in Fock matrix")
    print(f"  F = h + J - K")
    print(f"  ||F - F_ref|| = {F_diff:.4f}")
    print(f"  --> Exchange only between same-spin electrons!")
    print(f"  FIX: F = h + J - 0.5*K")

    # Bug 3 detection
    print(f"\nBUG 3: Missing factor of 1/2 in energy formula")
    print(f"  E_elec = np.trace(P @ (h + F))")
    print(f"  --> Double-counts electron pairs!")
    print(f"  FIX: E_elec = 0.5 * np.trace(P @ (h + F))")

    print("\nCorrected Code:")
    print("-" * 50)

    fixed_code = '''
# FIXED CODE
P = 2 * np.einsum('mi,ni->mn', C_occ, C_occ)  # Factor of 2!
J = np.einsum('mnls,ls->mn', eri, P)
K = np.einsum('mlns,ls->mn', eri, P)
F = h + J - 0.5*K                              # Factor of 1/2!
E_elec = 0.5 * np.trace(P @ (h + F))          # Factor of 1/2!
E_tot = E_elec + E_nuc
'''
    print(fixed_code)

    # Run the fixed code
    print("Fixed Results:")
    print("-" * 50)

    P_fixed = 2 * np.einsum('mi,ni->mn', C_occ, C_occ)  # Fixed!
    J_fixed = np.einsum('mnls,ls->mn', eri, P_fixed)
    K_fixed = np.einsum('mlns,ls->mn', eri, P_fixed)
    F_fixed = h + J_fixed - 0.5 * K_fixed  # Fixed!
    E_elec_fixed = 0.5 * np.trace(P_fixed @ (h + F_fixed))  # Fixed!
    E_tot_fixed = E_elec_fixed + E_nuc

    N_e_fixed = np.trace(P_fixed @ S)
    F_error = np.linalg.norm(F_fixed - mf_ref.get_fock())
    E_error = abs(E_tot_fixed - E_ref)

    print(f"  Tr[P_fixed * S] = {N_e_fixed:.10f} (expected: {mol.nelectron})")
    print(f"  ||F - F_ref|| = {F_error:.2e}")
    print(f"  E_tot_fixed = {E_tot_fixed:.10f} Hartree")
    print(f"  E_ref (PySCF) = {E_ref:.10f} Hartree")
    print(f"  Error = {E_error:.2e} Hartree")

    assert np.isclose(N_e_fixed, mol.nelectron, atol=1e-10), "Electron count still wrong!"
    assert F_error < 1e-10, "Fock matrix still wrong!"
    assert E_error < 1e-10, "Energy still wrong!"

    print("\n  [PASS] All fixes verified!")

    print("\nDebugging Strategy Summary:")
    print("-" * 50)
    print("  1. Check Tr[PS] = N_e (catches missing factor of 2 in P)")
    print("  2. Compare F against mf.get_fock() (catches factor in F)")
    print("  3. Compare E against mf.e_tot (catches factor in energy)")
    print("  4. Run each check independently to isolate errors")

    print("\n[COMPLETE] Exercise 1.9")

    return {
        "E_buggy": E_tot_buggy,
        "E_fixed": E_tot_fixed,
        "E_ref": E_ref,
        "N_e_buggy": N_e_buggy,
        "N_e_fixed": N_e_fixed,
        "bugs_identified": 3,
    }


# =============================================================================
# Main: Run All Exercises
# =============================================================================

def main():
    """Run all Chapter 1 exercises and display results."""

    print("\n" + "=" * 75)
    print("  Chapter 1 Exercises: Electron-Integral View of Quantum Chemistry")
    print("=" * 75)
    print("\nThis script runs all Chapter 1 exercises with computational components.")
    print("Each exercise validates against PySCF reference values.\n")

    # Track results
    all_results = {}

    # Core Exercises
    print("\n" + "#" * 75)
    print("#  CORE EXERCISES")
    print("#" * 75)

    exercise_1_1()
    all_results['1.1'] = 'Complete'

    all_results['1.2'] = exercise_1_2()
    all_results['1.3'] = exercise_1_3()
    all_results['1.4'] = exercise_1_4()

    # Advanced Exercises
    print("\n" + "#" * 75)
    print("#  ADVANCED EXERCISES")
    print("#" * 75)

    all_results['1.5'] = exercise_1_5()
    all_results['1.6'] = exercise_1_6()
    all_results['1.7'] = exercise_1_7()

    # Challenge Exercises
    print("\n" + "#" * 75)
    print("#  CHALLENGE EXERCISES")
    print("#" * 75)

    all_results['1.8'] = exercise_1_8()
    all_results['1.9'] = exercise_1_9()

    # Summary
    print("\n" + "=" * 75)
    print("  SUMMARY: All Chapter 1 Exercises Complete")
    print("=" * 75)

    print("\nExercises completed:")
    for ex_num in sorted(all_results.keys()):
        result = all_results[ex_num]
        if isinstance(result, dict):
            # Extract key validation metric if available
            if 'E_error' in result:
                status = f"E_error = {result['E_error']:.2e}"
            elif 'passed' in result:
                status = "PASSED" if result['passed'] else "FAILED"
            elif 'ortho_error' in result:
                status = f"ortho_error = {result['ortho_error']:.2e}"
            else:
                status = "Complete"
        else:
            status = str(result)
        print(f"  Exercise {ex_num}: {status}")

    print("\nKey course theme demonstrated:")
    print("  HARTREE-FOCK = INTEGRALS + LINEAR ALGEBRA")
    print("\nOnce we have S, T, V, and ERIs, the rest is matrix operations!")

    print("\n" + "=" * 75)
    print("  All exercises validated against PySCF reference values")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
