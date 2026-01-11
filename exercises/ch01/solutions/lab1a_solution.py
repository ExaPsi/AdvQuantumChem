#!/usr/bin/env python3
"""
Lab 1A Solution: AO Integral Inventory and Sanity Checks
=========================================================

This script serves as the answer key for Lab 1A from Chapter 1 of the
Advanced Quantum Chemistry lecture notes. It demonstrates the integral-driven
approach to Hartree-Fock theory.

Learning Objectives:
--------------------
1. Extract one-electron (S, T, V) and two-electron (ERI) integrals from PySCF
2. Understand and verify the symmetry properties of these matrices
3. Verify all 8-fold ERI symmetries explicitly
4. Compute electron count via Tr[PS] as a sanity check
5. Reconstruct the HF energy from integrals and compare to PySCF

Central Theme:
--------------
Everything in Hartree-Fock reduces to computing integrals + linear algebra!

Part of: 2302638 Advanced Quantum Chemistry
Instructor: Viwat Vchirawongkwin, Chulalongkorn University
"""

import numpy as np
from pyscf import gto, scf

# For reproducibility and cleaner output
np.set_printoptions(precision=8, linewidth=100, suppress=True)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# PART 1: MOLECULE AND BASIS SET SETUP
# =============================================================================

print_section("PART 1: Molecule and Basis Set Setup")

# Build H2 molecule at equilibrium bond length
# Using STO-3G (minimal basis) for clarity
mol = gto.M(
    atom="H 0.0 0.0 0.0; H 0.0 0.0 0.74",  # Bond along z-axis
    basis="sto-3g",
    unit="Angstrom",
    verbose=0  # Suppress PySCF output for clean display
)

print(f"Molecule: H2")
print(f"Bond length: 0.74 Angstrom (equilibrium geometry)")
print(f"Basis set: STO-3G (minimal basis)")
print(f"Number of atoms: {mol.natm}")
print(f"Number of electrons: {mol.nelectron}")
print(f"Number of AO basis functions (N): {mol.nao}")
print(f"Multiplicity: {mol.spin + 1} (singlet)")

# For STO-3G, each H atom contributes 1 s-function (1 contracted from 3 primitives)
# So N = 2 for H2 in STO-3G
assert mol.nao == 2, "Expected 2 AO basis functions for H2/STO-3G"


# =============================================================================
# PART 2: ONE-ELECTRON INTEGRAL EXTRACTION AND ANALYSIS
# =============================================================================

print_section("PART 2: One-Electron Integrals")

# Extract one-electron integrals using PySCF's intor interface
# These are the building blocks for the core Hamiltonian

# Overlap matrix: S_uv = <chi_u | chi_v>
# Measures the spatial overlap between basis functions
S = mol.intor("int1e_ovlp")

# Kinetic energy matrix: T_uv = <chi_u | -1/2 nabla^2 | chi_v>
# Note: PySCF includes the 1/2 factor (atomic units)
T = mol.intor("int1e_kin")

# Nuclear attraction matrix: V_uv = <chi_u | -sum_A Z_A/|r-R_A| | chi_v>
# Describes electron-nuclear Coulomb attraction
V = mol.intor("int1e_nuc")

# Core Hamiltonian: h = T + V
# One-electron part of the Fock matrix
H_core = T + V

print_subsection("Matrix Dimensions")
print(f"S (overlap) shape: {S.shape}")
print(f"T (kinetic) shape: {T.shape}")
print(f"V (nuclear attraction) shape: {V.shape}")
print(f"h (core Hamiltonian) shape: {H_core.shape}")

print_subsection("Overlap Matrix S")
print("Physical meaning: S_uv = integral of chi_u(r) * chi_v(r) over all space")
print("Diagonal elements should be 1 (normalized basis functions)")
print("Off-diagonal measures overlap between different functions")
print(f"\nS =\n{S}")

print_subsection("Kinetic Energy Matrix T")
print("Physical meaning: T_uv = -1/2 * integral of chi_u * nabla^2(chi_v)")
print("Always positive (kinetic energy is positive)")
print(f"\nT =\n{T}")

print_subsection("Nuclear Attraction Matrix V")
print("Physical meaning: V_uv = -sum_A Z_A * integral of chi_u * chi_v / |r - R_A|")
print("Should be negative (attractive potential)")
print(f"\nV =\n{V}")

print_subsection("Core Hamiltonian h = T + V")
print(f"\nh =\n{H_core}")

# Verify symmetry of one-electron matrices
print_subsection("Symmetry Verification")
print("All one-electron matrices should be symmetric: A = A^T")

S_symmetric = np.allclose(S, S.T, atol=1e-14)
T_symmetric = np.allclose(T, T.T, atol=1e-14)
V_symmetric = np.allclose(V, V.T, atol=1e-14)
h_symmetric = np.allclose(H_core, H_core.T, atol=1e-14)

print(f"  S symmetric: {S_symmetric} (||S - S^T|| = {np.linalg.norm(S - S.T):.2e})")
print(f"  T symmetric: {T_symmetric} (||T - T^T|| = {np.linalg.norm(T - T.T):.2e})")
print(f"  V symmetric: {V_symmetric} (||V - V^T|| = {np.linalg.norm(V - V.T):.2e})")
print(f"  h symmetric: {h_symmetric} (||h - h^T|| = {np.linalg.norm(H_core - H_core.T):.2e})")

assert S_symmetric and T_symmetric and V_symmetric and h_symmetric, \
    "Symmetry check failed!"
print("\n[PASS] All one-electron matrices are symmetric")


# =============================================================================
# PART 3: TWO-ELECTRON INTEGRALS (ERIs)
# =============================================================================

print_section("PART 3: Two-Electron Integrals (ERIs)")

# Extract full 4-index ERI tensor using aosym='s1' (no symmetry packing)
# ERI notation: (uv|ls) = integral of chi_u(1)*chi_v(1) * 1/r12 * chi_l(2)*chi_s(2)
# This is CHEMIST'S notation (Mulliken notation), NOT physicist's notation!
eri = mol.intor("int2e", aosym="s1")

print_subsection("ERI Tensor Properties")
print(f"ERI shape: {eri.shape}")
print(f"ERI dtype: {eri.dtype}")
print(f"Number of elements: {eri.size}")
print(f"Memory usage: {eri.nbytes / 1024:.2f} KB")

# For larger systems, memory scales as O(N^4)
N = mol.nao
theoretical_size_bytes = N**4 * 8  # 8 bytes per float64
print(f"\nScaling analysis:")
print(f"  N = {N} basis functions")
print(f"  Full ERI tensor: N^4 = {N**4} elements")
print(f"  Theoretical memory: {theoretical_size_bytes / 1024:.2f} KB")
print(f"  Actual memory: {eri.nbytes / 1024:.2f} KB")

# Show the ERI tensor (small enough to display for H2/STO-3G)
print_subsection("ERI Tensor Values")
print("Using chemist's notation (uv|ls):")
for u in range(N):
    for v in range(N):
        for l in range(N):
            for s in range(N):
                val = eri[u, v, l, s]
                if abs(val) > 1e-10:
                    print(f"  ({u}{v}|{l}{s}) = {val:.10f}")


# =============================================================================
# PART 4: VERIFICATION OF 8-FOLD ERI SYMMETRIES
# =============================================================================

print_section("PART 4: 8-Fold ERI Symmetry Verification")

print("""
The ERI (uv|ls) has 8-fold symmetry due to:
1. Real-valued basis functions: chi_u = chi_u*
2. Hermitian Coulomb operator: 1/r12 = (1/r12)*
3. Symmetric r12: |r1 - r2| = |r2 - r1|

The 8 symmetries are:
  (uv|ls) = (vu|ls)   swap u <-> v (bra exchange)
  (uv|ls) = (uv|sl)   swap l <-> s (ket exchange)
  (uv|ls) = (vu|sl)   combined
  (uv|ls) = (ls|uv)   bra <-> ket exchange (electron 1 <-> 2)
  (uv|ls) = (sl|uv)   combined
  (uv|ls) = (ls|vu)   combined
  (uv|ls) = (sl|vu)   combined
""")

# Test all 8 symmetries for every unique index combination
print_subsection("Exhaustive Symmetry Test")

n_tests = 0
n_passed = 0
max_error = 0.0

for u in range(N):
    for v in range(N):
        for l in range(N):
            for s in range(N):
                base = eri[u, v, l, s]

                # All 8 permutations
                perms = [
                    ("(uv|ls)", eri[u, v, l, s]),
                    ("(vu|ls)", eri[v, u, l, s]),  # swap u <-> v
                    ("(uv|sl)", eri[u, v, s, l]),  # swap l <-> s
                    ("(vu|sl)", eri[v, u, s, l]),  # swap both bra and ket
                    ("(ls|uv)", eri[l, s, u, v]),  # bra <-> ket
                    ("(sl|uv)", eri[s, l, u, v]),  # bra <-> ket + swap l <-> s
                    ("(ls|vu)", eri[l, s, v, u]),  # bra <-> ket + swap u <-> v
                    ("(sl|vu)", eri[s, l, v, u]),  # all swaps
                ]

                for name, val in perms:
                    n_tests += 1
                    error = abs(val - base)
                    max_error = max(max_error, error)
                    if error < 1e-14:
                        n_passed += 1

print(f"Total symmetry tests: {n_tests}")
print(f"Tests passed: {n_passed}")
print(f"Maximum error: {max_error:.2e}")

assert n_passed == n_tests, "Some ERI symmetries failed!"
print("\n[PASS] All 8-fold ERI symmetries verified!")

# Demonstrate with a specific example
print_subsection("Specific Example: (01|01)")
u, v, l, s = 0, 1, 0, 1
print(f"Testing index (u,v,l,s) = ({u},{v},{l},{s}):")
print(f"  (01|01) = {eri[0,1,0,1]:.10f}")
print(f"  (10|01) = {eri[1,0,0,1]:.10f}  (swap u<->v)")
print(f"  (01|10) = {eri[0,1,1,0]:.10f}  (swap l<->s)")
print(f"  (10|10) = {eri[1,0,1,0]:.10f}  (swap both)")
print(f"  (01|01) = {eri[0,1,0,1]:.10f}  (bra<->ket)")
print(f"  (10|01) = {eri[1,0,0,1]:.10f}  (combined)")
print(f"  (01|10) = {eri[0,1,1,0]:.10f}  (combined)")
print(f"  (10|10) = {eri[1,0,1,0]:.10f}  (all swaps)")


# =============================================================================
# PART 5: REFERENCE HF CALCULATION
# =============================================================================

print_section("PART 5: Reference HF Calculation")

# Run PySCF RHF for reference values
mf = scf.RHF(mol)
mf.verbose = 0  # Suppress output
E_pyscf = mf.kernel()

print(f"PySCF RHF converged: {mf.converged}")
print(f"Number of SCF iterations: {mf.cycles}")
print(f"Total energy (PySCF): {E_pyscf:.10f} Hartree")

# Extract converged quantities
C = mf.mo_coeff       # MO coefficients (AO x MO)
occ = mf.mo_occ       # Orbital occupations
eps = mf.mo_energy    # Orbital energies
P = mf.make_rdm1()    # Density matrix

print_subsection("Converged MO Information")
print(f"MO coefficient matrix shape: {C.shape}")
print(f"Orbital occupations: {occ}")
print(f"Orbital energies (Hartree): {eps}")

print_subsection("MO Coefficient Matrix C")
print("C transforms from AO to MO basis: |phi_p> = sum_u C_up |chi_u>")
print(f"\nC =\n{C}")

print_subsection("Density Matrix P")
print("For RHF: P_uv = 2 * sum_i C_ui * C_vi (factor of 2 for double occupation)")
print(f"\nP =\n{P}")


# =============================================================================
# PART 6: ELECTRON COUNT VERIFICATION
# =============================================================================

print_section("PART 6: Electron Count Verification via Tr[PS]")

print("""
The electron count formula Tr[PS] arises from:
  N_e = sum_uv P_uv S_uv = Tr[PS]

This works because:
  N_e = 2 * sum_i <phi_i|phi_i>_S         (sum over occupied MOs)
      = 2 * sum_i sum_uv C_ui C_vi S_uv   (expand in AO basis)
      = sum_uv P_uv S_uv                   (definition of P)
      = Tr[PS]
""")

# Compute electron count
n_elec_computed = np.trace(P @ S)
n_elec_expected = mol.nelectron

print(f"Computed electron count: Tr[P*S] = {n_elec_computed:.10f}")
print(f"Expected electron count: {n_elec_expected}")
print(f"Difference: {abs(n_elec_computed - n_elec_expected):.2e}")

assert np.isclose(n_elec_computed, n_elec_expected, atol=1e-10), \
    "Electron count mismatch!"
print("\n[PASS] Electron count verified!")


# =============================================================================
# PART 7: MO ORTHONORMALITY VERIFICATION
# =============================================================================

print_section("PART 7: MO Orthonormality Verification")

print("""
MOs should be orthonormal in the S-metric:
  <phi_p|phi_q>_S = sum_uv C_up S_uv C_vq = delta_pq

In matrix form: C^T * S * C = I
""")

# Compute C^T S C
orthonorm_check = C.T @ S @ C

print(f"C^T S C =\n{orthonorm_check}")
print(f"\nDeviation from identity: ||C^T S C - I|| = {np.linalg.norm(orthonorm_check - np.eye(N)):.2e}")

assert np.allclose(orthonorm_check, np.eye(N), atol=1e-10), \
    "MO orthonormality check failed!"
print("\n[PASS] MOs are orthonormal in S-metric!")


# =============================================================================
# PART 8: HF ENERGY RECONSTRUCTION FROM INTEGRALS
# =============================================================================

print_section("PART 8: HF Energy Reconstruction from Integrals")

print("""
The HF energy formula:
  E_HF = E_elec + E_nuc

where the electronic energy is:
  E_elec = (1/2) Tr[P(h + F)]
         = Tr[P*h] + (1/2) Tr[P*J] - (1/4) Tr[P*K]

and the Fock matrix is:
  F = h + J - (1/2)*K

The J and K matrices are built from ERIs:
  J_uv = sum_ls (uv|ls) P_ls    (Coulomb)
  K_uv = sum_ls (ul|vs) P_ls    (Exchange)
""")

print_subsection("Step 1: Build Coulomb Matrix J")
print("J_uv = sum_ls (uv|ls) * P_ls")
print("Einstein notation: J = einsum('uvls,ls->uv', eri, P)")

J = np.einsum('uvls,ls->uv', eri, P)
print(f"\nJ =\n{J}")

print_subsection("Step 2: Build Exchange Matrix K")
print("K_uv = sum_ls (ul|vs) * P_ls")
print("Einstein notation: K = einsum('ulvs,ls->uv', eri, P)")

K = np.einsum('ulvs,ls->uv', eri, P)
print(f"\nK =\n{K}")

print_subsection("Step 3: Build Fock Matrix")
print("F = h + J - (1/2)*K")

F = H_core + J - 0.5 * K
print(f"\nF =\n{F}")

# Compare to PySCF Fock matrix
F_pyscf = mf.get_fock()
print(f"\nPySCF Fock matrix:\n{F_pyscf}")
print(f"\n||F - F_pyscf|| = {np.linalg.norm(F - F_pyscf):.2e}")

assert np.allclose(F, F_pyscf, atol=1e-10), "Fock matrix mismatch!"
print("[PASS] Fock matrix matches PySCF!")

print_subsection("Step 4: Compute Electronic Energy")
print("E_elec = (1/2) * Tr[P * (h + F)]")

E_elec = 0.5 * np.einsum('uv,uv->', P, H_core + F)
print(f"\nE_elec = {E_elec:.10f} Hartree")

# Alternative computation to verify
E_1e = np.einsum('uv,uv->', P, H_core)  # One-electron energy
E_J = 0.5 * np.einsum('uv,uv->', P, J)  # Coulomb energy
E_K = -0.25 * np.einsum('uv,uv->', P, K)  # Exchange energy

print(f"\nEnergy breakdown:")
print(f"  One-electron energy Tr[P*h]:   {E_1e:.10f} Hartree")
print(f"  Coulomb energy (1/2)Tr[P*J]:   {E_J:.10f} Hartree")
print(f"  Exchange energy -(1/4)Tr[P*K]: {E_K:.10f} Hartree")
print(f"  Sum:                           {E_1e + E_J + E_K:.10f} Hartree")
print(f"  E_elec (from formula):         {E_elec:.10f} Hartree")

assert np.isclose(E_elec, E_1e + E_J + E_K, atol=1e-14), \
    "Energy breakdown inconsistency!"

print_subsection("Step 5: Nuclear Repulsion Energy")
print("E_nuc = sum_{A<B} Z_A * Z_B / R_AB")

E_nuc = mol.energy_nuc()
print(f"\nE_nuc = {E_nuc:.10f} Hartree")

# Manual calculation for verification
R_H2 = 0.74 / 0.529177  # Convert Angstrom to Bohr
E_nuc_manual = 1.0 * 1.0 / R_H2  # Z_H = 1 for both atoms
print(f"E_nuc (manual): {E_nuc_manual:.10f} Hartree")
print(f"H-H distance: {R_H2:.6f} Bohr = 0.74 Angstrom")

print_subsection("Step 6: Total Energy")
print("E_total = E_elec + E_nuc")

E_total = E_elec + E_nuc
print(f"\nE_total = {E_total:.10f} Hartree")


# =============================================================================
# PART 9: VALIDATION AGAINST PYSCF
# =============================================================================

print_section("PART 9: Final Validation")

print(f"Our implementation:  {E_total:.10f} Hartree")
print(f"PySCF reference:     {E_pyscf:.10f} Hartree")
print(f"Difference:          {abs(E_total - E_pyscf):.2e} Hartree")

assert np.isclose(E_total, E_pyscf, atol=1e-10), "Final energy mismatch!"
print("\n[PASS] Energy validated against PySCF!")

# Get PySCF's energy components for comparison
E_elec_pyscf = mf.energy_elec()[0]
print(f"\nComponent comparison:")
print(f"  E_elec (ours):  {E_elec:.10f} Hartree")
print(f"  E_elec (PySCF): {E_elec_pyscf:.10f} Hartree")
print(f"  E_nuc:          {E_nuc:.10f} Hartree")


# =============================================================================
# PART 10: SUMMARY AND KEY TAKEAWAYS
# =============================================================================

print_section("SUMMARY: Key Takeaways")

print("""
1. INTEGRAL EXTRACTION
   - S, T, V matrices via mol.intor("int1e_*")
   - ERIs via mol.intor("int2e", aosym="s1")
   - All one-electron matrices are symmetric

2. ERI SYMMETRIES (8-fold)
   - Swap indices within bra: (uv|ls) = (vu|ls)
   - Swap indices within ket: (uv|ls) = (uv|sl)
   - Exchange bra and ket: (uv|ls) = (ls|uv)
   - All combinations of above
   - Origin: real basis + Hermitian operator + symmetric r12

3. MEMORY SCALING
   - Full ERI tensor: O(N^4) storage
   - For N = 2 (H2/STO-3G): 16 elements = 128 bytes
   - For N = 100: 10^8 elements = 800 MB
   - For N = 1000: 10^12 elements = 8 TB (!)
   - This motivates symmetry packing, direct SCF, density fitting

4. SANITY CHECKS
   - Electron count: Tr[PS] = N_e
   - MO orthonormality: C^T S C = I
   - Matrix symmetries: S = S^T, T = T^T, V = V^T

5. HF ENERGY FORMULA
   - J_uv = sum_ls (uv|ls) P_ls  [Coulomb]
   - K_uv = sum_ls (ul|vs) P_ls  [Exchange]
   - F = h + J - (1/2)K          [Fock matrix]
   - E_elec = (1/2) Tr[P(h + F)] [Electronic energy]
   - E_total = E_elec + E_nuc    [Total energy]

6. CENTRAL THEME
   Hartree-Fock = Computing integrals + Linear algebra!

   Once we have S, T, V, and ERIs, the rest is just matrix operations.
   This is why efficient integral evaluation (Rys quadrature) is so important.
""")

print("=" * 70)
print("  Lab 1A Complete - All validations passed!")
print("=" * 70)
