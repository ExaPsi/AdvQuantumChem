#!/usr/bin/env python3
"""
Numerical Verification: Trace Formulas in Quantum Chemistry

This script demonstrates key trace formulas:
  1. N_e = Tr[PS]               (electron count)
  2. E_elec = (1/2)Tr[P(h+F)]   (electronic energy)
  3. E_elec = Tr[Ph] + (1/2)Tr[PG]  (alternative form, G = J - 1/2 K)
  4. Tr[ABC] = Tr[BCA] = Tr[CAB]    (cyclic property)
  5. mu = Tr[P*r]               (dipole moment)

Reference: Proof document in src/proofs/trace-importance.tex
"""
import numpy as np
from pyscf import gto, scf

# =============================================================================
# Step 1: Build molecule and run RHF
# =============================================================================

# Water molecule with STO-3G minimal basis (same as mo_orthonormality_verify.py)
mol = gto.M(
    atom="""
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
    """,
    basis="sto-3g",
    unit="Angstrom",
    verbose=0,
)

print("=" * 65)
print("Numerical Verification: Trace Formulas in Quantum Chemistry")
print("=" * 65)
print(f"\nMolecule: H2O")
print(f"Basis:    STO-3G")
print(f"Number of AOs:  {mol.nao_nr()}")
print(f"Number of electrons: {mol.nelectron}")

# Run RHF calculation
mf = scf.RHF(mol)
E_hf = mf.kernel()
print(f"\nRHF Total Energy: {E_hf:.10f} Hartree")

# =============================================================================
# Step 2: Extract all matrices
# =============================================================================

# One-electron integrals
S = mol.intor("int1e_ovlp")  # Overlap matrix
T = mol.intor("int1e_kin")   # Kinetic energy
V = mol.intor("int1e_nuc")   # Nuclear attraction
h = T + V                     # Core Hamiltonian

# Two-electron integrals (full tensor, chemist's notation)
eri = mol.intor("int2e", aosym="s1")  # (mu nu | lambda sigma)

# MO coefficients and density matrix
C = mf.mo_coeff
n_occ = mol.nelectron // 2
C_occ = C[:, :n_occ]
P = 2.0 * C_occ @ C_occ.T  # RHF density matrix

# Build J and K matrices from ERIs
# J_uv = sum_{ls} (uv|ls) P_ls
J = np.einsum('ijkl,kl->ij', eri, P)
# K_uv = sum_{ls} (ul|vs) P_ls
K = np.einsum('ikjl,kl->ij', eri, P)

# G matrix (electron-electron interaction part of Fock matrix)
# G = J - (1/2)K for RHF
G = J - 0.5 * K

# Fock matrix: F = h + G
F = h + G

# Nuclear repulsion energy
E_nuc = mol.energy_nuc()

print(f"\n--- Matrix Dimensions ---")
print(f"Overlap S:       {S.shape}")
print(f"Core Hamiltonian h: {h.shape}")
print(f"Density P:       {P.shape}")
print(f"Fock F:          {F.shape}")
print(f"ERI tensor:      {eri.shape}")

# =============================================================================
# Test 1: Verify N_e = Tr[PS]
# =============================================================================

print("\n" + "=" * 65)
print("Test 1: Electron Count N_e = Tr[PS]")
print("=" * 65)

N_e_expected = mol.nelectron
N_e_trace = np.trace(P @ S)

print(f"\nN_e (expected from molecule):  {N_e_expected}")
print(f"N_e = Tr[PS]:                  {N_e_trace:.10f}")
print(f"Difference:                    {abs(N_e_trace - N_e_expected):.2e}")

assert np.isclose(N_e_trace, N_e_expected, atol=1e-10), "Electron count mismatch!"
print("\nVERIFIED: N_e = Tr[PS]")

# =============================================================================
# Test 2: Verify E_elec = (1/2)Tr[P(h+F)]
# =============================================================================

print("\n" + "=" * 65)
print("Test 2: Electronic Energy E_elec = (1/2)Tr[P(h+F)]")
print("=" * 65)

# PySCF reference electronic energy
E_elec_pyscf = mf.energy_elec()[0]

# Our trace formula
E_elec_trace = 0.5 * np.trace(P @ (h + F))

# Total energy check
E_total_check = E_elec_trace + E_nuc

print(f"\nE_elec (PySCF):                {E_elec_pyscf:.10f} Hartree")
print(f"E_elec = (1/2)Tr[P(h+F)]:      {E_elec_trace:.10f} Hartree")
print(f"Difference:                    {abs(E_elec_trace - E_elec_pyscf):.2e} Hartree")

print(f"\nE_nuc:                         {E_nuc:.10f} Hartree")
print(f"E_total = E_elec + E_nuc:      {E_total_check:.10f} Hartree")
print(f"E_total (PySCF):               {E_hf:.10f} Hartree")

assert np.isclose(E_elec_trace, E_elec_pyscf, atol=1e-8), "E_elec mismatch!"
print("\nVERIFIED: E_elec = (1/2)Tr[P(h+F)]")

# =============================================================================
# Test 3: Verify E_elec = Tr[Ph] + (1/2)Tr[PG]
# =============================================================================

print("\n" + "=" * 65)
print("Test 3: Alternative Form E_elec = Tr[Ph] + (1/2)Tr[PG]")
print("=" * 65)

# One-electron energy
E_1e = np.trace(P @ h)

# Two-electron energy (G = J - 0.5K)
E_2e = 0.5 * np.trace(P @ G)

# Total electronic energy from this decomposition
E_elec_alt = E_1e + E_2e

print(f"\nE_1e = Tr[Ph]:                 {E_1e:.10f} Hartree")
print(f"E_2e = (1/2)Tr[PG]:            {E_2e:.10f} Hartree")
print(f"E_elec = E_1e + E_2e:          {E_elec_alt:.10f} Hartree")
print(f"E_elec (PySCF):                {E_elec_pyscf:.10f} Hartree")
print(f"Difference:                    {abs(E_elec_alt - E_elec_pyscf):.2e} Hartree")

# Also verify the two formulas are equivalent
print(f"\n(1/2)Tr[P(h+F)] vs Tr[Ph] + (1/2)Tr[PG]:")
print(f"  Formula 1: {E_elec_trace:.10f}")
print(f"  Formula 2: {E_elec_alt:.10f}")
print(f"  Difference: {abs(E_elec_trace - E_elec_alt):.2e}")

assert np.isclose(E_elec_alt, E_elec_pyscf, atol=1e-8), "Alternative E_elec mismatch!"
print("\nVERIFIED: E_elec = Tr[Ph] + (1/2)Tr[PG]")

# =============================================================================
# Test 4: Verify cyclic property Tr[ABC] = Tr[BCA] = Tr[CAB]
# =============================================================================

print("\n" + "=" * 65)
print("Test 4: Cyclic Property Tr[ABC] = Tr[BCA] = Tr[CAB]")
print("=" * 65)

# Use actual matrices from the calculation
A, B, C_mat = P, S, F  # Choose three real matrices

tr_ABC = np.trace(A @ B @ C_mat)
tr_BCA = np.trace(B @ C_mat @ A)
tr_CAB = np.trace(C_mat @ A @ B)

print(f"\nUsing A=P, B=S, C=F:")
print(f"  Tr[ABC] = Tr[PSF] = {tr_ABC:.10f}")
print(f"  Tr[BCA] = Tr[SFP] = {tr_BCA:.10f}")
print(f"  Tr[CAB] = Tr[FPS] = {tr_CAB:.10f}")
print(f"\n  |Tr[ABC] - Tr[BCA]| = {abs(tr_ABC - tr_BCA):.2e}")
print(f"  |Tr[ABC] - Tr[CAB]| = {abs(tr_ABC - tr_CAB):.2e}")

# Also test with random matrices
np.random.seed(42)
n = 5
A_rand = np.random.randn(n, n)
B_rand = np.random.randn(n, n)
C_rand = np.random.randn(n, n)

tr_ABC_rand = np.trace(A_rand @ B_rand @ C_rand)
tr_BCA_rand = np.trace(B_rand @ C_rand @ A_rand)
tr_CAB_rand = np.trace(C_rand @ A_rand @ B_rand)

print(f"\nUsing random 5x5 matrices:")
print(f"  Tr[ABC] = {tr_ABC_rand:.10f}")
print(f"  Tr[BCA] = {tr_BCA_rand:.10f}")
print(f"  Tr[CAB] = {tr_CAB_rand:.10f}")
print(f"\n  |Tr[ABC] - Tr[BCA]| = {abs(tr_ABC_rand - tr_BCA_rand):.2e}")
print(f"  |Tr[ABC] - Tr[CAB]| = {abs(tr_ABC_rand - tr_CAB_rand):.2e}")

assert np.isclose(tr_ABC, tr_BCA, atol=1e-10), "Cyclic property failed (real)"
assert np.isclose(tr_ABC, tr_CAB, atol=1e-10), "Cyclic property failed (real)"
assert np.isclose(tr_ABC_rand, tr_BCA_rand, atol=1e-10), "Cyclic property failed (random)"
print("\nVERIFIED: Tr[ABC] = Tr[BCA] = Tr[CAB]")

# =============================================================================
# Test 5: Dipole Moment mu = Tr[P*r]
# =============================================================================

print("\n" + "=" * 65)
print("Test 5: Dipole Moment mu = Tr[P*r]")
print("=" * 65)

# Set origin for dipole integrals (center of mass)
mol.set_common_origin([0, 0, 0])

# Dipole integrals: <mu|r_x|nu>, <mu|r_y|nu>, <mu|r_z|nu>
# Shape: (3, nao, nao)
r_ints = mol.intor("int1e_r")

# Electronic contribution to dipole: mu_elec = -Tr[P*r] (negative for electrons)
mu_elec = -np.array([np.trace(P @ r_ints[i]) for i in range(3)])

# Nuclear contribution to dipole
charges = mol.atom_charges()
coords = mol.atom_coords()  # Already in Bohr
mu_nuc = np.einsum('i,ix->x', charges, coords)

# Total dipole moment
mu_total = mu_elec + mu_nuc

# PySCF reference
from pyscf import dft
mu_pyscf = mf.dip_moment(unit='AU', verbose=0)

print(f"\nDipole integrals shape: {r_ints.shape}")
print(f"\nElectronic dipole (a.u.):      [{mu_elec[0]:10.6f}, {mu_elec[1]:10.6f}, {mu_elec[2]:10.6f}]")
print(f"Nuclear dipole (a.u.):         [{mu_nuc[0]:10.6f}, {mu_nuc[1]:10.6f}, {mu_nuc[2]:10.6f}]")
print(f"\nTotal dipole (our calc, a.u.): [{mu_total[0]:10.6f}, {mu_total[1]:10.6f}, {mu_total[2]:10.6f}]")
print(f"Total dipole (PySCF, a.u.):    [{mu_pyscf[0]:10.6f}, {mu_pyscf[1]:10.6f}, {mu_pyscf[2]:10.6f}]")

mu_diff = np.linalg.norm(mu_total - mu_pyscf)
print(f"\n||mu_ours - mu_pyscf|| = {mu_diff:.2e} a.u.")

# Convert to Debye for physical interpretation
au_to_debye = 2.541746  # 1 a.u. = 2.541746 Debye
mu_debye = np.linalg.norm(mu_total) * au_to_debye
print(f"\nDipole magnitude: {np.linalg.norm(mu_total):.6f} a.u. = {mu_debye:.4f} Debye")

assert np.allclose(mu_total, mu_pyscf, atol=1e-6), "Dipole moment mismatch!"
print("\nVERIFIED: mu = Tr[P*r] (with nuclear contribution)")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 65)
print("Summary")
print("=" * 65)
print(f"""
  Test 1: N_e = Tr[PS]
          {N_e_trace:.6f} = {N_e_expected} electrons

  Test 2: E_elec = (1/2)Tr[P(h+F)]
          {E_elec_trace:.10f} Hartree (diff: {abs(E_elec_trace - E_elec_pyscf):.2e})

  Test 3: E_elec = Tr[Ph] + (1/2)Tr[PG]
          {E_elec_alt:.10f} Hartree (diff: {abs(E_elec_alt - E_elec_pyscf):.2e})

  Test 4: Tr[ABC] = Tr[BCA] = Tr[CAB]
          Verified with both real and random matrices

  Test 5: mu = Tr[P*r]
          Dipole = {mu_debye:.4f} Debye (diff: {mu_diff:.2e} a.u.)

All tests PASSED - Trace formulas verified!
""")
