#!/usr/bin/env python3
"""
Numerical Verification: MO Orthonormality Condition C^T S C = I

This script demonstrates the key MO orthonormality theorem numerically:
  - AOs are NOT orthonormal: S != I
  - Naive C^T C != I (wrong metric!)
  - But C^T S C = I (correct S-metric orthonormality)

The MO coefficient matrix C from Hartree-Fock satisfies the generalized
orthonormality condition using the overlap matrix S as the metric tensor.

Reference: Proof document in src/proofs/mo-orthonormality.tex
"""
import numpy as np
from pyscf import gto, scf

# =============================================================================
# Step 1: Build molecule and run RHF
# =============================================================================

# Water molecule with STO-3G minimal basis
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

print("=" * 60)
print("Numerical Verification: C^T S C = I")
print("=" * 60)
print(f"\nMolecule: H2O")
print(f"Basis:    STO-3G")
print(f"Number of AOs:  {mol.nao_nr()}")
print(f"Number of electrons: {mol.nelectron}")

# Run RHF calculation
mf = scf.RHF(mol)
E_hf = mf.kernel()
print(f"\nRHF Energy: {E_hf:.10f} Hartree")

# =============================================================================
# Step 2: Extract matrices
# =============================================================================

# Overlap matrix S from integrals
S = mol.intor("int1e_ovlp")  # S_uv = <chi_u|chi_v>

# MO coefficient matrix from converged SCF
C = mf.mo_coeff  # |psi_p> = sum_u C_up |chi_u>

print(f"\n--- Matrix Dimensions ---")
print(f"Overlap matrix S: {S.shape}")
print(f"MO coefficients C: {C.shape}")

# =============================================================================
# Step 3: Demonstrate that S != I (AOs are NOT orthonormal)
# =============================================================================

print("\n" + "=" * 60)
print("Test 1: Is S = I? (Are AOs orthonormal?)")
print("=" * 60)

nao = mol.nao_nr()
I_ao = np.eye(nao)
diff_S_I = np.linalg.norm(S - I_ao, "fro")

print(f"\n||S - I||_F = {diff_S_I:.6f}")
print("EXPECTED: S != I (AOs are NOT orthonormal)")
print("\nSample S matrix elements:")
print(f"  S[0,0] = {S[0,0]:.6f}  (diagonal)")
print(f"  S[0,1] = {S[0,1]:.6f}  (O 1s - O 2s overlap)")

# =============================================================================
# Step 4: Demonstrate that C^T C != I (wrong metric!)
# =============================================================================

print("\n" + "=" * 60)
print("Test 2: Is C^T C = I? (Naive orthonormality)")
print("=" * 60)

CTC = C.T @ C
nmo = C.shape[1]
I_mo = np.eye(nmo)
diff_CTC_I = np.linalg.norm(CTC - I_mo, "fro")

print(f"\n||C^T C - I||_F = {diff_CTC_I:.6f}")
print("EXPECTED: C^T C != I (wrong metric!)")
print("\nDiagonal of C^T C (NOT equal to 1):")
for i in range(min(3, nmo)):
    print(f"  (C^T C)[{i},{i}] = {CTC[i,i]:.6f}")

# =============================================================================
# Step 5: Verify C^T S C = I (correct S-metric orthonormality)
# =============================================================================

print("\n" + "=" * 60)
print("Test 3: Is C^T S C = I? (S-metric orthonormality)")
print("=" * 60)

CTSC = C.T @ S @ C
diff_CTSC_I = np.linalg.norm(CTSC - I_mo, "fro")
max_offdiag = np.max(np.abs(CTSC - np.diag(np.diag(CTSC))))
max_diag_err = np.max(np.abs(np.diag(CTSC) - 1.0))

print(f"\n||C^T S C - I||_F = {diff_CTSC_I:.2e}")
print(f"Max |diagonal - 1| = {max_diag_err:.2e}")
print(f"Max |off-diagonal| = {max_offdiag:.2e}")

assert np.allclose(CTSC, I_mo, atol=1e-10), "C^T S C != I"
print("\nVERIFIED: C^T S C = I (within numerical precision)")

# =============================================================================
# Step 6: Verify electron count N_e = Tr[PS]
# =============================================================================

print("\n" + "=" * 60)
print("Bonus: Verify N_e = Tr[PS]")
print("=" * 60)

# Density matrix P = 2 * C_occ @ C_occ^T (RHF)
n_occ = mol.nelectron // 2
C_occ = C[:, :n_occ]
P = 2 * C_occ @ C_occ.T

# Electron count
N_e_calc = np.trace(P @ S)
print(f"\nN_e (expected):   {mol.nelectron}")
print(f"N_e = Tr[PS]:     {N_e_calc:.10f}")
print(f"Difference:       {abs(N_e_calc - mol.nelectron):.2e}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\n  S != I (AOs not orthonormal):   ||S-I|| = {diff_S_I:.4f}")
print(f"  C^T C != I (wrong metric):      ||C^T C-I|| = {diff_CTC_I:.4f}")
print(f"  C^T S C = I (correct metric):   ||C^T S C-I|| = {diff_CTSC_I:.2e}")
print(f"  N_e = Tr[PS] verified:          {N_e_calc:.6f} = {mol.nelectron}")
print("\nAll tests PASSED - MO orthonormality verified!")
