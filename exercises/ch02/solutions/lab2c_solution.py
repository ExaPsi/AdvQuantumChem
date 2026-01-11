#!/usr/bin/env python3
"""
Lab 2C Solution: Solve FC = SCe via Orthogonalization

This script demonstrates solving the generalized eigenvalue problem
FC = SCe (Roothaan-Hall equations) by transforming to an orthonormal basis.

The key insight: in an orthonormal basis, S becomes I, and the generalized
eigenproblem becomes a standard one that NumPy can solve directly.

Workflow:
    1. Build orthogonalizer X with X^T S X = I
    2. Transform: F' = X^T F X
    3. Solve standard eigenproblem: F' C' = C' e
    4. Back-transform: C = X C'
    5. Verify: C^T S C = I

Test molecule: H2O (water)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 2: Gaussian Basis Sets and Orthonormalization
"""

import numpy as np
from pyscf import gto, scf

# =============================================================================
# Section 1: Orthogonalization Utilities (from Lab 2B)
# =============================================================================

def canonical_orthogonalizer(S: np.ndarray,
                              threshold: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build canonical orthogonalizer X = U s^(-1/2) with eigenvalue thresholding.

    Args:
        S: Overlap matrix (N x N)
        threshold: Eigenvalues below this are dropped

    Returns:
        X: Orthogonalizer (N x M)
        eigenvalues_kept: Kept eigenvalues of S
        n_dropped: Number of dropped eigenvalues
    """
    eigenvalues, U = np.linalg.eigh(S)
    keep_mask = eigenvalues > threshold
    n_dropped = np.sum(~keep_mask)

    eigenvalues_kept = eigenvalues[keep_mask]
    U_kept = U[:, keep_mask]
    X = U_kept @ np.diag(eigenvalues_kept ** -0.5)

    return X, eigenvalues_kept, n_dropped


def symmetric_orthogonalizer(S: np.ndarray,
                              threshold: float = 1e-10) -> tuple[np.ndarray, int]:
    """
    Build symmetric (Lowdin) orthogonalizer X = S^(-1/2).

    Args:
        S: Overlap matrix (N x N)
        threshold: Eigenvalue cutoff

    Returns:
        X_sym: Symmetric orthogonalizer
        n_dropped: Number of near-zero eigenvalues
    """
    eigenvalues, U = np.linalg.eigh(S)
    keep_mask = eigenvalues > threshold
    n_dropped = np.sum(~keep_mask)

    if n_dropped > 0:
        eigenvalues_kept = eigenvalues[keep_mask]
        U_kept = U[:, keep_mask]
        X_sym = U_kept @ np.diag(eigenvalues_kept ** -0.5) @ U_kept.T
    else:
        X_sym = U @ np.diag(eigenvalues ** -0.5) @ U.T

    return X_sym, n_dropped


# =============================================================================
# Section 2: Generalized Eigenvalue Problem Solver
# =============================================================================

def solve_generalized_eigenproblem(F: np.ndarray, S: np.ndarray,
                                    threshold: float = 1e-10,
                                    method: str = "canonical") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve FC = SCe via orthogonalization.

    Algorithm 2.3 from Chapter 2:
        1. Build X from S (with thresholding)
        2. Form F' = X^T F X
        3. Diagonalize F' C' = C' e
        4. Back-transform C = X C'

    Args:
        F: Fock-like matrix (N x N), symmetric
        S: Overlap matrix (N x N), symmetric positive definite
        threshold: Eigenvalue cutoff for S
        method: "canonical" or "symmetric"

    Returns:
        eigenvalues: Orbital energies (M,)
        C: MO coefficients in AO basis (N x M)
        X: The orthogonalizer used
    """
    N = S.shape[0]

    # Step 1: Build orthogonalizer X with X^T S X = I
    if method == "canonical":
        X, _, n_dropped = canonical_orthogonalizer(S, threshold)
    elif method == "symmetric":
        X, n_dropped = symmetric_orthogonalizer(S, threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    M = X.shape[1]  # Dimension after thresholding

    if n_dropped > 0:
        print(f"  [Orthogonalizer: {method}, dropped {n_dropped} near-dependent functions]")
        print(f"  [Working in {M}-dimensional subspace (original: {N})]")

    # Step 2: Transform F to orthonormal basis
    # F' = X^T F X (M x M matrix)
    F_prime = X.T @ F @ X

    # Step 3: Solve standard eigenproblem F' C' = C' e
    # eigh returns eigenvalues in ascending order
    eigenvalues, C_prime = np.linalg.eigh(F_prime)

    # Step 4: Back-transform to AO basis
    # C = X C' (N x M matrix)
    C = X @ C_prime

    return eigenvalues, C, X


def verify_eigenproblem_solution(F: np.ndarray, S: np.ndarray,
                                  C: np.ndarray, eigenvalues: np.ndarray) -> dict:
    """
    Verify that (C, e) solves FC = SCe.

    Checks:
    1. C^T S C = I (orthonormality)
    2. FC - SC diag(e) = 0 (eigenproblem residual)
    """
    M = C.shape[1]

    # Check 1: Orthonormality
    CtSC = C.T @ S @ C
    ortho_error = np.linalg.norm(CtSC - np.eye(M))

    # Check 2: Residual ||FC - SC diag(e)||
    # Note: FC should equal S C diag(e)
    FC = F @ C
    SCe = S @ C @ np.diag(eigenvalues)
    residual_error = np.linalg.norm(FC - SCe)

    return {
        "ortho_error": ortho_error,
        "residual_error": residual_error,
        "passed": ortho_error < 1e-10 and residual_error < 1e-10
    }


# =============================================================================
# Section 3: Comparison with PySCF
# =============================================================================

def compare_with_pyscf(mol: gto.Mole, C_our: np.ndarray,
                        eps_our: np.ndarray) -> dict:
    """
    Compare our solution with PySCF's HF calculation.

    Note: We solve h C = S C e (core Hamiltonian), which is NOT the same as
    converged HF (which uses F = h + J - K/2). However, we can compare:
    - Eigenvalues should match if we use the same input matrix
    - Orthonormality properties
    """
    # Run PySCF RHF
    mf = scf.RHF(mol)
    mf.kernel()

    # Get PySCF's MO coefficients and energies
    C_pyscf = mf.mo_coeff
    eps_pyscf = mf.mo_energy

    # Also get the orthogonalizer PySCF used
    # PySCF stores it in mf._orth_ao (may need to access differently)
    S = mol.intor("int1e_ovlp")

    return {
        "eps_pyscf": eps_pyscf,
        "C_pyscf": C_pyscf,
        "S": S
    }


# =============================================================================
# Section 4: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 2C demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 2C: Solve FC = SCe via Orthogonalization" + " " * 25 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Water molecule geometry
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    # ==========================================================================
    # Part 1: Solve Core Hamiltonian Eigenproblem
    # ==========================================================================

    print("=" * 75)
    print("Part 1: Solve Core Hamiltonian Eigenproblem h C = S C e")
    print("=" * 75)

    mol = gto.M(
        atom=h2o_geometry,
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )

    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V  # Core Hamiltonian

    nao = mol.nao_nr()
    n_elec = mol.nelectron
    n_occ = n_elec // 2  # Closed-shell, doubly occupied

    print(f"\nMolecule: H2O")
    print(f"Basis: cc-pVDZ")
    print(f"Number of AOs: {nao}")
    print(f"Number of electrons: {n_elec}")
    print(f"Number of occupied orbitals (RHF): {n_occ}")

    # Solve using our implementation
    print("\n--- Solving h C = S C e using canonical orthogonalization ---")
    eps_can, C_can, X_can = solve_generalized_eigenproblem(h, S, method="canonical")

    print(f"\nOrbital energies (Hartree):")
    print(f"  Lowest 5: {eps_can[:5]}")
    print(f"  Highest 3: {eps_can[-3:]}")

    # Verify solution
    result_can = verify_eigenproblem_solution(h, S, C_can, eps_can)
    print(f"\nVerification (canonical):")
    print(f"  ||C^T S C - I|| = {result_can['ortho_error']:.2e}")
    print(f"  ||hC - SCe||    = {result_can['residual_error']:.2e}")
    print(f"  Status: {'PASS' if result_can['passed'] else 'FAIL'}")

    # ==========================================================================
    # Part 2: Compare Canonical vs Symmetric Orthogonalization
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 2: Compare Canonical vs Symmetric Orthogonalization")
    print("=" * 75)

    eps_sym, C_sym, X_sym = solve_generalized_eigenproblem(h, S, method="symmetric")

    print("\nEigenvalue comparison:")
    print(f"  ||eps_can - eps_sym|| = {np.linalg.norm(eps_can - eps_sym):.2e}")
    print("  (Should be ~0: same eigenvalues, different eigenvectors)")

    # Compare eigenvectors
    # Eigenvectors can differ by sign or unitary transformation for degenerate eigenvalues
    # For non-degenerate case, |C_can| ~ |C_sym| up to signs
    print(f"\nMO coefficient comparison (first 3 MOs):")
    for i in range(3):
        # Check if columns match up to sign
        diff1 = np.linalg.norm(C_can[:, i] - C_sym[:, i])
        diff2 = np.linalg.norm(C_can[:, i] + C_sym[:, i])
        min_diff = min(diff1, diff2)
        print(f"  MO {i+1}: min(||C_can - C_sym||, ||C_can + C_sym||) = {min_diff:.2e}")

    result_sym = verify_eigenproblem_solution(h, S, C_sym, eps_sym)
    print(f"\nVerification (symmetric):")
    print(f"  ||C^T S C - I|| = {result_sym['ortho_error']:.2e}")
    print(f"  ||hC - SCe||    = {result_sym['residual_error']:.2e}")

    # ==========================================================================
    # Part 3: Compare with PySCF HF MOs
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 3: Compare with PySCF Hartree-Fock")
    print("=" * 75)

    print("\nNote: We solved h C = S C e (core Hamiltonian eigenproblem).")
    print("PySCF solves F C = S C e where F = h + J - K/2 (self-consistent).")
    print("Eigenvalues will differ, but the ALGORITHM is the same.")

    # Run PySCF HF
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_hf = mf.kernel()

    eps_pyscf = mf.mo_energy
    C_pyscf = mf.mo_coeff

    print(f"\nPySCF HF converged: E = {E_hf:.10f} Hartree")

    # Verify PySCF MOs satisfy orthonormality
    CtSC_pyscf = C_pyscf.T @ S @ C_pyscf
    ortho_error_pyscf = np.linalg.norm(CtSC_pyscf - np.eye(nao))
    print(f"\nPySCF MO orthonormality:")
    print(f"  ||C^T S C - I|| = {ortho_error_pyscf:.2e}")

    # Now solve the CONVERGED Fock matrix eigenproblem
    print("\n--- Solving converged Fock eigenproblem F C = S C e ---")

    # Get the converged Fock matrix from PySCF
    F_converged = mf.get_fock()

    # Solve it with our method
    eps_fock, C_fock, _ = solve_generalized_eigenproblem(F_converged, S, method="canonical")

    # Compare eigenvalues
    print(f"\nOrbital energy comparison (Converged Fock matrix):")
    print(f"  ||eps_our - eps_pyscf|| = {np.linalg.norm(eps_fock - eps_pyscf):.2e}")

    # Should match to high precision
    print(f"\n  Our eps (first 5):   {eps_fock[:5]}")
    print(f"  PySCF eps (first 5): {eps_pyscf[:5]}")

    # Compare MO coefficients
    print(f"\nMO coefficient comparison (converged Fock):")
    for i in range(min(5, nao)):
        diff1 = np.linalg.norm(C_fock[:, i] - C_pyscf[:, i])
        diff2 = np.linalg.norm(C_fock[:, i] + C_pyscf[:, i])
        min_diff = min(diff1, diff2)
        print(f"  MO {i+1}: {min_diff:.2e}", end="")
        if min_diff < 1e-8:
            print(" [MATCH]")
        else:
            print(" [differs in phase/order]")

    # Verify our solution
    result_fock = verify_eigenproblem_solution(F_converged, S, C_fock, eps_fock)
    print(f"\nVerification (our Fock solution):")
    print(f"  ||C^T S C - I|| = {result_fock['ortho_error']:.2e}")
    print(f"  ||FC - SCe||    = {result_fock['residual_error']:.2e}")

    assert result_fock["passed"], "Fock eigenproblem solution failed!"
    print("\n[OK] Successfully reproduced PySCF orbital energies!")

    # ==========================================================================
    # Part 4: Ill-Conditioned Basis (aug-cc-pVDZ)
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 4: Ill-Conditioned Basis with Eigenvalue Thresholding")
    print("=" * 75)

    mol_aug = gto.M(
        atom=h2o_geometry,
        basis="aug-cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )

    S_aug = mol_aug.intor("int1e_ovlp")
    T_aug = mol_aug.intor("int1e_kin")
    V_aug = mol_aug.intor("int1e_nuc")
    h_aug = T_aug + V_aug

    nao_aug = mol_aug.nao_nr()
    eigs_S = np.linalg.eigvalsh(S_aug)

    print(f"\nBasis: aug-cc-pVDZ")
    print(f"Number of AOs: {nao_aug}")
    print(f"Smallest S eigenvalue: {eigs_S.min():.6e}")
    print(f"Condition number: {eigs_S.max()/eigs_S.min():.3e}")

    # Solve with different thresholds
    print("\n--- Effect of threshold on solution dimension ---")
    print(f"\n{'Threshold':>12} {'Dim':>5} {'||C^T S C - I||':>18} {'Status':>8}")
    print("-" * 50)

    for thresh in [1e-14, 1e-10, 1e-8, 1e-6]:
        eps_t, C_t, X_t = solve_generalized_eigenproblem(h_aug, S_aug,
                                                          threshold=thresh,
                                                          method="canonical")
        result_t = verify_eigenproblem_solution(h_aug, S_aug, C_t, eps_t)
        dim = C_t.shape[1]
        status = "PASS" if result_t["passed"] else "FAIL"
        print(f"{thresh:>12.0e} {dim:>5} {result_t['ortho_error']:>18.2e} {status:>8}")

    # ==========================================================================
    # Part 5: Complete Example - From Integrals to MO Energies
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 5: Complete Workflow Summary")
    print("=" * 75)

    workflow = """
ALGORITHM 2.3: Solving FC = SCe via Orthogonalization
------------------------------------------------------

INPUTS:
  - Symmetric F (Fock-like matrix, N x N)
  - Symmetric positive-definite S (overlap, N x N)
  - Threshold tau for near-linear dependence

STEPS:
  1. Eigendecompose S: S = U s U^T

  2. Build orthogonalizer with thresholding:
     - Keep eigenpairs with s_i > tau
     - X = U_kept @ diag(s_kept^(-1/2))   [canonical]
     - or X = U_kept @ diag(s_kept^(-1/2)) @ U_kept^T   [symmetric]

  3. Transform to orthonormal basis:
     F' = X^T F X   (M x M where M <= N)

  4. Solve standard eigenproblem:
     F' C' = C' e   (use np.linalg.eigh)

  5. Back-transform:
     C = X C'   (N x M)

OUTPUTS:
  - Eigenvalues e (M,)
  - Eigenvectors C (N x M) satisfying:
    * C^T S C = I  (orthonormality in S-metric)
    * F C = S C diag(e)  (generalized eigenproblem)

WHY THIS WORKS:
  X^T S X = I transforms the S-metric to Euclidean
  The transformed problem F' C' = C' e is standard
  Back-transformation C = X C' restores AO representation
  Orthonormality is preserved: (XC')^T S (XC') = C'^T X^T S X C' = C'^T C' = I
"""
    print(workflow)

    # ==========================================================================
    # Part 6: What You Should Observe
    # ==========================================================================

    print("=" * 75)
    print("What You Should Observe")
    print("=" * 75)

    observations = """
1. EIGENVALUE CONSISTENCY:
   - Canonical and symmetric methods give IDENTICAL eigenvalues
   - Eigenvectors may differ by phase (sign) or rotation (for degenerate e)
   - Our solution matches PySCF's when using the same input F

2. ORTHONORMALITY:
   - C^T S C = I should hold to ~10^(-14) for well-conditioned S
   - For ill-conditioned S with thresholding:
     * C is N x M (reduced dimension)
     * Still satisfies C^T S C = I_M

3. RESIDUAL:
   - ||FC - SC diag(e)|| should be ~10^(-14)
   - This verifies we actually solved the eigenproblem

4. THRESHOLDING EFFECT:
   - tau = 1e-14: Keep all, may have numerical issues
   - tau = 1e-10 to 1e-8: Safe default for most cases
   - tau = 1e-6: Too aggressive, may lose important functions

5. PRACTICAL IMPLICATIONS:
   - This is exactly what happens inside mf.kernel()
   - SCF iteration repeats this solve with updated F(P)
   - Robust orthogonalization is essential for stable SCF

6. PHYSICAL MEANING:
   - Eigenvalues e are orbital energies (in Hartree)
   - Columns of C are MO coefficients: phi_p = sum_mu C_mu,p chi_mu
   - Occupied MOs (lowest n_occ) determine the density matrix
"""
    print(observations)

    print()
    print("=" * 75)
    print("Lab 2C Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
