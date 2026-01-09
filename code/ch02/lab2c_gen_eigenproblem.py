#!/usr/bin/env python3
"""
Lab 2C: Solve FC = SCε via Orthogonalization

This script accompanies Chapter 2 of the Advanced Quantum Chemistry lecture notes.
It demonstrates how to:
  1. Build the core Hamiltonian H = T + V from one-electron integrals
  2. Construct a canonical orthogonalizer X such that X^T S X = I
  3. Transform the generalized eigenproblem FC = SCε to ordinary form F'C' = C'ε
  4. Verify that the resulting MO coefficients satisfy C^T S C = I
  5. Validate against scipy.linalg.eigh and PySCF

Theoretical Background:
  - Section 2.9: Generalized Eigenproblems in the AO Metric
  - Section 2.10: Hands-on Python (Lab 2C)

Algorithm (from Section 2.9, Algorithm 2.3):
  1. Build X from S (Algorithm 2.2) with threshold τ
  2. Form F' = X^T F X (transformation to orthonormal basis)
  3. Diagonalize: solve F'C' = C'ε (standard eigenproblem)
  4. Back-transform: C = X C' (return to AO basis)

Note: In this demo, we use the core Hamiltonian H = T + V as "F" to illustrate
the algorithm. In Hartree-Fock, F is the density-dependent Fock matrix.

Usage:
    python lab2c_gen_eigenproblem.py
"""

import numpy as np
import scipy.linalg
from pyscf import gto, scf


def build_core_hamiltonian(mol):
    """
    Build the core Hamiltonian from one-electron integrals.

    The core Hamiltonian is H = T + V where:
      - T is the kinetic energy integral matrix
      - V is the nuclear attraction integral matrix

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.

    Returns
    -------
    H : ndarray (N, N)
        Core Hamiltonian matrix.
    S : ndarray (N, N)
        Overlap matrix (returned for convenience).
    T : ndarray (N, N)
        Kinetic energy matrix.
    V : ndarray (N, N)
        Nuclear attraction matrix.
    """
    S = mol.intor("int1e_ovlp")  # Overlap: S_uv = <u|v>
    T = mol.intor("int1e_kin")   # Kinetic: T_uv = <u|-1/2 nabla^2|v>
    V = mol.intor("int1e_nuc")   # Nuclear: V_uv = <u|V_nuc|v>
    H = T + V                    # Core Hamiltonian
    return H, S, T, V


def canonical_orthogonalizer(S, thresh=1e-10):
    """
    Build the canonical orthogonalizer X such that X^T S X = I.

    Implements Algorithm 2.2 from the lecture notes.

    The canonical orthogonalization uses the eigendecomposition of S:
        S = U diag(e) U^T
    Then X = U diag(e^{-1/2}) so that X^T S X = I.

    Eigenvalues below thresh are discarded to handle near-linear dependence.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix (symmetric, positive semi-definite).
    thresh : float
        Threshold for discarding small eigenvalues. Default 1e-10.

        Note: The lecture notes recommend 1e-6 to 1e-8 for production codes.
        The default 1e-10 is more aggressive and appropriate for demonstrations.

    Returns
    -------
    X : ndarray (N, M)
        Orthogonalizer matrix, where M <= N.
    eigenvalues : ndarray (M,)
        Retained eigenvalues of S.
    n_removed : int
        Number of eigenvectors removed due to linear dependence.
    """
    # Diagonalize S: S = U diag(e) U^T
    eigenvalues, U = np.linalg.eigh(S)

    # Keep only eigenvectors with eigenvalue > thresh
    keep = eigenvalues > thresh
    n_removed = np.sum(~keep)

    U_keep = U[:, keep]
    e_keep = eigenvalues[keep]

    # Build orthogonalizer: X = U_keep diag(e_keep^{-1/2})
    X = U_keep @ np.diag(e_keep ** (-0.5))

    return X, e_keep, n_removed


def solve_generalized_eigenproblem(F, S, thresh=1e-10):
    """
    Solve the generalized eigenproblem FC = SCe via orthogonalization.

    Algorithm:
        1. Build orthogonalizer X such that X^T S X = I
        2. Transform to orthonormal basis: F' = X^T F X
        3. Solve ordinary eigenproblem: F' C' = C' e
        4. Back-transform: C = X C'

    The resulting MO coefficients C satisfy:
        - F C = S C e  (eigenvalue equation)
        - C^T S C = I   (orthonormality under S-metric)

    Parameters
    ----------
    F : ndarray (N, N)
        Fock-like matrix (or core Hamiltonian) in AO basis.
    S : ndarray (N, N)
        Overlap matrix.
    thresh : float
        Threshold for eigenvalue cutoff in orthogonalizer.

    Returns
    -------
    eigenvalues : ndarray (M,)
        Orbital energies (eigenvalues).
    C : ndarray (N, M)
        MO coefficient matrix in AO basis.
    info : dict
        Additional information including X, n_removed, and condition number.
    """
    N = S.shape[0]

    # Step 1: Build orthogonalizer
    X, S_eigs, n_removed = canonical_orthogonalizer(S, thresh)
    M = X.shape[1]

    # Step 2: Transform F to orthonormal basis
    # F' = X^T F X
    F_prime = X.T @ F @ X

    # Step 3: Solve ordinary eigenproblem
    eigenvalues, C_prime = np.linalg.eigh(F_prime)

    # Step 4: Back-transform to AO basis
    # C = X C'
    C = X @ C_prime

    # Compute condition number of S for diagnostics
    cond = S_eigs.max() / S_eigs.min()

    info = {
        "X": X,
        "n_removed": n_removed,
        "n_kept": M,
        "S_eigenvalues": S_eigs,
        "condition_number": cond,
    }

    return eigenvalues, C, info


def verify_mo_orthonormality(C, S):
    """
    Verify that MO coefficients satisfy C^T S C = I.

    In a non-orthonormal AO basis, MO orthonormality is expressed as:
        C^T S C = I
    This means the MOs are orthonormal under the S-metric.

    Parameters
    ----------
    C : ndarray (N, M)
        MO coefficient matrix in AO basis.
    S : ndarray (N, N)
        Overlap matrix.

    Returns
    -------
    error_norm : float
        Frobenius norm ||C^T S C - I||.
    CtSC : ndarray (M, M)
        The matrix C^T S C for inspection.
    """
    M = C.shape[1]
    CtSC = C.T @ S @ C
    I = np.eye(M)
    error_norm = np.linalg.norm(CtSC - I, ord="fro")
    return error_norm, CtSC


def verify_eigenvalue_equation(F, S, C, eigenvalues):
    """
    Verify that FC = SCe (the generalized eigenvalue equation).

    The residual is computed as:
        R = FC - SC diag(e)
    and the error metric is ||R||_F.

    Parameters
    ----------
    F : ndarray (N, N)
        Fock-like matrix in AO basis.
    S : ndarray (N, N)
        Overlap matrix.
    C : ndarray (N, M)
        MO coefficient matrix.
    eigenvalues : ndarray (M,)
        Orbital energies.

    Returns
    -------
    residual_norm : float
        Frobenius norm of the residual ||FC - SCe||.
    max_residual : float
        Maximum absolute value of any element in the residual.
    """
    # FC - SC diag(e)
    FC = F @ C
    SCe = S @ C @ np.diag(eigenvalues)
    residual = FC - SCe

    residual_norm = np.linalg.norm(residual, ord="fro")
    max_residual = np.abs(residual).max()

    return residual_norm, max_residual


def compare_with_scipy(F, S):
    """
    Solve FC = SCe using scipy.linalg.eigh for comparison.

    scipy.linalg.eigh(A, B) solves the generalized eigenproblem Ax = Bx*lambda.

    Parameters
    ----------
    F : ndarray (N, N)
        Fock-like matrix.
    S : ndarray (N, N)
        Overlap matrix.

    Returns
    -------
    eigenvalues : ndarray (N,)
        Eigenvalues from scipy.
    C : ndarray (N, N)
        Eigenvectors from scipy (MO coefficients).
    """
    eigenvalues, C = scipy.linalg.eigh(F, S)
    return eigenvalues, C


def compare_with_pyscf(mol):
    """
    Run PySCF RHF and extract orbital energies for comparison.

    Note: RHF uses the Fock matrix (which depends on density), not the
    core Hamiltonian. For H2 at the first SCF iteration with zero density,
    F = H_core. After convergence, the Fock matrix differs from H_core.

    This function returns both:
      - Core Hamiltonian eigenvalues (comparable to our demo)
      - Converged RHF orbital energies (for reference)

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.

    Returns
    -------
    eps_hcore : ndarray
        Eigenvalues of H_core (generalized eigenproblem).
    eps_rhf : ndarray
        Converged RHF orbital energies.
    mf : pyscf.scf.RHF
        Converged RHF object.
    """
    # Get integrals
    S = mol.intor("int1e_ovlp")
    H = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    # Solve H_core eigenproblem with scipy
    eps_hcore, _ = scipy.linalg.eigh(H, S)

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()
    eps_rhf = mf.mo_energy

    return eps_hcore, eps_rhf, mf


def run_demo_h2():
    """
    Demonstrate generalized eigenproblem solution for H2.

    Returns
    -------
    success : bool
        True if all validations pass.
    """
    print("\n" + "=" * 60)
    print("Demo: H2 molecule (STO-3G)")
    print("=" * 60)

    # Build molecule
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print(f"Molecule: H2")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {mol.nao_nr()}")

    # Build core Hamiltonian
    H, S, T, V = build_core_hamiltonian(mol)
    print(f"\nCore Hamiltonian H = T + V:")
    print(f"  H shape: {H.shape}")
    print(f"  S shape: {S.shape}")

    # Solve generalized eigenproblem using our implementation
    print("\n--- Our Implementation (via orthogonalization) ---")
    eigenvalues, C, info = solve_generalized_eigenproblem(H, S, thresh=1e-10)

    print(f"Orthogonalizer X shape: {info['X'].shape}")
    print(f"Eigenvalues removed: {info['n_removed']}")
    print(f"Condition number of S: {info['condition_number']:.6e}")
    print(f"\nEigenvalues (orbital energies):")
    for i, eps in enumerate(eigenvalues):
        print(f"  e_{i}: {eps:12.8f} Hartree")

    # Verify orthonormality
    print("\n--- Orthonormality Verification ---")
    ortho_err, CtSC = verify_mo_orthonormality(C, S)
    print(f"||C^T S C - I|| = {ortho_err:.2e}")
    print("C^T S C =")
    print(CtSC)

    # Verify eigenvalue equation
    print("\n--- Eigenvalue Equation Verification ---")
    resid_norm, max_resid = verify_eigenvalue_equation(H, S, C, eigenvalues)
    print(f"||HC - SCe||_F = {resid_norm:.2e}")
    print(f"max|HC - SCe|  = {max_resid:.2e}")

    # Compare with scipy
    print("\n--- Comparison with scipy.linalg.eigh ---")
    eps_scipy, C_scipy = compare_with_scipy(H, S)
    print("Eigenvalues comparison:")
    print(f"  {'Our impl':>12s}  {'scipy':>12s}  {'Difference':>12s}")
    for i in range(len(eigenvalues)):
        diff = abs(eigenvalues[i] - eps_scipy[i])
        print(f"  {eigenvalues[i]:12.8f}  {eps_scipy[i]:12.8f}  {diff:.2e}")

    # Verify scipy orthonormality too
    ortho_err_scipy, _ = verify_mo_orthonormality(C_scipy, S)
    print(f"scipy ||C^T S C - I|| = {ortho_err_scipy:.2e}")

    # Compare with PySCF
    print("\n--- Comparison with PySCF ---")
    eps_hcore_pyscf, eps_rhf_pyscf, mf = compare_with_pyscf(mol)
    print("H_core eigenvalues (PySCF/scipy):")
    print(f"  {'Our impl':>12s}  {'PySCF H_core':>12s}  {'PySCF RHF':>12s}")
    for i in range(len(eigenvalues)):
        print(f"  {eigenvalues[i]:12.8f}  {eps_hcore_pyscf[i]:12.8f}  {eps_rhf_pyscf[i]:12.8f}")
    print("(Note: RHF orbital energies differ because F != H_core after SCF)")

    # Summary checks
    success = True
    checks = [
        ("Orthonormality", ortho_err < 1e-10),
        ("Eigenvalue equation", resid_norm < 1e-10),
        ("scipy agreement", np.allclose(eigenvalues, eps_scipy, atol=1e-10)),
        ("PySCF H_core agreement", np.allclose(eigenvalues, eps_hcore_pyscf, atol=1e-10)),
    ]

    print("\n--- Summary ---")
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            success = False

    return success


def run_demo_h2o():
    """
    Demonstrate generalized eigenproblem solution for H2O.

    This example has more basis functions and shows the method
    scales correctly.

    Returns
    -------
    success : bool
        True if all validations pass.
    """
    print("\n" + "=" * 60)
    print("Demo: H2O molecule (cc-pVDZ)")
    print("=" * 60)

    # Build molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    print(f"Molecule: H2O")
    print(f"Basis: cc-pVDZ")
    print(f"Number of AOs: {mol.nao_nr()}")
    print(f"Number of electrons: {mol.nelectron}")

    # Build core Hamiltonian
    H, S, T, V = build_core_hamiltonian(mol)
    print(f"\nCore Hamiltonian H = T + V:")
    print(f"  H shape: {H.shape}")

    # Analyze overlap matrix conditioning
    S_eigs = np.linalg.eigvalsh(S)
    print(f"\nOverlap matrix eigenvalues:")
    print(f"  min(e): {S_eigs.min():.6e}")
    print(f"  max(e): {S_eigs.max():.6e}")
    print(f"  cond(S): {S_eigs.max()/S_eigs.min():.6e}")

    # Solve generalized eigenproblem
    print("\n--- Our Implementation ---")
    eigenvalues, C, info = solve_generalized_eigenproblem(H, S, thresh=1e-10)

    print(f"Eigenvalues removed: {info['n_removed']}")
    print(f"Kept {info['n_kept']} of {S.shape[0]} basis functions")

    print(f"\nLowest 5 eigenvalues:")
    for i in range(min(5, len(eigenvalues))):
        print(f"  e_{i}: {eigenvalues[i]:12.8f} Hartree")
    print(f"  ...")
    print(f"Highest eigenvalue: e_{len(eigenvalues)-1}: {eigenvalues[-1]:12.8f} Hartree")

    # Verify orthonormality
    ortho_err, _ = verify_mo_orthonormality(C, S)
    print(f"\n||C^T S C - I|| = {ortho_err:.2e}")

    # Verify eigenvalue equation
    resid_norm, max_resid = verify_eigenvalue_equation(H, S, C, eigenvalues)
    print(f"||HC - SCe||_F = {resid_norm:.2e}")

    # Compare with scipy
    eps_scipy, C_scipy = compare_with_scipy(H, S)
    max_diff = np.abs(eigenvalues - eps_scipy).max()
    print(f"\nmax|e_ours - e_scipy| = {max_diff:.2e}")

    # Summary
    success = True
    checks = [
        ("Orthonormality", ortho_err < 1e-10),
        ("Eigenvalue equation", resid_norm < 1e-10),
        ("scipy agreement", max_diff < 1e-10),
    ]

    print("\n--- Summary ---")
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            success = False

    return success


def run_demo_augmented():
    """
    Demonstrate behavior with near-linear dependence (aug-cc-pVDZ).

    Augmented basis sets often have near-linear dependence, making
    eigenvalue thresholding important.

    Returns
    -------
    success : bool
        True if all validations pass.
    """
    print("\n" + "=" * 60)
    print("Demo: H2O with aug-cc-pVDZ (near-linear dependence)")
    print("=" * 60)

    # Build molecule with augmented basis
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="aug-cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    print(f"Molecule: H2O")
    print(f"Basis: aug-cc-pVDZ")
    print(f"Number of AOs: {mol.nao_nr()}")

    # Build core Hamiltonian
    H, S, T, V = build_core_hamiltonian(mol)

    # Analyze overlap matrix
    S_eigs = np.linalg.eigvalsh(S)
    print(f"\nOverlap matrix analysis:")
    print(f"  min(e): {S_eigs.min():.6e}")
    print(f"  max(e): {S_eigs.max():.6e}")
    print(f"  cond(S): {S_eigs.max()/S_eigs.min():.6e}")

    # Count near-zero eigenvalues
    n_tiny = np.sum(S_eigs < 1e-6)
    print(f"  Eigenvalues < 1e-6: {n_tiny}")

    # Solve with different thresholds
    print("\n--- Effect of threshold on dimension ---")
    for thresh in [1e-14, 1e-10, 1e-6]:
        eigenvalues, C, info = solve_generalized_eigenproblem(H, S, thresh=thresh)
        ortho_err, _ = verify_mo_orthonormality(C, S)
        print(f"  thresh={thresh:.0e}: kept {info['n_kept']:2d}/{S.shape[0]}, "
              f"||C^T S C - I|| = {ortho_err:.2e}")

    # Use recommended threshold
    print("\n--- Recommended threshold (1e-10) ---")
    eigenvalues, C, info = solve_generalized_eigenproblem(H, S, thresh=1e-10)

    ortho_err, _ = verify_mo_orthonormality(C, S)
    resid_norm, _ = verify_eigenvalue_equation(H, S, C, eigenvalues)

    # Compare with scipy (which uses all directions)
    try:
        eps_scipy, C_scipy = compare_with_scipy(H, S)
        # scipy keeps all, so compare only kept eigenvalues
        # Note: ordering might differ, so compare sorted arrays
        eps_ours_sorted = np.sort(eigenvalues)
        eps_scipy_sorted = np.sort(eps_scipy)[:len(eigenvalues)]
        max_diff = np.abs(eps_ours_sorted - eps_scipy_sorted).max()
        scipy_success = True
    except np.linalg.LinAlgError:
        print("  scipy.linalg.eigh failed (ill-conditioned)")
        max_diff = np.nan
        scipy_success = False

    print(f"\n||C^T S C - I|| = {ortho_err:.2e}")
    print(f"||HC - SCe||_F = {resid_norm:.2e}")
    if scipy_success:
        print(f"max|e_ours - e_scipy| = {max_diff:.2e} (comparing {len(eigenvalues)} eigenvalues)")

    success = ortho_err < 1e-10 and resid_norm < 1e-10
    print(f"\nValidation: {'PASS' if success else 'FAIL'}")

    return success


def main():
    """Main function demonstrating Lab 2C concepts."""
    print("=" * 60)
    print("Lab 2C: Solve FC = SCe via Orthogonalization")
    print("=" * 60)
    print("\nThis lab demonstrates how to solve the generalized eigenproblem")
    print("that arises in quantum chemistry (Roothaan-Hall equations) by")
    print("transforming to an orthonormal basis.")
    print("\nKey steps:")
    print("  1. Build orthogonalizer X such that X^T S X = I")
    print("  2. Transform: F' = X^T F X")
    print("  3. Solve ordinary eigenproblem: F' C' = C' e")
    print("  4. Back-transform: C = X C'")

    # Run demonstrations
    success_h2 = run_demo_h2()
    success_h2o = run_demo_h2o()
    success_aug = run_demo_augmented()

    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    all_passed = success_h2 and success_h2o and success_aug
    results = [
        ("H2 (STO-3G)", success_h2),
        ("H2O (cc-pVDZ)", success_h2o),
        ("H2O (aug-cc-pVDZ)", success_aug),
    ]
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_passed:
        print("\nAll validation checks PASSED!")
    else:
        print("\nSome validation checks FAILED. Review output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
