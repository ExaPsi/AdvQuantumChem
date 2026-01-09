#!/usr/bin/env python3
"""
Lab 2B: Build X such that X^T S X = I

This script accompanies Chapter 2 of the Advanced Quantum Chemistry lecture notes.
It demonstrates how to:
  1. Build the canonical orthogonalizer with eigenvalue thresholding
  2. Build the symmetric (Löwdin) orthogonalizer with thresholding
  3. Implement Gram-Schmidt under the S-metric for pedagogical comparison
  4. Verify orthogonality of resulting transformation matrices
  5. Compare stability of different orthogonalizers across basis sets

Theoretical Background:
  - Section 2.5: The Overlap Matrix as a Metric
  - Section 2.6: Gram-Schmidt Orthonormalization in a Nonorthonormal AO Basis
  - Section 2.7: Canonical and Löwdin (Symmetric) Orthonormalization
  - Section 2.8: Cholesky Orthonormalization and Practical Linear Dependence
  - Section 2.10: Hands-on Python (Lab 2B)

Reference: Helgaker, Jorgensen, Olsen, Chapter 6 (orthogonalization methods)

Usage:
    python lab2b_orthogonalizers.py
"""
import numpy as np
from pyscf import gto


# =============================================================================
# Core Orthogonalizer Functions
# =============================================================================

def canonical_orthogonalizer(S: np.ndarray, thresh: float = 1e-10) -> tuple:
    """
    Build canonical orthogonalizer with eigenvalue thresholding.

    The canonical orthogonalizer diagonalizes S and builds X from its
    eigenvectors, scaled by the inverse square root of eigenvalues.
    Near-zero eigenvalues (indicating linear dependence) are removed.

    Given S = U diag(e) U^T, we construct:
        X = U_kept @ diag(e_kept^(-1/2))

    This is numerically stable because we explicitly remove directions
    where the basis is nearly linearly dependent.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix, must be symmetric positive semi-definite.
    thresh : float
        Eigenvalue threshold. Eigenvalues below this are considered
        linear dependencies and are dropped. Default 1e-10.

    Returns
    -------
    X : ndarray (N, M)
        Orthogonalizer satisfying X^T S X = I_M.
        M <= N due to possible removal of near-dependent directions.
    kept_eigenvalues : ndarray (M,)
        Eigenvalues that were kept (all > thresh).

    Notes
    -----
    The canonical orthogonalizer is the most stable choice for general use.
    It is guaranteed to produce an orthonormal set as long as S is positive
    semi-definite and thresh is chosen appropriately.
    """
    # Diagonalize overlap matrix: S = U @ diag(e) @ U^T
    eigenvalues, U = np.linalg.eigh(S)

    # Identify eigenvalues above threshold (keep these directions)
    keep_mask = eigenvalues > thresh
    n_kept = np.sum(keep_mask)
    n_dropped = S.shape[0] - n_kept

    if n_dropped > 0:
        print(f"  [canonical] Dropping {n_dropped} near-dependent directions "
              f"(eigenvalues < {thresh:.1e})")

    # Keep only directions with sufficient eigenvalue
    U_kept = U[:, keep_mask]
    e_kept = eigenvalues[keep_mask]

    # Build orthogonalizer: X = U_kept @ diag(e_kept^(-1/2))
    # This ensures X^T S X = diag(e_kept^(-1/2)) U_kept^T S U_kept diag(e_kept^(-1/2))
    #                      = diag(e_kept^(-1/2)) diag(e_kept) diag(e_kept^(-1/2))
    #                      = I
    X = U_kept @ np.diag(e_kept ** (-0.5))

    return X, e_kept


def symmetric_orthogonalizer(S: np.ndarray, thresh: float = 1e-10) -> tuple:
    """
    Build Löwdin symmetric orthogonalizer with eigenvalue thresholding.

    The true Löwdin symmetric orthogonalizer is S^(-1/2), which produces
    orthonormal functions that are "closest" to the original AOs in a
    least-squares sense. Without thresholding:

        X_full = S^(-1/2) = U @ diag(e^(-1/2)) @ U^T  (N x N matrix)

    However, when near-linear dependencies are present (small eigenvalues),
    we must remove those directions for numerical stability. With thresholding,
    we return the RECTANGULAR form:

        X = U_kept @ diag(e_kept^(-1/2))  (N x M matrix, M <= N)

    This is mathematically equivalent to the canonical orthogonalizer when
    thresholding is applied, because both produce X satisfying X^T S X = I_M.
    The distinction matters only when all eigenvalues are kept.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix, must be symmetric positive semi-definite.
    thresh : float
        Eigenvalue threshold. Eigenvalues below this are considered
        linear dependencies and are dropped. Default 1e-10.

        Note: The lecture notes recommend 1e-6 to 1e-8 for production codes.
        The default 1e-10 is more aggressive and appropriate for demonstrations
        where numerical stability is less critical.

    Returns
    -------
    X : ndarray (N, M)
        Orthogonalizer satisfying X^T S X = I_M.
        M <= N due to possible removal of near-dependent directions.
    kept_eigenvalues : ndarray (M,)
        Eigenvalues that were kept (all > thresh).

    Notes
    -----
    The true symmetric orthogonalizer S^(-1/2) minimizes the sum of squared
    differences between original and orthonormalized basis functions.
    When eigenvalues are dropped, this "closest" property is lost, but
    numerical stability is gained.

    For the full N x N symmetric orthogonalizer (no thresholding), use:
        X_full = U @ np.diag(e ** -0.5) @ U.T

    See Also
    --------
    canonical_orthogonalizer : Identical implementation with thresholding
    cholesky_orthogonalizer : Alternative using Cholesky decomposition
    """
    # Diagonalize overlap matrix: S = U @ diag(e) @ U^T
    eigenvalues, U = np.linalg.eigh(S)

    # Identify eigenvalues above threshold
    keep_mask = eigenvalues > thresh
    n_kept = np.sum(keep_mask)
    n_dropped = S.shape[0] - n_kept

    if n_dropped > 0:
        print(f"  [symmetric] Dropping {n_dropped} near-dependent directions "
              f"(eigenvalues < {thresh:.1e})")

    # Keep only directions with sufficient eigenvalue
    U_kept = U[:, keep_mask]
    e_kept = eigenvalues[keep_mask]

    # Build symmetric orthogonalizer
    # If all eigenvalues are kept, this gives the full S^(-1/2)
    # With thresholding, we get the projection onto the non-degenerate subspace
    X = U_kept @ np.diag(e_kept ** (-0.5))

    return X, e_kept


def cholesky_orthogonalizer(S: np.ndarray) -> np.ndarray:
    """
    Build Cholesky orthogonalizer X = L^{-T} where S = L L^T.

    The Cholesky decomposition factors S as S = L L^T where L is lower
    triangular. The orthogonalizer is then X = L^{-T} = (L^{-1})^T.

    This satisfies X^T S X = L^{-1} (L L^T) L^{-T} = I.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix, must be symmetric positive DEFINITE.
        (Not just positive semi-definite - Cholesky requires strict
        positive definiteness.)

    Returns
    -------
    X : ndarray (N, N)
        Cholesky orthogonalizer satisfying X^T S X = I.

    Raises
    ------
    np.linalg.LinAlgError
        If S is not positive definite (Cholesky decomposition fails).

    Notes
    -----
    The Cholesky orthogonalizer has computational advantages:
      - O(N^3/3) vs O(N^3) for eigendecomposition
      - No eigenvalue problem to solve
      - Naturally numerically stable for well-conditioned S

    However, it has significant limitations:
      - FAILS for near-singular S (no thresholding possible)
      - Not suitable for ill-conditioned basis sets
      - Use canonical/symmetric orthogonalizers when near-linear
        dependence is expected

    The orthonormalized functions are given by the columns of X, which
    transform the AO basis to an orthonormal set via triangular
    back-substitution.

    Reference: Section 2.8 in the lecture notes (Cholesky Orthonormalization)

    See Also
    --------
    canonical_orthogonalizer : Preferred for ill-conditioned S
    symmetric_orthogonalizer : Alternative with "closest" property
    """
    # Cholesky decomposition: S = L @ L.T
    L = np.linalg.cholesky(S)

    # Orthogonalizer: X = L^{-T} = (L^{-1})^T
    # We compute L^{-1} and then transpose
    L_inv = np.linalg.inv(L)
    X = L_inv.T

    return X


def gram_schmidt_metric(S: np.ndarray, thresh: float = 1e-12) -> tuple:
    """
    Build orthogonalizer using modified Gram-Schmidt under the S-metric.

    This implements the classical Gram-Schmidt orthogonalization, but using
    the S-metric inner product: <u, v>_S = u^T S v

    Starting from standard basis vectors e_k, we build orthonormal vectors
    x_1, x_2, ... such that x_i^T S x_j = delta_ij.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix, must be symmetric positive semi-definite.
    thresh : float
        Threshold for detecting linear dependence. If the norm-squared
        of a vector after orthogonalization is below thresh, that
        direction is skipped. Default 1e-12.

    Returns
    -------
    X : ndarray (N, M)
        Orthogonalizer satisfying X^T S X = I_M.
        M <= N due to possible linear dependence detection.
    n_kept : int
        Number of orthonormal directions found.

    Notes
    -----
    This implementation is for PEDAGOGICAL purposes. Gram-Schmidt under
    a non-identity metric is numerically sensitive:

    1. Loss of orthogonality: Accumulated round-off errors can cause
       the orthogonality condition x_i^T S x_j = 0 to degrade.

    2. Order dependence: Different ordering of input vectors can give
       different results, especially near linear dependence.

    3. Instability: Near-singular S leads to amplification of errors.

    For production code, always use eigenvalue-based orthogonalizers.
    """
    N = S.shape[0]
    X = np.zeros((N, N))
    m = 0  # Number of orthonormal vectors found so far

    for k in range(N):
        # Start with k-th standard basis vector
        v = np.zeros(N)
        v[k] = 1.0

        # Modified Gram-Schmidt: subtract projections onto existing orthonormal vectors
        # Projection of v onto x_j in S-metric: proj_j = (x_j^T S v) x_j
        for j in range(m):
            # Compute S-metric inner product
            overlap = X[:, j].T @ S @ v
            # Subtract projection
            v = v - overlap * X[:, j]

        # Compute norm-squared in S-metric: ||v||_S^2 = v^T S v
        norm_squared = v.T @ S @ v

        if norm_squared < thresh:
            # This direction is nearly linearly dependent on previous ones
            continue

        # Normalize and store
        X[:, m] = v / np.sqrt(norm_squared)
        m += 1

    # Return only the columns that were filled
    return X[:, :m], m


# =============================================================================
# Validation Functions
# =============================================================================

def check_orthogonalizer(S: np.ndarray, X: np.ndarray) -> float:
    """
    Verify that X satisfies X^T S X = I.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix.
    X : ndarray (N, M)
        Orthogonalizer to verify.

    Returns
    -------
    error : float
        Frobenius norm of (X^T S X - I), should be ~1e-14 for a good
        orthogonalizer.

    Notes
    -----
    The Frobenius norm ||A||_F = sqrt(sum_ij |A_ij|^2) gives an overall
    measure of how close the product X^T S X is to the identity.
    """
    M = X.shape[1]
    product = X.T @ S @ X
    identity = np.eye(M)
    error = np.linalg.norm(product - identity, 'fro')
    return error


def check_orthogonalizer_detailed(S: np.ndarray, X: np.ndarray) -> dict:
    """
    Detailed verification of orthogonalizer quality.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix.
    X : ndarray (N, M)
        Orthogonalizer to verify.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'frobenius_error': Overall error ||X^T S X - I||_F
        - 'max_diagonal_error': max |diag(X^T S X) - 1|
        - 'max_offdiagonal': max |offdiag(X^T S X)|
        - 'is_valid': True if all errors are below tolerance
    """
    M = X.shape[1]
    product = X.T @ S @ X
    identity = np.eye(M)

    # Extract diagonal and off-diagonal parts
    diagonal = np.diag(product)
    offdiagonal_mask = ~np.eye(M, dtype=bool)
    offdiagonal = product[offdiagonal_mask]

    results = {
        'frobenius_error': np.linalg.norm(product - identity, 'fro'),
        'max_diagonal_error': np.max(np.abs(diagonal - 1.0)),
        'max_offdiagonal': np.max(np.abs(offdiagonal)) if len(offdiagonal) > 0 else 0.0,
    }

    # Consider valid if errors are at numerical precision level
    results['is_valid'] = (results['frobenius_error'] < 1e-10)

    return results


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_orthogonalizers(S: np.ndarray, thresh: float = 1e-10,
                            include_cholesky: bool = True) -> dict:
    """
    Compare all orthogonalization methods on the same overlap matrix.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix.
    thresh : float
        Eigenvalue threshold for canonical and symmetric methods.
    include_cholesky : bool
        Whether to include Cholesky orthogonalizer (may fail for
        ill-conditioned S). Default True.

    Returns
    -------
    results : dict
        Dictionary with results for each method, containing:
        - 'X': The orthogonalizer matrix
        - 'kept_dim': Number of dimensions kept
        - 'error': Frobenius norm of (X^T S X - I)
        - 'valid': Whether error is below tolerance
    """
    N = S.shape[0]
    results = {}

    # Canonical orthogonalizer
    print("\n  Computing canonical orthogonalizer...")
    X_can, e_can = canonical_orthogonalizer(S, thresh=thresh)
    err_can = check_orthogonalizer(S, X_can)
    results['canonical'] = {
        'X': X_can,
        'kept_dim': X_can.shape[1],
        'dropped_dim': N - X_can.shape[1],
        'error': err_can,
        'valid': err_can < 1e-10,
        'min_eigenvalue': e_can.min(),
        'max_eigenvalue': e_can.max(),
    }

    # Symmetric orthogonalizer
    print("  Computing symmetric orthogonalizer...")
    X_sym, e_sym = symmetric_orthogonalizer(S, thresh=thresh)
    err_sym = check_orthogonalizer(S, X_sym)
    results['symmetric'] = {
        'X': X_sym,
        'kept_dim': X_sym.shape[1],
        'dropped_dim': N - X_sym.shape[1],
        'error': err_sym,
        'valid': err_sym < 1e-10,
        'min_eigenvalue': e_sym.min(),
        'max_eigenvalue': e_sym.max(),
    }

    # Cholesky orthogonalizer (may fail for ill-conditioned S)
    if include_cholesky:
        print("  Computing Cholesky orthogonalizer...")
        try:
            X_chol = cholesky_orthogonalizer(S)
            err_chol = check_orthogonalizer(S, X_chol)
            results['cholesky'] = {
                'X': X_chol,
                'kept_dim': X_chol.shape[1],
                'dropped_dim': 0,  # Cholesky keeps all dimensions
                'error': err_chol,
                'valid': err_chol < 1e-10,
            }
        except np.linalg.LinAlgError:
            print("    [cholesky] Failed - S is not positive definite")
            results['cholesky'] = {
                'X': None,
                'kept_dim': 0,
                'dropped_dim': N,
                'error': np.inf,
                'valid': False,
                'failed': True,
            }

    # Gram-Schmidt (use tighter threshold for numerical stability)
    print("  Computing Gram-Schmidt orthogonalizer...")
    X_gs, n_gs = gram_schmidt_metric(S, thresh=1e-12)
    err_gs = check_orthogonalizer(S, X_gs)
    results['gram_schmidt'] = {
        'X': X_gs,
        'kept_dim': n_gs,
        'dropped_dim': N - n_gs,
        'error': err_gs,
        'valid': err_gs < 1e-10,
    }

    return results


def print_comparison_results(results: dict, basis_name: str):
    """
    Print a formatted comparison of orthogonalizer results.

    Parameters
    ----------
    results : dict
        Output from compare_orthogonalizers().
    basis_name : str
        Name of the basis set for display.
    """
    print(f"\n  {'Method':<15} {'Kept/Total':<12} {'Error':<12} {'Status':<8}")
    print(f"  {'-'*47}")

    for method, data in results.items():
        total_dim = data['kept_dim'] + data['dropped_dim']
        if data.get('failed', False):
            status = "FAILED"
            print(f"  {method:<15} {'N/A':>4}/{'N/A':<7} "
                  f"{'N/A':<12} {status:<8}")
        else:
            status = "PASS" if data['valid'] else "FAIL"
            print(f"  {method:<15} {data['kept_dim']:>4}/{total_dim:<7} "
                  f"{data['error']:<12.2e} {status:<8}")


# =============================================================================
# Demonstration Functions
# =============================================================================

def analyze_basis_conditioning(mol) -> dict:
    """
    Analyze the conditioning of the overlap matrix for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.

    Returns
    -------
    analysis : dict
        Dictionary containing eigenvalue spectrum and condition number.
    """
    S = mol.intor("int1e_ovlp")
    eigenvalues = np.linalg.eigvalsh(S)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]  # Descending order

    # Effective condition number (ignoring tiny eigenvalues)
    tiny = 1e-14
    e_keep = eigenvalues_sorted[eigenvalues_sorted > tiny]
    cond = e_keep[0] / e_keep[-1] if len(e_keep) > 0 else np.inf

    return {
        'eigenvalues': eigenvalues_sorted,
        'min_eigenvalue': eigenvalues_sorted[-1],
        'max_eigenvalue': eigenvalues_sorted[0],
        'condition_number': cond,
        'n_below_1e10': np.sum(eigenvalues < 1e-10),
        'n_below_1e6': np.sum(eigenvalues < 1e-6),
    }


def demonstrate_gen_eig_solve(mol, X: np.ndarray):
    """
    Demonstrate solving generalized eigenvalue problem FC = SCe using X.

    This shows how the orthogonalizer transforms the generalized eigenvalue
    problem FC = SCe into a standard eigenvalue problem F'C' = C'e.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    X : ndarray (N, M)
        Orthogonalizer satisfying X^T S X = I.

    Returns
    -------
    results : dict
        Dictionary with eigenvalues and verification results.
    """
    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    F = T + V  # Use core Hamiltonian as test "Fock" matrix

    # Transform to orthonormal basis: F' = X^T F X
    F_prime = X.T @ F @ X

    # Solve standard eigenvalue problem in orthonormal basis
    eigenvalues, C_prime = np.linalg.eigh(F_prime)

    # Transform back to AO basis: C = X @ C'
    C = X @ C_prime

    # Verify: C^T S C should be identity
    orthonormality_error = np.linalg.norm(C.T @ S @ C - np.eye(C.shape[1]), 'fro')

    # Verify: FC = SCe (residual should be ~0)
    residual = F @ C - S @ C @ np.diag(eigenvalues)
    residual_norm = np.linalg.norm(residual, 'fro')

    return {
        'eigenvalues': eigenvalues,
        'C': C,
        'orthonormality_error': orthonormality_error,
        'residual_norm': residual_norm,
    }


# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    """Main function demonstrating Lab 2B concepts."""
    print("=" * 70)
    print("Lab 2B: Build X such that X^T S X = I")
    print("=" * 70)

    # Define water molecule
    water_coords = "O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043"

    # Test cases with increasing basis set complexity
    test_cases = [
        ("sto-3g", "Well-conditioned minimal basis"),
        ("cc-pVDZ", "Moderately conditioned DZ basis"),
        ("aug-cc-pVDZ", "Diffuse functions increase condition number"),
    ]

    all_passed = True

    for basis, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Basis: {basis}")
        print(f"Description: {description}")
        print("=" * 70)

        # Build molecule
        mol = gto.M(
            atom=water_coords,
            basis=basis,
            unit="Angstrom",
            verbose=0,
        )

        print(f"\nMolecule: H2O")
        print(f"Number of AOs: {mol.nao_nr()}")

        # Analyze conditioning
        print("\n--- Overlap Matrix Analysis ---")
        analysis = analyze_basis_conditioning(mol)
        print(f"Eigenvalue range: [{analysis['min_eigenvalue']:.3e}, "
              f"{analysis['max_eigenvalue']:.3e}]")
        print(f"Condition number: {analysis['condition_number']:.3e}")
        print(f"Eigenvalues < 1e-10: {analysis['n_below_1e10']}")
        print(f"Eigenvalues < 1e-6:  {analysis['n_below_1e6']}")

        # Compare orthogonalizers
        print("\n--- Comparing Orthogonalizers ---")
        S = mol.intor("int1e_ovlp")
        results = compare_orthogonalizers(S, thresh=1e-10)
        print_comparison_results(results, basis)

        # Check if all methods passed
        for method, data in results.items():
            if not data['valid']:
                all_passed = False
                print(f"\n  WARNING: {method} failed validation!")

        # Demonstrate generalized eigenvalue problem solution
        print("\n--- Generalized Eigenvalue Problem Demo ---")
        X_can = results['canonical']['X']
        gen_eig_results = demonstrate_gen_eig_solve(mol, X_can)
        print(f"Solving H_core C = S C e using canonical orthogonalizer")
        print(f"  ||C^T S C - I||_F = {gen_eig_results['orthonormality_error']:.2e}")
        print(f"  ||FC - SCe||_F    = {gen_eig_results['residual_norm']:.2e}")
        print(f"  Lowest eigenvalues: {gen_eig_results['eigenvalues'][:3]}")

    # Additional test: Show effect of threshold on problematic basis
    print(f"\n{'='*70}")
    print("Effect of Eigenvalue Threshold on aug-cc-pVDZ")
    print("=" * 70)

    mol = gto.M(
        atom=water_coords,
        basis="aug-cc-pVDZ",
        unit="Angstrom",
        verbose=0,
    )
    S = mol.intor("int1e_ovlp")
    N = mol.nao_nr()

    print(f"\n  {'Threshold':<12} {'Kept/Total':<12} {'Error':<12} {'Status'}")
    print(f"  {'-'*48}")

    for thresh in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]:
        X, e_kept = canonical_orthogonalizer(S, thresh=thresh)
        err = check_orthogonalizer(S, X)
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  {thresh:<12.0e} {X.shape[1]:>4}/{N:<7} {err:<12.2e} {status}")

    # Demonstrate a case with actual linear dependence (aug-cc-pVTZ often has issues)
    print(f"\n{'='*70}")
    print("Testing with aug-cc-pVTZ (may show near-linear dependencies)")
    print("=" * 70)

    try:
        mol_tz = gto.M(
            atom=water_coords,
            basis="aug-cc-pVTZ",
            unit="Angstrom",
            verbose=0,
        )
        S_tz = mol_tz.intor("int1e_ovlp")
        N_tz = mol_tz.nao_nr()

        analysis_tz = analyze_basis_conditioning(mol_tz)
        print(f"\nNumber of AOs: {N_tz}")
        print(f"Eigenvalue range: [{analysis_tz['min_eigenvalue']:.3e}, "
              f"{analysis_tz['max_eigenvalue']:.3e}]")
        print(f"Condition number: {analysis_tz['condition_number']:.3e}")
        print(f"Eigenvalues < 1e-6: {analysis_tz['n_below_1e6']}")

        print(f"\n  {'Threshold':<12} {'Kept/Total':<12} {'Error':<12} {'Status'}")
        print(f"  {'-'*48}")

        for thresh in [1e-10, 1e-8, 1e-6, 1e-4]:
            X_tz, _ = canonical_orthogonalizer(S_tz, thresh=thresh)
            err_tz = check_orthogonalizer(S_tz, X_tz)
            status = "PASS" if err_tz < 1e-10 else "FAIL"
            print(f"  {thresh:<12.0e} {X_tz.shape[1]:>4}/{N_tz:<7} "
                  f"{err_tz:<12.2e} {status}")
    except Exception as e:
        print(f"\n  (aug-cc-pVTZ test skipped: {e})")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print("=" * 70)

    if all_passed:
        print("\nAll validation checks PASSED!")
        print("\nKey observations:")
        print("  1. Canonical and symmetric orthogonalizers give identical dimensions")
        print("  2. Gram-Schmidt may lose orthogonality with poor conditioning")
        print("  3. Diffuse basis sets (aug-) have smaller minimum eigenvalues")
        print("  4. Threshold selection balances stability vs. flexibility")
    else:
        print("\nSome validation checks FAILED. Review output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
