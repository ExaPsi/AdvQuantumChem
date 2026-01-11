#!/usr/bin/env python3
"""
Lab 2B Solution: Build Orthogonalizer X such that X^T S X = I

This script implements three orthogonalization methods:
1. Canonical orthogonalization via S^(-1/2) = U s^(-1/2)
2. Symmetric (Lowdin) orthogonalization: X = U s^(-1/2) U^T
3. Gram-Schmidt orthogonalization under the S-metric (pedagogical)

All methods produce X satisfying X^T S X = I, enabling transformation
of generalized eigenproblems to standard form.

Learning objectives:
1. Understand the role of orthogonalizers in quantum chemistry
2. Implement canonical and symmetric orthogonalization
3. Implement Gram-Schmidt with the S-metric (and see its limitations)
4. Apply eigenvalue thresholding for near-linear dependence

Test molecule: H2O (water)

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 2: Gaussian Basis Sets and Orthonormalization
"""

import numpy as np
from pyscf import gto
from typing import Optional

# =============================================================================
# Section 1: Canonical Orthogonalization
# =============================================================================

def canonical_orthogonalizer(S: np.ndarray,
                              threshold: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build canonical orthogonalizer X_can = U s^(-1/2) with eigenvalue thresholding.

    The canonical orthogonalizer transforms AO basis to an orthonormal basis
    where each new basis function is an eigenvector of S. This is numerically
    robust because we can safely discard directions with small eigenvalues.

    Mathematical basis:
        S = U s U^T  (eigendecomposition)
        X = U_k s_k^(-1/2)  where k indexes kept eigenpairs
        X^T S X = s_k^(-1/2) U_k^T U s U^T U_k s_k^(-1/2)
                = s_k^(-1/2) s_k s_k^(-1/2) = I

    Args:
        S: Overlap matrix (N x N), symmetric positive semi-definite
        threshold: Eigenvalues below this are considered near-zero

    Returns:
        X: Orthogonalizer matrix (N x M where M <= N)
        eigenvalues_kept: The kept eigenvalues
        n_dropped: Number of eigenvalues/eigenvectors dropped
    """
    # Diagonalize S = U diag(s) U^T
    # eigvalsh returns eigenvalues in ascending order
    eigenvalues, U = np.linalg.eigh(S)

    # Identify significant eigenvalues (above threshold)
    # Small eigenvalues indicate near-linear dependence
    keep_mask = eigenvalues > threshold
    n_dropped = np.sum(~keep_mask)

    # Extract kept eigenpairs
    eigenvalues_kept = eigenvalues[keep_mask]
    U_kept = U[:, keep_mask]

    # Build orthogonalizer: X = U_k @ diag(s_k^(-1/2))
    # This is equivalent to X = U_k @ np.diag(eigenvalues_kept ** -0.5)
    # but more efficiently written as column scaling
    X = U_kept @ np.diag(eigenvalues_kept ** -0.5)

    return X, eigenvalues_kept, n_dropped


# =============================================================================
# Section 2: Symmetric (Lowdin) Orthogonalization
# =============================================================================

def symmetric_orthogonalizer(S: np.ndarray,
                              threshold: float = 1e-10) -> tuple[np.ndarray, int]:
    """
    Build symmetric (Lowdin) orthogonalizer X_sym = S^(-1/2) = U s^(-1/2) U^T.

    The symmetric orthogonalizer produces an orthonormal basis that is
    "closest" to the original AO basis in a least-squares sense.
    It is symmetric: X_sym = X_sym^T.

    Mathematical basis:
        S = U s U^T
        S^(-1/2) = U s^(-1/2) U^T
        (S^(-1/2))^T S (S^(-1/2)) = S^(-1/2) S S^(-1/2) = I

    Properties of Lowdin orthogonalization:
    - The new basis minimizes sum of squared differences from original AOs
    - X_sym is symmetric (unlike X_can)
    - Each new AO is a "democratically averaged" combination of old AOs

    Args:
        S: Overlap matrix (N x N)
        threshold: Eigenvalue cutoff for thresholding

    Returns:
        X_sym: Symmetric orthogonalizer (N x N, or N x M with thresholding)
        n_dropped: Number of near-zero eigenvalues detected
    """
    eigenvalues, U = np.linalg.eigh(S)

    # Apply threshold
    keep_mask = eigenvalues > threshold
    n_dropped = np.sum(~keep_mask)

    if n_dropped > 0:
        # With thresholding: use pseudo-inverse form
        # X_sym = U_k s_k^(-1/2) U_k^T (projects onto kept subspace)
        eigenvalues_kept = eigenvalues[keep_mask]
        U_kept = U[:, keep_mask]
        X_sym = U_kept @ np.diag(eigenvalues_kept ** -0.5) @ U_kept.T
    else:
        # No thresholding needed: full S^(-1/2)
        X_sym = U @ np.diag(eigenvalues ** -0.5) @ U.T

    return X_sym, n_dropped


# =============================================================================
# Section 3: Gram-Schmidt Orthogonalization (Pedagogical)
# =============================================================================

def gram_schmidt_s_metric(S: np.ndarray,
                           threshold: float = 1e-12) -> tuple[np.ndarray, int]:
    """
    Build orthogonalizer using Modified Gram-Schmidt under the S-metric.

    This is a PEDAGOGICAL implementation to understand orthogonalization.
    For production code, use canonical/symmetric methods.

    The S-metric inner product is: <u, v>_S = u^T S v
    We orthonormalize the standard basis vectors e_1, ..., e_N.

    Algorithm (Modified Gram-Schmidt):
        for k = 1, ..., N:
            v = e_k
            for j = 1, ..., m:  # m = number of kept vectors so far
                r = x_j^T S v
                v = v - r * x_j
            norm_sq = v^T S v
            if norm_sq > threshold:
                x_{m+1} = v / sqrt(norm_sq)
                m = m + 1

    LIMITATIONS:
    - Order-dependent (reordering basis changes result)
    - Can lose orthogonality for ill-conditioned S
    - May skip valid directions or keep redundant ones

    Args:
        S: Overlap matrix (N x N)
        threshold: Norm-squared cutoff for detecting near-dependence

    Returns:
        X: Orthogonalizer matrix (N x M)
        n_dropped: Number of directions dropped due to near-dependence
    """
    N = S.shape[0]
    X = np.zeros((N, N))
    m = 0  # Number of orthonormal vectors found so far

    for k in range(N):
        # Start with k-th standard basis vector
        v = np.zeros(N)
        v[k] = 1.0

        # Modified Gram-Schmidt: orthogonalize against all previous vectors
        # "Modified" = update v after each subtraction (more stable than classical)
        for j in range(m):
            # Compute projection: r = <x_j, v>_S = x_j^T S v
            r = X[:, j].T @ S @ v
            # Subtract projection
            v = v - r * X[:, j]

        # Compute S-norm squared of residual
        norm_sq = v.T @ S @ v

        if norm_sq < threshold:
            # This direction is (nearly) linearly dependent on previous ones
            continue

        # Normalize and store
        X[:, m] = v / np.sqrt(norm_sq)
        m += 1

    # Return only the m kept columns
    n_dropped = N - m
    return X[:, :m], n_dropped


# =============================================================================
# Section 4: Validation Functions
# =============================================================================

def verify_orthonormalizer(S: np.ndarray, X: np.ndarray,
                            name: str = "X") -> dict:
    """
    Verify that X satisfies X^T S X = I to numerical precision.

    Args:
        S: Overlap matrix
        X: Proposed orthogonalizer
        name: Label for printing

    Returns:
        Dictionary with verification metrics
    """
    M = X.shape[1]  # Number of orthonormal vectors

    # Compute X^T S X
    XtSX = X.T @ S @ X

    # Compare to identity
    I_M = np.eye(M)
    error_matrix = XtSX - I_M

    # Metrics
    frobenius_error = np.linalg.norm(error_matrix, ord='fro')
    max_diag_error = np.max(np.abs(np.diag(XtSX) - 1.0))
    max_offdiag_error = np.max(np.abs(error_matrix - np.diag(np.diag(error_matrix))))

    return {
        "name": name,
        "input_dim": S.shape[0],
        "output_dim": M,
        "frobenius_error": frobenius_error,
        "max_diag_error": max_diag_error,
        "max_offdiag_error": max_offdiag_error,
        "passed": frobenius_error < 1e-10
    }


def compare_orthogonalizers(X_can: np.ndarray, X_sym: np.ndarray,
                             X_gs: Optional[np.ndarray] = None) -> None:
    """Compare properties of different orthogonalizers."""
    print("\nComparison of Orthogonalizer Properties:")
    print("-" * 60)

    # Canonical
    print(f"\nCanonical X_can (shape {X_can.shape}):")
    print(f"  Is symmetric: {np.allclose(X_can, X_can.T)}")
    print(f"  Condition number: {np.linalg.cond(X_can):.3e}")
    print(f"  Frobenius norm: {np.linalg.norm(X_can):.4f}")

    # Symmetric (only if square)
    print(f"\nSymmetric X_sym (shape {X_sym.shape}):")
    print(f"  Is symmetric: {np.allclose(X_sym, X_sym.T)}")
    if X_sym.shape[0] == X_sym.shape[1]:
        print(f"  Condition number: {np.linalg.cond(X_sym):.3e}")
    print(f"  Frobenius norm: {np.linalg.norm(X_sym):.4f}")

    # Gram-Schmidt
    if X_gs is not None:
        print(f"\nGram-Schmidt X_gs (shape {X_gs.shape}):")
        print(f"  Is symmetric: {np.allclose(X_gs, X_gs.T)}")
        print(f"  Condition number: {np.linalg.cond(X_gs):.3e}")
        print(f"  Frobenius norm: {np.linalg.norm(X_gs):.4f}")


# =============================================================================
# Section 5: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 2B demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 2B: Build Orthogonalizer X such that X^T S X = I" + " " * 17 + "*")
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
    # Part 1: Well-conditioned basis (cc-pVDZ)
    # ==========================================================================

    print("=" * 75)
    print("Part 1: Well-Conditioned Basis (cc-pVDZ)")
    print("=" * 75)

    mol_dz = gto.M(
        atom=h2o_geometry,
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    S_dz = mol_dz.intor("int1e_ovlp")
    nao_dz = mol_dz.nao_nr()

    print(f"\nMolecule: H2O")
    print(f"Basis: cc-pVDZ")
    print(f"Number of AOs: {nao_dz}")

    # Compute condition number
    eigs_dz = np.linalg.eigvalsh(S_dz)
    kappa_dz = eigs_dz.max() / eigs_dz.min()
    print(f"Condition number kappa(S) = {kappa_dz:.3e}")
    print(f"Smallest eigenvalue: {eigs_dz.min():.6e}")

    # Build all three orthogonalizers
    print("\n--- Building Orthogonalizers ---")

    X_can_dz, eigs_kept_can, n_drop_can = canonical_orthogonalizer(S_dz, threshold=1e-10)
    print(f"\n1. Canonical: X_can shape = {X_can_dz.shape}, dropped = {n_drop_can}")

    X_sym_dz, n_drop_sym = symmetric_orthogonalizer(S_dz, threshold=1e-10)
    print(f"2. Symmetric: X_sym shape = {X_sym_dz.shape}, dropped = {n_drop_sym}")

    X_gs_dz, n_drop_gs = gram_schmidt_s_metric(S_dz, threshold=1e-12)
    print(f"3. Gram-Schmidt: X_gs shape = {X_gs_dz.shape}, dropped = {n_drop_gs}")

    # Verify each orthogonalizer
    print("\n--- Verification: X^T S X = I ---")

    for X, name in [(X_can_dz, "Canonical"), (X_sym_dz, "Symmetric"), (X_gs_dz, "Gram-Schmidt")]:
        result = verify_orthonormalizer(S_dz, X, name)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{name:15} ||X^T S X - I|| = {result['frobenius_error']:.2e}  [{status}]")

    # Compare properties
    compare_orthogonalizers(X_can_dz, X_sym_dz, X_gs_dz)

    # ==========================================================================
    # Part 2: Ill-conditioned basis (aug-cc-pVDZ)
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 2: Ill-Conditioned Basis (aug-cc-pVDZ with diffuse functions)")
    print("=" * 75)

    mol_aug = gto.M(
        atom=h2o_geometry,
        basis="aug-cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    S_aug = mol_aug.intor("int1e_ovlp")
    nao_aug = mol_aug.nao_nr()

    print(f"\nBasis: aug-cc-pVDZ")
    print(f"Number of AOs: {nao_aug}")

    eigs_aug = np.linalg.eigvalsh(S_aug)
    kappa_aug = eigs_aug.max() / eigs_aug.min()
    print(f"Condition number kappa(S) = {kappa_aug:.3e}")
    print(f"Smallest eigenvalue: {eigs_aug.min():.6e}")

    # Build orthogonalizers with thresholding
    print("\n--- Building Orthogonalizers with Threshold = 1e-8 ---")

    threshold_aug = 1e-8  # More aggressive threshold for ill-conditioned S

    X_can_aug, _, n_drop_can_aug = canonical_orthogonalizer(S_aug, threshold=threshold_aug)
    print(f"\n1. Canonical: X_can shape = {X_can_aug.shape}, dropped = {n_drop_can_aug}")

    X_sym_aug, n_drop_sym_aug = symmetric_orthogonalizer(S_aug, threshold=threshold_aug)
    print(f"2. Symmetric: X_sym shape = {X_sym_aug.shape}, dropped = {n_drop_sym_aug}")

    X_gs_aug, n_drop_gs_aug = gram_schmidt_s_metric(S_aug, threshold=1e-10)
    print(f"3. Gram-Schmidt: X_gs shape = {X_gs_aug.shape}, dropped = {n_drop_gs_aug}")

    # Verify
    print("\n--- Verification ---")
    for X, name in [(X_can_aug, "Canonical"), (X_sym_aug, "Symmetric"), (X_gs_aug, "Gram-Schmidt")]:
        result = verify_orthonormalizer(S_aug, X, name)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{name:15} ||X^T S X - I|| = {result['frobenius_error']:.2e}  [{status}]")

    # ==========================================================================
    # Part 3: Effect of Threshold Choice
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 3: Effect of Eigenvalue Threshold on Dimension")
    print("=" * 75)

    print(f"\nBasis: aug-cc-pVDZ (N = {nao_aug} AOs)")
    print(f"\n{'Threshold':>12} {'Kept':>6} {'Dropped':>8} {'||X^T S X - I||':>18}")
    print("-" * 50)

    thresholds = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

    for thresh in thresholds:
        X_test, _, n_drop = canonical_orthogonalizer(S_aug, threshold=thresh)
        result = verify_orthonormalizer(S_aug, X_test)
        kept = X_test.shape[1]
        print(f"{thresh:>12.0e} {kept:>6} {n_drop:>8} {result['frobenius_error']:>18.2e}")

    # ==========================================================================
    # Part 4: Validation Against scipy.linalg.fractional_matrix_power
    # ==========================================================================

    print()
    print("=" * 75)
    print("Part 4: Validation Against scipy.linalg Reference")
    print("=" * 75)

    from scipy.linalg import fractional_matrix_power

    # Use well-conditioned basis for clean comparison
    S_inv_sqrt_scipy = fractional_matrix_power(S_dz, -0.5)

    # Our symmetric orthogonalizer should match
    X_sym_full, _ = symmetric_orthogonalizer(S_dz, threshold=0)  # No thresholding

    diff = np.linalg.norm(X_sym_full - S_inv_sqrt_scipy)
    print(f"\n||X_sym - scipy S^(-1/2)|| = {diff:.2e}")

    # Verify both satisfy X^T S X = I
    result_ours = verify_orthonormalizer(S_dz, X_sym_full)
    result_scipy = verify_orthonormalizer(S_dz, S_inv_sqrt_scipy)

    print(f"\nOur X_sym:    ||X^T S X - I|| = {result_ours['frobenius_error']:.2e}")
    print(f"scipy S^-1/2: ||X^T S X - I|| = {result_scipy['frobenius_error']:.2e}")

    assert result_ours["passed"], "Our symmetric orthogonalizer failed!"
    assert result_scipy["passed"], "scipy reference failed!"
    print("\n[OK] Both implementations validated!")

    # ==========================================================================
    # Part 5: What You Should Observe
    # ==========================================================================

    print()
    print("=" * 75)
    print("What You Should Observe")
    print("=" * 75)

    observations = """
1. CANONICAL vs SYMMETRIC:
   - Both satisfy X^T S X = I
   - Canonical: X = U s^(-1/2), NOT symmetric
   - Symmetric: X = U s^(-1/2) U^T = S^(-1/2), IS symmetric
   - Symmetric orthogonalization preserves more "character" of original AOs

2. GRAM-SCHMIDT LIMITATIONS:
   - Works well for well-conditioned S
   - For ill-conditioned S:
     * May lose orthogonality (accumulated rounding errors)
     * Order-dependent (different AO ordering gives different X)
     * May drop more or fewer functions than eigenvalue methods

3. THRESHOLDING:
   - Too small threshold (1e-14): Keep near-dependent functions
     -> Orthogonalizer has large elements, numerical instability
   - Too large threshold (1e-4): Drop too many functions
     -> Lose degrees of freedom, may affect results
   - Typical: 1e-8 to 1e-10 is a reasonable balance

4. DIMENSION REDUCTION:
   - When M < N (functions dropped), we work in reduced space
   - The "missing" directions are truly redundant
   - Energy should NOT change (within threshold precision)

5. WHICH TO USE IN PRACTICE:
   - Production codes: Canonical with thresholding (robust, efficient)
   - Understanding: Symmetric (intuitive, close to original basis)
   - Never: Naive Gram-Schmidt (pedagogical only)
"""
    print(observations)

    print()
    print("=" * 75)
    print("Lab 2B Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
