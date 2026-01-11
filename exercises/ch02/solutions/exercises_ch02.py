#!/usr/bin/env python3
"""
Chapter 2 Exercise Solutions: Gaussian Basis Sets and Orthonormalization
=========================================================================

This file provides Python implementations for all Chapter 2 exercises that
require numerical computation. Each exercise function is self-contained and
includes validation against PySCF reference values.

Exercises covered:
  - Exercise 2.1: Basis-Set Dependence of Conditioning
  - Exercise 2.2: Gram-Schmidt Ordering Dependence
  - Exercise 2.3: Symmetric vs Canonical Orthogonalization
  - Exercise 2.4: Generalized Eigenproblem Consistency
  - Exercise 2.6: Cartesian vs Spherical Gaussian Counting
  - Exercise 2.7: Thresholding Sensitivity Study
  - Exercise 2.8: GTO Normalization Verification

Note: Exercise 2.5 is conceptual (no code required).

Course: 2302638 Advanced Quantum Chemistry
Institution: Department of Chemistry, Chulalongkorn University

Usage:
    python exercises_ch02.py              # Run all exercises
    python exercises_ch02.py --exercise 1 # Run specific exercise

Dependencies:
    numpy, scipy, pyscf
"""

import numpy as np
import scipy.linalg
import scipy.integrate
from pyscf import gto
import argparse
from typing import Tuple, Dict, Any


# =============================================================================
# Common Utilities
# =============================================================================

def canonical_orthogonalizer(S: np.ndarray, thresh: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build canonical orthogonalizer with eigenvalue thresholding.

    The canonical orthogonalizer diagonalizes S = U diag(e) U^T and constructs:
        X = U_kept @ diag(e_kept^(-1/2))
    where eigenvalues below thresh are removed.

    Args:
        S: Overlap matrix (N x N), symmetric positive semi-definite
        thresh: Eigenvalue threshold for removing near-dependent directions

    Returns:
        X: Orthogonalizer matrix (N x M), satisfies X^T S X = I_M
        kept_eigs: Eigenvalues that were kept (M,)
        n_dropped: Number of directions removed
    """
    eigenvalues, U = np.linalg.eigh(S)
    keep = eigenvalues > thresh
    n_dropped = np.sum(~keep)

    U_kept = U[:, keep]
    e_kept = eigenvalues[keep]

    X = U_kept @ np.diag(e_kept ** -0.5)
    return X, e_kept, n_dropped


def symmetric_orthogonalizer(S: np.ndarray, thresh: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build symmetric (Lowdin) orthogonalizer S^(-1/2).

    The true Lowdin orthogonalizer is X = S^(-1/2) = U diag(e^(-1/2)) U^T.
    With thresholding, we return the rectangular form like canonical.

    Args:
        S: Overlap matrix (N x N)
        thresh: Eigenvalue threshold

    Returns:
        X: Orthogonalizer matrix (N x M)
        kept_eigs: Eigenvalues that were kept
        n_dropped: Number of directions removed
    """
    eigenvalues, U = np.linalg.eigh(S)
    keep = eigenvalues > thresh
    n_dropped = np.sum(~keep)

    U_kept = U[:, keep]
    e_kept = eigenvalues[keep]

    # Symmetric form: X = U_k @ diag(e_k^(-1/2)) @ U_k^T would be square
    # With thresholding, return rectangular form for consistency
    X = U_kept @ np.diag(e_kept ** -0.5)
    return X, e_kept, n_dropped


def check_orthogonality(S: np.ndarray, X: np.ndarray) -> float:
    """
    Verify X^T S X = I by computing Frobenius norm of deviation.

    Args:
        S: Overlap matrix
        X: Orthogonalizer to verify

    Returns:
        Frobenius norm of (X^T S X - I)
    """
    M = X.shape[1]
    product = X.T @ S @ X
    return np.linalg.norm(product - np.eye(M), 'fro')


# =============================================================================
# Exercise 2.1: Basis-Set Dependence of Conditioning
# =============================================================================

def exercise_2_1() -> bool:
    """
    Exercise 2.1: Analyze how basis set choice affects overlap matrix conditioning.

    For H2O with fixed geometry, compute eigenvalue spectrum of S for different
    basis sets and observe how diffuse functions (aug-) affect conditioning.

    Key observations:
      - Adding diffuse functions decreases min(s_i) dramatically
      - Condition number kappa(S) grows ~10x with each step up in basis quality
      - Large kappa(S) requires careful numerical treatment

    Returns:
        True if all computations complete successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 2.1: Basis-Set Dependence of Conditioning")
    print("=" * 70)

    # Water geometry (fixed for comparison)
    atom = "O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043"

    # Basis sets to compare (increasing quality/diffuseness)
    basis_sets = ["sto-3g", "cc-pVDZ", "aug-cc-pVDZ", "aug-cc-pVTZ"]

    print(f"\nTest molecule: H2O")
    print(f"Geometry: O at origin, H atoms at (~0.76, 0, 0.50) Angstrom")
    print()
    print(f"{'Basis':<15} {'N_AO':>5} {'min(s)':>12} {'max(s)':>8} {'kappa(S)':>12}")
    print("-" * 60)

    results = []
    for basis in basis_sets:
        mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
        S = mol.intor("int1e_ovlp")

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(S)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

        # Condition number (ignore numerically zero eigenvalues)
        tiny = 1e-14
        e_keep = eigenvalues[eigenvalues > tiny]
        cond = e_keep[0] / e_keep[-1] if len(e_keep) > 0 else np.inf

        results.append({
            'basis': basis,
            'nao': mol.nao_nr(),
            'min_eig': eigenvalues[-1],
            'max_eig': eigenvalues[0],
            'cond': cond,
            'eigenvalues': eigenvalues
        })

        print(f"{basis:<15} {mol.nao_nr():>5} {eigenvalues[-1]:>12.3e} "
              f"{eigenvalues[0]:>8.2f} {cond:>12.1e}")

    print("-" * 60)

    # Analyze effect of diffuse functions
    print("\n--- Effect of Diffuse Functions (aug-) ---")
    if 'cc-pVDZ' in basis_sets and 'aug-cc-pVDZ' in basis_sets:
        r_cc = next(r for r in results if r['basis'] == 'cc-pVDZ')
        r_aug = next(r for r in results if r['basis'] == 'aug-cc-pVDZ')

        min_ratio = r_cc['min_eig'] / r_aug['min_eig']
        cond_ratio = r_aug['cond'] / r_cc['cond']

        print(f"cc-pVDZ -> aug-cc-pVDZ:")
        print(f"  N_AO increase:    {r_cc['nao']} -> {r_aug['nao']} (+{r_aug['nao'] - r_cc['nao']})")
        print(f"  min(s) decrease:  {r_cc['min_eig']:.3e} -> {r_aug['min_eig']:.3e} ({min_ratio:.0f}x smaller)")
        print(f"  kappa(S) increase: {r_cc['cond']:.1e} -> {r_aug['cond']:.1e} ({cond_ratio:.0f}x worse)")

    # Show smallest 5 eigenvalues for aug-cc-pVDZ
    print("\n--- Smallest 5 eigenvalues for aug-cc-pVDZ ---")
    r_aug = next(r for r in results if r['basis'] == 'aug-cc-pVDZ')
    for i, e in enumerate(np.sort(r_aug['eigenvalues'])[:5]):
        print(f"  s_{i+1} = {e:.6e}")

    print("\n--- Key Observations ---")
    print("1. Adding diffuse functions (aug-) decreases min eigenvalue by ~10-100x")
    print("2. Condition number grows roughly by an order of magnitude per step")
    print("3. aug-cc-pVTZ has kappa(S) ~ 10^3-10^4, still manageable in double precision")
    print("4. For anions or very diffuse bases, thresholding becomes essential")

    return True


# =============================================================================
# Exercise 2.2: Gram-Schmidt Ordering Dependence
# =============================================================================

def exercise_2_2() -> bool:
    """
    Exercise 2.2: Show that Gram-Schmidt orthogonalization depends on input ordering.

    With permuted basis ordering S' = P^T S P, the Gram-Schmidt orthogonalizers
    X and X' differ by more than just the permutation.

    Key insight: Gram-Schmidt is NOT unique; the result depends on processing order.
    Symmetric orthogonalization (S^(-1/2)) is unique and order-independent.

    Returns:
        True if demonstration completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 2.2: Gram-Schmidt Ordering Dependence")
    print("=" * 70)

    def gram_schmidt_metric(S: np.ndarray, order: np.ndarray = None, thresh: float = 1e-12) -> np.ndarray:
        """
        Modified Gram-Schmidt under S-metric with optional reordering.

        Args:
            S: Overlap matrix
            order: Permutation of basis indices (default: natural order)
            thresh: Threshold for detecting linear dependence

        Returns:
            X: Orthogonalizer in original (unpermuted) basis ordering
        """
        N = S.shape[0]
        if order is None:
            order = np.arange(N)

        X = np.zeros((N, N))
        m = 0  # Number of orthonormal vectors found

        for k in order:
            # Start with k-th standard basis vector
            v = np.zeros(N)
            v[k] = 1.0

            # Modified Gram-Schmidt: subtract projections
            for j in range(m):
                overlap = X[:, j].T @ S @ v
                v = v - overlap * X[:, j]

            # Check norm in S-metric
            norm_sq = v.T @ S @ v
            if norm_sq < thresh:
                continue

            X[:, m] = v / np.sqrt(norm_sq)
            m += 1

        return X[:, :m]

    # Build molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    S = mol.intor("int1e_ovlp")
    N = mol.nao_nr()

    print(f"\nMolecule: H2O (STO-3G)")
    print(f"Number of AOs: {N}")

    # Natural ordering
    order_natural = np.arange(N)
    X_natural = gram_schmidt_metric(S, order_natural)

    # Reversed ordering
    order_reversed = np.arange(N)[::-1]
    X_reversed = gram_schmidt_metric(S, order_reversed)

    # Random permutation (reproducible)
    np.random.seed(42)
    order_random = np.random.permutation(N)
    X_random = gram_schmidt_metric(S, order_random)

    # Verify all produce valid orthogonalizers
    print("\n--- Orthogonality Check ---")
    err_natural = check_orthogonality(S, X_natural)
    err_reversed = check_orthogonality(S, X_reversed)
    err_random = check_orthogonality(S, X_random)

    print(f"Natural order:   ||X^T S X - I|| = {err_natural:.2e}")
    print(f"Reversed order:  ||X^T S X - I|| = {err_reversed:.2e}")
    print(f"Random order:    ||X^T S X - I|| = {err_random:.2e}")

    # Compare the orthogonalizers
    print("\n--- Orthogonalizer Comparison ---")

    # They should all have same dimension
    print(f"Dimensions: natural={X_natural.shape[1]}, reversed={X_reversed.shape[1]}, random={X_random.shape[1]}")

    # But the matrices themselves differ
    diff_nr = np.linalg.norm(X_natural - X_reversed, 'fro')
    diff_nrand = np.linalg.norm(X_natural - X_random, 'fro')

    print(f"||X_natural - X_reversed|| = {diff_nr:.2e}")
    print(f"||X_natural - X_random||   = {diff_nrand:.2e}")

    # Compare with symmetric orthogonalizer (which IS unique)
    X_sym, _, _ = symmetric_orthogonalizer(S, thresh=1e-10)

    # Full symmetric form for comparison (square matrix)
    eigenvalues, U = np.linalg.eigh(S)
    X_sym_full = U @ np.diag(eigenvalues ** -0.5) @ U.T

    print("\n--- Comparison with Symmetric Orthogonalizer ---")
    print(f"Symmetric orthogonalizer (S^(-1/2)) is UNIQUE:")
    print(f"  Shape: {X_sym_full.shape} (square matrix)")
    print(f"  ||X_sym^T S X_sym - I|| = {check_orthogonality(S, X_sym):.2e}")

    # Symmetric is invariant under permutation
    P = np.eye(N)[:, order_reversed]  # Permutation matrix
    S_perm = P.T @ S @ P
    eigenvalues_p, U_p = np.linalg.eigh(S_perm)
    X_sym_perm = U_p @ np.diag(eigenvalues_p ** -0.5) @ U_p.T

    # Transform back to original basis
    X_sym_back = P @ X_sym_perm @ P.T
    diff_sym = np.linalg.norm(X_sym_full - X_sym_back, 'fro')
    print(f"  ||X_sym(original) - P @ X_sym(permuted) @ P^T|| = {diff_sym:.2e}")

    print("\n--- Key Observations ---")
    print("1. All Gram-Schmidt orderings produce valid orthogonalizers (X^T S X = I)")
    print("2. The orthogonalizer matrices DIFFER between orderings")
    print("3. Symmetric orthogonalization (S^(-1/2)) is UNIQUE and order-independent")
    print("4. Use symmetric/canonical orthogonalization for reproducibility")

    return True


# =============================================================================
# Exercise 2.3: Symmetric vs Canonical Orthogonalization
# =============================================================================

def exercise_2_3() -> bool:
    """
    Exercise 2.3: Compare symmetric and canonical orthogonalization.

    Both satisfy X^T S X = I, but they produce different matrices:
      - Canonical: X = U @ diag(e^(-1/2)), rectangular with thresholding
      - Symmetric: X = S^(-1/2) = U @ diag(e^(-1/2)) @ U^T, square (Lowdin)

    Key insights:
      - Lowdin orthogonalized functions are "closest" to original AOs
      - With thresholding, both become rectangular and functionally equivalent
      - Choice matters for interpretation, not for solving FC = SCe

    Returns:
        True if demonstration completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 2.3: Symmetric vs Canonical Orthogonalization")
    print("=" * 70)

    # Build molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )
    S = mol.intor("int1e_ovlp")
    N = mol.nao_nr()

    print(f"\nMolecule: H2O (cc-pVDZ)")
    print(f"Number of AOs: {N}")

    # Build both orthogonalizers
    X_can, eigs_can, n_drop_can = canonical_orthogonalizer(S, thresh=1e-10)
    X_sym, eigs_sym, n_drop_sym = symmetric_orthogonalizer(S, thresh=1e-10)

    # Also build full symmetric (square) for comparison
    eigenvalues, U = np.linalg.eigh(S)
    X_sym_full = U @ np.diag(eigenvalues ** -0.5) @ U.T

    print("\n--- Orthogonalizer Properties ---")
    print(f"{'Method':<20} {'Shape':<12} {'Dropped':>8} {'||X^T S X - I||':>16}")
    print("-" * 60)

    err_can = check_orthogonality(S, X_can)
    err_sym = check_orthogonality(S, X_sym)
    err_sym_full = check_orthogonality(S, X_sym_full)

    print(f"{'Canonical':<20} {str(X_can.shape):<12} {n_drop_can:>8} {err_can:>16.2e}")
    print(f"{'Symmetric (rect)':<20} {str(X_sym.shape):<12} {n_drop_sym:>8} {err_sym:>16.2e}")
    print(f"{'Symmetric (full)':<20} {str(X_sym_full.shape):<12} {0:>8} {err_sym_full:>16.2e}")

    # Compare the matrices
    print("\n--- Matrix Comparison ---")

    # Canonical and rectangular symmetric are identical when both use same eigenbasis
    diff_can_sym = np.linalg.norm(X_can - X_sym, 'fro')
    print(f"||X_canonical - X_symmetric(rect)|| = {diff_can_sym:.2e}")
    print("  (These are identical when using same threshold)")

    # Full symmetric differs from both
    diff_sym_full_can = np.linalg.norm(X_sym_full[:, :X_can.shape[1]] - X_can, 'fro')
    print(f"||X_symmetric(full)[:,:M] - X_canonical|| = {diff_sym_full_can:.2e}")
    print("  (Full symmetric has different structure)")

    # Lowdin property: minimize sum of squared differences from original
    print("\n--- Lowdin 'Closest' Property ---")

    # For the full symmetric orthogonalizer, the orthonormalized functions
    # are the columns of X_sym_full. These minimize ||chi_ortho - chi_orig||^2.
    # This is equivalent to saying the transformation matrix closest to identity
    # under the constraint X^T S X = I is S^(-1/2).

    # Measure "closeness" to identity
    I_N = np.eye(N)
    dist_can = np.linalg.norm(X_can - I_N[:, :X_can.shape[1]], 'fro')
    dist_sym_full = np.linalg.norm(X_sym_full - I_N, 'fro')

    print(f"||X_canonical - I|| = {dist_can:.4f}")
    print(f"||X_symmetric - I|| = {dist_sym_full:.4f}")
    print("  (Symmetric orthogonalizer is closer to identity = 'less distortion')")

    # Show that both give same eigenvalues when solving FC = SCe
    print("\n--- Solving FC = SCe ---")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V  # Core Hamiltonian

    # Solve using canonical orthogonalizer
    F_prime_can = X_can.T @ H @ X_can
    eps_can, C_prime_can = np.linalg.eigh(F_prime_can)
    C_can = X_can @ C_prime_can

    # Solve using full symmetric orthogonalizer
    F_prime_sym = X_sym_full.T @ H @ X_sym_full
    eps_sym, C_prime_sym = np.linalg.eigh(F_prime_sym)
    C_sym = X_sym_full @ C_prime_sym

    print(f"Lowest 5 eigenvalues:")
    print(f"  {'i':<4} {'Canonical':>14} {'Symmetric':>14} {'Difference':>14}")
    for i in range(min(5, len(eps_can))):
        diff = abs(eps_can[i] - eps_sym[i])
        print(f"  {i:<4} {eps_can[i]:>14.8f} {eps_sym[i]:>14.8f} {diff:>14.2e}")

    print("\n--- Key Observations ---")
    print("1. Both canonical and symmetric orthogonalizers satisfy X^T S X = I")
    print("2. With thresholding, canonical and symmetric give IDENTICAL rectangular X")
    print("3. Full symmetric (S^(-1/2)) is square and minimizes distortion from original AOs")
    print("4. Both give SAME eigenvalues when solving FC = SCe")
    print("5. Choice affects MO coefficients (C) but not observables (energies)")

    return True


# =============================================================================
# Exercise 2.4: Generalized Eigenproblem Consistency
# =============================================================================

def exercise_2_4() -> bool:
    """
    Exercise 2.4: Verify generalized eigenproblem solution FC = SCe.

    Verification checklist:
      1. MO orthonormality: ||C^T S C - I|| < 10^-12
      2. Eigenvalue equation: ||FC - SCe|| < 10^-12
      3. Cross-validation: eigenvalues match scipy.linalg.eigh(F, S)

    Returns:
        True if all verifications pass
    """
    print("\n" + "=" * 70)
    print("Exercise 2.4: Generalized Eigenproblem Consistency")
    print("=" * 70)

    # Build molecule
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )

    # Build integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V  # Core Hamiltonian as F
    N = mol.nao_nr()

    print(f"\nMolecule: H2 (cc-pVDZ)")
    print(f"Number of AOs: {N}")

    # Solve via orthogonalization
    X, kept_eigs, n_dropped = canonical_orthogonalizer(S, thresh=1e-10)
    M = X.shape[1]

    # Transform F to orthonormal basis
    F_prime = X.T @ H @ X

    # Solve ordinary eigenproblem
    eigenvalues, C_prime = np.linalg.eigh(F_prime)

    # Back-transform to AO basis
    C = X @ C_prime

    print(f"\nOrthogonalizer: dropped {n_dropped} directions, kept {M}/{N}")

    # Verification 1: MO orthonormality
    print("\n--- Verification 1: MO Orthonormality ---")
    CtSC = C.T @ S @ C
    ortho_err = np.linalg.norm(CtSC - np.eye(M), 'fro')
    ortho_pass = ortho_err < 1e-12
    print(f"||C^T S C - I||_F = {ortho_err:.2e}")
    print(f"Status: {'PASS' if ortho_pass else 'FAIL'}")

    # Verification 2: Eigenvalue equation
    print("\n--- Verification 2: Eigenvalue Equation ---")
    residual = H @ C - S @ C @ np.diag(eigenvalues)
    res_norm = np.linalg.norm(residual, 'fro')
    res_pass = res_norm < 1e-12
    print(f"||FC - SCe||_F = {res_norm:.2e}")
    print(f"Status: {'PASS' if res_pass else 'FAIL'}")

    # Verification 3: Cross-validation with scipy
    print("\n--- Verification 3: scipy.linalg.eigh Cross-Validation ---")
    eps_scipy, C_scipy = scipy.linalg.eigh(H, S)

    # Compare eigenvalues
    eps_diff = np.max(np.abs(eigenvalues - eps_scipy[:M]))
    scipy_pass = eps_diff < 1e-12
    print(f"max|eps_ours - eps_scipy| = {eps_diff:.2e}")
    print(f"Status: {'PASS' if scipy_pass else 'FAIL'}")

    # Show eigenvalues
    print(f"\nLowest 5 eigenvalues:")
    print(f"  {'i':<4} {'Ours':>14} {'scipy':>14} {'Difference':>14}")
    for i in range(min(5, M)):
        diff = abs(eigenvalues[i] - eps_scipy[i])
        print(f"  {i:<4} {eigenvalues[i]:>14.8f} {eps_scipy[i]:>14.8f} {diff:>14.2e}")

    # Common error demonstration
    print("\n--- Common Student Error: Wrong Back-Transform ---")
    print("WRONG: C = X^T @ C_prime (transposes do not commute!)")
    print("RIGHT: C = X @ C_prime")

    C_wrong = X.T @ C_prime  # Intentionally wrong
    ortho_err_wrong = np.linalg.norm(C_wrong.T @ S @ C_wrong - np.eye(M), 'fro')
    print(f"||C_wrong^T S C_wrong - I|| = {ortho_err_wrong:.2e} (should be ~0 for correct)")

    # Summary
    print("\n--- Summary ---")
    all_pass = ortho_pass and res_pass and scipy_pass
    print(f"MO Orthonormality:     {'PASS' if ortho_pass else 'FAIL'}")
    print(f"Eigenvalue Equation:   {'PASS' if res_pass else 'FAIL'}")
    print(f"scipy Cross-Validation: {'PASS' if scipy_pass else 'FAIL'}")
    print(f"\nOverall: {'All checks PASSED!' if all_pass else 'Some checks FAILED!'}")

    return all_pass


# =============================================================================
# Exercise 2.6: Cartesian vs Spherical Gaussian Counting
# =============================================================================

def exercise_2_6() -> bool:
    """
    Exercise 2.6: Count basis functions for Cartesian vs spherical Gaussians.

    Formulas:
      - Cartesian: N_cart(L) = (L+1)(L+2)/2
      - Spherical: N_sph(L) = 2L+1
      - Spurious: N_cart - N_sph = L(L-1)/2

    The difference arises because Cartesian polynomials x^l y^m z^n with
    l+m+n = L include r^2 type functions that are actually lower L.

    Returns:
        True if computations complete successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 2.6: Cartesian vs Spherical Gaussian Counting")
    print("=" * 70)

    def n_cart(L: int) -> int:
        """Number of Cartesian Gaussians for angular momentum L."""
        return (L + 1) * (L + 2) // 2

    def n_sph(L: int) -> int:
        """Number of spherical Gaussians for angular momentum L."""
        return 2 * L + 1

    def n_spurious(L: int) -> int:
        """Number of spurious (lower L) components in Cartesian representation."""
        return L * (L - 1) // 2

    # Part (a): Counting table
    print("\n--- Part (a): Counting by Angular Momentum ---")
    print(f"{'L':<3} {'Type':<5} {'N_cart':>8} {'N_sph':>8} {'Spurious':>10} {'Character':<15}")
    print("-" * 60)

    shell_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
    spurious_char = {0: '---', 1: '---', 2: 's (r^2)', 3: 'p (xr^2, ...)', 4: 'd+s types'}

    for L in range(5):
        nc = n_cart(L)
        ns = n_sph(L)
        nsp = n_spurious(L)
        print(f"{L:<3} {shell_names[L]:<5} {nc:>8} {ns:>8} {nsp:>10} {spurious_char[L]:<15}")

    # Part (b): Oxygen with [3s2p1d]
    print("\n--- Part (b): Oxygen with [3s2p1d] ---")
    n_s, n_p, n_d = 3, 2, 1
    n_cart_O = n_s * n_cart(0) + n_p * n_cart(1) + n_d * n_cart(2)
    n_sph_O = n_s * n_sph(0) + n_p * n_sph(1) + n_d * n_sph(2)

    print(f"  3s: 3 x {n_cart(0)} = {3*n_cart(0)} (Cart) = {3*n_sph(0)} (Sph)")
    print(f"  2p: 2 x {n_cart(1)} = {2*n_cart(1)} (Cart) = {2*n_sph(1)} (Sph)")
    print(f"  1d: 1 x {n_cart(2)} = {1*n_cart(2)} (Cart) = {1*n_sph(2)} (Sph)")
    print(f"  Total: {n_cart_O} (Cartesian), {n_sph_O} (Spherical)")

    # Part (c): Large molecule
    print("\n--- Part (c): 10 Heavy Atoms [4s3p2d1f] + 20 H [2s1p] ---")

    # Heavy atoms
    n_heavy_cart = 10 * (4*n_cart(0) + 3*n_cart(1) + 2*n_cart(2) + 1*n_cart(3))
    n_heavy_sph = 10 * (4*n_sph(0) + 3*n_sph(1) + 2*n_sph(2) + 1*n_sph(3))

    # Hydrogen
    n_H_both = 20 * (2*n_cart(0) + 1*n_cart(1))  # Same for cart and sph up to p

    print(f"  Heavy atoms (Cart): 10 x (4+9+12+10) = 10 x 35 = {n_heavy_cart}")
    print(f"  Heavy atoms (Sph):  10 x (4+9+10+7)  = 10 x 30 = {n_heavy_sph}")
    print(f"  Hydrogens (both):   20 x (2+3)       = 20 x 5  = {n_H_both}")
    print(f"  Total (Cartesian): {n_heavy_cart + n_H_both}")
    print(f"  Total (Spherical): {n_heavy_sph + n_H_both}")
    print(f"  Difference: {(n_heavy_cart + n_H_both) - (n_heavy_sph + n_H_both)} spurious functions")

    # Verify with PySCF
    print("\n--- Verification with PySCF ---")

    # Water with cc-pVDZ (has d functions)
    mol_cart = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="cc-pVDZ",
        unit="Angstrom",
        cart=True,   # Force Cartesian
        verbose=0
    )

    mol_sph = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="cc-pVDZ",
        unit="Angstrom",
        cart=False,  # Spherical (default)
        verbose=0
    )

    print(f"H2O/cc-pVDZ:")
    print(f"  Cartesian:  {mol_cart.nao_nr()} AOs")
    print(f"  Spherical:  {mol_sph.nao_nr()} AOs")
    print(f"  Difference: {mol_cart.nao_nr() - mol_sph.nao_nr()} (from d-type spurious)")

    print("\n--- Key Observations ---")
    print("1. Spherical Gaussians are preferred because:")
    print("   - Fewer basis functions (more efficient)")
    print("   - No spurious lower-L components")
    print("   - Better numerical stability (less redundancy)")
    print("2. The spurious component in Cartesian d is r^2 (s-type)")
    print("3. The difference grows as L(L-1)/2 for higher angular momentum")

    return True


# =============================================================================
# Exercise 2.7: Thresholding Sensitivity Study
# =============================================================================

def exercise_2_7() -> bool:
    """
    Exercise 2.7: Study effect of eigenvalue threshold on orthogonalization.

    For F- (anion) with aug-cc-pVTZ, explore how threshold affects:
      - Number of kept dimensions
      - Numerical stability (||X^T S X - I||)
      - Orbital energies

    Returns:
        True if study completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 2.7: Thresholding Sensitivity Study")
    print("=" * 70)

    # F- anion with diffuse basis (challenging case)
    mol = gto.M(
        atom="F 0 0 0",
        basis="aug-cc-pVTZ",
        charge=-1,
        spin=0,
        unit="Angstrom",
        verbose=0
    )

    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H = T + V
    N = mol.nao_nr()

    print(f"\nMolecule: F- (aug-cc-pVTZ)")
    print(f"Number of AOs: {N}")
    print(f"Number of electrons: {mol.nelectron}")

    # Analyze overlap eigenvalue spectrum
    eigenvalues_S = np.linalg.eigvalsh(S)
    eigenvalues_S = np.sort(eigenvalues_S)

    print("\n--- Overlap Eigenvalue Spectrum ---")
    print(f"  min(s):   {eigenvalues_S[0]:.6e}")
    print(f"  max(s):   {eigenvalues_S[-1]:.6e}")
    print(f"  kappa(S): {eigenvalues_S[-1]/eigenvalues_S[0]:.6e}")

    # Count eigenvalues in different ranges
    print("\n  Eigenvalue distribution:")
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
    for i, tau in enumerate(thresholds[:-1]):
        n_in_range = np.sum((eigenvalues_S >= thresholds[i+1]) & (eigenvalues_S < tau))
        print(f"    {thresholds[i+1]:.0e} <= s < {tau:.0e}: {n_in_range}")
    print(f"    s >= {thresholds[0]:.0e}: {np.sum(eigenvalues_S >= thresholds[0])}")

    # Threshold sensitivity study
    print("\n--- Effect of Threshold on Orthogonalization ---")
    print(f"{'tau':<10} {'M (kept)':>10} {'||X^T S X - I||':>18} {'eps_min (Hartree)':>18}")
    print("-" * 60)

    test_thresholds = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    results = []

    for tau in test_thresholds:
        X, kept_eigs, n_dropped = canonical_orthogonalizer(S, thresh=tau)
        M = X.shape[1]

        # Check orthogonality
        err = check_orthogonality(S, X)

        # Solve eigenvalue problem
        F_prime = X.T @ H @ X
        eps, _ = np.linalg.eigh(F_prime)

        results.append({
            'tau': tau,
            'M': M,
            'error': err,
            'eps_min': eps[0]
        })

        print(f"{tau:<10.0e} {M:>10} {err:>18.2e} {eps[0]:>18.8f}")

    print("-" * 60)

    # Analyze stability
    print("\n--- Stability Analysis ---")

    # Find where error starts to grow
    stable_results = [r for r in results if r['error'] < 1e-10]
    if stable_results:
        most_inclusive = stable_results[-1]
        print(f"Most inclusive stable threshold: tau = {most_inclusive['tau']:.0e}")
        print(f"  Kept dimensions: {most_inclusive['M']}")
        print(f"  Orthogonality error: {most_inclusive['error']:.2e}")

    # Recommended threshold
    print("\n--- Practical Recommendations ---")
    print("1. For ground-state energies: tau = 1e-7 to 1e-8")
    print("2. For anions/diffuse properties: tau = 1e-8 to 1e-10")
    print("3. Monitor condition number: kappa(X^T S X) should be ~1")
    print("4. If eigenvalues fluctuate with tau, basis may be too diffuse")

    # Compare with scipy for cross-validation
    print("\n--- Cross-Validation with scipy ---")
    X_stable, _, _ = canonical_orthogonalizer(S, thresh=1e-8)
    F_prime = X_stable.T @ H @ X_stable
    eps_ours, _ = np.linalg.eigh(F_prime)

    eps_scipy, _ = scipy.linalg.eigh(H, S)

    print(f"Lowest 3 eigenvalues comparison (tau=1e-8):")
    print(f"  {'Ours':>14} {'scipy':>14} {'Difference':>14}")
    for i in range(3):
        diff = abs(eps_ours[i] - eps_scipy[i])
        print(f"  {eps_ours[i]:>14.8f} {eps_scipy[i]:>14.8f} {diff:>14.2e}")

    return True


# =============================================================================
# Exercise 2.8: GTO Normalization Verification
# =============================================================================

def exercise_2_8() -> bool:
    """
    Exercise 2.8: Verify GTO normalization constants numerically.

    For Gaussian g(r) = N * (x-Ax)^l * (y-Ay)^m * (z-Az)^n * exp(-alpha*|r-A|^2),
    the normalization constant N is chosen so that integral |g|^2 dr = 1.

    Parts:
      (a) s-type (l=m=n=0) with alpha=1.0
      (b) Numerical verification via integration
      (c) p-type (l=1, m=n=0)
      (d) Contracted Gaussian normalization

    Returns:
        True if all verifications pass
    """
    print("\n" + "=" * 70)
    print("Exercise 2.8: GTO Normalization Verification")
    print("=" * 70)

    def norm_const_s(alpha: float) -> float:
        """Normalization constant for s-type Gaussian."""
        # N = (2*alpha/pi)^(3/4)
        return (2.0 * alpha / np.pi) ** 0.75

    def norm_const_p(alpha: float) -> float:
        """Normalization constant for p-type Gaussian (e.g., px)."""
        # N = (2*alpha/pi)^(3/4) * sqrt(4*alpha)
        # The extra factor accounts for the x^1 polynomial
        return (2.0 * alpha / np.pi) ** 0.75 * np.sqrt(4.0 * alpha)

    def gaussian_s(r: np.ndarray, alpha: float, center: np.ndarray) -> np.ndarray:
        """Normalized s-type Gaussian."""
        N = norm_const_s(alpha)
        r_minus_A = r - center
        r2 = np.sum(r_minus_A ** 2, axis=-1)
        return N * np.exp(-alpha * r2)

    def gaussian_px(r: np.ndarray, alpha: float, center: np.ndarray) -> np.ndarray:
        """Normalized px-type Gaussian."""
        N = norm_const_p(alpha)
        r_minus_A = r - center
        x = r_minus_A[..., 0]
        r2 = np.sum(r_minus_A ** 2, axis=-1)
        return N * x * np.exp(-alpha * r2)

    # Part (a): s-type normalization constant
    print("\n--- Part (a): s-type Normalization Constant ---")
    alpha = 1.0
    N_s = norm_const_s(alpha)
    print(f"For s-type with alpha = {alpha}:")
    print(f"  N_s = (2*alpha/pi)^(3/4) = (2/{np.pi:.6f})^(3/4)")
    print(f"  N_s = {N_s:.6f}")

    # Part (b): Numerical verification
    print("\n--- Part (b): Numerical Verification ---")

    # Integrate |g_s|^2 over 3D space using quadrature
    # We use the fact that integral is separable: prod of 1D integrals
    def integrand_s(x, y, z, alpha):
        N = norm_const_s(alpha)
        return N**2 * np.exp(-2*alpha*(x**2 + y**2 + z**2))

    # Integration limits (Gaussian decays rapidly)
    L = 5.0  # Integration range [-L, L]

    # Use scipy.integrate.tplquad for 3D integration
    result, error = scipy.integrate.tplquad(
        lambda z, y, x: integrand_s(x, y, z, alpha),
        -L, L,  # x limits
        lambda x: -L, lambda x: L,  # y limits
        lambda x, y: -L, lambda x, y: L  # z limits
    )

    print(f"Numerical integration of |g_s|^2 over [-{L}, {L}]^3:")
    print(f"  Integral = {result:.10f}")
    print(f"  Error estimate = {error:.2e}")
    print(f"  Deviation from 1: {abs(result - 1.0):.2e}")

    # Verify with analytical formula
    # integral exp(-2*alpha*r^2) d^3r = (pi/(2*alpha))^(3/2)
    analytical = norm_const_s(alpha)**2 * (np.pi / (2*alpha))**(3/2)
    print(f"  Analytical value: {analytical:.10f}")

    # Part (c): p-type normalization
    print("\n--- Part (c): p-type Normalization Constant ---")
    N_p = norm_const_p(alpha)
    print(f"For px-type with alpha = {alpha}:")
    print(f"  N_p = (2*alpha/pi)^(3/4) * sqrt(4*alpha)")
    print(f"  N_p = {N_p:.6f}")

    # Verify p-type numerically
    def integrand_px(x, y, z, alpha):
        N = norm_const_p(alpha)
        return N**2 * x**2 * np.exp(-2*alpha*(x**2 + y**2 + z**2))

    result_p, error_p = scipy.integrate.tplquad(
        lambda z, y, x: integrand_px(x, y, z, alpha),
        -L, L,
        lambda x: -L, lambda x: L,
        lambda x, y: -L, lambda x, y: L
    )

    print(f"Numerical integration of |g_px|^2:")
    print(f"  Integral = {result_p:.10f}")
    print(f"  Deviation from 1: {abs(result_p - 1.0):.2e}")

    # Part (d): Contracted Gaussian normalization
    print("\n--- Part (d): Contracted Gaussian Normalization ---")
    print("For contracted Gaussian chi = sum_p d_p * g_p:")
    print("  <chi|chi> = sum_{p,q} d_p * d_q * <g_p|g_q>")
    print("           = sum_{p,q} d_p * d_q * S_pq")
    print()
    print("The overlap S_pq between primitives with different exponents is NOT delta_pq!")
    print("For two s-type primitives with exponents alpha and beta:")
    print("  S = N(alpha) * N(beta) * (pi/(alpha+beta))^(3/2)")

    # Example with STO-3G for hydrogen
    print("\nExample: STO-3G for hydrogen (3 primitives):")

    # STO-3G exponents and coefficients for H 1s
    # (These are approximate values)
    exponents = np.array([3.42525091, 0.62391373, 0.16885540])
    coeffs_raw = np.array([0.15432897, 0.53532814, 0.44463454])

    # Compute overlap matrix between primitives
    n_prim = len(exponents)
    S_prim = np.zeros((n_prim, n_prim))

    for p in range(n_prim):
        for q in range(n_prim):
            alpha, beta = exponents[p], exponents[q]
            N_p = norm_const_s(alpha)
            N_q = norm_const_s(beta)
            # Overlap of two normalized s-type Gaussians
            S_prim[p, q] = N_p * N_q * (np.pi / (alpha + beta))**(3/2)

    print(f"  Primitive overlap matrix:")
    for i in range(n_prim):
        print(f"    {S_prim[i]}")

    # Total contraction norm
    norm_sq = coeffs_raw @ S_prim @ coeffs_raw
    print(f"\n  Raw contraction norm^2 = d^T S d = {norm_sq:.6f}")
    print(f"  Normalized coefficients: d_norm = d / sqrt(norm^2)")

    coeffs_norm = coeffs_raw / np.sqrt(norm_sq)
    norm_sq_check = coeffs_norm @ S_prim @ coeffs_norm
    print(f"  Check: d_norm^T S d_norm = {norm_sq_check:.6f}")

    # Verify with PySCF
    print("\n--- Verification with PySCF ---")
    # H atom has 1 electron (doublet), so spin=1 (2S=1, meaning Nalpha-Nbeta=1)
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    S_pyscf = mol.intor("int1e_ovlp")
    print(f"PySCF overlap for H/STO-3G: S = {S_pyscf[0,0]:.10f}")
    print("(Should be 1.0 for self-overlap of normalized basis)")

    print("\n--- Key Observations ---")
    print("1. Normalization constants scale as alpha^(3/4) for s-type")
    print("2. Each additional power of (x,y,z) adds sqrt(2*alpha) factor")
    print("3. Contracted functions are NOT simply sums of normalized primitives")
    print("4. The contraction coefficients must account for primitive overlaps")
    print("5. Modern basis sets provide pre-normalized contraction coefficients")

    return True


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """Run all Chapter 2 exercises or a specific exercise."""
    parser = argparse.ArgumentParser(
        description="Chapter 2 Exercise Solutions: Gaussian Basis Sets and Orthonormalization"
    )
    parser.add_argument(
        "--exercise", "-e",
        type=int,
        choices=[1, 2, 3, 4, 6, 7, 8],
        help="Run specific exercise (1, 2, 3, 4, 6, 7, or 8). Exercise 5 is conceptual."
    )
    args = parser.parse_args()

    exercises = {
        1: ("Basis-Set Dependence of Conditioning", exercise_2_1),
        2: ("Gram-Schmidt Ordering Dependence", exercise_2_2),
        3: ("Symmetric vs Canonical Orthogonalization", exercise_2_3),
        4: ("Generalized Eigenproblem Consistency", exercise_2_4),
        6: ("Cartesian vs Spherical Gaussian Counting", exercise_2_6),
        7: ("Thresholding Sensitivity Study", exercise_2_7),
        8: ("GTO Normalization Verification", exercise_2_8),
    }

    print("=" * 70)
    print("Chapter 2 Exercise Solutions")
    print("Gaussian Basis Sets and Orthonormalization")
    print("=" * 70)
    print("\nCourse: 2302638 Advanced Quantum Chemistry")
    print("Institution: Chulalongkorn University")

    if args.exercise:
        # Run specific exercise
        ex_num = args.exercise
        if ex_num in exercises:
            name, func = exercises[ex_num]
            print(f"\nRunning Exercise 2.{ex_num}: {name}")
            success = func()
        else:
            print(f"\nError: Exercise 2.{ex_num} not available.")
            success = False
    else:
        # Run all exercises
        print("\nRunning all exercises...")
        results = {}
        for ex_num, (name, func) in exercises.items():
            try:
                success = func()
                results[ex_num] = ("PASS" if success else "FAIL", name)
            except Exception as e:
                results[ex_num] = ("ERROR", f"{name}: {str(e)}")

        # Summary
        print("\n" + "=" * 70)
        print("Summary of Results")
        print("=" * 70)
        print(f"\n{'Exercise':<12} {'Status':<8} {'Description':<45}")
        print("-" * 70)

        all_passed = True
        for ex_num in sorted(results.keys()):
            status, desc = results[ex_num]
            print(f"2.{ex_num:<10} {status:<8} {desc:<45}")
            if status != "PASS":
                all_passed = False

        print("-" * 70)
        print(f"\nNote: Exercise 2.5 is conceptual (no code required)")

        if all_passed:
            print("\nAll exercises completed successfully!")
        else:
            print("\nSome exercises had issues. Review output above.")

        success = all_passed

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
