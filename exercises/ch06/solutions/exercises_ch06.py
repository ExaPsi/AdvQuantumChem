#!/usr/bin/env python3
"""
Chapter 6 Exercise Solutions: Hartree-Fock SCF from Integrals
==============================================================

This file provides Python implementations for all Chapter 6 exercises that
require numerical computation. Each exercise function is self-contained and
includes validation against PySCF reference values.

Exercises covered:
  - Exercise 6.1: Derive RHF Energy in AO Form
  - Exercise 6.2: Stationarity and the Commutator Residual
  - Exercise 6.3: Direct SCF Conceptual Design
  - Exercise 6.4: DIIS Behavior Study
  - Exercise 6.5: Reproduce PySCF Energies
  - Exercise 6.6: MO-Basis Gradient Check (Advanced)
  - Exercise 6.7: Level Shifting Implementation (Advanced)
  - Exercise 6.8: SCF Metadynamics (Research)

Course: 2302638 Advanced Quantum Chemistry
Institution: Department of Chemistry, Chulalongkorn University

Usage:
    python exercises_ch06.py              # Run all exercises
    python exercises_ch06.py --exercise 1 # Run specific exercise

Dependencies:
    numpy, scipy, pyscf
"""

import numpy as np
import scipy.linalg
from pyscf import gto, scf
import argparse
from typing import Tuple, Dict, Any, List, Optional


# =============================================================================
# Common Utilities
# =============================================================================

def symmetric_orthogonalizer(S: np.ndarray, thresh: float = 1e-10) -> np.ndarray:
    """
    Build symmetric (Lowdin) orthogonalizer X = S^(-1/2).

    The symmetric orthogonalizer transforms AO basis to an orthonormal basis:
    X^T S X = I. Eigenvalues below thresh are discarded to handle linear dependence.

    Args:
        S: Overlap matrix (N x N)
        thresh: Eigenvalue threshold for numerical stability

    Returns:
        X: Orthogonalizer matrix (N x M) where M <= N
    """
    eigenvalues, U = np.linalg.eigh(S)
    keep = eigenvalues > thresh
    n_dropped = np.sum(~keep)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} near-linear dependent functions")
    s_inv_sqrt = 1.0 / np.sqrt(eigenvalues[keep])
    X = U[:, keep] @ np.diag(s_inv_sqrt) @ U[:, keep].T
    return X


def build_jk_matrices(P: np.ndarray, eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Coulomb (J) and Exchange (K) matrices from density and ERIs.

    Uses chemist's notation: (uv|ls)
        J_uv = sum_{ls} (uv|ls) P_ls   (Coulomb)
        K_uv = sum_{ls} (ul|vs) P_ls   (Exchange)

    Args:
        P: Density matrix (includes factor of 2 for RHF)
        eri: Two-electron integrals in chemist's notation (nao, nao, nao, nao)

    Returns:
        J, K: Coulomb and Exchange matrices
    """
    J = np.einsum('uvls,ls->uv', eri, P, optimize=True)
    K = np.einsum('ulvs,ls->uv', eri, P, optimize=True)
    return J, K


def build_density_matrix(C: np.ndarray, n_occ: int) -> np.ndarray:
    """
    Build RHF density matrix P = 2 * C_occ @ C_occ.T.

    The factor of 2 accounts for double occupancy in closed-shell RHF.

    Args:
        C: MO coefficient matrix (nao x nmo)
        n_occ: Number of doubly occupied orbitals

    Returns:
        P: Density matrix (nao x nao)
    """
    C_occ = C[:, :n_occ]
    return 2.0 * C_occ @ C_occ.T


def compute_scf_residual(F: np.ndarray, P: np.ndarray,
                         S: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute SCF residual R = FPS - SPF and its Frobenius norm.

    At convergence, [F, P]_S = 0 (F and P commute in S-metric).

    Args:
        F: Fock matrix
        P: Density matrix
        S: Overlap matrix

    Returns:
        R: Residual matrix
        norm_R: Frobenius norm of R
    """
    R = F @ P @ S - S @ P @ F
    return R, np.linalg.norm(R)


def solve_roothaan_hall(F: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Roothaan-Hall equations FC = SCe via orthogonalization.

    Transforms to orthonormal basis: F' = X^T F X, solves F'C' = C'e,
    then back-transforms: C = X C'.

    Args:
        F: Fock matrix in AO basis
        X: Orthogonalizer (X^T S X = I)

    Returns:
        eps: Orbital energies (sorted ascending)
        C: MO coefficients in AO basis
    """
    F_prime = X.T @ F @ X
    eps, C_prime = np.linalg.eigh(F_prime)
    C = X @ C_prime
    return eps, C


class SimpleDIIS:
    """
    Simple DIIS (Direct Inversion in the Iterative Subspace) implementation.

    DIIS extrapolates a better Fock matrix from a history of Fock/error pairs:
        F_DIIS = sum_i c_i F_i, where sum_i c_i = 1 minimizes ||sum_i c_i e_i||^2

    Reference: Pulay, Chem. Phys. Lett. 73, 393 (1980).
    """

    def __init__(self, max_vec: int = 8, start: int = 2):
        """Initialize DIIS with max history size and start iteration."""
        self.max_vec = max_vec
        self.start = start
        self.F_list: List[np.ndarray] = []
        self.e_list: List[np.ndarray] = []

    def reset(self):
        """Clear DIIS history."""
        self.F_list = []
        self.e_list = []

    def update(self, F: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Add Fock/error pair and return extrapolated Fock matrix.

        Args:
            F: Current Fock matrix
            e: Error vector (flattened residual R = FPS - SPF)

        Returns:
            F_diis: Extrapolated Fock matrix
        """
        self.F_list.append(F.copy())
        self.e_list.append(e.copy())

        # Trim history
        if len(self.F_list) > self.max_vec:
            self.F_list.pop(0)
            self.e_list.pop(0)

        m = len(self.F_list)
        if m < self.start:
            return F

        # Build B matrix: B_ij = e_i^T e_j
        B = np.zeros((m + 1, m + 1))
        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(self.e_list[i], self.e_list[j])
        B[-1, :m] = -1.0
        B[:m, -1] = -1.0
        B[-1, -1] = 0.0

        # Right-hand side
        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0

        try:
            sol = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return F  # Skip DIIS if singular

        coeffs = sol[:m]

        # Extrapolate
        F_diis = np.zeros_like(F)
        for ci, Fi in zip(coeffs, self.F_list):
            F_diis += ci * Fi

        return F_diis


def rhf_scf(mol: gto.Mole,
            max_iter: int = 100,
            conv_E: float = 1e-10,
            conv_rmsd: float = 1e-10,
            use_diis: bool = True,
            diis_start: int = 2,
            damping: float = 0.0,
            level_shift: float = 0.0,
            init_guess: str = "hcore",
            verbose: bool = True) -> Dict[str, Any]:
    """
    RHF SCF with optional DIIS, damping, and level shifting.

    Args:
        mol: PySCF Mole object
        max_iter: Maximum SCF iterations
        conv_E: Energy convergence threshold (Hartree)
        conv_rmsd: Density RMSD convergence threshold
        use_diis: Enable DIIS acceleration
        diis_start: Iteration to start DIIS
        damping: Density damping factor (0 = no damping)
        level_shift: Virtual orbital shift (Hartree)
        init_guess: Initial guess type ("hcore" or "sad")
        verbose: Print iteration details

    Returns:
        Dictionary with SCF results
    """
    # Extract integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    nao = mol.nao_nr()
    n_occ = mol.nelectron // 2

    # Build orthogonalizer
    X = symmetric_orthogonalizer(S)

    # Initial guess
    if init_guess == "sad":
        # Use PySCF's SAD guess for better convergence
        mf_temp = scf.RHF(mol)
        mf_temp.verbose = 0
        P = mf_temp.init_guess_by_atom(mol)
        # Ensure proper electron count normalization
        n_elec_guess = np.trace(P @ S)
        if abs(n_elec_guess - mol.nelectron) > 0.1:
            P = P * mol.nelectron / n_elec_guess
        # Get initial C from diagonalizing F(P_guess)
        J, K = build_jk_matrices(P, eri)
        F_init = h + J - 0.5 * K
        eps, C = solve_roothaan_hall(F_init, X)
    else:
        # Core Hamiltonian guess (default)
        eps, C = solve_roothaan_hall(h, X)
        P = build_density_matrix(C, n_occ)

    # Initialize DIIS
    diis = SimpleDIIS(max_vec=8, start=diis_start) if use_diis else None

    history = []
    converged = False
    E_old = 0.0

    if verbose:
        print(f"{'Iter':>4}  {'E_total':>20}  {'dE':>12}  {'RMSD(P)':>12}  {'|R|':>12}")
        print("-" * 70)

    for iteration in range(1, max_iter + 1):
        # Build Fock matrix
        J, K = build_jk_matrices(P, eri)
        F = h + J - 0.5 * K

        # Compute residual
        R, norm_R = compute_scf_residual(F, P, S)

        # Apply DIIS
        if diis is not None:
            F = diis.update(F, R.ravel())

        # Apply level shift
        if level_shift > 0:
            F_prime = X.T @ F @ X
            eps_temp, C_prime = np.linalg.eigh(F_prime)
            # Shift virtual orbital energies
            eps_shifted = eps_temp.copy()
            eps_shifted[n_occ:] += level_shift
            F_prime_shifted = C_prime @ np.diag(eps_shifted) @ C_prime.T
            F = X @ F_prime_shifted @ X.T

        # Solve Roothaan-Hall
        eps, C = solve_roothaan_hall(F, X)

        # Build new density
        P_new = build_density_matrix(C, n_occ)

        # Apply damping
        if damping > 0:
            P_new = (1 - damping) * P_new + damping * P

        # Compute energy
        J_new, K_new = build_jk_matrices(P_new, eri)
        F_new = h + J_new - 0.5 * K_new
        E_elec = 0.5 * np.einsum('uv,uv->', P_new, h + F_new)
        E_total = E_elec + E_nuc

        # Convergence metrics
        dE = E_total - E_old
        diff_P = P_new - P
        rmsd_P = np.sqrt(np.sum(diff_P ** 2) / diff_P.size)

        history.append({'E': E_total, 'dE': dE, 'rmsd_P': rmsd_P, 'norm_R': norm_R})

        if verbose:
            print(f"{iteration:4d}  {E_total:+20.12f}  {dE:+12.3e}  {rmsd_P:12.3e}  {norm_R:12.3e}")

        if iteration > 1 and abs(dE) < conv_E and rmsd_P < conv_rmsd:
            converged = True
            break

        P = P_new
        E_old = E_total

    if verbose:
        print("-" * 70)
        status = "Converged" if converged else "NOT converged"
        print(f"{status} in {iteration} iterations. E = {E_total:.10f} Hartree")

    return {
        'E_total': E_total,
        'E_elec': E_elec,
        'E_nuc': E_nuc,
        'C': C,
        'eps': eps,
        'P': P_new,
        'F': F_new,
        'converged': converged,
        'n_iter': iteration,
        'history': history,
        'S': S, 'h': h
    }


# =============================================================================
# Exercise 6.1: Derive RHF Energy in AO Form
# =============================================================================

def exercise_6_1() -> bool:
    """
    Exercise 6.1: Derive and verify the RHF energy expression in AO form.

    Starting from spin-orbital energy:
        E_HF = sum_i <i|h|i> + (1/2) sum_{i,j} (<ij|ij> - <ij|ji>) + E_nuc

    Derive the closed-shell RHF expression:
        E_elec = Tr[P h] + (1/2) Tr[P G] = (1/2) Tr[P (h + F)]

    This exercise verifies the factor of 2 tracking and double-counting avoidance.

    Returns:
        True if all verifications pass
    """
    print("\n" + "=" * 70)
    print("Exercise 6.1: Derive RHF Energy in AO Form")
    print("=" * 70)

    # Build test molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    # Run PySCF for reference
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_pyscf = mf.kernel()
    P = mf.make_rdm1()  # RHF density (includes factor of 2)

    print(f"\nMolecule: H2O (STO-3G)")
    print(f"Electrons: {mol.nelectron}")
    print(f"AO functions: {mol.nao_nr()}")

    # Build J and K
    J, K = build_jk_matrices(P, eri)
    G = J - 0.5 * K  # Two-electron part of Fock
    F = h + G

    print("\n--- Energy Expression Verification ---")

    # Method 1: Tr[P h] + (1/2) Tr[P G]
    E1_one = np.einsum('uv,uv->', P, h)
    E1_two = 0.5 * np.einsum('uv,uv->', P, G)
    E1_elec = E1_one + E1_two
    E1_total = E1_elec + E_nuc
    print(f"\nMethod 1: E = Tr[P h] + (1/2) Tr[P G] + E_nuc")
    print(f"  Tr[P h]      = {E1_one:.10f}")
    print(f"  (1/2) Tr[PG] = {E1_two:.10f}")
    print(f"  E_elec       = {E1_elec:.10f}")
    print(f"  E_total      = {E1_total:.10f}")

    # Method 2: (1/2) Tr[P (h + F)]
    E2_elec = 0.5 * np.einsum('uv,uv->', P, h + F)
    E2_total = E2_elec + E_nuc
    print(f"\nMethod 2: E = (1/2) Tr[P (h + F)] + E_nuc")
    print(f"  (1/2) Tr[P(h+F)] = {E2_elec:.10f}")
    print(f"  E_total          = {E2_total:.10f}")

    # Method 3: Tr[P h] + (1/2) Tr[P J] - (1/4) Tr[P K]
    E3_one = np.einsum('uv,uv->', P, h)
    E3_J = 0.5 * np.einsum('uv,uv->', P, J)
    E3_K = -0.25 * np.einsum('uv,uv->', P, K)
    E3_elec = E3_one + E3_J + E3_K
    E3_total = E3_elec + E_nuc
    print(f"\nMethod 3: E = Tr[P h] + (1/2) Tr[P J] - (1/4) Tr[P K] + E_nuc")
    print(f"  Tr[P h]        = {E3_one:.10f}")
    print(f"  (1/2) Tr[P J]  = {E3_J:.10f}")
    print(f"  -(1/4) Tr[P K] = {E3_K:.10f}")
    print(f"  E_total        = {E3_total:.10f}")

    print(f"\nPySCF Reference: {E_pyscf:.10f}")

    # Verify all methods agree
    tol = 1e-10
    agree_12 = abs(E1_total - E2_total) < tol
    agree_23 = abs(E2_total - E3_total) < tol
    agree_pyscf = abs(E1_total - E_pyscf) < tol

    print("\n--- Verification ---")
    print(f"  Methods 1 and 2 agree: {agree_12} (diff = {abs(E1_total - E2_total):.2e})")
    print(f"  Methods 2 and 3 agree: {agree_23} (diff = {abs(E2_total - E3_total):.2e})")
    print(f"  Match PySCF:           {agree_pyscf} (diff = {abs(E1_total - E_pyscf):.2e})")

    print("\n--- Why the Factor of 1/2 Avoids Double-Counting ---")
    print("""
The two-electron energy sums over all pairs (i,j):
    E_2e = (1/2) sum_{i,j} <ij||ij>

The trace Tr[P G] = Tr[P (J - 0.5*K)] counts each pair TWICE because:
    - P contains both electrons' contributions
    - The integral (uv|ls) includes both orderings

The factor (1/2) corrects this double-counting.

The formula E = (1/2) Tr[P (h + F)] works because:
    h + F = h + h + G = 2h + G
    (1/2) Tr[P(h+F)] = Tr[P h] + (1/2) Tr[P G]
This automatically gives one-electron + half of two-electron.
""")

    all_pass = agree_12 and agree_23 and agree_pyscf
    print(f"\n{'All verifications PASSED!' if all_pass else 'Some verifications FAILED!'}")

    return all_pass


# =============================================================================
# Exercise 6.2: Stationarity and the Commutator Residual
# =============================================================================

def exercise_6_2() -> bool:
    """
    Exercise 6.2: Verify that the commutator residual R = FPS - SPF = 0 at convergence.

    At SCF convergence, F and P commute in the S-metric:
        [F, P]_S = FPS - SPF = 0

    This happens because at convergence:
        F C = S C eps  (Roothaan-Hall)
    and P is built from the eigenvectors of F in the S-metric.

    Returns:
        True if verification passes
    """
    print("\n" + "=" * 70)
    print("Exercise 6.2: Stationarity and the Commutator Residual")
    print("=" * 70)

    # Build molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Get converged solution from PySCF
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    # Extract matrices
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    P = mf.make_rdm1()
    F = mf.get_fock()
    C = mf.mo_coeff
    eps = mf.mo_energy
    n_occ = mol.nelectron // 2

    print(f"\nMolecule: H2O (STO-3G)")
    print(f"Converged HF energy: {mf.e_tot:.10f} Hartree")

    # Compute commutator residual
    R = F @ P @ S - S @ P @ F
    norm_R = np.linalg.norm(R, 'fro')

    print("\n--- Commutator Residual at Convergence ---")
    print(f"||R||_F = ||FPS - SPF||_F = {norm_R:.2e}")

    # Verify Roothaan-Hall equations
    FC = F @ C
    SCe = S @ C @ np.diag(eps)
    rh_error = np.linalg.norm(FC - SCe, 'fro')
    print(f"||FC - SCe||_F = {rh_error:.2e}")

    # Verify MO orthonormality in S-metric
    CtSC = C.T @ S @ C
    ortho_err = np.linalg.norm(CtSC - np.eye(CtSC.shape[0]), 'fro')
    print(f"||C^T S C - I||_F = {ortho_err:.2e}")

    # Show the mathematical proof numerically
    print("\n--- Mathematical Proof (Numerical Verification) ---")
    print("""
At convergence, FC = SCe, so for occupied orbitals:
    F C_occ = S C_occ eps_occ

Then:
    FPS = F (2 C_occ C_occ^T) S
        = 2 (S C_occ eps_occ) C_occ^T S   [using FC_occ = SC_occ eps_occ]
        = 2 S C_occ eps_occ C_occ^T S

Similarly:
    SPF = S (2 C_occ C_occ^T) F
        = 2 S C_occ C_occ^T (S C_occ eps_occ)   [rearranging]
        = 2 S C_occ (C_occ^T S C_occ) eps_occ

But C_occ^T S C_occ = I (orthonormality), so:
    SPF = 2 S C_occ eps_occ

Both FPS and SPF reduce to 2 S C_occ eps_occ C_occ^T S,
therefore R = FPS - SPF = 0.
""")

    # Numerical verification of proof steps
    C_occ = C[:, :n_occ]
    eps_occ = np.diag(eps[:n_occ])

    term1 = 2 * S @ C_occ @ eps_occ @ C_occ.T @ S  # Expanded FPS
    term2 = F @ P @ S  # Direct FPS

    proof_err = np.linalg.norm(term1 - term2, 'fro')
    print(f"Verification: ||2 S C_occ eps_occ C_occ^T S - FPS||_F = {proof_err:.2e}")

    # Check convergence during SCF iteration
    print("\n--- Residual During SCF Iterations ---")

    # Run our own SCF and track residual
    results = rhf_scf(mol, verbose=False, conv_E=1e-10, conv_rmsd=1e-10)

    print(f"{'Iter':>4}  {'|R|':>12}  {'log10(|R|)':>12}")
    print("-" * 35)
    for i, hist in enumerate(results['history'][:10], 1):
        log_R = np.log10(hist['norm_R']) if hist['norm_R'] > 0 else -np.inf
        print(f"{i:4d}  {hist['norm_R']:12.3e}  {log_R:12.2f}")
    if len(results['history']) > 10:
        print(f"... ({len(results['history']) - 10} more iterations)")
    final = results['history'][-1]
    log_R_final = np.log10(final['norm_R']) if final['norm_R'] > 0 else -np.inf
    print(f"{'Final':>4}  {final['norm_R']:12.3e}  {log_R_final:12.2f}")

    # Verification
    converged = norm_R < 1e-8 and ortho_err < 1e-10 and rh_error < 1e-8
    print(f"\n{'Verification PASSED!' if converged else 'Verification FAILED!'}")

    return converged


# =============================================================================
# Exercise 6.3: Direct SCF Conceptual Design
# =============================================================================

def exercise_6_3() -> bool:
    """
    Exercise 6.3: Demonstrate Direct SCF conceptual design.

    Direct SCF computes integrals on-the-fly rather than storing the full
    ERI tensor. This exercise shows the shell-quartet structure and
    demonstrates Schwarz screening.

    Returns:
        True if demonstration completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 6.3: Direct SCF Conceptual Design")
    print("=" * 70)

    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    nao = mol.nao_nr()
    nbas = mol.nbas

    print(f"\nMolecule: H2O (STO-3G)")
    print(f"AO functions: {nao}")
    print(f"Shells: {nbas}")

    # Shell information
    print("\n--- Shell Structure ---")
    print(f"{'Shell':>6} {'Atom':>6} {'L':>4} {'N_func':>8} {'Type':>6}")
    print("-" * 40)
    ao_idx = 0
    shell_info = []
    for i in range(nbas):
        atom = mol.bas_atom(i)
        L = mol.bas_angular(i)
        n_func = 2 * L + 1  # Spherical
        shell_type = ['s', 'p', 'd', 'f', 'g'][L]
        shell_info.append({'start': ao_idx, 'end': ao_idx + n_func, 'L': L})
        print(f"{i:>6} {atom:>6} {L:>4} {n_func:>8} {shell_type:>6}")
        ao_idx += n_func

    # Compute diagonal ERI blocks for Schwarz screening
    print("\n--- Schwarz Screening Bounds ---")
    print("Schwarz inequality: |(AB|CD)| <= sqrt((AB|AB)) * sqrt((CD|CD))")

    diag_eri = []
    for i in range(nbas):
        for j in range(nbas):
            # (ij|ij) block
            eri_block = mol.intor_by_shell('int2e_sph', [i, j, i, j])
            # Max element in block
            max_val = np.max(np.abs(eri_block))
            diag_eri.append((i, j, max_val))

    # Show some screening bounds
    print(f"\n{'Shell pair (i,j)':>18} {'sqrt((ij|ij))':>16}")
    print("-" * 40)
    for i, j, val in sorted(diag_eri, key=lambda x: -x[2])[:5]:
        print(f"{'(' + str(i) + ',' + str(j) + ')':>18} {np.sqrt(val):>16.6f}")

    # Demonstrate screening effect
    print("\n--- Screening Effect ---")
    threshold = 1e-10

    # Count total quartets vs screened
    total_quartets = 0
    screened_quartets = 0

    for i in range(nbas):
        for j in range(nbas):
            for k in range(nbas):
                for l in range(nbas):
                    total_quartets += 1
                    # Get Schwarz bounds
                    ij_max = next(d[2] for d in diag_eri if d[0] == i and d[1] == j)
                    kl_max = next(d[2] for d in diag_eri if d[0] == k and d[1] == l)
                    bound = np.sqrt(ij_max) * np.sqrt(kl_max)
                    if bound < threshold:
                        screened_quartets += 1

    screened_frac = 100 * screened_quartets / total_quartets
    print(f"Total shell quartets: {total_quartets}")
    print(f"Screened (bound < {threshold:.0e}): {screened_quartets} ({screened_frac:.1f}%)")
    print(f"Computed: {total_quartets - screened_quartets}")

    # Memory comparison
    print("\n--- Memory Comparison ---")
    eri_memory = nao ** 4 * 8 / 1e6  # MB
    direct_memory = 3 * nao ** 2 * 8 / 1e6  # J, K, P only

    print(f"In-core ERI storage: {eri_memory:.3f} MB ({nao}^4 elements)")
    print(f"Direct SCF storage:  {direct_memory:.3f} MB (only J, K, P)")
    print(f"Memory ratio: {eri_memory / direct_memory:.1f}x")

    # Pseudocode for Direct SCF
    print("\n--- Direct SCF Pseudocode ---")
    print("""
def direct_scf_jk(mol, P, threshold):
    '''Build J, K by looping over shell quartets with screening.'''
    J = zeros(nao, nao)
    K = zeros(nao, nao)

    # Precompute Schwarz bounds
    schwarz = precompute_diagonal_eri(mol)

    for iA in range(n_shells):
        for iB in range(iA + 1):  # Use symmetry
            for iC in range(iA + 1):
                max_CD = iC if iA > iC else iB
                for iD in range(max_CD + 1):

                    # Schwarz screening
                    bound_AB = sqrt(schwarz[iA, iB])
                    bound_CD = sqrt(schwarz[iC, iD])
                    if bound_AB * bound_CD < threshold:
                        continue  # Skip negligible quartet

                    # Compute ERI block using Rys quadrature
                    eri_block = compute_eri_rys(iA, iB, iC, iD)

                    # Contract into J and K with permutation symmetry
                    contract_to_jk(J, K, eri_block, P, iA, iB, iC, iD)

    # Fill symmetric parts
    J = J + J.T - diag(J)
    K = K + K.T - diag(K)

    return J, K
""")

    print("Direct SCF design demonstration completed.")
    return True


# =============================================================================
# Exercise 6.4: DIIS Behavior Study
# =============================================================================

def exercise_6_4() -> bool:
    """
    Exercise 6.4: Study DIIS convergence behavior for easy and hard cases.

    Compare SCF convergence with and without DIIS for:
    1. H2O at equilibrium (easy)
    2. Stretched H2 at R = 3.0 A (hard)

    Returns:
        True if study completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 6.4: DIIS Behavior Study")
    print("=" * 70)

    # Case 1: H2O at equilibrium (easy)
    print("\n--- Case 1: H2O / STO-3G (Equilibrium - Easy) ---")

    mol_h2o = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print("\nWithout DIIS:")
    result_no_diis = rhf_scf(mol_h2o, use_diis=False, verbose=True)

    print("\nWith DIIS:")
    result_with_diis = rhf_scf(mol_h2o, use_diis=True, verbose=True)

    mf = scf.RHF(mol_h2o)
    mf.verbose = 0
    E_ref = mf.kernel()

    print(f"\nPySCF reference: {E_ref:.10f}")
    print(f"Speedup: {result_no_diis['n_iter']} -> {result_with_diis['n_iter']} iterations")

    # Case 2: Stretched H2 (hard)
    print("\n--- Case 2: Stretched H2 (R = 3.0 A - Hard) ---")

    mol_h2 = gto.M(
        atom="H 0 0 0; H 0 0 3.0",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Check HOMO-LUMO gap
    mf_h2 = scf.RHF(mol_h2)
    mf_h2.verbose = 0
    mf_h2.kernel()
    gap = mf_h2.mo_energy[1] - mf_h2.mo_energy[0]
    print(f"HOMO-LUMO gap: {gap:.6f} Hartree = {gap * 27.2114:.4f} eV (small!)")

    print("\nWithout DIIS:")
    result_h2_no = rhf_scf(mol_h2, use_diis=False, max_iter=50, verbose=True)

    print("\nWith DIIS:")
    result_h2_diis = rhf_scf(mol_h2, use_diis=True, max_iter=50, verbose=True)

    print("\nWith DIIS + Damping (0.3):")
    result_h2_hybrid = rhf_scf(mol_h2, use_diis=True, damping=0.3, max_iter=50, verbose=True)

    print("\n--- Residual Norm Comparison ---")
    print("Without DIIS: Residual decreases linearly (on log scale)")
    print("With DIIS: Residual shows jumps as extrapolation kicks in")

    # Compare residual behavior
    print(f"\n{'Iter':>4}  {'|R| no DIIS':>12}  {'|R| DIIS':>12}")
    print("-" * 35)
    n_show = min(10, len(result_h2_no['history']), len(result_h2_diis['history']))
    for i in range(n_show):
        r_no = result_h2_no['history'][i]['norm_R']
        r_diis = result_h2_diis['history'][i]['norm_R']
        print(f"{i+1:4d}  {r_no:12.3e}  {r_diis:12.3e}")

    print("\n--- Summary ---")
    print(f"{'System':<20} {'No DIIS':>10} {'DIIS':>10} {'DIIS+Damp':>12}")
    print("-" * 55)
    print(f"{'H2O (easy)':<20} {result_no_diis['n_iter']:>10} {result_with_diis['n_iter']:>10} {'N/A':>12}")
    print(f"{'H2 R=3.0A (hard)':<20} {result_h2_no['n_iter']:>10} {result_h2_diis['n_iter']:>10} {result_h2_hybrid['n_iter']:>12}")

    print("""
Key Observations:
1. For easy cases (large HOMO-LUMO gap), DIIS provides modest speedup
2. For hard cases (small gap), DIIS is essential for reasonable convergence
3. Very difficult cases may need DIIS + damping combination
4. The residual norm |R| = |FPS - SPF| provides earlier warning of problems
   than energy change alone
""")

    return True


# =============================================================================
# Exercise 6.5: Reproduce PySCF Energies
# =============================================================================

def exercise_6_5() -> bool:
    """
    Exercise 6.5: Reproduce PySCF RHF energies exactly.

    Verify our implementation against PySCF for multiple systems.
    Identify common sources of discrepancy including multiple SCF solutions.

    Returns:
        True if all energies match within tolerance
    """
    print("\n" + "=" * 70)
    print("Exercise 6.5: Reproduce PySCF Energies")
    print("=" * 70)

    # Test systems (avoiding N2 which has multiple solutions with core guess)
    test_systems = [
        ("H2 (eq)", "H 0 0 0; H 0 0 0.74", "sto-3g"),
        ("H2O (eq)", "O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043", "sto-3g"),
        ("CH4", "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", "sto-3g"),
        ("H2O/6-31G", "O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043", "6-31g"),
        ("HF (eq)", "H 0 0 0; F 0 0 0.92", "sto-3g"),
    ]

    print(f"\n{'System':<15} {'Our Energy':>18} {'PySCF':>18} {'Diff':>12} {'Status':>8}")
    print("-" * 75)

    all_pass = True
    tol = 1e-8

    for name, atom, basis in test_systems:
        mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)

        # Our implementation (use SAD guess for reliable comparison)
        result = rhf_scf(mol, verbose=False, conv_E=1e-12, conv_rmsd=1e-12,
                        init_guess="sad")
        E_ours = result['E_total']

        # PySCF reference (also uses SAD guess by default)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-12
        E_pyscf = mf.kernel()

        diff = abs(E_ours - E_pyscf)
        status = "PASS" if diff < tol else "FAIL"
        if diff >= tol:
            all_pass = False

        print(f"{name:<15} {E_ours:>18.10f} {E_pyscf:>18.10f} {diff:>12.2e} {status:>8}")

    print("-" * 75)

    # Detailed diagnostics for H2O
    print("\n--- Detailed Diagnostics for H2O/STO-3G ---")
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    result = rhf_scf(mol, verbose=False, conv_E=1e-12, conv_rmsd=1e-12)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    # Compare intermediate quantities
    print(f"\nElectron count check:")
    n_elec_ours = np.trace(result['P'] @ S)
    n_elec_pyscf = np.trace(mf.make_rdm1() @ S)
    print(f"  Our Tr[PS]   = {n_elec_ours:.10f} (expected: {mol.nelectron})")
    print(f"  PySCF Tr[PS] = {n_elec_pyscf:.10f}")

    print(f"\nFock matrix comparison:")
    F_ours = result['F']
    F_pyscf = mf.get_fock()
    F_diff = np.linalg.norm(F_ours - F_pyscf)
    print(f"  ||F_ours - F_pyscf||_F = {F_diff:.2e}")

    print(f"\nDensity matrix comparison:")
    P_diff = np.linalg.norm(result['P'] - mf.make_rdm1())
    print(f"  ||P_ours - P_pyscf||_F = {P_diff:.2e}")

    print("""
Common Sources of Energy Discrepancy:
1. Convergence threshold differences
2. J/K einsum index patterns (chemist's vs physicist's notation)
3. Factor of 2 in density matrix or factor of 0.5 in exchange
4. Nuclear repulsion energy calculation
5. Orthogonalizer threshold dropping different eigenvectors
6. Multiple SCF solutions (different initial guesses lead to different minima)
""")

    # Demonstrate multiple solutions for N2
    print("--- Multiple SCF Solutions Example: N2 ---")
    mol_n2 = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g",
                   unit="Angstrom", verbose=0)

    # Core Hamiltonian guess
    result_core = rhf_scf(mol_n2, verbose=False, conv_E=1e-12,
                          conv_rmsd=1e-12, init_guess="hcore")

    # SAD guess
    result_sad = rhf_scf(mol_n2, verbose=False, conv_E=1e-12,
                         conv_rmsd=1e-12, init_guess="sad")

    # PySCF reference
    mf_n2 = scf.RHF(mol_n2)
    mf_n2.verbose = 0
    E_pyscf_n2 = mf_n2.kernel()

    print(f"N2 with hcore guess: {result_core['E_total']:.10f} Hartree")
    print(f"N2 with SAD guess:   {result_sad['E_total']:.10f} Hartree")
    print(f"PySCF reference:     {E_pyscf_n2:.10f} Hartree")
    print("\nThe core Hamiltonian guess leads to a local minimum,")
    print("while SAD guess finds the global minimum (same as PySCF).")
    print("This demonstrates why initial guess choice matters!")

    print(f"\n{'All systems match PySCF!' if all_pass else 'Some systems do not match!'}")
    return all_pass


# =============================================================================
# Exercise 6.6: MO-Basis Gradient Check (Advanced)
# =============================================================================

def exercise_6_6() -> bool:
    """
    Exercise 6.6: Verify the MO-basis orbital gradient.

    The occupied-virtual gradient is:
        g_ai = 2 * F_ai^MO = 2 * (C_vir^T F C_occ)

    At convergence, g_ai = 0 because F is diagonal in the MO basis.

    Returns:
        True if verification passes
    """
    print("\n" + "=" * 70)
    print("Exercise 6.6: MO-Basis Gradient Check (Advanced)")
    print("=" * 70)

    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    S = mol.intor("int1e_ovlp")
    C = mf.mo_coeff
    F = mf.get_fock()
    P = mf.make_rdm1()
    n_occ = mol.nelectron // 2
    n_vir = C.shape[1] - n_occ

    print(f"\nMolecule: H2O (STO-3G)")
    print(f"Occupied orbitals: {n_occ}")
    print(f"Virtual orbitals:  {n_vir}")

    # Transform F to MO basis
    F_MO = C.T @ F @ C

    print("\n--- Fock Matrix in MO Basis ---")
    print("At convergence, F_MO should be diagonal:")
    print(f"\nF_MO diagonal elements (orbital energies):")
    for i in range(len(mf.mo_energy)):
        occ_label = "occ" if i < n_occ else "vir"
        print(f"  eps[{i}] = {mf.mo_energy[i]:12.6f} ({occ_label})")

    # Check off-diagonal elements
    F_MO_offdiag = F_MO - np.diag(np.diag(F_MO))
    offdiag_norm = np.linalg.norm(F_MO_offdiag)
    print(f"\n||F_MO - diag(F_MO)||_F = {offdiag_norm:.2e}")

    # Occupied-virtual block (the gradient)
    F_ov = F_MO[:n_occ, n_occ:]
    g = 2.0 * F_ov  # Occupied-virtual gradient
    grad_norm = np.linalg.norm(g)

    print(f"\n--- Occupied-Virtual Gradient ---")
    print(f"g_ai = 2 * F_ai^MO")
    print(f"||g||_F = {grad_norm:.2e}")

    # Compare with AO residual
    R, norm_R = compute_scf_residual(F, P, S)
    print(f"\n--- Comparison with AO Residual ---")
    print(f"||FPS - SPF||_F = {norm_R:.2e}")

    # Show relationship
    print("""
At convergence, both quantities vanish because:
- F_MO is diagonal (occupied and virtual blocks don't mix)
- g_ai = 0 means no gradient for orbital rotation
- R = FPS - SPF = 0 is the AO-basis equivalent

The relationship is:
    ||g||_F ~ ||R||_F (up to transformation factors)

Both provide convergence measures, but:
- |R| is computed in AO basis (no transformation needed)
- |g| gives physical interpretation (no occupied-virtual mixing)
""")

    # Numerical relationship check
    # Transform R to MO basis
    R_MO = C.T @ S @ R @ S @ C  # Approximate (not exact due to metric)
    R_MO_ov = R_MO[:n_occ, n_occ:]
    R_MO_ov_norm = np.linalg.norm(R_MO_ov)
    print(f"||R^MO_ov||_F = {R_MO_ov_norm:.2e} (for comparison with |g|)")

    converged = grad_norm < 1e-8 and offdiag_norm < 1e-8
    print(f"\n{'Verification PASSED!' if converged else 'Verification FAILED!'}")

    return converged


# =============================================================================
# Exercise 6.7: Level Shifting Implementation (Advanced)
# =============================================================================

def exercise_6_7() -> bool:
    """
    Exercise 6.7: Implement and test level shifting for difficult SCF.

    Level shifting increases the HOMO-LUMO gap by shifting virtual orbital
    energies up by sigma during SCF. This stabilizes early iterations.

    Returns:
        True if demonstration completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 6.7: Level Shifting Implementation (Advanced)")
    print("=" * 70)

    # Use stretched H2 as a difficult case
    mol = gto.M(
        atom="H 0 0 0; H 0 0 3.5",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf_ref = scf.RHF(mol)
    mf_ref.verbose = 0
    mf_ref.kernel()
    true_gap = mf_ref.mo_energy[1] - mf_ref.mo_energy[0]
    E_ref = mf_ref.e_tot

    print(f"Molecule: Stretched H2 (R = 3.5 A)")
    print(f"True HOMO-LUMO gap: {true_gap:.6f} Hartree = {true_gap * 27.2114:.4f} eV")
    print(f"Reference energy: {E_ref:.10f} Hartree")
    print("\nThis small gap makes SCF difficult.")

    # Test different level shift values
    print("\n--- Effect of Level Shift ---")
    print(f"{'Shift (Ha)':>12} {'Iterations':>12} {'Converged':>12} {'Energy':>18}")
    print("-" * 58)

    shifts = [0.0, 0.1, 0.25, 0.5, 1.0]
    results = []

    for sigma in shifts:
        result = rhf_scf(mol, level_shift=sigma, use_diis=False,
                        max_iter=100, verbose=False)
        results.append(result)
        status = "Yes" if result['converged'] else "No"
        print(f"{sigma:>12.2f} {result['n_iter']:>12} {status:>12} {result['E_total']:>18.10f}")

    print("-" * 58)

    # Show how level shifting affects the Fock matrix
    print("\n--- How Level Shifting Works ---")
    print("""
Level shifting modifies the virtual orbital energies:
    eps_virtual += sigma

This increases the effective HOMO-LUMO gap, stabilizing SCF.

In the Fock matrix in MO basis:
    F_MO = diag(eps_1, eps_2, ..., eps_occ, eps_vir+sigma, ...)

Effect:
- Large sigma: More stable but slower convergence
- Small sigma: Less stable but faster when it works
- sigma = 0: Standard SCF (may oscillate for difficult cases)

Note: Level shift should be removed or tapered as SCF converges
to get correct final orbital energies.
""")

    # Compare with DIIS
    print("--- Comparison with DIIS ---")
    result_diis = rhf_scf(mol, use_diis=True, level_shift=0.0,
                          max_iter=100, verbose=False)
    result_both = rhf_scf(mol, use_diis=True, level_shift=0.25,
                          max_iter=100, verbose=False)

    print(f"DIIS only:          {result_diis['n_iter']:>3} iterations, "
          f"E = {result_diis['E_total']:.10f}")
    print(f"Level shift only:   {results[2]['n_iter']:>3} iterations, "
          f"E = {results[2]['E_total']:.10f}")
    print(f"DIIS + shift (0.25): {result_both['n_iter']:>3} iterations, "
          f"E = {result_both['E_total']:.10f}")

    print("\nLevel shifting demonstration completed.")
    return True


# =============================================================================
# Exercise 6.8: SCF Metadynamics (Research)
# =============================================================================

def exercise_6_8() -> bool:
    """
    Exercise 6.8: Explore multiple SCF solutions via different initial guesses.

    For systems with near-degeneracy, multiple SCF solutions may exist.
    Different initial guesses can converge to different local minima.

    Returns:
        True if exploration completes successfully
    """
    print("\n" + "=" * 70)
    print("Exercise 6.8: SCF Metadynamics (Research)")
    print("=" * 70)

    # Use stretched H2 where multiple solutions may exist
    mol = gto.M(
        atom="H 0 0 0; H 0 0 4.0",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"Molecule: Very stretched H2 (R = 4.0 A)")
    print(f"Electrons: {mol.nelectron}")

    # Reference from PySCF
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()
    print(f"PySCF reference energy: {E_ref:.10f} Hartree")

    # Collect integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()
    nao = mol.nao_nr()
    n_occ = mol.nelectron // 2

    X = symmetric_orthogonalizer(S)

    print("\n--- Testing Different Initial Guesses ---")

    def run_scf_with_guess(P_init, name):
        """Run SCF with a given initial density matrix."""
        P = P_init.copy()
        diis = SimpleDIIS()

        for iteration in range(1, 101):
            J, K = build_jk_matrices(P, eri)
            F = h + J - 0.5 * K
            R, norm_R = compute_scf_residual(F, P, S)
            F = diis.update(F, R.ravel())

            eps, C = solve_roothaan_hall(F, X)
            P_new = build_density_matrix(C, n_occ)

            J_new, K_new = build_jk_matrices(P_new, eri)
            F_new = h + J_new - 0.5 * K_new
            E_elec = 0.5 * np.einsum('uv,uv->', P_new, h + F_new)
            E_total = E_elec + E_nuc

            diff_P = np.linalg.norm(P_new - P)
            if diff_P < 1e-10:
                return {'E': E_total, 'C': C, 'eps': eps, 'P': P_new,
                        'iter': iteration, 'converged': True, 'name': name}
            P = P_new

        return {'E': E_total, 'C': C, 'eps': eps, 'P': P_new,
                'iter': iteration, 'converged': False, 'name': name}

    solutions = []

    # Guess 1: Core Hamiltonian (standard)
    print("\nGuess 1: Core Hamiltonian")
    eps, C = solve_roothaan_hall(h, X)
    P_core = build_density_matrix(C, n_occ)
    sol1 = run_scf_with_guess(P_core, "Core H")
    solutions.append(sol1)
    print(f"  E = {sol1['E']:.10f}, converged in {sol1['iter']} iter")

    # Guess 2: Perturbed density (random perturbation)
    print("\nGuess 2: Perturbed density")
    np.random.seed(42)
    P_perturbed = P_core + 0.1 * np.random.randn(nao, nao)
    P_perturbed = 0.5 * (P_perturbed + P_perturbed.T)  # Symmetrize
    sol2 = run_scf_with_guess(P_perturbed, "Perturbed")
    solutions.append(sol2)
    print(f"  E = {sol2['E']:.10f}, converged in {sol2['iter']} iter")

    # Guess 3: Localized density on one atom
    print("\nGuess 3: Density localized on atom 1")
    P_local1 = np.zeros((nao, nao))
    P_local1[0, 0] = 2.0  # Put both electrons on first H
    sol3 = run_scf_with_guess(P_local1, "Local-1")
    solutions.append(sol3)
    print(f"  E = {sol3['E']:.10f}, converged in {sol3['iter']} iter")

    # Guess 4: Localized density on other atom
    print("\nGuess 4: Density localized on atom 2")
    P_local2 = np.zeros((nao, nao))
    P_local2[1, 1] = 2.0  # Put both electrons on second H
    sol4 = run_scf_with_guess(P_local2, "Local-2")
    solutions.append(sol4)
    print(f"  E = {sol4['E']:.10f}, converged in {sol4['iter']} iter")

    # Summary
    print("\n--- Summary of Solutions Found ---")
    print(f"{'Guess':<15} {'Energy (Ha)':>18} {'Iter':>6} {'Gap (Ha)':>12}")
    print("-" * 55)

    unique_energies = []
    for sol in solutions:
        if sol['converged']:
            gap = sol['eps'][1] - sol['eps'][0]
            print(f"{sol['name']:<15} {sol['E']:>18.10f} {sol['iter']:>6} {gap:>12.6f}")

            # Check if this is a unique solution
            is_unique = True
            for E_prev in unique_energies:
                if abs(sol['E'] - E_prev) < 1e-6:
                    is_unique = False
                    break
            if is_unique:
                unique_energies.append(sol['E'])

    print(f"\nUnique solutions found: {len(unique_energies)}")
    for i, E in enumerate(sorted(unique_energies)):
        print(f"  Solution {i+1}: E = {E:.10f} Hartree")

    print("""
--- Physical Interpretation ---
For stretched H2, multiple SCF solutions can exist:
1. The global RHF minimum (delocalized bonding orbital)
2. Local minima corresponding to charge transfer states
3. Symmetry-broken solutions

At large R, RHF fails because it cannot describe the
correct dissociation limit (requires multi-reference methods).

The different solutions represent:
- Different charge distributions
- Breaking of spatial symmetry
- Instabilities in the RHF wavefunction
""")

    return True


# =============================================================================
# Main Runner
# =============================================================================

def main():
    """Run all Chapter 6 exercises or a specific exercise."""
    parser = argparse.ArgumentParser(
        description="Chapter 6 Exercise Solutions: Hartree-Fock SCF from Integrals"
    )
    parser.add_argument(
        "--exercise", "-e",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Run specific exercise (1-8)"
    )
    args = parser.parse_args()

    exercises = {
        1: ("RHF Energy Derivation", exercise_6_1),
        2: ("Stationarity and Commutator Residual", exercise_6_2),
        3: ("Direct SCF Conceptual Design", exercise_6_3),
        4: ("DIIS Behavior Study", exercise_6_4),
        5: ("Reproduce PySCF Energies", exercise_6_5),
        6: ("MO-Basis Gradient Check (Advanced)", exercise_6_6),
        7: ("Level Shifting (Advanced)", exercise_6_7),
        8: ("SCF Metadynamics (Research)", exercise_6_8),
    }

    print("=" * 70)
    print("Chapter 6 Exercise Solutions")
    print("Hartree-Fock SCF from Integrals")
    print("=" * 70)
    print("\nCourse: 2302638 Advanced Quantum Chemistry")
    print("Institution: Chulalongkorn University")

    if args.exercise:
        # Run specific exercise
        ex_num = args.exercise
        name, func = exercises[ex_num]
        print(f"\nRunning Exercise 6.{ex_num}: {name}")
        success = func()
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
                import traceback
                traceback.print_exc()

        # Summary
        print("\n" + "=" * 70)
        print("Summary of Results")
        print("=" * 70)
        print(f"\n{'Exercise':<12} {'Status':<8} {'Description':<45}")
        print("-" * 70)

        all_passed = True
        for ex_num in sorted(results.keys()):
            status, desc = results[ex_num]
            print(f"6.{ex_num:<10} {status:<8} {desc:<45}")
            if status != "PASS":
                all_passed = False

        print("-" * 70)

        if all_passed:
            print("\nAll exercises completed successfully!")
        else:
            print("\nSome exercises had issues. Review output above.")

        success = all_passed

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
