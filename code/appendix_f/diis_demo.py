#!/usr/bin/env python3
"""
DIIS (Direct Inversion in the Iterative Subspace) Demonstration

Demonstrates DIIS acceleration for SCF convergence, including error vector
computation and comparison of convergence with and without DIIS.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

References:
  P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
  P. Pulay, J. Comput. Chem. 3, 556 (1982)

Key equations:
  Error vector: e = FPS - SPF (commutator, vanishes at convergence)
  DIIS: F_new = sum_i c_i F_i, minimizing ||sum_i c_i e_i||^2
        subject to sum_i c_i = 1
"""

import numpy as np
from scipy.linalg import eigh, solve
from pyscf import gto, scf


def compute_diis_error(F: np.ndarray, P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute DIIS error vector e = FPS - SPF.

    At SCF convergence, [F, PS] = 0 (commutator vanishes).
    The error vector measures deviation from this condition.

    Args:
        F: Fock matrix
        P: Density matrix
        S: Overlap matrix

    Returns:
        e: Error vector (flattened)
    """
    FPS = F @ P @ S
    SPF = S @ P @ F
    error = FPS - SPF
    return error.flatten()


def solve_diis_coefficients(errors: list) -> np.ndarray:
    """Solve DIIS linear equations for coefficients.

    Minimize ||sum_i c_i e_i||^2 subject to sum_i c_i = 1.

    This is equivalent to solving:
    [B  -1] [c]   [0]
    [-1  0] [L] = [-1]

    where B_ij = <e_i|e_j>

    Args:
        errors: List of error vectors

    Returns:
        c: DIIS coefficients
    """
    n = len(errors)

    # Build B matrix: B_ij = e_i . e_j
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            B[i, j] = np.dot(errors[i], errors[j])

    # Add Lagrange multiplier row/column
    B[:n, n] = -1.0
    B[n, :n] = -1.0
    B[n, n] = 0.0

    # RHS: [0, 0, ..., 0, -1]
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0

    # Solve for coefficients
    try:
        solution = solve(B, rhs)
        c = solution[:n]
    except np.linalg.LinAlgError:
        # Fallback: use only latest Fock matrix
        c = np.zeros(n)
        c[-1] = 1.0

    return c


def extrapolate_fock(fock_list: list, coefficients: np.ndarray) -> np.ndarray:
    """Extrapolate Fock matrix using DIIS coefficients.

    F_new = sum_i c_i F_i

    Args:
        fock_list: List of Fock matrices
        coefficients: DIIS coefficients

    Returns:
        F_diis: Extrapolated Fock matrix
    """
    F_diis = np.zeros_like(fock_list[0])
    for c, F in zip(coefficients, fock_list):
        F_diis += c * F
    return F_diis


def run_scf_with_diis(mol, max_iter: int = 50, conv_tol: float = 1e-10,
                      diis_start: int = 1, diis_space: int = 6,
                      verbose: bool = True) -> dict:
    """Run RHF SCF with DIIS acceleration.

    Args:
        mol: PySCF molecule object
        max_iter: Maximum iterations
        conv_tol: Energy convergence threshold
        diis_start: Iteration to start DIIS
        diis_space: Maximum DIIS vectors to keep
        verbose: Print progress

    Returns:
        Dictionary with results and convergence history
    """
    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    n_occ = mol.nelectron // 2

    # Initial guess: core Hamiltonian
    _e, C = eigh(h, S)
    C_occ = C[:, :n_occ]
    P = 2.0 * C_occ @ C_occ.T

    # Storage for DIIS
    fock_list = []
    error_list = []
    energy_history = []
    error_norm_history = []

    E_old = 0.0

    if verbose:
        print("\n  Iter    E_total           dE           ||error||      DIIS")
        print("  " + "-" * 65)

    for iteration in range(1, max_iter + 1):
        # Build Fock matrix
        J = np.einsum('ijkl,kl->ij', eri, P)
        K = np.einsum('ikjl,kl->ij', eri, P)
        F = h + J - 0.5 * K

        # Compute energy
        E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)
        E_tot = E_elec + E_nuc

        # DIIS error
        error = compute_diis_error(F, P, S)
        error_norm = np.linalg.norm(error)

        energy_history.append(E_tot)
        error_norm_history.append(error_norm)

        # Store for DIIS
        fock_list.append(F.copy())
        error_list.append(error.copy())

        # Limit DIIS space
        if len(fock_list) > diis_space:
            fock_list.pop(0)
            error_list.pop(0)

        # Apply DIIS extrapolation
        use_diis = iteration >= diis_start and len(fock_list) >= 2
        if use_diis:
            c = solve_diis_coefficients(error_list)
            F = extrapolate_fock(fock_list, c)
            diis_str = f"({len(fock_list)} vecs)"
        else:
            diis_str = ""

        dE = E_tot - E_old

        if verbose:
            print(f"  {iteration:4d}   {E_tot:16.10f}   {dE:12.2e}   {error_norm:12.2e}   {diis_str}")

        # Check convergence
        if abs(dE) < conv_tol and error_norm < conv_tol * 10:
            if verbose:
                print("  " + "-" * 65)
                print(f"  Converged!")
            break

        # Diagonalize Fock matrix
        _e, C = eigh(F, S)
        C_occ = C[:, :n_occ]
        P = 2.0 * C_occ @ C_occ.T

        E_old = E_tot

    return {
        'e_tot': E_tot,
        'iterations': iteration,
        'converged': iteration < max_iter,
        'energy_history': energy_history,
        'error_history': error_norm_history
    }


def run_scf_without_diis(mol, max_iter: int = 50, conv_tol: float = 1e-10,
                         verbose: bool = True) -> dict:
    """Run RHF SCF without DIIS (simple iteration)."""
    # Get integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    n_occ = mol.nelectron // 2

    # Initial guess
    _e, C = eigh(h, S)
    C_occ = C[:, :n_occ]
    P = 2.0 * C_occ @ C_occ.T

    energy_history = []
    error_norm_history = []
    E_old = 0.0

    if verbose:
        print("\n  Iter    E_total           dE           ||error||")
        print("  " + "-" * 55)

    for iteration in range(1, max_iter + 1):
        J = np.einsum('ijkl,kl->ij', eri, P)
        K = np.einsum('ikjl,kl->ij', eri, P)
        F = h + J - 0.5 * K

        E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)
        E_tot = E_elec + E_nuc

        error = compute_diis_error(F, P, S)
        error_norm = np.linalg.norm(error)

        energy_history.append(E_tot)
        error_norm_history.append(error_norm)

        dE = E_tot - E_old

        if verbose:
            print(f"  {iteration:4d}   {E_tot:16.10f}   {dE:12.2e}   {error_norm:12.2e}")

        if abs(dE) < conv_tol and error_norm < conv_tol * 10:
            if verbose:
                print("  " + "-" * 55)
                print(f"  Converged!")
            break

        _e, C = eigh(F, S)
        C_occ = C[:, :n_occ]
        P = 2.0 * C_occ @ C_occ.T

        E_old = E_tot

    return {
        'e_tot': E_tot,
        'iterations': iteration,
        'converged': iteration < max_iter,
        'energy_history': energy_history,
        'error_history': error_norm_history
    }


def main():
    print("=" * 70)
    print("DIIS Acceleration Demonstration")
    print("=" * 70)

    # =========================================================================
    # Test System 1: H2O (usually converges well)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: H2O / STO-3G")
    print("=" * 50)

    mol_h2o = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print("\n--- Without DIIS ---")
    result_no_diis = run_scf_without_diis(mol_h2o, max_iter=30)

    print("\n--- With DIIS ---")
    result_diis = run_scf_with_diis(mol_h2o, max_iter=30)

    # PySCF reference
    mf = scf.RHF(mol_h2o)
    E_ref = mf.kernel()

    print(f"\n  Summary:")
    print(f"    Without DIIS: {result_no_diis['iterations']} iterations, "
          f"E = {result_no_diis['e_tot']:.10f}")
    print(f"    With DIIS:    {result_diis['iterations']} iterations, "
          f"E = {result_diis['e_tot']:.10f}")
    print(f"    PySCF:        {mf.cycles} iterations, E = {E_ref:.10f}")

    # =========================================================================
    # Test System 2: More challenging system
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: HF / cc-pVDZ (can be slower to converge)")
    print("=" * 50)

    mol_hf = gto.M(
        atom="H 0 0 0; F 0 0 0.92",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print("\n--- Without DIIS ---")
    result_no_diis_hf = run_scf_without_diis(mol_hf, max_iter=40)

    print("\n--- With DIIS ---")
    result_diis_hf = run_scf_with_diis(mol_hf, max_iter=40)

    mf_hf = scf.RHF(mol_hf)
    E_ref_hf = mf_hf.kernel()

    print(f"\n  Summary:")
    print(f"    Without DIIS: {result_no_diis_hf['iterations']} iterations, "
          f"E = {result_no_diis_hf['e_tot']:.10f}")
    print(f"    With DIIS:    {result_diis_hf['iterations']} iterations, "
          f"E = {result_diis_hf['e_tot']:.10f}")
    print(f"    PySCF:        {mf_hf.cycles} iterations, E = {E_ref_hf:.10f}")

    # =========================================================================
    # Section 3: Visualize Convergence
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Convergence Comparison (H2O)")
    print("=" * 50)

    print("\n  Error norm reduction:")
    print("  " + "-" * 45)
    print(f"  {'Iter':>4} | {'No DIIS':>12} | {'With DIIS':>12}")
    print("  " + "-" * 45)

    max_show = max(len(result_no_diis['error_history']),
                   len(result_diis['error_history']))
    for i in range(min(max_show, 15)):
        e_no = result_no_diis['error_history'][i] if i < len(result_no_diis['error_history']) else np.nan
        e_yes = result_diis['error_history'][i] if i < len(result_diis['error_history']) else np.nan
        print(f"  {i+1:4d} | {e_no:12.4e} | {e_yes:12.4e}")

    # =========================================================================
    # Section 4: DIIS Error Vector Analysis
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. DIIS Error Vector Analysis")
    print("=" * 50)

    # Get converged quantities
    S = mol_h2o.intor("int1e_ovlp")
    F = mf.get_fock()
    P = mf.make_rdm1()

    # Compute error at convergence
    error_converged = compute_diis_error(F, P, S)
    error_norm_converged = np.linalg.norm(error_converged)

    print(f"\n  At convergence:")
    print(f"    ||FPS - SPF|| = {error_norm_converged:.2e}")
    print(f"    (Should be ~0 at self-consistency)")

    # Show that FPS = SPF at convergence
    FPS = F @ P @ S
    SPF = S @ P @ F
    print(f"\n  Commutator check:")
    print(f"    ||FPS - SPF||_max = {np.max(np.abs(FPS - SPF)):.2e}")

    # Show error during iteration
    print("\n  Error vector properties during SCF (H2O):")

    # Re-run a few iterations capturing errors
    T = mol_h2o.intor("int1e_kin")
    V = mol_h2o.intor("int1e_nuc")
    h = T + V
    eri = mol_h2o.intor("int2e", aosym="s1")
    n_occ = mol_h2o.nelectron // 2

    _e, C = eigh(h, S)
    C_occ = C[:, :n_occ]
    P_iter = 2.0 * C_occ @ C_occ.T

    print(f"\n  Iter | ||error||  | max|error|")
    print("  " + "-" * 35)

    for it in range(1, 6):
        J = np.einsum('ijkl,kl->ij', eri, P_iter)
        K = np.einsum('ikjl,kl->ij', eri, P_iter)
        F_iter = h + J - 0.5 * K

        error = compute_diis_error(F_iter, P_iter, S)
        error_mat = error.reshape(mol_h2o.nao, mol_h2o.nao)

        print(f"  {it:4d} | {np.linalg.norm(error):10.4e} | {np.max(np.abs(error_mat)):10.4e}")

        _, C = eigh(F_iter, S)
        C_occ = C[:, :n_occ]
        P_iter = 2.0 * C_occ @ C_occ.T

    print("\n" + "=" * 70)
    print("DIIS demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
