#!/usr/bin/env python3
"""
lab6a_rhf_scf.py - Minimal RHF SCF from AO Integrals (Lab 6A)

This module implements a complete Restricted Hartree-Fock SCF algorithm
from scratch using AO integrals extracted from PySCF. The implementation
follows Algorithm 6.1 from Chapter 6 exactly.

The SCF loop:
    P -> F(P) -> (epsilon, C) -> P_new -> repeat

Key equations (RHF closed-shell):
    - Density matrix:    P = 2 * C_occ @ C_occ.T
    - Fock matrix:       F = h + J - (1/2) K
    - Roothaan-Hall:     F C = S C epsilon
    - Electronic energy: E_elec = (1/2) Tr[P(h + F)]
    - SCF residual:      R = FPS - SPF

The orthogonalization transforms the generalized eigenvalue problem:
    F C = S C epsilon  -->  F' C' = C' epsilon
    where F' = X.T @ F @ X, C = X @ C', and X = S^{-1/2}

References:
    - Chapter 6, Section 5: SCF iteration framework
    - Chapter 6, Section 6: SCF convergence and acceleration
    - Algorithm 6.1: Minimal RHF SCF from integrals
    - Listings 6.1-6.2: Code implementation

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import Tuple, Optional


def symm_orth(S: np.ndarray, thresh: float = 1e-10) -> np.ndarray:
    """
    Symmetric orthogonalizer X = S^{-1/2} with linear-dependence threshold.

    The symmetric orthogonalizer preserves maximum similarity to the
    original AO basis while ensuring orthonormality: X.T @ S @ X = I.

    For near-linear-dependent basis sets, small eigenvalues of S are
    discarded to maintain numerical stability.

    Parameters
    ----------
    S : np.ndarray
        Overlap matrix (symmetric positive semi-definite)
    thresh : float
        Threshold below which eigenvalues are considered linear-dependent

    Returns
    -------
    X : np.ndarray
        Orthogonalizer matrix such that X.T @ S @ X = I
    """
    e, U = np.linalg.eigh(S)
    keep = e > thresh
    n_removed = len(e) - np.sum(keep)
    if n_removed > 0:
        print(f"  Warning: Removed {n_removed} near-linear-dependent basis function(s)")
    X = U[:, keep] @ np.diag(e[keep]**-0.5) @ U[:, keep].T
    return X


def build_jk(P: np.ndarray, eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Coulomb (J) and Exchange (K) matrices from full ERIs.

    J_mn = sum_{ls} (mn|ls) P_ls     (classical Coulomb repulsion)
    K_mn = sum_{ls} (ml|ns) P_ls     (quantum exchange)

    Parameters
    ----------
    P : np.ndarray
        Density matrix (includes factor of 2 for RHF)
    eri : np.ndarray
        Full ERI tensor (pq|rs) with shape (N, N, N, N)

    Returns
    -------
    J : np.ndarray
        Coulomb matrix
    K : np.ndarray
        Exchange matrix
    """
    J = np.einsum("pqrs,rs->pq", eri, P, optimize=True)
    K = np.einsum("prqs,rs->pq", eri, P, optimize=True)
    return J, K


def rhf_scf(
    S: np.ndarray,
    h: np.ndarray,
    eri: np.ndarray,
    nelec: int,
    Enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    diis: Optional[object] = None,
    verbose: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Educational RHF SCF using full ERIs.

    Implements Algorithm 6.1: Minimal RHF SCF from integrals.

    Parameters
    ----------
    S : np.ndarray
        Overlap matrix
    h : np.ndarray
        Core Hamiltonian (kinetic + nuclear attraction)
    eri : np.ndarray
        Full ERI tensor (pq|rs)
    nelec : int
        Number of electrons
    Enuc : float
        Nuclear repulsion energy
    max_cycle : int
        Maximum number of SCF iterations
    conv_tol : float
        Convergence tolerance for energy
    diis : object, optional
        DIIS accelerator with .update(F, R) -> F_new method
    verbose : bool
        Print iteration information

    Returns
    -------
    E_tot : float
        Total HF energy
    eps : np.ndarray
        Orbital energies
    C : np.ndarray
        MO coefficient matrix
    P : np.ndarray
        Converged density matrix

    Raises
    ------
    RuntimeError
        If SCF does not converge within max_cycle iterations
    """
    nbf = S.shape[0]
    nocc = nelec // 2

    if verbose:
        print(f"\nRHF SCF Calculation")
        print(f"  Basis functions: {nbf}")
        print(f"  Electrons: {nelec}")
        print(f"  Occupied orbitals: {nocc}")
        print("-" * 70)

    # Build orthogonalizer
    X = symm_orth(S)

    # Initial guess: diagonalize h in orthogonal basis
    Fp = X.T @ h @ X
    eps, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    Cocc = C[:, :nocc]
    P = 2.0 * (Cocc @ Cocc.T)

    E_last = None

    for it in range(1, max_cycle + 1):
        # Build J and K matrices
        J, K = build_jk(P, eri)
        G = J - 0.5 * K
        F = h + G

        # Compute SCF residual: R = FPS - SPF
        R = F @ P @ S - S @ P @ F
        rnorm = np.linalg.norm(R)

        # Apply DIIS extrapolation if provided
        if diis is not None:
            F = diis.update(F, R)

        # Solve Roothaan-Hall via orthogonalization
        Fp = X.T @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp

        # Build new density matrix
        Cocc = C[:, :nocc]
        P_new = 2.0 * (Cocc @ Cocc.T)

        # Compute electronic energy: E_elec = (1/2) Tr[P(h + F)]
        # Here we use G (not extrapolated F) for energy consistency
        F_energy = h + G
        E_elec = 0.5 * np.einsum("pq,pq->", P_new, h + F_energy, optimize=True)
        E_tot = E_elec + Enuc

        dP = np.linalg.norm(P_new - P)
        dE = None if E_last is None else (E_tot - E_last)

        if verbose:
            if dE is None:
                print(f"  it={it:3d}  E={E_tot:+.12f}  |R|={rnorm:.3e}  |dP|={dP:.3e}")
            else:
                print(f"  it={it:3d}  E={E_tot:+.12f}  dE={dE:+.3e}  |R|={rnorm:.3e}  |dP|={dP:.3e}")

        # Check convergence
        if E_last is not None and abs(dE) < conv_tol and rnorm < np.sqrt(conv_tol):
            if verbose:
                print("-" * 70)
                print(f"  SCF converged in {it} iterations")
                print(f"  Final energy: {E_tot:.10f} Hartree")
            return E_tot, eps, C, P_new

        P = P_new
        E_last = E_tot

    raise RuntimeError("SCF did not converge. Try DIIS, damping, or a better initial guess.")


# =============================================================================
# Validation against PySCF
# =============================================================================

def validate_h2o_sto3g():
    """
    Validate minimal RHF SCF against PySCF for H2O/STO-3G.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Cannot run validation.")
        return False

    print("=" * 70)
    print("Lab 6A: Minimal RHF SCF from AO Integrals")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Create molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: H2O")
    print(f"Basis: STO-3G ({mol.nao} basis functions)")
    print(f"Electrons: {mol.nelectron}")

    # Extract integrals from PySCF
    print("\nExtracting integrals from PySCF...")
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")  # Full tensor for small basis
    Enuc = mol.energy_nuc()

    print(f"  S shape: {S.shape}")
    print(f"  h shape: {h.shape}")
    print(f"  ERI shape: {eri.shape}")
    print(f"  E_nuc = {Enuc:.10f} Hartree")

    # Run our educational RHF
    print("\n" + "=" * 70)
    print("Running educational RHF SCF...")
    print("=" * 70)
    E_tot, eps, C, P = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, diis=None, verbose=True)

    # Run PySCF RHF for reference
    print("\n" + "=" * 70)
    print("Running PySCF RHF for reference...")
    print("=" * 70)
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()
    print(f"  PySCF RHF energy: {E_ref:.10f} Hartree")
    print(f"  PySCF iterations: {mf.cycles}")

    # Compare results
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    diff = E_tot - E_ref
    print(f"  Educational RHF: {E_tot:.10f} Hartree")
    print(f"  PySCF RHF:       {E_ref:.10f} Hartree")
    print(f"  Difference:      {diff:+.2e} Hartree")

    # Verify electron count
    n_elec = np.trace(P @ S)
    print(f"\n  Electron count (Tr[PS]): {n_elec:.10f} (expected: {mol.nelectron})")

    # Verify density matrix against PySCF
    P_ref = mf.make_rdm1()
    P_diff = np.linalg.norm(P - P_ref)
    print(f"  ||P - P_ref||_F: {P_diff:.2e}")

    # Verify orbital energies
    eps_ref = mf.mo_energy
    eps_diff = np.linalg.norm(eps - eps_ref)
    print(f"  ||eps - eps_ref||: {eps_diff:.2e}")

    success = abs(diff) < 1e-8
    print("\n" + "=" * 70)
    if success:
        print("VALIDATION PASSED: Energy agrees within 1e-8 Hartree")
    else:
        print("VALIDATION FAILED: Energy difference exceeds tolerance")
    print("=" * 70)

    return success


def validate_h2_minimal():
    """
    Minimal validation for H2/STO-3G (2 AOs, 2 electrons).
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed.")
        return False

    print("\n" + "=" * 70)
    print("Minimal validation: H2/STO-3G")
    print("=" * 70)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri = mol.intor("int2e", aosym="s1")
    Enuc = mol.energy_nuc()

    E_tot, eps, C, P = rhf_scf(S, h, eri, mol.nelectron, Enuc=Enuc, verbose=False)

    mf = scf.RHF(mol)
    mf.verbose = 0
    E_ref = mf.kernel()

    diff = E_tot - E_ref
    print(f"  NAO: {mol.nao}, Electrons: {mol.nelectron}")
    print(f"  Educational RHF: {E_tot:.10f}")
    print(f"  PySCF RHF:       {E_ref:.10f}")
    print(f"  Difference:      {diff:+.2e}")

    success = abs(diff) < 1e-8
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return success


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Lab 6A: Minimal RHF SCF from AO Integrals")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    success1 = validate_h2_minimal()
    success2 = validate_h2o_sto3g()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  H2/STO-3G:  {'PASS' if success1 else 'FAIL'}")
    print(f"  H2O/STO-3G: {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print("\nAll validations PASSED!")
        print("\nKey takeaways:")
        print("  1. RHF SCF is a fixed-point iteration: P -> F(P) -> C -> P")
        print("  2. Orthogonalization via S^{-1/2} converts to standard eigenvalue problem")
        print("  3. The SCF residual R = FPS - SPF vanishes at convergence")
        print("  4. Energy: E = (1/2)Tr[P(h+F)] + E_nuc")
    else:
        print("\nSome validations FAILED")
    print("=" * 70)
