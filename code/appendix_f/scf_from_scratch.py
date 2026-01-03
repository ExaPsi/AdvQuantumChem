#!/usr/bin/env python3
"""
Minimal SCF Implementation Using Only PySCF Integrals

Implements a complete RHF SCF loop using only PySCF for integral evaluation.
All Fock matrix construction and diagonalization is done manually.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

Key equations:
  Core Hamiltonian: h = T + V
  Density matrix:   P = 2 * C_occ @ C_occ.T  (RHF, closed-shell)
  Coulomb:          J_ij = sum_kl (ij|kl) P_kl
  Exchange:         K_ij = sum_kl (ik|jl) P_kl
  Fock matrix:      F = h + J - 0.5*K
  Energy:           E = 0.5 * Tr[P(h+F)] + E_nuc
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import gto, scf


def symmetric_orthogonalizer(S: np.ndarray, thresh: float = 1e-8) -> np.ndarray:
    """Compute symmetric orthogonalization matrix X = S^(-1/2).

    Args:
        S: Overlap matrix
        thresh: Eigenvalue threshold for linear dependence

    Returns:
        X: Orthogonalization matrix such that X^T S X = I
    """
    # Diagonalize S: S = U @ diag(s) @ U^T
    s, U = np.linalg.eigh(S)

    # Check for near-linear dependence
    n_dep = np.sum(s < thresh)
    if n_dep > 0:
        print(f"  Warning: {n_dep} near-linear-dependent basis functions")
        # Remove linear dependencies
        mask = s >= thresh
        s = s[mask]
        U = U[:, mask]

    # X = U @ diag(s^(-1/2)) @ U^T
    X = U @ np.diag(1.0 / np.sqrt(s)) @ U.T

    return X


def build_density_matrix(C: np.ndarray, n_occ: int) -> np.ndarray:
    """Build RHF density matrix from MO coefficients.

    Args:
        C: MO coefficient matrix (nao x nmo)
        n_occ: Number of doubly-occupied orbitals

    Returns:
        P: Density matrix (nao x nao)
    """
    # P_ij = 2 * sum_a C_ia C_ja (sum over occupied orbitals)
    C_occ = C[:, :n_occ]
    P = 2.0 * C_occ @ C_occ.T
    return P


def build_fock_matrix(h: np.ndarray, eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build Fock matrix from integrals and density.

    Args:
        h: Core Hamiltonian (nao x nao)
        eri: Two-electron integrals (nao x nao x nao x nao)
        P: Density matrix (nao x nao)

    Returns:
        F: Fock matrix (nao x nao)
    """
    # Coulomb: J_ij = sum_kl (ij|kl) P_kl
    J = np.einsum('ijkl,kl->ij', eri, P)

    # Exchange: K_ij = sum_kl (ik|jl) P_kl
    K = np.einsum('ikjl,kl->ij', eri, P)

    # Fock: F = h + J - 0.5*K (factor of 0.5 for RHF)
    F = h + J - 0.5 * K

    return F


def compute_hf_energy(P: np.ndarray, h: np.ndarray, F: np.ndarray,
                      E_nuc: float) -> float:
    """Compute total HF energy.

    Args:
        P: Density matrix
        h: Core Hamiltonian
        F: Fock matrix
        E_nuc: Nuclear repulsion energy

    Returns:
        E_tot: Total HF energy
    """
    # E_elec = 0.5 * Tr[P(h+F)]
    E_elec = 0.5 * np.einsum('ij,ij->', P, h + F)
    E_tot = E_elec + E_nuc
    return E_tot


def solve_roothaan_hall(F: np.ndarray, S: np.ndarray) -> tuple:
    """Solve Roothaan-Hall equations FC = SCe.

    Args:
        F: Fock matrix
        S: Overlap matrix

    Returns:
        e: Orbital energies
        C: MO coefficients
    """
    # Solve generalized eigenvalue problem
    e, C = eigh(F, S)
    return e, C


def run_scf(mol, max_iter: int = 50, conv_tol: float = 1e-10,
            verbose: bool = True) -> dict:
    """Run RHF SCF calculation.

    Args:
        mol: PySCF molecule object
        max_iter: Maximum SCF iterations
        conv_tol: Energy convergence threshold
        verbose: Print iteration info

    Returns:
        Dictionary with results
    """
    if verbose:
        print("\n" + "=" * 50)
        print("Starting SCF from scratch")
        print("=" * 50)

    # Extract integrals
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    h = T + V
    eri = mol.intor("int2e", aosym="s1")
    E_nuc = mol.energy_nuc()

    nao = mol.nao
    n_occ = mol.nelectron // 2

    if verbose:
        print(f"\n  nao = {nao}, n_elec = {mol.nelectron}, n_occ = {n_occ}")
        print(f"  E_nuc = {E_nuc:.10f} Hartree")

    # Initial guess: diagonalize core Hamiltonian
    if verbose:
        print("\n  Initial guess: Core Hamiltonian")
    e, C = solve_roothaan_hall(h, S)
    P = build_density_matrix(C, n_occ)

    # SCF loop
    E_old = 0.0
    converged = False

    if verbose:
        print("\n  Iter      E_total          dE            ||dP||")
        print("  " + "-" * 50)

    for iteration in range(1, max_iter + 1):
        # Build Fock matrix
        F = build_fock_matrix(h, eri, P)

        # Compute energy
        E_tot = compute_hf_energy(P, h, F, E_nuc)

        # Solve Roothaan-Hall
        e, C = solve_roothaan_hall(F, S)

        # New density
        P_new = build_density_matrix(C, n_occ)

        # Convergence check
        dE = E_tot - E_old
        dP = np.linalg.norm(P_new - P)

        if verbose:
            print(f"  {iteration:4d}   {E_tot:16.10f}   {dE:12.2e}   {dP:12.2e}")

        if abs(dE) < conv_tol and dP < conv_tol:
            converged = True
            if verbose:
                print("  " + "-" * 50)
                print(f"  Converged in {iteration} iterations!")
            break

        E_old = E_tot
        P = P_new

    if not converged and verbose:
        print(f"  Warning: SCF did not converge in {max_iter} iterations")

    return {
        'converged': converged,
        'e_tot': E_tot,
        'e_elec': E_tot - E_nuc,
        'mo_energy': e,
        'mo_coeff': C,
        'dm': P,
        'fock': F,
        'iterations': iteration
    }


def main():
    print("=" * 70)
    print("Minimal SCF Implementation Using Only PySCF Integrals")
    print("=" * 70)

    # =========================================================================
    # Test 1: H2 / STO-3G
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: H2 / STO-3G")
    print("=" * 50)

    mol_h2 = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Our implementation
    result_h2 = run_scf(mol_h2)

    # PySCF reference
    mf_h2 = scf.RHF(mol_h2)
    E_pyscf_h2 = mf_h2.kernel()

    print(f"\n  Validation:")
    print(f"    Our energy:    {result_h2['e_tot']:.10f} Hartree")
    print(f"    PySCF energy:  {E_pyscf_h2:.10f} Hartree")
    print(f"    Difference:    {abs(result_h2['e_tot'] - E_pyscf_h2):.2e} Hartree")

    assert np.isclose(result_h2['e_tot'], E_pyscf_h2, atol=1e-8), "H2 energy mismatch!"
    print("    PASSED")

    # =========================================================================
    # Test 2: H2O / STO-3G
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: H2O / STO-3G")
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

    result_h2o = run_scf(mol_h2o)

    mf_h2o = scf.RHF(mol_h2o)
    E_pyscf_h2o = mf_h2o.kernel()

    print(f"\n  Validation:")
    print(f"    Our energy:    {result_h2o['e_tot']:.10f} Hartree")
    print(f"    PySCF energy:  {E_pyscf_h2o:.10f} Hartree")
    print(f"    Difference:    {abs(result_h2o['e_tot'] - E_pyscf_h2o):.2e} Hartree")

    assert np.isclose(result_h2o['e_tot'], E_pyscf_h2o, atol=1e-8), "H2O energy mismatch!"
    print("    PASSED")

    # =========================================================================
    # Test 3: LiH / cc-pVDZ
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: LiH / cc-pVDZ")
    print("=" * 50)

    mol_lih = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    result_lih = run_scf(mol_lih)

    mf_lih = scf.RHF(mol_lih)
    E_pyscf_lih = mf_lih.kernel()

    print(f"\n  Validation:")
    print(f"    Our energy:    {result_lih['e_tot']:.10f} Hartree")
    print(f"    PySCF energy:  {E_pyscf_lih:.10f} Hartree")
    print(f"    Difference:    {abs(result_lih['e_tot'] - E_pyscf_lih):.2e} Hartree")

    assert np.isclose(result_lih['e_tot'], E_pyscf_lih, atol=1e-8), "LiH energy mismatch!"
    print("    PASSED")

    # =========================================================================
    # Detailed Output for H2
    # =========================================================================
    print("\n" + "=" * 50)
    print("Detailed Analysis: H2 / STO-3G")
    print("=" * 50)

    print(f"\n  Orbital energies (Hartree):")
    for i, e in enumerate(result_h2['mo_energy']):
        occ = "occ" if i < mol_h2.nelectron // 2 else "vir"
        print(f"    MO {i}: {e:12.6f}  ({occ})")

    print(f"\n  MO coefficients:")
    print(f"    C =\n{result_h2['mo_coeff']}")

    print(f"\n  HOMO-LUMO gap:")
    n_occ = mol_h2.nelectron // 2
    homo = result_h2['mo_energy'][n_occ - 1]
    lumo = result_h2['mo_energy'][n_occ]
    gap = lumo - homo
    print(f"    HOMO = {homo:.6f} Hartree")
    print(f"    LUMO = {lumo:.6f} Hartree")
    print(f"    Gap  = {gap:.6f} Hartree = {gap * 27.211:.2f} eV")

    # Verify electron count
    S_h2 = mol_h2.intor("int1e_ovlp")
    n_elec = np.trace(result_h2['dm'] @ S_h2)
    print(f"\n  Electron count: Tr[PS] = {n_elec:.6f} (expected: {mol_h2.nelectron})")

    print("\n" + "=" * 70)
    print("All SCF tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
