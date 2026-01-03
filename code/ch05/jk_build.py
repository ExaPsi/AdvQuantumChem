#!/usr/bin/env python3
"""
jk_build.py - Coulomb (J) and Exchange (K) Matrix Construction (Lab 5D)

This module implements the construction of Coulomb and Exchange matrices
from electron repulsion integrals (ERIs) and a density matrix.

The definitions (chemist's notation):
    J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}
    K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

These contractions are the core computational step in Hartree-Fock theory.
The Fock matrix is:
    F = h + J - (1/2) K    (for RHF)

where h is the core Hamiltonian (kinetic + nuclear attraction).

Key insights:
1. J represents the classical Coulomb repulsion of the electron density
2. K represents quantum mechanical exchange (fermionic antisymmetry)
3. The "crossed" indices in K reflect exchange of electrons

References:
    - Chapter 5, Section 7: From ERIs to Coulomb and exchange
    - Eq. (5.45): J contraction
    - Eq. (5.46): K contraction
    - Algorithm 5.2: Build J and K from full ERIs

Course: 2302638 Advanced Quantum Chemistry
"""

import numpy as np
from typing import Tuple


def build_J_explicit(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build Coulomb matrix J by explicit summation (educational, slow).

    J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}

    This is O(N^4) and used only for understanding. Production codes
    use more efficient algorithms (screening, density fitting).

    Parameters
    ----------
    eri : np.ndarray
        Full ERI tensor of shape (N, N, N, N)
    P : np.ndarray
        Density matrix of shape (N, N)

    Returns
    -------
    J : np.ndarray
        Coulomb matrix of shape (N, N)
    """
    nao = P.shape[0]
    J = np.zeros((nao, nao))

    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    J[mu, nu] += eri[mu, nu, lam, sig] * P[lam, sig]

    return J


def build_K_explicit(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build Exchange matrix K by explicit summation (educational, slow).

    K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

    Note the "crossed" indices compared to J: (mu lambda|nu sigma) vs (mu nu|lambda sigma).
    This reflects the exchange nature of the integral.

    Parameters
    ----------
    eri : np.ndarray
        Full ERI tensor of shape (N, N, N, N)
    P : np.ndarray
        Density matrix of shape (N, N)

    Returns
    -------
    K : np.ndarray
        Exchange matrix of shape (N, N)
    """
    nao = P.shape[0]
    K = np.zeros((nao, nao))

    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    K[mu, nu] += eri[mu, lam, nu, sig] * P[lam, sig]

    return K


def build_JK_einsum(eri: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K matrices using numpy einsum (efficient and clear).

    J_ij = sum_kl (ij|kl) P_lk  =>  einsum('ijkl,lk->ij', eri, P)
    K_ij = sum_kl (ik|jl) P_lk  =>  einsum('ikjl,lk->ij', eri, P)

    Note: einsum with optimize=True is much faster for large arrays.

    Parameters
    ----------
    eri : np.ndarray
        Full ERI tensor of shape (N, N, N, N)
    P : np.ndarray
        Density matrix of shape (N, N)

    Returns
    -------
    J : np.ndarray
        Coulomb matrix
    K : np.ndarray
        Exchange matrix
    """
    # J_mu,nu = (mu nu|lambda sigma) P_sigma,lambda
    # Using transposed density (lk instead of kl) to match standard conventions
    J = np.einsum('ijkl,lk->ij', eri, P, optimize=True)

    # K_mu,nu = (mu lambda|nu sigma) P_sigma,lambda
    K = np.einsum('ikjl,lk->ij', eri, P, optimize=True)

    return J, K


def build_veff(J: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Build the effective potential (two-electron part of Fock matrix).

    V_eff = J - (1/2) K    (for RHF)

    The factor of 1/2 arises from the closed-shell RHF formalism where
    each orbital is doubly occupied.

    Parameters
    ----------
    J : np.ndarray
        Coulomb matrix
    K : np.ndarray
        Exchange matrix

    Returns
    -------
    V_eff : np.ndarray
        Effective potential matrix
    """
    return J - 0.5 * K


def verify_symmetry(J: np.ndarray, K: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verify that J and K are symmetric (required for real orbitals).

    Parameters
    ----------
    J, K : np.ndarray
        Coulomb and exchange matrices
    tol : float
        Tolerance for symmetry check

    Returns
    -------
    bool
        True if both matrices are symmetric
    """
    J_symm = np.allclose(J, J.T, atol=tol)
    K_symm = np.allclose(K, K.T, atol=tol)

    if not J_symm:
        print(f"WARNING: J is not symmetric! ||J - J^T|| = {np.linalg.norm(J - J.T):.2e}")
    if not K_symm:
        print(f"WARNING: K is not symmetric! ||K - K^T|| = {np.linalg.norm(K - K.T):.2e}")

    return J_symm and K_symm


# =============================================================================
# Validation against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate J and K construction against PySCF's internal routines.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("Validation: J and K matrices against PySCF")
    print("=" * 70)

    # Create H2O molecule with STO-3G basis
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"Molecule: H2O")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {mol.nao}")
    print(f"Number of electrons: {mol.nelectron}")
    print("-" * 70)

    # Run RHF to get converged density matrix
    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    print(f"RHF converged in {mf.cycles} iterations")
    print(f"RHF energy: {mf.e_tot:.10f} Hartree")
    print("-" * 70)

    # Get full ERI tensor (only feasible for small basis!)
    eri = mol.intor("int2e", aosym="s1")
    print(f"ERI tensor shape: {eri.shape}")
    print(f"ERI tensor memory: {eri.nbytes / 1e6:.2f} MB")

    # Build J and K using einsum
    J, K = build_JK_einsum(eri, P)

    # Get PySCF reference
    J_ref, K_ref = mf.get_jk(mol, P)

    # Compare
    J_diff = np.linalg.norm(J - J_ref)
    K_diff = np.linalg.norm(K - K_ref)

    print("\nComparison with PySCF get_jk():")
    print(f"  ||J - J_ref||_F = {J_diff:.2e}")
    print(f"  ||K - K_ref||_F = {K_diff:.2e}")

    # Also compare V_eff
    V_eff = build_veff(J, K)
    V_eff_ref = mf.get_veff(mol, P)
    V_eff_diff = np.linalg.norm(V_eff - V_eff_ref)

    print(f"  ||V_eff - V_eff_ref||_F = {V_eff_diff:.2e}")

    # Verify symmetry
    print("\nSymmetry check:")
    symm_ok = verify_symmetry(J, K)

    # Check individual matrix elements
    print("\nSample matrix elements (first 3x3 block):")
    print("J matrix:")
    print(J[:3, :3])
    print("\nK matrix:")
    print(K[:3, :3])

    success = J_diff < 1e-10 and K_diff < 1e-10 and V_eff_diff < 1e-10 and symm_ok

    print("-" * 70)
    if success:
        print("VALIDATION PASSED: J and K agree with PySCF to machine precision")
    else:
        print("VALIDATION FAILED: Discrepancy detected")

    return success


def compare_explicit_vs_einsum():
    """
    Compare explicit loop implementation vs einsum for correctness.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Skipping comparison.")
        return

    print("\nComparison: Explicit loops vs einsum")
    print("=" * 70)

    # Small molecule for feasible explicit computation
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    eri = mol.intor("int2e", aosym="s1")

    # Explicit (slow)
    J_explicit = build_J_explicit(eri, P)
    K_explicit = build_K_explicit(eri, P)

    # Einsum (fast)
    J_einsum, K_einsum = build_JK_einsum(eri, P)

    print(f"||J_explicit - J_einsum|| = {np.linalg.norm(J_explicit - J_einsum):.2e}")
    print(f"||K_explicit - K_einsum|| = {np.linalg.norm(K_explicit - K_einsum):.2e}")


def demonstrate_jk_properties():
    """
    Demonstrate physical properties of J and K matrices.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Skipping demonstration.")
        return

    print("\nPhysical properties of J and K")
    print("=" * 70)

    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    eri = mol.intor("int2e", aosym="s1")
    J, K = build_JK_einsum(eri, P)

    # Property 1: J and K are symmetric
    print("\n1. Symmetry:")
    print(f"   ||J - J^T|| = {np.linalg.norm(J - J.T):.2e}")
    print(f"   ||K - K^T|| = {np.linalg.norm(K - K.T):.2e}")

    # Property 2: Energy contributions
    E_J = 0.5 * np.einsum('ij,ij->', P, J)  # Coulomb energy (1/2 factor for double counting)
    E_K = 0.5 * np.einsum('ij,ij->', P, K)  # Exchange energy

    print("\n2. Energy contributions:")
    print(f"   (1/2) Tr[P*J] = {E_J:12.8f} Hartree (Coulomb)")
    print(f"   (1/4) Tr[P*K] = {0.5*E_K:12.8f} Hartree (Exchange, with 1/2 from RHF)")
    print(f"   Note: Full 2e energy = (1/2) Tr[P*J] - (1/4) Tr[P*K] = {E_J - 0.5*E_K:12.8f} Hartree")

    # Property 3: Comparison of J and K magnitudes
    print("\n3. Magnitude comparison:")
    print(f"   ||J||_F = {np.linalg.norm(J):.6f}")
    print(f"   ||K||_F = {np.linalg.norm(K):.6f}")
    print(f"   ||J||/||K|| = {np.linalg.norm(J)/np.linalg.norm(K):.4f}")

    # Property 4: Eigenvalue spectrum
    print("\n4. Eigenvalue spectrum:")
    eig_J = np.linalg.eigvalsh(J)
    eig_K = np.linalg.eigvalsh(K)
    print(f"   J eigenvalues: min = {eig_J.min():.6f}, max = {eig_J.max():.6f}")
    print(f"   K eigenvalues: min = {eig_K.min():.6f}, max = {eig_K.max():.6f}")


def understand_index_crossing():
    """
    Demonstrate why K has 'crossed' indices compared to J.

    J_mn = sum_ls (mn|ls) P_ls   -- electron 1 at (m,n), electron 2 at (l,s)
    K_mn = sum_ls (ml|ns) P_ls   -- exchange: swap n<->l

    The exchange integral represents the quantum mechanical exchange of
    electrons due to fermionic antisymmetry.
    """
    print("\nUnderstanding index crossing in J vs K")
    print("=" * 70)

    print("""
The Coulomb integral J represents the classical electrostatic interaction:
    J_mn = sum_ls (mn|ls) P_ls

    - Electron 1 density: rho_1 = sum_mn P_mn * chi_m(r1) * chi_n(r1)
    - Electron 2 density: rho_2 = sum_ls P_ls * chi_l(r2) * chi_s(r2)
    - J represents: integral rho_1(r1) * (1/r12) * rho_2(r2) dr1 dr2

The Exchange integral K arises from fermionic antisymmetry:
    K_mn = sum_ls (ml|ns) P_ls

    - Notice the 'crossed' indices: (ml|ns) instead of (mn|ls)
    - This comes from the antisymmetric structure of Slater determinants
    - It represents quantum mechanical indistinguishability of electrons

In physicist's notation:
    J_mn = <mn|ls> P_ls   (electron 1 at positions m,l; electron 2 at n,s)
    K_mn = <ml|ns> P_ls   (swapped to show exchange)

The exchange term has no classical analog - it is purely quantum mechanical!
""")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lab 5D: Coulomb (J) and Exchange (K) Matrix Construction")
    print("Course: 2302638 Advanced Quantum Chemistry")
    print("=" * 70)

    # Run validation
    success = validate_against_pyscf()

    # Compare implementations
    compare_explicit_vs_einsum()

    # Demonstrate properties
    demonstrate_jk_properties()

    # Educational discussion
    understand_index_crossing()

    print("\n" + "=" * 70)
    if success:
        print("All validations PASSED")
    else:
        print("Validation FAILED (or PySCF not available)")
    print("=" * 70)
