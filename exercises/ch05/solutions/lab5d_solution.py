#!/usr/bin/env python3
"""
Lab 5D Solution: Build J and K Matrices from ERIs

This script demonstrates the construction of Coulomb (J) and Exchange (K)
matrices from electron repulsion integrals (ERIs) and a density matrix.

The definitions in chemist's notation:
    J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}
    K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

These contractions are the core computational step in Hartree-Fock theory.
The Fock matrix for RHF is:
    F = h + J - (1/2) K

where h is the core Hamiltonian (kinetic + nuclear attraction).

Key insights:
1. J represents the classical Coulomb repulsion of the electron density
2. K represents quantum mechanical exchange (fermionic antisymmetry)
3. The "crossed" indices in K reflect exchange of electrons

Learning objectives:
1. Understand the physical meaning of J and K contractions
2. Implement J and K using explicit loops (educational) and einsum (efficient)
3. Verify symmetry properties of J and K
4. Compare to PySCF Fock matrix

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 5: Rys Quadrature in Practice
"""

import numpy as np
from typing import Tuple

# =============================================================================
# Section 1: Explicit Loop Implementations (Educational)
# =============================================================================

def build_J_explicit(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build Coulomb matrix J by explicit summation (educational, slow).

    The Coulomb matrix represents the classical electrostatic interaction:

        J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}

    Physical interpretation:
    - Electron 1 occupies orbital product chi_mu * chi_nu
    - Electron 2's density is P_{lambda sigma} * chi_lambda * chi_sigma
    - J measures the Coulomb repulsion between these distributions

    This O(N^4) implementation is used only for understanding.
    Production codes use screening, density fitting, or integral-direct methods.

    Args:
        eri: Full ERI tensor of shape (N, N, N, N) in chemist's notation (ij|kl)
        P: Density matrix of shape (N, N)

    Returns:
        J: Coulomb matrix of shape (N, N)
    """
    nao = P.shape[0]
    J = np.zeros((nao, nao))

    # J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    J[mu, nu] += eri[mu, nu, lam, sig] * P[lam, sig]

    return J


def build_K_explicit(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build Exchange matrix K by explicit summation (educational, slow).

    The Exchange matrix arises from fermionic antisymmetry:

        K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

    Note the "crossed" indices compared to J: (mu lambda|nu sigma) vs (mu nu|lambda sigma).

    Physical interpretation:
    - K has no classical analog - it's purely quantum mechanical
    - Arises from the Slater determinant requirement (antisymmetry)
    - Represents the exchange interaction between electrons

    Index crossing explanation:
    - In J: electron 1 at positions (mu, nu), electron 2 at (lambda, sigma)
    - In K: electron 1 at positions (mu, lambda), electron 2 at (nu, sigma)
    - The swap nu <-> lambda reflects "exchange" of electron coordinates

    Args:
        eri: Full ERI tensor of shape (N, N, N, N)
        P: Density matrix of shape (N, N)

    Returns:
        K: Exchange matrix of shape (N, N)
    """
    nao = P.shape[0]
    K = np.zeros((nao, nao))

    # K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}
    for mu in range(nao):
        for nu in range(nao):
            for lam in range(nao):
                for sig in range(nao):
                    K[mu, nu] += eri[mu, lam, nu, sig] * P[lam, sig]

    return K


# =============================================================================
# Section 2: Efficient einsum Implementations
# =============================================================================

def build_JK_einsum(eri: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build J and K matrices using numpy einsum (efficient and clear).

    The einsum notation makes the index contractions explicit:

        J_ij = sum_kl (ij|kl) P_lk  =>  einsum('ijkl,lk->ij', eri, P)
        K_ij = sum_kl (ik|jl) P_lk  =>  einsum('ikjl,lk->ij', eri, P)

    Note: We use P_lk (transposed) because of the symmetry P = P^T for real orbitals.
    Using 'lk' instead of 'kl' gives the same result due to P's symmetry.

    The einsum with optimize=True uses an efficient contraction order.

    Args:
        eri: Full ERI tensor of shape (N, N, N, N)
        P: Density matrix of shape (N, N)

    Returns:
        J: Coulomb matrix
        K: Exchange matrix
    """
    # J_mu,nu = (mu nu|lambda sigma) P_sigma,lambda
    # Index labels: i=mu, j=nu, k=lambda, l=sigma
    J = np.einsum('ijkl,lk->ij', eri, P, optimize=True)

    # K_mu,nu = (mu lambda|nu sigma) P_sigma,lambda
    # Index labels: i=mu, k=lambda, j=nu, l=sigma
    K = np.einsum('ikjl,lk->ij', eri, P, optimize=True)

    return J, K


def build_J_einsum(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build only the Coulomb matrix J using einsum.

    J_mu,nu = sum_{lambda,sigma} (mu nu|lambda sigma) P_{lambda sigma}

    Args:
        eri: Full ERI tensor of shape (N, N, N, N)
        P: Density matrix of shape (N, N)

    Returns:
        J: Coulomb matrix
    """
    return np.einsum('ijkl,kl->ij', eri, P, optimize=True)


def build_K_einsum(eri: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Build only the Exchange matrix K using einsum.

    K_mu,nu = sum_{lambda,sigma} (mu lambda|nu sigma) P_{lambda sigma}

    Args:
        eri: Full ERI tensor of shape (N, N, N, N)
        P: Density matrix of shape (N, N)

    Returns:
        K: Exchange matrix
    """
    return np.einsum('ikjl,kl->ij', eri, P, optimize=True)


# =============================================================================
# Section 3: Fock Matrix Construction
# =============================================================================

def build_veff(J: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Build the effective potential (two-electron part of Fock matrix).

    For RHF:
        V_eff = J - (1/2) K

    The factor of 1/2 arises from the closed-shell RHF formalism where
    each spatial orbital is doubly occupied. The exchange interaction
    only occurs between electrons of the same spin.

    Args:
        J: Coulomb matrix
        K: Exchange matrix

    Returns:
        V_eff: Effective potential matrix
    """
    return J - 0.5 * K


def build_fock(h: np.ndarray, J: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Build the Fock matrix for RHF.

        F = h + J - (1/2) K

    where h is the core Hamiltonian (h = T + V, kinetic + nuclear attraction).

    Args:
        h: Core Hamiltonian matrix
        J: Coulomb matrix
        K: Exchange matrix

    Returns:
        F: Fock matrix
    """
    return h + J - 0.5 * K


# =============================================================================
# Section 4: Symmetry and Property Verification
# =============================================================================

def verify_symmetry(J: np.ndarray, K: np.ndarray, tol: float = 1e-10) -> Tuple[bool, dict]:
    """
    Verify that J and K are symmetric (required for real orbitals).

    Both J and K should be symmetric: J = J^T and K = K^T.
    This follows from:
    - The 8-fold symmetry of ERIs
    - The symmetry of the density matrix P = P^T

    Args:
        J, K: Coulomb and exchange matrices
        tol: Tolerance for symmetry check

    Returns:
        passed: True if both matrices are symmetric
        details: Dictionary with symmetry errors
    """
    J_symm_error = np.linalg.norm(J - J.T)
    K_symm_error = np.linalg.norm(K - K.T)

    J_symm = J_symm_error < tol
    K_symm = K_symm_error < tol

    details = {
        "J_symm_error": J_symm_error,
        "K_symm_error": K_symm_error,
        "J_symmetric": J_symm,
        "K_symmetric": K_symm,
    }

    return J_symm and K_symm, details


def compute_energy_contributions(P: np.ndarray, h: np.ndarray,
                                  J: np.ndarray, K: np.ndarray) -> dict:
    """
    Compute energy contributions from each term.

    The RHF electronic energy is:
        E_elec = Tr[P*h] + (1/2)*Tr[P*J] - (1/4)*Tr[P*K]
               = Tr[P*h] + (1/2)*Tr[P*(J - (1/2)*K)]
               = (1/2)*Tr[P*(h + F)]

    Breaking down:
    - One-electron energy: E_1 = Tr[P*h]
    - Coulomb energy: E_J = (1/2)*Tr[P*J]
    - Exchange energy: E_K = (1/4)*Tr[P*K]  (with factor 1/2 from RHF)
    - Two-electron energy: E_2 = E_J - E_K

    Args:
        P: Density matrix
        h: Core Hamiltonian
        J: Coulomb matrix
        K: Exchange matrix

    Returns:
        Dictionary with energy contributions
    """
    # One-electron energy
    E_1 = np.einsum('ij,ij->', P, h)

    # Coulomb energy (factor 1/2 to avoid double counting)
    E_J = 0.5 * np.einsum('ij,ij->', P, J)

    # Exchange energy (factor 1/4 = 1/2 from double counting * 1/2 from RHF)
    E_K = 0.25 * np.einsum('ij,ij->', P, K)

    # Two-electron energy
    E_2 = E_J - E_K

    # Total electronic energy
    E_elec = E_1 + E_2

    return {
        "E_1": E_1,
        "E_J": E_J,
        "E_K": E_K,
        "E_2": E_2,
        "E_elec": E_elec,
    }


# =============================================================================
# Section 5: Validation Against PySCF
# =============================================================================

def validate_against_pyscf():
    """
    Validate J and K construction against PySCF's internal routines.

    We run an RHF calculation to get a converged density matrix, then
    compare our J and K matrices with PySCF's get_jk() method.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Skipping validation.")
        return False

    print("=" * 75)
    print("Validation Against PySCF")
    print("=" * 75)

    # Create H2O molecule with STO-3G basis (small enough for full ERI tensor)
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: H2O")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {mol.nao}")
    print(f"Number of electrons: {mol.nelectron}")
    print("-" * 75)

    # Run RHF to get converged density matrix
    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()

    print(f"\nRHF converged in {mf.cycles} iterations")
    print(f"RHF energy: {mf.e_tot:.10f} Hartree")
    print("-" * 75)

    # Get full ERI tensor (only feasible for small basis!)
    eri = mol.intor("int2e", aosym="s1")
    print(f"\nERI tensor shape: {eri.shape}")
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
    symm_ok, symm_details = verify_symmetry(J, K)
    print(f"  ||J - J^T|| = {symm_details['J_symm_error']:.2e}")
    print(f"  ||K - K^T|| = {symm_details['K_symm_error']:.2e}")
    print(f"  Symmetric: {symm_ok}")

    # Verify Fock matrix
    h = mf.get_hcore()
    F = build_fock(h, J, K)
    F_ref = mf.get_fock(dm=P)
    F_diff = np.linalg.norm(F - F_ref)

    print("\nFock matrix comparison:")
    print(f"  ||F - F_ref||_F = {F_diff:.2e}")

    success = J_diff < 1e-10 and K_diff < 1e-10 and V_eff_diff < 1e-10 and symm_ok

    print("-" * 75)
    if success:
        print("VALIDATION PASSED: J, K, and F agree with PySCF to machine precision")
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
        return True

    print("\n" + "=" * 75)
    print("Comparison: Explicit Loops vs einsum")
    print("=" * 75)

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

    J_diff = np.linalg.norm(J_explicit - J_einsum)
    K_diff = np.linalg.norm(K_explicit - K_einsum)

    print(f"\n||J_explicit - J_einsum|| = {J_diff:.2e}")
    print(f"||K_explicit - K_einsum|| = {K_diff:.2e}")

    passed = J_diff < 1e-12 and K_diff < 1e-12
    print(f"\nExplicit vs einsum: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Section 6: Physical Properties Demonstration
# =============================================================================

def demonstrate_jk_properties():
    """
    Demonstrate physical properties of J and K matrices.
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        print("PySCF not installed. Skipping demonstration.")
        return

    print("\n" + "=" * 75)
    print("Physical Properties of J and K")
    print("=" * 75)

    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    P = mf.make_rdm1()
    h = mf.get_hcore()

    eri = mol.intor("int2e", aosym="s1")
    J, K = build_JK_einsum(eri, P)

    # Property 1: Symmetry
    print("\n1. Symmetry:")
    print(f"   ||J - J^T|| = {np.linalg.norm(J - J.T):.2e}")
    print(f"   ||K - K^T|| = {np.linalg.norm(K - K.T):.2e}")

    # Property 2: Energy contributions
    energies = compute_energy_contributions(P, h, J, K)

    print("\n2. Energy contributions:")
    print(f"   E_1 = Tr[P*h]                = {energies['E_1']:12.8f} Hartree (one-electron)")
    print(f"   E_J = (1/2)*Tr[P*J]          = {energies['E_J']:12.8f} Hartree (Coulomb)")
    print(f"   E_K = (1/4)*Tr[P*K]          = {energies['E_K']:12.8f} Hartree (Exchange)")
    print(f"   E_2 = E_J - E_K              = {energies['E_2']:12.8f} Hartree (two-electron)")
    print(f"   E_elec = E_1 + E_2           = {energies['E_elec']:12.8f} Hartree")
    print(f"   E_nuc                        = {mol.energy_nuc():12.8f} Hartree")
    print(f"   E_total = E_elec + E_nuc     = {energies['E_elec'] + mol.energy_nuc():12.8f} Hartree")
    print(f"   PySCF E_total                = {mf.e_tot:12.8f} Hartree")

    # Property 3: Magnitude comparison
    print("\n3. Magnitude comparison:")
    print(f"   ||J||_F = {np.linalg.norm(J):.6f}")
    print(f"   ||K||_F = {np.linalg.norm(K):.6f}")
    print(f"   ||J||/||K|| = {np.linalg.norm(J)/np.linalg.norm(K):.4f}")
    print("   (J typically larger than K; both important for energetics)")

    # Property 4: Eigenvalue spectrum
    print("\n4. Eigenvalue spectrum:")
    eig_J = np.linalg.eigvalsh(J)
    eig_K = np.linalg.eigvalsh(K)
    print(f"   J eigenvalues: min = {eig_J.min():.6f}, max = {eig_J.max():.6f}")
    print(f"   K eigenvalues: min = {eig_K.min():.6f}, max = {eig_K.max():.6f}")

    # Property 5: Sample matrix elements
    print("\n5. Sample matrix elements (first 3x3 block):")
    print("   J matrix:")
    for i in range(min(3, mol.nao)):
        print(f"   {J[i, :min(3, mol.nao)]}")
    print("\n   K matrix:")
    for i in range(min(3, mol.nao)):
        print(f"   {K[i, :min(3, mol.nao)]}")


# =============================================================================
# Section 7: Physical Interpretation
# =============================================================================

def explain_index_crossing():
    """
    Explain why K has 'crossed' indices compared to J.
    """
    explanation = """
Understanding Index Crossing in J vs K
=======================================

The Coulomb and Exchange matrices differ in their index structure:

COULOMB MATRIX J:
-----------------
    J_mn = sum_ls (mn|ls) P_ls

    - ERI indices: electron 1 at (m,n), electron 2 at (l,s)
    - Physical: density of electron 1 at positions m,n interacts with
      density of electron 2 at positions l,s via 1/r_{12}
    - This is the classical electrostatic interaction

EXCHANGE MATRIX K:
------------------
    K_mn = sum_ls (ml|ns) P_ls

    - ERI indices: electron 1 at (m,l), electron 2 at (n,s)
    - Note the swap: n <-> l compared to J
    - Physical: "exchange" of electron positions

WHY THE SWAP?
-------------
The exchange term arises from the antisymmetry of the wavefunction.
For a Slater determinant, the two-electron density matrix includes:

    Gamma(1,2; 1',2') = P(1,1')P(2,2') - P(1,2')P(2,1')

The first term gives J, the second term (with swapped coordinates) gives K.

PHYSICAL INTERPRETATION:
------------------------
- J: Classical Coulomb repulsion between electron clouds
- K: Quantum mechanical exchange, arising from indistinguishability

The exchange term has NO classical analog. It represents the fact that
electrons are fermions and must have antisymmetric wavefunctions.

IN RHF:
-------
For closed-shell RHF with doubly occupied orbitals:
    F = h + J - (1/2) K

The factor of 1/2 arises because:
- Coulomb interaction occurs between ALL electron pairs
- Exchange interaction only between electrons of SAME SPIN
- In RHF, each spatial orbital has one alpha and one beta electron
- Exchange only affects same-spin pairs, hence 1/2 the "full" exchange
"""
    print(explanation)


# =============================================================================
# Section 8: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 5D demonstration."""

    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 5D: Build J and K Matrices from ERIs" + " " * 28 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Physical explanation
    explain_index_crossing()

    # Validate against PySCF
    pyscf_passed = validate_against_pyscf()

    # Compare implementations
    impl_passed = compare_explicit_vs_einsum()

    # Demonstrate properties
    demonstrate_jk_properties()

    # ==========================================================================
    # Section: einsum notation guide
    # ==========================================================================

    print("\n" + "=" * 75)
    print("einsum Notation Guide for J and K")
    print("=" * 75)

    guide = """
The numpy.einsum function provides a compact notation for tensor contractions.

For J and K matrices:

    # Coulomb: J_ij = sum_kl (ij|kl) P_kl
    J = np.einsum('ijkl,kl->ij', eri, P)

    # Exchange: K_ij = sum_kl (ik|jl) P_kl
    K = np.einsum('ikjl,kl->ij', eri, P)

Reading the einsum strings:
- Left of comma: index labels for input arrays
- Right of comma, before '->': second input's indices
- After '->': output indices

For J: 'ijkl,kl->ij'
- eri has indices i,j,k,l
- P has indices k,l (summed over)
- Output J has indices i,j

For K: 'ikjl,kl->ij'
- eri has indices i,k,j,l (note the rearrangement!)
- P has indices k,l (summed over)
- Output K has indices i,j

The rearranged indices in K reflect the "crossed" index structure:
    K_ij = (ik|jl) P_kl  vs  J_ij = (ij|kl) P_kl
"""
    print(guide)

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 75)
    print("Lab 5D Summary")
    print("=" * 75)
    print(f"PySCF validation:         {'PASS' if pyscf_passed else 'FAIL'}")
    print(f"Explicit vs einsum:       {'PASS' if impl_passed else 'FAIL'}")
    print("-" * 75)
    if pyscf_passed and impl_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("SOME VALIDATIONS FAILED - Review output above")
    print("=" * 75)


if __name__ == "__main__":
    main()
