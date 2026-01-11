#!/usr/bin/env python3
"""
Lab 6A Solution: Minimal RHF SCF from AO Integrals

This script implements Algorithm 6.1 from Chapter 6: a complete Restricted
Hartree-Fock (RHF) Self-Consistent Field procedure using in-core AO integrals.

Learning Objectives:
1. Understand the SCF workflow: P -> F(P) -> (eps, C) -> P_new -> repeat
2. Build the Fock matrix F = h + J - 0.5*K from density and integrals
3. Solve the Roothaan-Hall equations FC = SCe via orthogonalization
4. Update the density matrix P = 2 * C_occ @ C_occ.T (RHF closed-shell)
5. Track convergence using energy change and RMSD(P)

Test System: H2O in STO-3G basis

Physical Insight:
-----------------
The SCF procedure finds the optimal single-determinant wavefunction by
iteratively adjusting the molecular orbitals until they are eigenfunctions
of the Fock operator built from those same orbitals (self-consistency).

The Fock operator F = h + J - 0.5*K represents the effective one-electron
Hamiltonian where each electron "feels" the average field of all other
electrons through the Coulomb (J) and exchange (K) operators.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 6: Hartree-Fock Self-Consistent Field as an Integral-Driven Algorithm
"""

import numpy as np
from pyscf import gto, scf
from typing import Tuple, Optional


# =============================================================================
# Section 1: Core SCF Functions
# =============================================================================

def symmetric_orthogonalizer(S: np.ndarray,
                              threshold: float = 1e-10) -> np.ndarray:
    """
    Build the symmetric orthogonalizer X = S^(-1/2) with linear dependence handling.

    The symmetric orthogonalizer transforms from the non-orthogonal AO basis
    to an orthonormal basis: X^T S X = I.

    For near-linear dependent bases, eigenvalues below threshold are discarded
    to prevent numerical instability.

    Args:
        S: Overlap matrix (nao x nao)
        threshold: Eigenvalue threshold for discarding near-linear dependence

    Returns:
        X: Orthogonalizer matrix (nao x n_independent)
    """
    # Diagonalize S = U s U^T
    eigenvalues, U = np.linalg.eigh(S)

    # Keep only eigenvalues above threshold
    keep = eigenvalues > threshold
    n_discarded = np.sum(~keep)

    if n_discarded > 0:
        print(f"  Warning: Discarding {n_discarded} near-linear dependent basis functions")

    # Build X = U s^(-1/2) U^T (for kept eigenvalues)
    s_inv_sqrt = 1.0 / np.sqrt(eigenvalues[keep])
    X = U[:, keep] @ np.diag(s_inv_sqrt) @ U[:, keep].T

    return X


def build_jk_matrices(P: np.ndarray,
                      eri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Coulomb (J) and Exchange (K) matrices from density and ERIs.

    For RHF, the density matrix P includes the factor of 2 for double occupancy:
        P_uv = 2 * sum_i C_ui C_vi  (sum over occupied orbitals)

    The contractions use chemist's notation ERIs (uv|ls):
        J_uv = sum_{ls} (uv|ls) P_ls   (Coulomb: same indices on bra/ket)
        K_uv = sum_{ls} (ul|vs) P_ls   (Exchange: crossed indices)

    Args:
        P: Density matrix (nao x nao), includes factor of 2 for RHF
        eri: Two-electron integrals in chemist's notation (nao, nao, nao, nao)

    Returns:
        J: Coulomb matrix (nao x nao)
        K: Exchange matrix (nao x nao)
    """
    # J_uv = sum_{ls} (uv|ls) P_ls
    # Contraction over last two indices of ERI
    J = np.einsum('uvls,ls->uv', eri, P, optimize=True)

    # K_uv = sum_{ls} (ul|vs) P_ls
    # Note the index rearrangement: contract l with 2nd and s with 4th index
    K = np.einsum('ulvs,ls->uv', eri, P, optimize=True)

    return J, K


def build_fock_matrix(h: np.ndarray,
                      J: np.ndarray,
                      K: np.ndarray) -> np.ndarray:
    """
    Build the RHF Fock matrix F = h + J - 0.5*K.

    The Fock matrix is the effective one-electron Hamiltonian:
    - h: core Hamiltonian (kinetic + nuclear attraction)
    - J: Coulomb repulsion (classical electrostatics)
    - K: Exchange interaction (quantum mechanical, from antisymmetry)

    The factor of 0.5 on K arises from spin integration in closed-shell RHF.

    Args:
        h: Core Hamiltonian matrix (nao x nao)
        J: Coulomb matrix (nao x nao)
        K: Exchange matrix (nao x nao)

    Returns:
        F: Fock matrix (nao x nao)
    """
    F = h + J - 0.5 * K
    return F


def solve_roothaan_hall(F: np.ndarray,
                        X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Roothaan-Hall equations FC = SCe via orthogonalization.

    The generalized eigenvalue problem FC = SCe is transformed to an
    ordinary eigenvalue problem F'C' = C'e by the substitution:
        F' = X^T F X
        C = X C'

    This works because X^T S X = I, so X transforms to an orthonormal basis.

    Args:
        F: Fock matrix in AO basis (nao x nao)
        X: Orthogonalizer X = S^(-1/2) (nao x n_independent)

    Returns:
        orbital_energies: Eigenvalues (orbital energies) sorted ascending
        C: MO coefficients in AO basis (nao x nao)
    """
    # Transform Fock matrix to orthonormal basis
    F_prime = X.T @ F @ X

    # Solve ordinary eigenvalue problem
    orbital_energies, C_prime = np.linalg.eigh(F_prime)

    # Transform MO coefficients back to AO basis
    C = X @ C_prime

    return orbital_energies, C


def build_density_matrix(C: np.ndarray,
                         n_occ: int) -> np.ndarray:
    """
    Build the RHF density matrix from MO coefficients.

    For closed-shell RHF, each spatial orbital is doubly occupied:
        P_uv = 2 * sum_i C_ui C_vi  (i runs over occupied orbitals)

    The factor of 2 accounts for alpha and beta electrons in each orbital.

    Args:
        C: MO coefficients (nao x nmo)
        n_occ: Number of doubly occupied spatial orbitals (n_electrons // 2)

    Returns:
        P: Density matrix (nao x nao)
    """
    C_occ = C[:, :n_occ]
    P = 2.0 * C_occ @ C_occ.T
    return P


def compute_scf_energy(P: np.ndarray,
                       h: np.ndarray,
                       F: np.ndarray,
                       E_nuc: float) -> Tuple[float, float]:
    """
    Compute the SCF electronic and total energies.

    The RHF electronic energy can be written as:
        E_elec = (1/2) Tr[P(h + F)]

    This form avoids double-counting the two-electron interaction because
    F = h + G where G = J - 0.5*K, and:
        E_elec = Tr[P h] + (1/2) Tr[P G]
               = (1/2) Tr[P h] + (1/2) Tr[P (h + G)]
               = (1/2) Tr[P (h + F)]

    Args:
        P: Density matrix
        h: Core Hamiltonian
        F: Fock matrix
        E_nuc: Nuclear repulsion energy

    Returns:
        E_elec: Electronic energy
        E_total: Total energy (electronic + nuclear)
    """
    E_elec = 0.5 * np.einsum('uv,uv->', P, h + F, optimize=True)
    E_total = E_elec + E_nuc
    return E_elec, E_total


def compute_scf_residual(F: np.ndarray,
                         P: np.ndarray,
                         S: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the SCF residual R = FPS - SPF and its Frobenius norm.

    At SCF convergence, [F, P]_S = FPS - SPF = 0, meaning F and P commute
    in the S-metric (they share the same eigenvectors in the orthonormal basis).

    The norm ||R||_F provides a basis-invariant measure of self-consistency
    and is commonly used as a convergence criterion.

    Args:
        F: Fock matrix
        P: Density matrix
        S: Overlap matrix

    Returns:
        R: Residual matrix
        norm_R: Frobenius norm of R
    """
    R = F @ P @ S - S @ P @ F
    norm_R = np.linalg.norm(R)
    return R, norm_R


def compute_density_rmsd(P_new: np.ndarray,
                         P_old: np.ndarray) -> float:
    """
    Compute the RMSD between density matrices.

    RMSD = sqrt(sum_uv (P_new - P_old)^2 / n_elements)

    This provides a measure of how much the density changed between iterations.

    Args:
        P_new: New density matrix
        P_old: Previous density matrix

    Returns:
        rmsd: Root mean square deviation
    """
    diff = P_new - P_old
    n_elements = diff.size
    rmsd = np.sqrt(np.sum(diff**2) / n_elements)
    return rmsd


# =============================================================================
# Section 2: Main SCF Driver
# =============================================================================

def rhf_scf(mol: gto.Mole,
            max_iter: int = 50,
            conv_E: float = 1e-8,
            conv_rmsd: float = 1e-8,
            verbose: bool = True) -> dict:
    """
    Run a minimal RHF SCF calculation from AO integrals.

    This implements Algorithm 6.1 from the lecture notes:
    1. Extract integrals from PySCF
    2. Build initial guess from core Hamiltonian
    3. SCF loop: P -> F -> solve FC=SCe -> P_new -> check convergence

    Args:
        mol: PySCF Mole object with defined geometry and basis
        max_iter: Maximum SCF iterations
        conv_E: Energy convergence threshold (Hartree)
        conv_rmsd: Density RMSD convergence threshold
        verbose: Print iteration details

    Returns:
        Dictionary with SCF results:
            - E_total: Total energy
            - E_elec: Electronic energy
            - C: MO coefficients
            - eps: Orbital energies
            - P: Final density matrix
            - converged: Convergence flag
            - n_iter: Number of iterations
            - history: List of (E, rmsd_P) per iteration
    """
    # =========================================================================
    # Step 1: Extract integrals from PySCF
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("Minimal RHF SCF Implementation (Algorithm 6.1)")
        print("=" * 70)
        print(f"\nSystem: {mol.atom}")
        print(f"Basis: {mol.basis}")
        print(f"Electrons: {mol.nelectron}")
        print(f"AO functions: {mol.nao_nr()}")

    S = mol.intor("int1e_ovlp")        # Overlap matrix
    T = mol.intor("int1e_kin")         # Kinetic energy
    V = mol.intor("int1e_nuc")         # Nuclear attraction
    h = T + V                           # Core Hamiltonian
    eri = mol.intor("int2e", aosym="s1")  # Full ERI tensor (uv|ls)
    E_nuc = mol.energy_nuc()           # Nuclear repulsion

    nao = mol.nao_nr()
    n_elec = mol.nelectron
    n_occ = n_elec // 2  # Number of doubly occupied orbitals

    if verbose:
        print(f"\nIntegral shapes:")
        print(f"  S: {S.shape}")
        print(f"  h: {h.shape}")
        print(f"  ERI: {eri.shape}")
        print(f"  Nuclear repulsion: {E_nuc:.10f} Hartree")

    # =========================================================================
    # Step 2: Build orthogonalizer and initial guess
    # =========================================================================
    if verbose:
        print("\nBuilding orthogonalizer X = S^(-1/2)...")

    X = symmetric_orthogonalizer(S)

    # Initial guess: diagonalize h in the S-metric
    if verbose:
        print("Building initial guess from core Hamiltonian...")

    eps, C = solve_roothaan_hall(h, X)
    P = build_density_matrix(C, n_occ)

    # Initial energy
    J, K = build_jk_matrices(P, eri)
    F = build_fock_matrix(h, J, K)
    E_elec, E_total = compute_scf_energy(P, h, F, E_nuc)

    if verbose:
        print(f"\nInitial guess energy: {E_total:.10f} Hartree")
        print(f"\nStarting SCF iterations...")
        print("-" * 70)
        print(f"{'Iter':>4}  {'E_total (Hartree)':>20}  {'dE':>12}  {'RMSD(P)':>12}  {'|R|':>12}")
        print("-" * 70)

    # =========================================================================
    # Step 3: SCF iteration loop
    # =========================================================================
    history = []
    converged = False
    E_old = E_total

    for iteration in range(1, max_iter + 1):
        # Build Fock matrix from current density
        J, K = build_jk_matrices(P, eri)
        F = build_fock_matrix(h, J, K)

        # Compute residual for monitoring
        R, norm_R = compute_scf_residual(F, P, S)

        # Solve Roothaan-Hall equations
        eps, C = solve_roothaan_hall(F, X)

        # Build new density matrix
        P_new = build_density_matrix(C, n_occ)

        # Compute energy with new density
        # Note: We use the old Fock matrix with new density for energy
        # This is consistent with the half-trace formula
        J_new, K_new = build_jk_matrices(P_new, eri)
        F_new = build_fock_matrix(h, J_new, K_new)
        E_elec, E_total = compute_scf_energy(P_new, h, F_new, E_nuc)

        # Compute convergence metrics
        dE = E_total - E_old
        rmsd_P = compute_density_rmsd(P_new, P)

        # Store history
        history.append({'E': E_total, 'dE': dE, 'rmsd_P': rmsd_P, 'norm_R': norm_R})

        if verbose:
            print(f"{iteration:4d}  {E_total:+20.12f}  {dE:+12.3e}  {rmsd_P:12.3e}  {norm_R:12.3e}")

        # Check convergence
        if abs(dE) < conv_E and rmsd_P < conv_rmsd:
            converged = True
            if verbose:
                print("-" * 70)
                print(f"SCF converged in {iteration} iterations!")
            break

        # Update for next iteration
        P = P_new
        E_old = E_total

    if not converged and verbose:
        print("-" * 70)
        print(f"WARNING: SCF did not converge in {max_iter} iterations!")

    # =========================================================================
    # Step 4: Final results
    # =========================================================================
    # Verify electron count
    n_elec_computed = np.trace(P @ S)

    if verbose:
        print(f"\nFinal Results:")
        print(f"  Electronic energy: {E_elec:.10f} Hartree")
        print(f"  Nuclear repulsion: {E_nuc:.10f} Hartree")
        print(f"  Total energy:      {E_total:.10f} Hartree")
        print(f"  Electron count:    {n_elec_computed:.6f} (expected: {n_elec})")

    return {
        'E_total': E_total,
        'E_elec': E_elec,
        'E_nuc': E_nuc,
        'C': C,
        'eps': eps,
        'P': P,
        'F': F,
        'converged': converged,
        'n_iter': iteration,
        'history': history
    }


# =============================================================================
# Section 3: Validation Against PySCF
# =============================================================================

def validate_against_pyscf(mol: gto.Mole,
                           our_results: dict) -> bool:
    """
    Validate our SCF implementation against PySCF reference.

    Args:
        mol: PySCF Mole object
        our_results: Dictionary from rhf_scf()

    Returns:
        True if validation passes
    """
    print("\n" + "=" * 70)
    print("Validation Against PySCF Reference")
    print("=" * 70)

    # Run PySCF RHF
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_pyscf = mf.kernel()

    our_E = our_results['E_total']
    diff = abs(our_E - E_pyscf)

    print(f"\n  Our implementation:  {our_E:.10f} Hartree")
    print(f"  PySCF reference:     {E_pyscf:.10f} Hartree")
    print(f"  Difference:          {diff:.2e} Hartree")

    # Check Fock matrix agreement
    P = our_results['P']
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

    # PySCF Fock matrix
    J_ref, K_ref = mf.get_jk(mol, P)
    F_ref = h + J_ref - 0.5 * K_ref
    F_our = our_results['F']

    F_diff = np.linalg.norm(F_our - F_ref)
    print(f"  ||F_our - F_ref||:   {F_diff:.2e}")

    # Check orbital energies
    eps_ref = mf.mo_energy
    eps_our = our_results['eps']
    eps_diff = np.max(np.abs(eps_our - eps_ref))
    print(f"  max|eps_our - eps_ref|: {eps_diff:.2e}")

    # Validation criteria
    energy_ok = diff < 1e-8
    fock_ok = F_diff < 1e-8

    if energy_ok and fock_ok:
        print("\n  VALIDATION PASSED!")
        return True
    else:
        print("\n  VALIDATION FAILED!")
        return False


# =============================================================================
# Section 4: Main Demonstration
# =============================================================================

def main():
    """Run the complete Lab 6A demonstration."""

    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*   Lab 6A: Minimal RHF SCF from AO Integrals                      *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # =========================================================================
    # Part 1: H2O / STO-3G (standard test case)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test Case 1: H2O / STO-3G")
    print("=" * 70)

    # Water molecule geometry (near experimental)
    mol_h2o = gto.M(
        atom="""
            O   0.000000   0.000000   0.117369
            H   0.756950   0.000000  -0.469476
            H  -0.756950   0.000000  -0.469476
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Run our SCF
    results_h2o = rhf_scf(mol_h2o, max_iter=50, conv_E=1e-10, conv_rmsd=1e-10)

    # Validate against PySCF
    validate_against_pyscf(mol_h2o, results_h2o)

    # =========================================================================
    # Part 2: H2 / STO-3G (simplest case)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test Case 2: H2 / STO-3G (Equilibrium)")
    print("=" * 70)

    mol_h2 = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    results_h2 = rhf_scf(mol_h2, max_iter=50, conv_E=1e-10, conv_rmsd=1e-10)
    validate_against_pyscf(mol_h2, results_h2)

    # =========================================================================
    # Part 3: Analysis of Convergence Behavior
    # =========================================================================
    print("\n" + "=" * 70)
    print("Convergence Analysis for H2O")
    print("=" * 70)

    print("\nIteration-by-iteration convergence:")
    print("-" * 50)
    print(f"{'Iter':>4}  {'E_total':>16}  {'dE':>12}  {'RMSD(P)':>12}")
    print("-" * 50)

    for i, h in enumerate(results_h2o['history'], 1):
        print(f"{i:4d}  {h['E']:16.10f}  {h['dE']:+12.3e}  {h['rmsd_P']:12.3e}")

    # =========================================================================
    # Part 4: What You Should Observe
    # =========================================================================
    print("\n" + "=" * 70)
    print("What You Should Observe")
    print("=" * 70)

    observations = """
1. CONVERGENCE BEHAVIOR:
   - Energy should decrease monotonically (for a good initial guess)
   - RMSD(P) decreases as the density approaches self-consistency
   - The SCF residual ||R|| = ||FPS - SPF|| -> 0 at convergence

2. ITERATION COUNT:
   - H2O/STO-3G typically converges in 10-15 iterations without DIIS
   - H2/STO-3G converges faster (simpler system)
   - Convergence rate depends on HOMO-LUMO gap

3. ENERGY COMPONENTS:
   - Nuclear repulsion E_nuc is positive (nuclei repel)
   - Electronic energy E_elec is negative (bound electrons)
   - Total energy E_tot = E_elec + E_nuc is negative for bound systems

4. VALIDATION:
   - Our implementation matches PySCF to ~10^(-10) Hartree
   - Small differences can arise from convergence thresholds
   - Fock matrices should agree to machine precision

5. PHYSICAL MEANING:
   - The Fock matrix eigenvalues (orbital energies) follow Koopmans' theorem:
     - eps_HOMO approximates -IP (ionization potential)
     - eps_LUMO approximates -EA (electron affinity)
   - The density matrix trace gives electron count: Tr[PS] = N_elec
"""
    print(observations)

    # =========================================================================
    # Part 5: Display orbital energies
    # =========================================================================
    print("\n" + "=" * 70)
    print("Orbital Energies for H2O (Hartree)")
    print("=" * 70)

    eps = results_h2o['eps']
    n_occ = mol_h2o.nelectron // 2

    print("\nOccupied orbitals:")
    for i in range(n_occ):
        occ_label = "HOMO" if i == n_occ - 1 else f"MO {i+1}"
        print(f"  {occ_label:>8}: {eps[i]:12.6f}")

    print("\nVirtual orbitals:")
    for i in range(n_occ, len(eps)):
        virt_label = "LUMO" if i == n_occ else f"MO {i+1}"
        print(f"  {virt_label:>8}: {eps[i]:12.6f}")

    homo_lumo_gap = eps[n_occ] - eps[n_occ - 1]
    print(f"\n  HOMO-LUMO gap: {homo_lumo_gap:.6f} Hartree = {homo_lumo_gap * 27.2114:.2f} eV")

    print("\n" + "=" * 70)
    print("Lab 6A Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
