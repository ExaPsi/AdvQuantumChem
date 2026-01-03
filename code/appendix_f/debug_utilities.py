#!/usr/bin/env python3
"""
Debug Utilities for Hartree-Fock Calculations

Provides validation functions for students to check their HF implementations:
- Electron count validation
- Energy validation
- Convergence validation
- Comprehensive HF calculation checker

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference
"""

import numpy as np
from pyscf import gto, scf


def validate_electron_count(dm: np.ndarray, S: np.ndarray, n_elec: int,
                            tol: float = 1e-6, verbose: bool = True) -> bool:
    """Validate electron count from density matrix.

    Checks: Tr[P*S] = N_electrons

    Args:
        dm: Density matrix (nao x nao)
        S: Overlap matrix (nao x nao)
        n_elec: Expected number of electrons
        tol: Tolerance for comparison
        verbose: Print diagnostic information

    Returns:
        True if validation passes
    """
    n_calc = np.trace(dm @ S)

    passed = np.isclose(n_calc, n_elec, atol=tol)

    if verbose:
        print("\n  Electron Count Validation")
        print("  " + "-" * 40)
        print(f"    Tr[P*S] = {n_calc:.8f}")
        print(f"    Expected = {n_elec}")
        print(f"    Difference = {abs(n_calc - n_elec):.2e}")
        print(f"    Status: {'PASSED' if passed else 'FAILED'}")

    return passed


def validate_density_matrix(dm: np.ndarray, S: np.ndarray, n_elec: int,
                            verbose: bool = True) -> dict:
    """Comprehensive density matrix validation.

    Checks:
    1. Symmetry: P = P^T
    2. Electron count: Tr[PS] = N
    3. Idempotency: PSP = P (for pure state)
    4. Eigenvalues: 0 <= lambda <= 2 for RHF

    Args:
        dm: Density matrix
        S: Overlap matrix
        n_elec: Expected number of electrons
        verbose: Print diagnostics

    Returns:
        Dictionary with validation results
    """
    results = {}

    if verbose:
        print("\n  Density Matrix Validation")
        print("  " + "-" * 40)

    # 1. Symmetry
    sym_error = np.linalg.norm(dm - dm.T)
    results['symmetric'] = sym_error < 1e-10
    if verbose:
        print(f"    Symmetry ||P - P^T|| = {sym_error:.2e} "
              f"({'PASSED' if results['symmetric'] else 'FAILED'})")

    # 2. Electron count
    n_calc = np.trace(dm @ S)
    results['electron_count'] = np.isclose(n_calc, n_elec, atol=1e-6)
    results['n_electrons_calc'] = n_calc
    if verbose:
        print(f"    Electron count Tr[PS] = {n_calc:.6f} (expected {n_elec}) "
              f"({'PASSED' if results['electron_count'] else 'FAILED'})")

    # 3. Idempotency (for RHF: PSP = 2P due to factor of 2 in density)
    # Actually P_normalized S P_normalized = P_normalized where P_normalized = P/2
    PSP = dm @ S @ dm
    # For RHF with P = 2*C_occ*C_occ^T: PSP = 2P
    idem_error = np.linalg.norm(PSP - 2*dm)
    results['idempotent'] = idem_error < 1e-6
    results['idempotency_error'] = idem_error
    if verbose:
        print(f"    Idempotency ||PSP - 2P|| = {idem_error:.2e} "
              f"({'PASSED' if results['idempotent'] else 'FAILED'})")

    # 4. Eigenvalue bounds (natural occupation numbers)
    # For RHF: P = 2 C_occ C_occ^T with C^T S C = I
    # Eigenvalues of (1/2)PS should be 0 or 1 for pure state
    # Equivalently, eigenvalues of PS should be 0 or 2
    PS = dm @ S
    nat_occ = np.linalg.eigvals(PS).real
    nat_occ = np.sort(nat_occ)

    results['nat_occ_min'] = nat_occ.min()
    results['nat_occ_max'] = nat_occ.max()
    # Check that eigenvalues are close to 0 or 2
    valid_occ = np.all((np.abs(nat_occ) < 1e-6) | (np.abs(nat_occ - 2.0) < 1e-6))
    results['eigenvalue_bounds'] = valid_occ
    if verbose:
        print(f"    Natural occupations (PS): [{nat_occ.min():.6f}, {nat_occ.max():.6f}] "
              f"(should be 0 or 2) ({'PASSED' if results['eigenvalue_bounds'] else 'FAILED'})")

    results['all_passed'] = all([results['symmetric'], results['electron_count'],
                                  results['idempotent'], results['eigenvalue_bounds']])

    return results


def validate_energy(dm: np.ndarray, h: np.ndarray, J: np.ndarray, K: np.ndarray,
                    E_nuc: float, E_ref: float, tol: float = 1e-8,
                    verbose: bool = True) -> dict:
    """Validate HF energy calculation.

    Checks energy formula: E = Tr[Ph] + 0.5*Tr[PJ] - 0.25*Tr[PK] + E_nuc

    Args:
        dm: Density matrix
        h: Core Hamiltonian
        J: Coulomb matrix
        K: Exchange matrix
        E_nuc: Nuclear repulsion energy
        E_ref: Reference energy to compare against
        tol: Energy tolerance
        verbose: Print diagnostics

    Returns:
        Dictionary with energy components and validation status
    """
    results = {}

    # Energy components
    E_1e = np.einsum('ij,ji->', dm, h)
    E_J = 0.5 * np.einsum('ij,ji->', dm, J)
    E_K = -0.25 * np.einsum('ij,ji->', dm, K)
    E_elec = E_1e + E_J + E_K
    E_tot = E_elec + E_nuc

    results['E_1e'] = E_1e
    results['E_J'] = E_J
    results['E_K'] = E_K
    results['E_elec'] = E_elec
    results['E_nuc'] = E_nuc
    results['E_tot'] = E_tot
    results['error'] = abs(E_tot - E_ref)
    results['passed'] = results['error'] < tol

    if verbose:
        print("\n  Energy Validation")
        print("  " + "-" * 40)
        print(f"    E_1e (one-electron):  {E_1e:16.10f} Hartree")
        print(f"    E_J  (Coulomb/2):     {E_J:16.10f} Hartree")
        print(f"    E_K  (-Exchange/4):   {E_K:16.10f} Hartree")
        print(f"    E_elec:               {E_elec:16.10f} Hartree")
        print(f"    E_nuc:                {E_nuc:16.10f} Hartree")
        print(f"    E_tot:                {E_tot:16.10f} Hartree")
        print(f"    E_ref:                {E_ref:16.10f} Hartree")
        print(f"    |Error|:              {results['error']:.2e} Hartree")
        print(f"    Status: {'PASSED' if results['passed'] else 'FAILED'}")

    return results


def validate_convergence(F: np.ndarray, P: np.ndarray, S: np.ndarray,
                         tol: float = 1e-5, verbose: bool = True) -> dict:
    """Validate SCF convergence using commutator criterion.

    At convergence: [F, PS] = FPS - SPF = 0

    Args:
        F: Fock matrix
        P: Density matrix
        S: Overlap matrix
        tol: Convergence tolerance
        verbose: Print diagnostics

    Returns:
        Dictionary with convergence metrics
    """
    results = {}

    # Commutator FPS - SPF
    FPS = F @ P @ S
    SPF = S @ P @ F
    commutator = FPS - SPF

    results['commutator_norm'] = np.linalg.norm(commutator)
    results['commutator_max'] = np.max(np.abs(commutator))
    results['converged'] = results['commutator_norm'] < tol

    if verbose:
        print("\n  Convergence Validation")
        print("  " + "-" * 40)
        print(f"    ||FPS - SPF|| = {results['commutator_norm']:.2e}")
        print(f"    max|FPS - SPF| = {results['commutator_max']:.2e}")
        print(f"    Tolerance: {tol:.2e}")
        print(f"    Status: {'CONVERGED' if results['converged'] else 'NOT CONVERGED'}")

    return results


def validate_orthonormality(C: np.ndarray, S: np.ndarray,
                            verbose: bool = True) -> dict:
    """Validate MO coefficient orthonormality.

    MOs should satisfy: C^T S C = I

    Args:
        C: MO coefficient matrix (nao x nmo)
        S: Overlap matrix
        verbose: Print diagnostics

    Returns:
        Dictionary with orthonormality metrics
    """
    results = {}

    # C^T S C should be identity
    CTSC = C.T @ S @ C
    identity = np.eye(CTSC.shape[0])
    error = CTSC - identity

    results['error_norm'] = np.linalg.norm(error)
    results['error_max'] = np.max(np.abs(error))
    results['diagonal_range'] = (np.diag(CTSC).min(), np.diag(CTSC).max())
    results['orthonormal'] = results['error_norm'] < 1e-10

    if verbose:
        print("\n  MO Orthonormality Validation")
        print("  " + "-" * 40)
        print(f"    ||C^T S C - I|| = {results['error_norm']:.2e}")
        print(f"    max|C^T S C - I| = {results['error_max']:.2e}")
        print(f"    Diagonal range: [{results['diagonal_range'][0]:.6f}, "
              f"{results['diagonal_range'][1]:.6f}]")
        print(f"    Status: {'PASSED' if results['orthonormal'] else 'FAILED'}")

    return results


def check_hf_calculation(mol, mf=None, verbose: bool = True) -> dict:
    """Comprehensive check of HF calculation.

    Runs all validation checks on a completed HF calculation.

    Args:
        mol: PySCF molecule object
        mf: Completed HF calculation (if None, runs new calculation)
        verbose: Print all diagnostics

    Returns:
        Dictionary with all validation results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Comprehensive HF Calculation Check")
        print("=" * 60)

    # Run HF if not provided
    if mf is None:
        mf = scf.RHF(mol)
        mf.kernel()

    # Get all quantities
    S = mol.intor("int1e_ovlp")
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    dm = mf.make_rdm1()
    C = mf.mo_coeff
    F = mf.get_fock()
    J, K = mf.get_jk()
    E_nuc = mol.energy_nuc()

    results = {}

    if verbose:
        print(f"\n  Molecule: {mol.nelectron} electrons, {mol.nao} AOs")
        print(f"  HF energy: {mf.e_tot:.10f} Hartree")
        print(f"  Converged: {mf.converged}")
        print(f"  Iterations: {mf.cycles}")

    # 1. Density matrix validation
    results['density'] = validate_density_matrix(dm, S, mol.nelectron, verbose)

    # 2. Energy validation
    results['energy'] = validate_energy(dm, h, J, K, E_nuc, mf.e_tot, verbose=verbose)

    # 3. Convergence validation
    results['convergence'] = validate_convergence(F, dm, S, verbose=verbose)

    # 4. MO orthonormality
    results['orthonormality'] = validate_orthonormality(C, S, verbose=verbose)

    # 5. Fock matrix symmetry
    F_sym_error = np.linalg.norm(F - F.T)
    results['fock_symmetric'] = F_sym_error < 1e-10
    if verbose:
        print(f"\n  Fock Matrix Symmetry")
        print("  " + "-" * 40)
        print(f"    ||F - F^T|| = {F_sym_error:.2e} "
              f"({'PASSED' if results['fock_symmetric'] else 'FAILED'})")

    # 6. J/K symmetry
    J_sym_error = np.linalg.norm(J - J.T)
    K_sym_error = np.linalg.norm(K - K.T)
    results['jk_symmetric'] = J_sym_error < 1e-10 and K_sym_error < 1e-10
    if verbose:
        print(f"\n  J/K Matrix Symmetry")
        print("  " + "-" * 40)
        print(f"    ||J - J^T|| = {J_sym_error:.2e}")
        print(f"    ||K - K^T|| = {K_sym_error:.2e}")
        print(f"    Status: {'PASSED' if results['jk_symmetric'] else 'FAILED'}")

    # Overall status
    all_passed = (results['density']['all_passed'] and
                  results['energy']['passed'] and
                  results['convergence']['converged'] and
                  results['orthonormality']['orthonormal'] and
                  results['fock_symmetric'] and
                  results['jk_symmetric'])

    results['all_passed'] = all_passed

    if verbose:
        print("\n" + "=" * 60)
        print(f"  OVERALL STATUS: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
        print("=" * 60)

    return results


def compare_implementations(mol, dm_user: np.ndarray, E_user: float,
                            verbose: bool = True) -> dict:
    """Compare user implementation against PySCF reference.

    Args:
        mol: PySCF molecule object
        dm_user: User's density matrix
        E_user: User's total energy
        verbose: Print diagnostics

    Returns:
        Dictionary with comparison results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Implementation Comparison Against PySCF")
        print("=" * 60)

    # Run PySCF reference
    mf = scf.RHF(mol)
    E_ref = mf.kernel()
    dm_ref = mf.make_rdm1()

    results = {}

    # Energy comparison
    results['energy_diff'] = abs(E_user - E_ref)
    results['energy_match'] = results['energy_diff'] < 1e-8

    # Density comparison
    results['dm_diff_norm'] = np.linalg.norm(dm_user - dm_ref)
    results['dm_diff_max'] = np.max(np.abs(dm_user - dm_ref))
    results['dm_match'] = results['dm_diff_norm'] < 1e-6

    if verbose:
        print(f"\n  Energy Comparison:")
        print(f"    User energy:  {E_user:.10f} Hartree")
        print(f"    PySCF energy: {E_ref:.10f} Hartree")
        print(f"    Difference:   {results['energy_diff']:.2e} Hartree")
        print(f"    Status: {'MATCH' if results['energy_match'] else 'MISMATCH'}")

        print(f"\n  Density Matrix Comparison:")
        print(f"    ||P_user - P_ref|| = {results['dm_diff_norm']:.2e}")
        print(f"    max|P_user - P_ref| = {results['dm_diff_max']:.2e}")
        print(f"    Status: {'MATCH' if results['dm_match'] else 'MISMATCH'}")

    results['all_match'] = results['energy_match'] and results['dm_match']

    if verbose:
        print(f"\n  Overall: {'IMPLEMENTATION CORRECT' if results['all_match'] else 'IMPLEMENTATION DIFFERS'}")

    return results


def main():
    print("=" * 70)
    print("Debug Utilities Demonstration")
    print("=" * 70)

    # =========================================================================
    # Test System
    # =========================================================================
    mol = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nTest system: H2O / STO-3G")
    print(f"  Number of electrons: {mol.nelectron}")
    print(f"  Number of AOs: {mol.nao}")

    # =========================================================================
    # Test 1: Comprehensive Check on Correct Calculation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: Comprehensive Check (Correct Calculation)")
    print("=" * 50)

    mf = scf.RHF(mol)
    mf.kernel()

    results = check_hf_calculation(mol, mf, verbose=True)

    # =========================================================================
    # Test 2: Detecting Common Errors
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: Detecting Common Errors")
    print("=" * 50)

    S = mol.intor("int1e_ovlp")
    dm_correct = mf.make_rdm1()

    # Error 1: Missing factor of 2 in density matrix
    print("\n--- Error: Missing factor of 2 in P ---")
    dm_wrong = dm_correct / 2  # Common RHF error
    validate_electron_count(dm_wrong, S, mol.nelectron, verbose=True)

    # Error 2: Non-symmetric density
    print("\n--- Error: Non-symmetric density matrix ---")
    dm_asym = dm_correct.copy()
    dm_asym[0, 1] += 0.1  # Break symmetry
    validate_density_matrix(dm_asym, S, mol.nelectron, verbose=True)

    # =========================================================================
    # Test 3: Using compare_implementations
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: Compare Implementations")
    print("=" * 50)

    # Simulate a "user implementation" with small error
    dm_user = dm_correct + np.random.randn(*dm_correct.shape) * 1e-10
    dm_user = 0.5 * (dm_user + dm_user.T)  # Keep symmetric
    E_user = mf.e_tot + 1e-12  # Tiny energy difference

    compare_implementations(mol, dm_user, E_user, verbose=True)

    # =========================================================================
    # Test 4: Individual Validation Functions
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 4: Individual Validation Functions")
    print("=" * 50)

    # Get all matrices
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    F = mf.get_fock()
    J, K = mf.get_jk()
    C = mf.mo_coeff
    E_nuc = mol.energy_nuc()

    print("\n  Individual function calls:")

    # Electron count
    validate_electron_count(dm_correct, S, mol.nelectron, verbose=True)

    # Energy
    validate_energy(dm_correct, h, J, K, E_nuc, mf.e_tot, verbose=True)

    # Convergence
    validate_convergence(F, dm_correct, S, verbose=True)

    # Orthonormality
    validate_orthonormality(C, S, verbose=True)

    print("\n" + "=" * 70)
    print("Debug utilities demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
