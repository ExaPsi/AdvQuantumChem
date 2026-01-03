#!/usr/bin/env python3
"""
Lab 1A: AO Integral Inventory and Sanity Checks

This script accompanies Chapter 1 of the Advanced Quantum Chemistry lecture notes.
It demonstrates how to:
  1. Build a molecule in PySCF
  2. Extract one-electron integrals (S, T, V) and two-electron integrals (ERIs)
  3. Verify integral symmetries
  4. Run an RHF calculation and validate the energy expression

Reference: Algorithm 1.1 in the lecture notes (Integral-Driven HF Energy Evaluation)

Usage:
    python lab1a_integrals.py
"""
import numpy as np
from pyscf import gto, scf


def build_molecule(atom_string="H 0 0 0; H 0 0 0.74", basis="sto-3g"):
    """
    Build a molecule in PySCF.

    Parameters
    ----------
    atom_string : str
        Atomic coordinates in format "Element x y z; Element x y z; ..."
        Default is H2 at approximately equilibrium bond length (0.74 Angstrom).
    basis : str
        Basis set name. Default is STO-3G (minimal basis for debugging).

    Returns
    -------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    """
    mol = gto.M(
        atom=atom_string,
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )
    return mol


def extract_one_electron_integrals(mol):
    """
    Extract one-electron integrals from a PySCF molecule.

    This corresponds to Step 1 of Algorithm 1.1: compute h = T + V.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.

    Returns
    -------
    S : ndarray (N, N)
        Overlap matrix S_uv = <u|v>
    T : ndarray (N, N)
        Kinetic energy matrix T_uv = <u|-1/2 nabla^2|v>
    V : ndarray (N, N)
        Nuclear attraction matrix V_uv = <u|V_nuc|v>
    h : ndarray (N, N)
        Core Hamiltonian h = T + V
    """
    S = mol.intor("int1e_ovlp")  # Overlap
    T = mol.intor("int1e_kin")   # Kinetic energy
    V = mol.intor("int1e_nuc")   # Nuclear attraction
    h = T + V                    # Core Hamiltonian
    return S, T, V, h


def extract_two_electron_integrals(mol, aosym="s1"):
    """
    Extract two-electron repulsion integrals (ERIs) from a PySCF molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    aosym : str
        Symmetry option for ERI storage:
        - "s1": Full 4-index tensor (N, N, N, N), no symmetry exploitation
        - "s4": 4-fold symmetry (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        - "s8": 8-fold symmetry, also includes (ij|kl) = (kl|ij)

    Returns
    -------
    eri : ndarray
        ERI tensor. Shape depends on aosym option.

    Notes
    -----
    PySCF uses chemist's notation: eri[mu,nu,lam,sig] = (mu nu | lam sig)
    where (mu nu | lam sig) = integral of chi_mu(1) chi_nu(1) (1/r12) chi_lam(2) chi_sig(2)
    """
    eri = mol.intor("int2e", aosym=aosym)
    return eri


def verify_symmetries(S, h, eri):
    """
    Verify expected symmetries of integrals.

    Parameters
    ----------
    S : ndarray (N, N)
        Overlap matrix.
    h : ndarray (N, N)
        Core Hamiltonian.
    eri : ndarray (N, N, N, N)
        Full ERI tensor (aosym="s1").

    Returns
    -------
    results : dict
        Dictionary of symmetry check results.
    """
    results = {}

    # One-electron symmetries: S and h should be symmetric
    results["S_symmetric"] = np.allclose(S, S.T, atol=1e-12)
    results["h_symmetric"] = np.allclose(h, h.T, atol=1e-12)

    # ERI 8-fold symmetry checks (for real orbitals)
    # Using indices 0 and 1 for H2/STO-3G (2 basis functions)
    if eri.shape[0] >= 2:
        # (mu nu | lam sig) = (nu mu | lam sig)  [swap mu <-> nu]
        results["eri_mu_nu_swap"] = np.allclose(
            eri[0, 1, 0, 1], eri[1, 0, 0, 1], atol=1e-12
        )
        # (mu nu | lam sig) = (mu nu | sig lam)  [swap lam <-> sig]
        results["eri_lam_sig_swap"] = np.allclose(
            eri[0, 1, 0, 1], eri[0, 1, 1, 0], atol=1e-12
        )
        # (mu nu | lam sig) = (lam sig | mu nu)  [swap bra <-> ket]
        results["eri_bra_ket_swap"] = np.allclose(
            eri[0, 1, 0, 1], eri[0, 1, 0, 1], atol=1e-12
        )
        # Combined: (01|01) = (10|10)
        results["eri_full_swap"] = np.allclose(
            eri[0, 1, 0, 1], eri[1, 0, 1, 0], atol=1e-12
        )

    return results


def run_rhf(mol):
    """
    Run restricted Hartree-Fock calculation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.

    Returns
    -------
    mf : pyscf.scf.RHF
        Converged RHF object.
    E_tot : float
        Total RHF energy in Hartree.
    dm : ndarray (N, N)
        Converged density matrix P_uv = 2 * sum_i C_ui C_vi (RHF).
    """
    mf = scf.RHF(mol)
    E_tot = mf.kernel()
    dm = mf.make_rdm1()
    return mf, E_tot, dm


def validate_electron_count(dm, S, expected_electrons):
    """
    Validate electron count via Tr[P*S].

    In a non-orthonormal AO basis, the electron count is:
        N_e = Tr[P*S] = sum_{mu,nu} P_{mu,nu} S_{nu,mu}

    This is NOT Tr[P] because the basis functions overlap.

    Parameters
    ----------
    dm : ndarray (N, N)
        Density matrix.
    S : ndarray (N, N)
        Overlap matrix.
    expected_electrons : int
        Expected number of electrons.

    Returns
    -------
    N_e : float
        Computed electron count.
    is_valid : bool
        True if electron count matches expected value.
    """
    N_e = np.einsum("ij,ji->", dm, S)  # Tr[P*S]
    is_valid = np.isclose(N_e, expected_electrons, atol=1e-10)
    return N_e, is_valid


def reconstruct_energy(mol, mf, dm, h):
    """
    Reconstruct electronic energy from integrals to validate Algorithm 1.1.

    This implements Steps 2-5 of Algorithm 1.1:
        E_elec = Tr[P*h] + (1/2) * Tr[P * (J - (1/2)K)]
               = (1/2) * Tr[P * (h + F)]

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    mf : pyscf.scf.RHF
        Converged RHF object.
    dm : ndarray (N, N)
        Density matrix.
    h : ndarray (N, N)
        Core Hamiltonian.

    Returns
    -------
    E_rebuilt : float
        Reconstructed total energy.
    difference : float
        Difference from SCF energy (should be ~0).
    """
    # Steps 2-3: Get J - (1/2)K from PySCF
    # For RHF, get_veff returns G = J - 0.5*K
    vhf = mf.get_veff(mol, dm)

    # Steps 4-5: Compute electronic energy
    # E_elec = Tr[P*h] + 0.5 * Tr[P*G] where G = J - 0.5*K
    E_one_electron = np.einsum("ij,ji->", dm, h)
    E_two_electron = 0.5 * np.einsum("ij,ji->", dm, vhf)
    E_elec = E_one_electron + E_two_electron

    # Add nuclear repulsion
    E_nuc = mol.energy_nuc()
    E_rebuilt = E_elec + E_nuc

    # Compare to SCF result
    difference = E_rebuilt - mf.e_tot

    return E_rebuilt, difference


def print_memory_analysis(mol, eri_full, eri_packed):
    """
    Print memory usage analysis for integrals.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF molecule object.
    eri_full : ndarray
        Full ERI tensor (aosym="s1").
    eri_packed : ndarray
        Symmetry-packed ERI tensor (aosym="s8").
    """
    N = mol.nao_nr()
    print(f"\n{'='*60}")
    print("Memory Analysis")
    print(f"{'='*60}")
    print(f"Number of AOs (N): {N}")
    print(f"One-electron matrices: {N}x{N} = {N**2} elements = {N**2 * 8 / 1024:.2f} KB each")
    print(f"ERI tensor (full): {N}^4 = {N**4} elements = {eri_full.nbytes / 1024**2:.4f} MB")
    print(f"ERI tensor (s8):   {eri_packed.shape[0]} elements = {eri_packed.nbytes / 1024**2:.4f} MB")
    print(f"Compression ratio: {eri_full.nbytes / eri_packed.nbytes:.1f}x")


def main():
    """Main function demonstrating Lab 1A concepts."""
    print("="*60)
    print("Lab 1A: AO Integral Inventory and Sanity Checks")
    print("="*60)

    # Build molecule
    print("\n--- Building H2 molecule ---")
    mol = build_molecule()
    print(f"Molecule: H2")
    print(f"Basis: STO-3G")
    print(f"Number of AOs: {mol.nao_nr()}")
    print(f"Number of electrons: {mol.nelectron}")

    # Extract one-electron integrals (Algorithm 1.1, Step 1)
    print("\n--- Extracting one-electron integrals ---")
    S, T, V, h = extract_one_electron_integrals(mol)
    print(f"Overlap matrix S shape: {S.shape}")
    print(f"Kinetic matrix T shape: {T.shape}")
    print(f"Nuclear attraction V shape: {V.shape}")
    print(f"Core Hamiltonian h = T + V shape: {h.shape}")

    # Extract two-electron integrals
    print("\n--- Extracting two-electron integrals ---")
    eri_full = extract_two_electron_integrals(mol, aosym="s1")
    eri_packed = extract_two_electron_integrals(mol, aosym="s8")
    print(f"ERI (full) shape: {eri_full.shape}")
    print(f"ERI (s8 packed) shape: {eri_packed.shape}")

    # Verify symmetries
    print("\n--- Verifying integral symmetries ---")
    symmetry_results = verify_symmetries(S, h, eri_full)
    for name, result in symmetry_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    # Run RHF
    print("\n--- Running RHF calculation ---")
    mf, E_tot, dm = run_rhf(mol)
    print(f"RHF converged: {mf.converged}")
    print(f"RHF total energy: {E_tot:.10f} Eh")

    # Validate electron count
    print("\n--- Validating electron count ---")
    N_e, is_valid = validate_electron_count(dm, S, mol.nelectron)
    print(f"Tr[P*S] = {N_e:.10f}")
    print(f"Expected: {mol.nelectron}")
    print(f"Electron count valid: {is_valid}")

    # Reconstruct energy (Algorithm 1.1, Steps 2-5)
    print("\n--- Reconstructing energy from integrals ---")
    E_rebuilt, difference = reconstruct_energy(mol, mf, dm, h)
    print(f"E_tot from SCF:     {E_tot:.10f} Eh")
    print(f"E_tot rebuilt:      {E_rebuilt:.10f} Eh")
    print(f"Difference:         {difference:.2e} Eh")
    print(f"Energy validated:   {abs(difference) < 1e-10}")

    # Memory analysis
    print_memory_analysis(mol, eri_full, eri_packed)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    all_passed = (
        all(symmetry_results.values()) and
        is_valid and
        abs(difference) < 1e-10
    )
    if all_passed:
        print("All validation checks PASSED!")
    else:
        print("Some validation checks FAILED. Review output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
