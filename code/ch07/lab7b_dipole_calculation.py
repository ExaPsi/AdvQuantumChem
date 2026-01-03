#!/usr/bin/env python3
"""
Lab 7B: Dipole Moment from Density Matrix and Integrals
========================================================

This script demonstrates the "integrals-first" approach to computing
molecular properties. The dipole moment is computed as:

    μ = μ_nuc + μ_el = Σ_A Z_A R_A - Tr[P · r]

where:
    - μ_nuc: nuclear contribution (positive charges at nuclear positions)
    - μ_el: electronic contribution (Tr[P · r] with position operator integrals)
    - P: density matrix from converged SCF
    - r: position operator integral matrix (int1e_r)

Key concepts from Chapter 7:
    - One-electron property formula: ⟨Ô⟩ = Tr[P · o]
    - Origin independence for neutral molecules
    - Origin dependence for charged species (ions)

Reference: Section 7.7 (One-Electron Properties from Density and Integrals)
"""

import numpy as np
from pyscf import gto, scf

# Conversion factor: atomic units to Debye
AU_TO_DEBYE = 2.5417464


def compute_dipole_from_density(mol, dm, origin=None):
    """
    Compute the molecular dipole moment from the density matrix.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    dm : ndarray
        Density matrix (nao x nao)
    origin : ndarray, optional
        Origin for dipole calculation (default: [0, 0, 0])

    Returns
    -------
    mu : ndarray
        Dipole moment vector in atomic units (3,)
    """
    if origin is None:
        origin = np.zeros(3)

    # Get position operator integrals relative to origin
    with mol.with_common_orig(origin):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)  # Shape: (3, nao, nao)

    # Electronic contribution: μ_el = Tr[P · r]
    # Note: this is the expectation value of the position operator
    mu_el = np.einsum('xij,ji->x', ao_r, dm).real

    # Nuclear contribution: μ_nuc = Σ_A Z_A (R_A - origin)
    charges = mol.atom_charges()
    coords = mol.atom_coords()  # In Bohr
    mu_nuc = np.einsum('a,ax->x', charges, coords - origin[None, :])

    # Total dipole: nuclear - electronic (electrons have charge -1)
    mu = mu_nuc - mu_el

    return mu


def compute_dipole_components(mol, dm, origin=None):
    """
    Compute dipole moment with detailed component breakdown.

    Returns
    -------
    dict with keys: mu_nuc, mu_el, mu_total, mu_norm, mu_debye
    """
    if origin is None:
        origin = np.zeros(3)

    with mol.with_common_orig(origin):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)

    mu_el = np.einsum('xij,ji->x', ao_r, dm).real

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    mu_nuc = np.einsum('a,ax->x', charges, coords - origin[None, :])

    mu_total = mu_nuc - mu_el
    mu_norm = np.linalg.norm(mu_total)

    return {
        'mu_nuc': mu_nuc,
        'mu_el': mu_el,
        'mu_total': mu_total,
        'mu_norm_au': mu_norm,
        'mu_norm_debye': mu_norm * AU_TO_DEBYE,
        'origin': origin,
    }


def demo_dipole_calculation():
    """
    Demonstrate dipole moment calculation for water.

    This corresponds to Lab 7B in the lecture notes.
    """
    print("=" * 70)
    print("Lab 7B: Dipole Moment from Density and Integrals")
    print("=" * 70)

    # Water molecule
    mol = gto.M(
        atom='''
        O  0.0000  0.0000  0.0000
        H  0.7586  0.0000  0.5043
        H -0.7586  0.0000  0.5043
        ''',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    print(f"\nMolecule: H2O")
    print(f"Basis: cc-pVDZ")
    print(f"RHF Energy: {mf.e_tot:.10f} Eh")

    # Compute dipole at origin
    origin = np.zeros(3)
    result = compute_dipole_components(mol, dm, origin)

    print("\n" + "-" * 50)
    print("Dipole Moment Calculation")
    print("-" * 50)
    print(f"Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}) Bohr")
    print(f"\nNuclear contribution (a.u.):")
    print(f"  μ_nuc = ({result['mu_nuc'][0]:+.6f}, "
          f"{result['mu_nuc'][1]:+.6f}, {result['mu_nuc'][2]:+.6f})")
    print(f"\nElectronic contribution (a.u.):")
    print(f"  μ_el  = ({result['mu_el'][0]:+.6f}, "
          f"{result['mu_el'][1]:+.6f}, {result['mu_el'][2]:+.6f})")
    print(f"\nTotal dipole moment:")
    print(f"  μ = μ_nuc - μ_el = ({result['mu_total'][0]:+.6f}, "
          f"{result['mu_total'][1]:+.6f}, {result['mu_total'][2]:+.6f}) a.u.")
    print(f"  |μ| = {result['mu_norm_au']:.6f} a.u. = {result['mu_norm_debye']:.4f} Debye")

    # Compare with PySCF
    pyscf_dipole = mf.dip_moment(unit='AU', origin=origin)
    print(f"\nPySCF reference:")
    print(f"  μ = ({pyscf_dipole[0]:+.6f}, {pyscf_dipole[1]:+.6f}, "
          f"{pyscf_dipole[2]:+.6f}) a.u.")

    # Verify agreement
    error = np.linalg.norm(result['mu_total'] - pyscf_dipole)
    print(f"\nDifference from PySCF: {error:.2e} a.u.")
    assert error < 1e-10, f"Dipole calculation error: {error}"
    print("[PASSED] Our calculation matches PySCF!")


def demo_origin_independence():
    """
    Demonstrate that the dipole moment is origin-independent for neutral molecules.

    For a charged species, the dipole depends on the origin.
    """
    print("\n" + "=" * 70)
    print("Origin Independence Test")
    print("=" * 70)

    # Neutral molecule: H2O
    mol_neutral = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf_neutral = scf.RHF(mol_neutral)
    mf_neutral.kernel()
    dm_neutral = mf_neutral.make_rdm1()

    origins = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([2.0, 3.0, -1.0]),
    ]

    print("\n1. Neutral molecule (H2O):")
    print("-" * 50)
    print(f"{'Origin (Bohr)':<25} {'|μ| (Debye)':>15}")
    print("-" * 50)

    dipoles_neutral = []
    for origin in origins:
        mu = compute_dipole_from_density(mol_neutral, dm_neutral, origin)
        mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_neutral.append(mu_debye)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"{mu_debye:>18.6f}")

    # Check origin independence
    max_diff = max(dipoles_neutral) - min(dipoles_neutral)
    print(f"\nMax difference: {max_diff:.2e} Debye")
    print("[PASSED] Dipole is origin-independent for neutral molecule!"
          if max_diff < 1e-10 else "[FAILED] Unexpected origin dependence!")

    # Charged species: HF+ (cation)
    print("\n2. Charged molecule (HF+):")
    print("-" * 50)

    mol_cation = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVDZ',
        unit='Angstrom',
        charge=1,
        spin=1,
        verbose=0
    )

    mf_cation = scf.UHF(mol_cation)
    mf_cation.kernel()
    dm_cation = mf_cation.make_rdm1()
    # For UHF, dm is (2, nao, nao); sum for total density
    dm_total = dm_cation[0] + dm_cation[1]

    print(f"{'Origin (Bohr)':<25} {'|μ| (Debye)':>15}")
    print("-" * 50)

    dipoles_cation = []
    for origin in origins:
        mu = compute_dipole_from_density(mol_cation, dm_total, origin)
        mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_cation.append(mu_debye)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"{mu_debye:>18.6f}")

    max_diff_cation = max(dipoles_cation) - min(dipoles_cation)
    print(f"\nMax difference: {max_diff_cation:.4f} Debye")
    print("[EXPECTED] Dipole IS origin-dependent for charged species!")

    # Physical explanation
    print("\n" + "=" * 70)
    print("Physical Explanation")
    print("=" * 70)
    print("""
For a system with total charge Q:
    μ(O + d) = μ(O) - Q·d

- Neutral molecules (Q = 0): Origin shift has no effect
- Charged species (Q ≠ 0): Origin shift changes the dipole by -Q·d

This is why conventions for reporting ion dipole moments must specify
the origin (e.g., center of mass, center of nuclear charge).
""")


def demo_property_pattern():
    """
    Demonstrate the general one-electron property pattern: ⟨Ô⟩ = Tr[P·o]
    """
    print("\n" + "=" * 70)
    print("One-Electron Property Pattern: ⟨Ô⟩ = Tr[P·o]")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    # Various one-electron properties
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')

    n_elec = np.einsum('ij,ji->', dm, S)
    e_kin = np.einsum('ij,ji->', dm, T)
    e_nuc = np.einsum('ij,ji->', dm, V)

    print("\nProperty calculations using Tr[P·o]:")
    print("-" * 50)
    print(f"Electron count:    N = Tr[P·S] = {n_elec:.6f}")
    print(f"                   (expected: {mol.nelectron})")
    print(f"\nKinetic energy:    <T> = Tr[P·T] = {e_kin:.10f} Eh")
    print(f"Nuclear attraction: <V> = Tr[P·V] = {e_nuc:.10f} Eh")
    print(f"One-electron energy: <h> = <T> + <V> = {e_kin + e_nuc:.10f} Eh")

    # Verify electron count
    assert abs(n_elec - mol.nelectron) < 1e-10, "Electron count mismatch!"
    print("\n[PASSED] All properties computed correctly!")


def validate_against_pyscf():
    """
    Comprehensive validation against PySCF reference values.
    """
    print("\n" + "=" * 70)
    print("Validation: Dipole Calculation vs PySCF")
    print("=" * 70)

    test_molecules = [
        ('HF', 'H 0 0 0; F 0 0 0.92', 'cc-pVDZ'),
        ('H2O', 'O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043', 'cc-pVDZ'),
        ('NH3', '''N 0 0 0.1173; H 0 0.9377 -0.2737;
                   H 0.8121 -0.4689 -0.2737; H -0.8121 -0.4689 -0.2737''', 'cc-pVDZ'),
    ]

    print(f"\n{'Molecule':<10} {'Basis':<12} {'Our |μ| (D)':>14} "
          f"{'PySCF |μ| (D)':>14} {'Error (D)':>12}")
    print("-" * 70)

    all_passed = True
    for name, atoms, basis in test_molecules:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm = mf.make_rdm1()

        # Our calculation
        mu_ours = compute_dipole_from_density(mol, dm)
        mu_ours_debye = np.linalg.norm(mu_ours) * AU_TO_DEBYE

        # PySCF reference
        mu_pyscf = mf.dip_moment(unit='DEBYE')
        mu_pyscf_norm = np.linalg.norm(mu_pyscf)

        error = abs(mu_ours_debye - mu_pyscf_norm)
        status = "OK" if error < 1e-6 else "FAIL"
        if error >= 1e-6:
            all_passed = False

        print(f"{name:<10} {basis:<12} {mu_ours_debye:>14.6f} "
              f"{mu_pyscf_norm:>14.6f} {error:>12.2e} [{status}]")

    print()
    if all_passed:
        print("[PASSED] All dipole calculations match PySCF!")
    else:
        print("[FAILED] Some calculations differ from PySCF!")


if __name__ == '__main__':
    demo_dipole_calculation()
    demo_origin_independence()
    demo_property_pattern()
    validate_against_pyscf()
