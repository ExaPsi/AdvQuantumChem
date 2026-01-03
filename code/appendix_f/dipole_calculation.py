#!/usr/bin/env python3
"""
Dipole Moment Calculation from Density Matrix

Demonstrates property calculation from the one-particle density matrix,
including electronic and nuclear contributions, and unit conversion.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference

Key equations:
  mu_elec = -Tr[P * r]  (negative for electron charge)
  mu_nuc  = sum_A Z_A * R_A
  mu_tot  = mu_elec + mu_nuc

Units: 1 a.u. (Debye) = 2.541746 Debye
"""

import numpy as np
from pyscf import gto, scf


def compute_electronic_dipole(mol, dm: np.ndarray, origin: np.ndarray = None) -> np.ndarray:
    """Compute electronic contribution to dipole moment.

    Args:
        mol: PySCF molecule object
        dm: One-particle density matrix
        origin: Origin for dipole calculation (default: center of mass)

    Returns:
        mu_elec: Electronic dipole moment [x, y, z] in atomic units
    """
    if origin is None:
        origin = np.zeros(3)

    # Set origin for integral calculation
    mol.set_common_origin(origin)

    # Get dipole integrals: <mu|r|nu> for r = x, y, z
    r_ints = mol.intor("int1e_r")  # Shape: (3, nao, nao)

    # Electronic dipole: -Tr[P * r] (negative for electron charge)
    mu_elec = -np.einsum('xij,ji->x', r_ints, dm)

    return mu_elec


def compute_nuclear_dipole(mol, origin: np.ndarray = None) -> np.ndarray:
    """Compute nuclear contribution to dipole moment.

    Args:
        mol: PySCF molecule object
        origin: Origin for dipole calculation

    Returns:
        mu_nuc: Nuclear dipole moment [x, y, z] in atomic units
    """
    if origin is None:
        origin = np.zeros(3)

    mu_nuc = np.zeros(3)
    for i in range(mol.natm):
        Z = mol.atom_charge(i)
        R = mol.atom_coord(i) - origin
        mu_nuc += Z * R

    return mu_nuc


def compute_dipole_moment(mol, dm: np.ndarray, origin: np.ndarray = None) -> dict:
    """Compute total dipole moment with all components.

    Args:
        mol: PySCF molecule object
        dm: One-particle density matrix
        origin: Origin for calculation

    Returns:
        Dictionary with dipole components and magnitude
    """
    if origin is None:
        origin = np.zeros(3)

    mu_elec = compute_electronic_dipole(mol, dm, origin)
    mu_nuc = compute_nuclear_dipole(mol, origin)
    mu_total = mu_elec + mu_nuc

    # Magnitude
    magnitude_au = np.linalg.norm(mu_total)
    magnitude_debye = magnitude_au * 2.541746  # 1 a.u. = 2.541746 Debye

    return {
        'electronic': mu_elec,
        'nuclear': mu_nuc,
        'total': mu_total,
        'magnitude_au': magnitude_au,
        'magnitude_debye': magnitude_debye,
        'origin': origin
    }


def main():
    print("=" * 70)
    print("Dipole Moment Calculation from Density Matrix")
    print("=" * 70)

    # =========================================================================
    # Test 1: H2O (has permanent dipole)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: H2O (polar molecule)")
    print("=" * 50)

    # Water with O at origin, molecule in yz-plane
    mol_h2o = gto.M(
        atom="""
            O   0.0000   0.0000   0.1173
            H   0.0000   0.7572  -0.4692
            H   0.0000  -0.7572  -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    print(f"\n  Molecule: H2O / cc-pVDZ")
    print(f"  Geometry: O near origin, H atoms in yz-plane")

    # Run HF
    mf = scf.RHF(mol_h2o)
    mf.kernel()
    dm = mf.make_rdm1()

    # Our calculation (with origin at [0,0,0])
    result = compute_dipole_moment(mol_h2o, dm, origin=np.zeros(3))

    print(f"\n  Our calculation:")
    print(f"    Origin: {result['origin']}")
    print(f"    Electronic: [{result['electronic'][0]:8.5f}, "
          f"{result['electronic'][1]:8.5f}, {result['electronic'][2]:8.5f}] a.u.")
    print(f"    Nuclear:    [{result['nuclear'][0]:8.5f}, "
          f"{result['nuclear'][1]:8.5f}, {result['nuclear'][2]:8.5f}] a.u.")
    print(f"    Total:      [{result['total'][0]:8.5f}, "
          f"{result['total'][1]:8.5f}, {result['total'][2]:8.5f}] a.u.")
    print(f"    Magnitude:  {result['magnitude_au']:.5f} a.u. = "
          f"{result['magnitude_debye']:.4f} Debye")

    # PySCF reference
    dip_pyscf = mf.dip_moment(unit='AU', verbose=0)
    dip_mag_pyscf = np.linalg.norm(dip_pyscf)

    print(f"\n  PySCF reference:")
    print(f"    Total:      [{dip_pyscf[0]:8.5f}, {dip_pyscf[1]:8.5f}, "
          f"{dip_pyscf[2]:8.5f}] a.u.")
    print(f"    Magnitude:  {dip_mag_pyscf:.5f} a.u. = "
          f"{dip_mag_pyscf * 2.541746:.4f} Debye")

    # Verify
    diff = np.linalg.norm(result['total'] - dip_pyscf)
    print(f"\n  Verification: ||our - PySCF|| = {diff:.2e}")
    assert diff < 1e-6, "Dipole mismatch!"
    print("  PASSED")

    # =========================================================================
    # Test 2: H2 (nonpolar by symmetry)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: H2 (nonpolar molecule)")
    print("=" * 50)

    mol_h2 = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    mf_h2 = scf.RHF(mol_h2)
    mf_h2.kernel()
    dm_h2 = mf_h2.make_rdm1()

    # Center of mass as origin
    com = np.mean(mol_h2.atom_coords(), axis=0)
    result_h2 = compute_dipole_moment(mol_h2, dm_h2, origin=com)

    print(f"\n  Molecule: H2 / cc-pVDZ")
    print(f"  Origin: Center of mass {com}")
    print(f"  Total dipole: [{result_h2['total'][0]:8.5f}, "
          f"{result_h2['total'][1]:8.5f}, {result_h2['total'][2]:8.5f}] a.u.")
    print(f"  Magnitude: {result_h2['magnitude_debye']:.6f} Debye")
    print(f"  (Should be ~0 for symmetric homonuclear diatomic)")

    # =========================================================================
    # Test 3: HF (large dipole)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: HF (highly polar molecule)")
    print("=" * 50)

    mol_hf = gto.M(
        atom="H 0 0 0; F 0 0 0.92",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    mf_hf = scf.RHF(mol_hf)
    mf_hf.kernel()
    dm_hf = mf_hf.make_rdm1()

    # Use center of mass
    com_hf = np.sum([mol_hf.atom_charge(i) * mol_hf.atom_coord(i)
                     for i in range(mol_hf.natm)], axis=0) / sum(
        mol_hf.atom_charge(i) for i in range(mol_hf.natm))

    result_hf = compute_dipole_moment(mol_hf, dm_hf, origin=np.zeros(3))

    print(f"\n  Molecule: HF / cc-pVDZ")
    print(f"  Origin: [0, 0, 0]")
    print(f"  Electronic: [{result_hf['electronic'][0]:8.5f}, "
          f"{result_hf['electronic'][1]:8.5f}, {result_hf['electronic'][2]:8.5f}] a.u.")
    print(f"  Nuclear:    [{result_hf['nuclear'][0]:8.5f}, "
          f"{result_hf['nuclear'][1]:8.5f}, {result_hf['nuclear'][2]:8.5f}] a.u.")
    print(f"  Total:      [{result_hf['total'][0]:8.5f}, "
          f"{result_hf['total'][1]:8.5f}, {result_hf['total'][2]:8.5f}] a.u.")
    print(f"  Magnitude:  {result_hf['magnitude_debye']:.4f} Debye")
    print(f"  (Experimental: ~1.83 Debye)")

    # =========================================================================
    # Test 4: Origin Dependence for Neutral Molecules
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 4: Origin Independence for Neutral Molecules")
    print("=" * 50)

    print(f"\n  For a neutral molecule, the dipole moment should be")
    print(f"  independent of the choice of origin.")

    origins = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([5.0, 5.0, 5.0]),
    ]

    print(f"\n  H2O dipole with different origins:")
    print(f"  {'Origin':<25} | {'Magnitude (Debye)':<20}")
    print("  " + "-" * 50)

    for origin in origins:
        result_test = compute_dipole_moment(mol_h2o, dm, origin=origin)
        print(f"  {str(origin):<25} | {result_test['magnitude_debye']:<20.6f}")

    print(f"\n  All magnitudes should be identical (within numerical precision)")

    # =========================================================================
    # Test 5: LiH (ionic character)
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 5: LiH (ionic character)")
    print("=" * 50)

    mol_lih = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    mf_lih = scf.RHF(mol_lih)
    mf_lih.kernel()
    dm_lih = mf_lih.make_rdm1()

    result_lih = compute_dipole_moment(mol_lih, dm_lih, origin=np.zeros(3))

    print(f"\n  Molecule: LiH / cc-pVDZ")
    print(f"  Bond length: 1.6 Angstrom")
    print(f"  Total dipole: [{result_lih['total'][0]:8.5f}, "
          f"{result_lih['total'][1]:8.5f}, {result_lih['total'][2]:8.5f}] a.u.")
    print(f"  Magnitude: {result_lih['magnitude_debye']:.4f} Debye")
    print(f"  (Experimental: ~5.88 Debye)")

    # =========================================================================
    # Test 6: Unit Conversion Reference
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 6: Unit Conversion Reference")
    print("=" * 50)

    print(f"\n  Dipole moment unit conversions:")
    print(f"    1 a.u. (e*a0) = 2.541746 Debye")
    print(f"    1 a.u. (e*a0) = 8.478353e-30 C*m")
    print(f"    1 Debye = 3.33564e-30 C*m")

    print(f"\n  Typical dipole moments:")
    print(f"    H2O:  ~1.85 Debye (experimental)")
    print(f"    HF:   ~1.83 Debye (experimental)")
    print(f"    LiH:  ~5.88 Debye (experimental)")
    print(f"    NH3:  ~1.47 Debye (experimental)")

    # =========================================================================
    # Test 7: Basis Set Dependence
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 7: Basis Set Dependence of Dipole Moment")
    print("=" * 50)

    print(f"\n  H2O dipole moment vs basis set:")
    print(f"  {'Basis':<12} | {'nAO':<5} | {'mu (Debye)':<12}")
    print("  " + "-" * 35)

    for basis in ['sto-3g', '6-31g', 'cc-pvdz', 'cc-pvtz', 'aug-cc-pvdz']:
        mol_test = gto.M(
            atom="""
                O   0.0000   0.0000   0.1173
                H   0.0000   0.7572  -0.4692
                H   0.0000  -0.7572  -0.4692
            """,
            basis=basis,
            unit="Angstrom",
            verbose=0
        )

        mf_test = scf.RHF(mol_test)
        mf_test.kernel()
        dip_test = mf_test.dip_moment(unit='DEBYE', verbose=0)
        mag_test = np.linalg.norm(dip_test)

        print(f"  {basis:<12} | {mol_test.nao:<5} | {mag_test:<12.4f}")

    print(f"\n  Note: Larger basis sets generally give more accurate dipoles.")
    print(f"  Diffuse functions (aug-) important for polarizability/dipole.")

    print("\n" + "=" * 70)
    print("Dipole calculation demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
