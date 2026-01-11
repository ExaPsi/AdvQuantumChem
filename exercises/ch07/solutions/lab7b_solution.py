#!/usr/bin/env python3
"""
Lab 7B Solution: Dipole Moment from Density Matrix and Integrals
=================================================================

This script demonstrates the "integrals-first" approach to computing molecular
properties. The dipole moment is computed directly from the density matrix and
one-electron position operator integrals.

Learning objectives:
1. Understand one-electron property formula: <O> = Tr[P * o]
2. Compute dipole moment: mu = Sum_A Z_A R_A - Tr[P * r]
3. Extract and use dipole integrals from PySCF
4. Verify against PySCF reference (mf.dip_moment())
5. Convert between atomic units and Debye
6. Explore origin dependence for neutral vs charged species

Test molecules: H2O, HF, NH3

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 7: Scaling and Properties
"""

import numpy as np
from pyscf import gto, scf

# =============================================================================
# Physical Constants and Unit Conversions
# =============================================================================

# Conversion factor: atomic units (e*a0) to Debye
# 1 Debye = 10^-18 esu*cm = 0.393430 a.u.
# 1 a.u. = 2.5417464 Debye
AU_TO_DEBYE = 2.5417464

# For reference: 1 a.u. of dipole = e * a0 = 8.478 x 10^-30 C*m


# =============================================================================
# Section 1: Core Dipole Calculation Functions
# =============================================================================


def compute_dipole_from_integrals(mol: gto.Mole, dm: np.ndarray,
                                   origin: np.ndarray = None) -> np.ndarray:
    """
    Compute molecular dipole moment from density matrix and integrals.

    The dipole moment is:
        mu = mu_nuc - mu_el
           = Sum_A Z_A * (R_A - origin) - Tr[P * r]

    where r are the position operator integrals (int1e_r).

    Args:
        mol: PySCF molecule object
        dm: Density matrix (nao x nao)
        origin: Origin for dipole calculation (default: [0,0,0])

    Returns:
        mu: Dipole moment vector in atomic units (3,)
    """
    if origin is None:
        origin = np.zeros(3)

    # =========================================================================
    # Step 1: Nuclear contribution
    # mu_nuc = Sum_A Z_A * (R_A - origin)
    # =========================================================================
    charges = mol.atom_charges()           # Array of nuclear charges Z_A
    coords = mol.atom_coords()             # Coordinates in Bohr (N_atoms x 3)
    mu_nuc = np.einsum('a,ax->x', charges, coords - origin[None, :])

    # =========================================================================
    # Step 2: Electronic contribution
    # mu_el = Tr[P * r] = Sum_mu_nu P_nu_mu * r_mu_nu
    #
    # The position operator integrals are:
    #   r_mu_nu = <chi_mu | (r - origin) | chi_nu>
    #
    # Note: We use mol.with_common_orig() to set the origin for integrals
    # =========================================================================
    with mol.with_common_orig(origin):
        # int1e_r gives <mu|r|nu> for each Cartesian component
        # Shape: (3, nao, nao) for x, y, z components
        ao_r = mol.intor_symmetric('int1e_r', comp=3)

    # Electronic dipole: Tr[P * r] for each component
    # Using einsum: 'xij,ji->x' contracts over AO indices
    mu_el = np.einsum('xij,ji->x', ao_r, dm).real

    # =========================================================================
    # Step 3: Total dipole
    # Electrons have charge -1, so total dipole is:
    # mu = mu_nuc - mu_el = Sum_A Z_A R_A - Tr[P * r]
    # =========================================================================
    mu_total = mu_nuc - mu_el

    return mu_total


def compute_dipole_components(mol: gto.Mole, dm: np.ndarray,
                               origin: np.ndarray = None) -> dict:
    """
    Compute dipole moment with detailed component breakdown.

    Args:
        mol: PySCF molecule object
        dm: Density matrix
        origin: Origin for calculation

    Returns:
        Dictionary with all dipole components and derived quantities
    """
    if origin is None:
        origin = np.zeros(3)

    # Nuclear contribution
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    mu_nuc = np.einsum('a,ax->x', charges, coords - origin[None, :])

    # Electronic contribution
    with mol.with_common_orig(origin):
        ao_r = mol.intor_symmetric('int1e_r', comp=3)
    mu_el = np.einsum('xij,ji->x', ao_r, dm).real

    # Total
    mu_total = mu_nuc - mu_el
    mu_norm = np.linalg.norm(mu_total)

    return {
        'mu_nuc': mu_nuc,
        'mu_el': mu_el,
        'mu_total': mu_total,
        'mu_norm_au': mu_norm,
        'mu_norm_debye': mu_norm * AU_TO_DEBYE,
        'origin': origin.copy(),
    }


# =============================================================================
# Section 2: Main Demonstration - Water Dipole
# =============================================================================


def demo_water_dipole() -> None:
    """
    Compute and analyze the dipole moment of water.

    Water has a significant dipole (~1.85 D experimentally) due to
    the bent geometry and O-H bond polarity.
    """
    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 7B: Dipole Moment from Density Matrix and Integrals" + " " * 13 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Water molecule
    mol = gto.M(
        atom='''
        O    0.000000    0.000000    0.117369
        H    0.756950    0.000000   -0.469476
        H   -0.756950    0.000000   -0.469476
        ''',
        basis='cc-pVTZ',
        unit='Angstrom',
        verbose=0
    )

    # Run RHF to get density matrix
    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    print("Molecule: H2O (Water)")
    print("-" * 50)
    print(f"Basis: cc-pVTZ (N_AO = {mol.nao})")
    print(f"RHF Energy: {mf.e_tot:.10f} Hartree")

    # Compute dipole at origin (0, 0, 0)
    origin = np.zeros(3)
    result = compute_dipole_components(mol, dm, origin)

    print()
    print("=" * 60)
    print("Dipole Moment Calculation")
    print("=" * 60)
    print()
    print(f"Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}) Bohr")

    print()
    print("Step 1: Nuclear contribution")
    print("  mu_nuc = Sum_A Z_A * R_A")
    print(f"  mu_nuc = ({result['mu_nuc'][0]:+.6f}, "
          f"{result['mu_nuc'][1]:+.6f}, {result['mu_nuc'][2]:+.6f}) a.u.")

    print()
    print("Step 2: Electronic contribution")
    print("  mu_el = Tr[P * r] (position operator expectation)")
    print(f"  mu_el = ({result['mu_el'][0]:+.6f}, "
          f"{result['mu_el'][1]:+.6f}, {result['mu_el'][2]:+.6f}) a.u.")

    print()
    print("Step 3: Total dipole")
    print("  mu = mu_nuc - mu_el")
    print(f"  mu = ({result['mu_total'][0]:+.6f}, "
          f"{result['mu_total'][1]:+.6f}, {result['mu_total'][2]:+.6f}) a.u.")
    print()
    print(f"  |mu| = {result['mu_norm_au']:.6f} a.u.")
    print(f"       = {result['mu_norm_debye']:.4f} Debye")

    # Comparison with PySCF
    print()
    print("-" * 60)
    print("Validation against PySCF")
    print("-" * 60)

    pyscf_dipole = mf.dip_moment(unit='AU')
    pyscf_debye = np.linalg.norm(pyscf_dipole) * AU_TO_DEBYE

    print(f"PySCF dipole: ({pyscf_dipole[0]:+.6f}, "
          f"{pyscf_dipole[1]:+.6f}, {pyscf_dipole[2]:+.6f}) a.u.")
    print(f"PySCF |mu|:   {np.linalg.norm(pyscf_dipole):.6f} a.u. = {pyscf_debye:.4f} Debye")

    error = np.linalg.norm(result['mu_total'] - pyscf_dipole)
    print()
    print(f"Difference from PySCF: {error:.2e} a.u.")

    if error < 1e-10:
        print("[PASS] Our calculation matches PySCF exactly!")
    else:
        print("[FAIL] Unexpected difference from PySCF!")

    # Physical interpretation
    print()
    print("=" * 60)
    print("Physical Interpretation")
    print("=" * 60)
    print("""
Water's dipole moment points from the oxygen toward the hydrogen atoms
(in the conventional chemistry direction, from - to +). The magnitude
of ~1.85 D (experimental) arises from:

1. Bond polarity: O is more electronegative than H, pulling electron
   density toward oxygen (mu_el large on O side)

2. Bent geometry: The two O-H bond dipoles don't cancel (as they
   would in linear H-O-H)

The HF/cc-pVTZ value (~1.93 D) is slightly higher than experiment
because HF overestimates ionic character in polar bonds.
""")


# =============================================================================
# Section 3: Origin Independence for Neutral Molecules
# =============================================================================


def demo_origin_independence() -> None:
    """
    Demonstrate that dipole moment is origin-independent for neutral molecules.

    For a system with total charge Q:
        mu(O + d) = mu(O) - Q * d

    For neutral molecules (Q = 0), the dipole is origin-independent.
    For charged species, it depends on the choice of origin.
    """
    print()
    print("=" * 75)
    print("Origin Independence Test")
    print("=" * 75)

    # Neutral molecule: H2O
    mol_neutral = gto.M(
        atom='O 0 0 0; H 0.7570 0 0.5043; H -0.7570 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf_neutral = scf.RHF(mol_neutral)
    mf_neutral.kernel()
    dm_neutral = mf_neutral.make_rdm1()

    # Test various origins
    origins = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    ]

    print()
    print("1. Neutral molecule (H2O, Q = 0):")
    print("-" * 55)
    print(f"{'Origin (Bohr)':<25} {'|mu| (Debye)':>15}")
    print("-" * 55)

    dipoles_neutral = []
    for origin in origins:
        mu = compute_dipole_from_integrals(mol_neutral, dm_neutral, origin)
        mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_neutral.append(mu_debye)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"{mu_debye:>18.8f}")

    max_diff_neutral = max(dipoles_neutral) - min(dipoles_neutral)
    print("-" * 55)
    print(f"Maximum difference: {max_diff_neutral:.2e} Debye")

    if max_diff_neutral < 1e-10:
        print("[PASS] Dipole is origin-independent for neutral molecule!")
    else:
        print("[FAIL] Unexpected origin dependence!")

    # Charged species: OH- (hydroxide ion)
    print()
    print("2. Charged molecule (OH-, Q = -1):")
    print("-" * 55)

    mol_anion = gto.M(
        atom='O 0 0 0; H 0 0 0.97',
        basis='cc-pVDZ',
        unit='Angstrom',
        charge=-1,
        verbose=0
    )

    mf_anion = scf.RHF(mol_anion)
    mf_anion.kernel()
    dm_anion = mf_anion.make_rdm1()

    print(f"{'Origin (Bohr)':<25} {'|mu| (Debye)':>15}")
    print("-" * 55)

    dipoles_anion = []
    for origin in origins:
        mu = compute_dipole_from_integrals(mol_anion, dm_anion, origin)
        mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE
        dipoles_anion.append(mu_debye)
        print(f"({origin[0]:+5.1f}, {origin[1]:+5.1f}, {origin[2]:+5.1f})"
              f"{mu_debye:>18.8f}")

    max_diff_anion = max(dipoles_anion) - min(dipoles_anion)
    print("-" * 55)
    print(f"Maximum difference: {max_diff_anion:.4f} Debye")
    print("[EXPECTED] Dipole IS origin-dependent for charged species!")

    # Explanation
    print()
    print("=" * 60)
    print("Mathematical Explanation")
    print("=" * 60)
    print("""
For a system with total charge Q, shifting the origin by d gives:

    mu(O + d) = mu(O) - Q * d

Proof:
  mu_nuc' = Sum_A Z_A (R_A - O - d) = mu_nuc - Z_tot * d
  mu_el'  = Tr[P * (r - O - d)]    = mu_el + N_e * d   (since Tr[P] = N_e)
  mu'     = mu_nuc' - mu_el'       = mu - (Z_tot - N_e) * d = mu - Q * d

For neutral molecules: Q = Z_tot - N_e = 0, so mu is origin-independent.
For ions: mu depends on origin, so one must specify a convention
(e.g., center of mass or center of nuclear charge).
""")


# =============================================================================
# Section 4: Basis Set Dependence
# =============================================================================


def demo_basis_set_comparison() -> None:
    """
    Compare dipole moments computed with different basis sets.

    Larger, more diffuse basis sets generally give more accurate dipole
    moments because they better describe the electron density tail.
    """
    print()
    print("=" * 75)
    print("Basis Set Dependence of Dipole Moment")
    print("=" * 75)

    # Water molecule
    mol_atoms = '''
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    '''

    basis_sets = ['STO-3G', '6-31G', '6-31G*', 'cc-pVDZ', 'cc-pVTZ', 'aug-cc-pVDZ']

    print()
    print("Molecule: H2O")
    print("-" * 65)
    print(f"{'Basis':<14} {'N_AO':>6} {'E_HF (Hartree)':>18} {'|mu| (Debye)':>14}")
    print("-" * 65)

    results = []
    for basis in basis_sets:
        mol = gto.M(atom=mol_atoms, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm = mf.make_rdm1()

        mu = compute_dipole_from_integrals(mol, dm)
        mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE

        results.append({
            'basis': basis,
            'nao': mol.nao,
            'energy': mf.e_tot,
            'dipole': mu_debye,
        })

        print(f"{basis:<14} {mol.nao:>6} {mf.e_tot:>18.10f} {mu_debye:>14.4f}")

    print("-" * 65)
    print()
    print("Experimental value: ~1.85 Debye")
    print()
    print("Observations:")
    print("  - STO-3G underestimates the dipole (poor description of tails)")
    print("  - Adding polarization functions (6-31G*) improves accuracy")
    print("  - Augmented basis (aug-cc-pVDZ) includes diffuse functions")
    print("  - HF systematically overestimates due to lack of correlation")


# =============================================================================
# Section 5: General One-Electron Property Pattern
# =============================================================================


def demo_property_pattern() -> None:
    """
    Demonstrate the general pattern for one-electron properties.

    Any one-electron operator O has expectation value:
        <O> = Tr[P * o]

    where o_mu_nu = <chi_mu | O | chi_nu>
    """
    print()
    print("=" * 75)
    print("One-Electron Property Pattern: <O> = Tr[P * o]")
    print("=" * 75)

    mol = gto.M(
        atom='O 0 0 0; H 0.7570 0 0.5043; H -0.7570 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    # Various one-electron properties using Tr[P * o]

    # 1. Electron count: O = identity, o = S (overlap)
    S = mol.intor('int1e_ovlp')
    n_elec = np.einsum('ij,ji->', dm, S)

    # 2. Kinetic energy: O = -1/2 nabla^2
    T = mol.intor('int1e_kin')
    e_kin = np.einsum('ij,ji->', dm, T)

    # 3. Nuclear attraction: O = Sum_A -Z_A/|r-R_A|
    V = mol.intor('int1e_nuc')
    e_nuc_attr = np.einsum('ij,ji->', dm, V)

    # 4. Quadrupole moment components (example of higher moment)
    with mol.with_common_orig([0, 0, 0]):
        # r_i * r_j integrals
        ao_rr = mol.intor('int1e_rr', comp=9).reshape(3, 3, mol.nao, mol.nao)

    Q_xx = np.einsum('ij,ji->', dm, ao_rr[0, 0])
    Q_yy = np.einsum('ij,ji->', dm, ao_rr[1, 1])
    Q_zz = np.einsum('ij,ji->', dm, ao_rr[2, 2])

    print()
    print(f"Molecule: H2O (cc-pVDZ, {mol.nao} AOs)")
    print("-" * 55)
    print()
    print("Property                    Operator         Value")
    print("-" * 55)
    print(f"Electron count              S               {n_elec:12.6f}")
    print(f"Expected electrons                          {mol.nelectron:12d}")
    print()
    print(f"Kinetic energy <T>          T               {e_kin:12.6f} Eh")
    print(f"Nuc. attraction <V_en>      V               {e_nuc_attr:12.6f} Eh")
    print(f"One-elec. energy <h>        T + V           {e_kin + e_nuc_attr:12.6f} Eh")
    print()
    print(f"<x^2> electronic            r_x * r_x       {Q_xx:12.6f} a.u.")
    print(f"<y^2> electronic            r_y * r_y       {Q_yy:12.6f} a.u.")
    print(f"<z^2> electronic            r_z * r_z       {Q_zz:12.6f} a.u.")

    # Validation
    print()
    print("-" * 55)
    print("Validation:")
    if abs(n_elec - mol.nelectron) < 1e-10:
        print(f"[PASS] Electron count matches ({n_elec:.10f})")
    else:
        print(f"[FAIL] Electron count mismatch!")

    print()
    print("=" * 55)
    print("Key pattern: For any one-electron operator O,")
    print("   <O> = Tr[P * o]  where  o_mu_nu = <mu|O|nu>")
    print("=" * 55)


# =============================================================================
# Section 6: Validation Against PySCF
# =============================================================================


def validate_dipole_calculation() -> None:
    """
    Comprehensive validation against PySCF for multiple molecules.
    """
    print()
    print("=" * 75)
    print("Validation: Dipole Calculations vs PySCF")
    print("=" * 75)

    test_cases = [
        ('HF', 'H 0 0 0; F 0 0 0.92', 'cc-pVDZ'),
        ('H2O', 'O 0 0 0; H 0.7570 0 0.5043; H -0.7570 0 0.5043', 'cc-pVDZ'),
        ('NH3', '''N 0 0 0.1173;
                   H 0 0.9377 -0.2737;
                   H 0.8121 -0.4689 -0.2737;
                   H -0.8121 -0.4689 -0.2737''', 'cc-pVDZ'),
        ('CO', 'C 0 0 0; O 0 0 1.128', 'cc-pVTZ'),
    ]

    print()
    print(f"{'Molecule':<10} {'Basis':<12} {'Our |mu| (D)':>14} "
          f"{'PySCF |mu| (D)':>14} {'Error':>12}")
    print("-" * 70)

    all_passed = True
    for name, atoms, basis in test_cases:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        dm = mf.make_rdm1()

        # Our calculation
        mu_ours = compute_dipole_from_integrals(mol, dm)
        mu_ours_debye = np.linalg.norm(mu_ours) * AU_TO_DEBYE

        # PySCF reference
        mu_pyscf = mf.dip_moment(unit='DEBYE')
        mu_pyscf_norm = np.linalg.norm(mu_pyscf)

        error = abs(mu_ours_debye - mu_pyscf_norm)
        status = "[OK]" if error < 1e-6 else "[FAIL]"
        if error >= 1e-6:
            all_passed = False

        print(f"{name:<10} {basis:<12} {mu_ours_debye:>14.6f} "
              f"{mu_pyscf_norm:>14.6f} {error:>10.2e} {status}")

    print("-" * 70)
    print()
    if all_passed:
        print("[PASS] All dipole calculations match PySCF reference!")
    else:
        print("[FAIL] Some calculations differ from PySCF!")


# =============================================================================
# Section 7: What You Should Observe
# =============================================================================


def print_observations() -> None:
    """Print summary of key observations from this lab."""

    observations = """
================================================================================
What You Should Observe (Lab 7B)
================================================================================

1. DIPOLE FORMULA:
   mu = mu_nuc - mu_el = Sum_A Z_A R_A - Tr[P * r]
   - Nuclear term: positive charges at nuclear positions
   - Electronic term: trace of density with position integrals
   - The minus sign: electrons have negative charge

2. ORIGIN INDEPENDENCE (neutral molecules):
   - For Q = 0: dipole is the same regardless of origin choice
   - This is a fundamental physical requirement
   - Mathematical proof: mu' = mu - Q*d, so for Q=0, mu'=mu

3. ORIGIN DEPENDENCE (ions):
   - For Q != 0: dipole changes with origin by -Q*d
   - Must specify convention (e.g., center of mass)
   - No unique "dipole moment" for charged species

4. UNIT CONVERSION:
   - 1 atomic unit (e*a0) = 2.5417 Debye
   - Experimental dipoles typically reported in Debye
   - HF tends to overestimate (lacks electron correlation)

5. BASIS SET EFFECTS:
   - Minimal bases underestimate (poor tail description)
   - Diffuse functions improve accuracy
   - HF values ~5-10% higher than experiment for polar molecules

6. GENERAL PROPERTY PATTERN:
   For any one-electron operator O:
     <O> = Tr[P * o]  where  o_mu_nu = <mu|O|nu>

   Examples:
     - Electron count: Tr[P * S]
     - Kinetic energy: Tr[P * T]
     - Dipole: mu = nuclear - Tr[P * r]
     - Quadrupole: involves Tr[P * r*r]

================================================================================
"""
    print(observations)


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Run the complete Lab 7B demonstration."""

    # Core demonstration
    demo_water_dipole()

    # Origin independence
    demo_origin_independence()

    # Basis set comparison
    demo_basis_set_comparison()

    # General property pattern
    demo_property_pattern()

    # Validation
    validate_dipole_calculation()

    # Summary
    print_observations()

    print()
    print("=" * 75)
    print("Lab 7B Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
