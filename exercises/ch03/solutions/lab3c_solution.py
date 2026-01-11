#!/usr/bin/env python3
"""
Lab 3C Solution: Dipole Integrals and Dipole Moment from Density

This solution script demonstrates the calculation of molecular dipole
moments as one-electron properties, using the density matrix and
position operator integrals.

KEY CONCEPTS:
=============

1. One-electron properties from density matrix:
   <O> = Tr[P * o]
   where P is the density matrix and o_uv = <u|O|v> are operator integrals.

2. Dipole moment formula:
   mu = sum_A Z_A * (R_A - O) - Tr[P * r(O)]

   - First term: nuclear contribution (positive charges)
   - Second term: electronic contribution (negative charges, note sign)
   - O: origin of coordinate system

3. Origin dependence:
   - Neutral molecules: dipole is origin-independent
   - Charged species: dipole depends on origin choice
   - Physical reason: only charge separation matters for neutral systems

4. Units:
   - Atomic units: e * a_0 (electron charge times Bohr radius)
   - Debye: 1 e*a_0 = 2.541746 Debye
   - Typical molecular dipoles: 0.5 - 5 Debye

Physical Insight:
-----------------
The dipole moment measures the separation of positive (nuclear) and
negative (electronic) charge centers. For water, the dipole points
from the oxygen toward the hydrogens because the electrons are pulled
toward the more electronegative oxygen, leaving a net positive charge
on the H atoms.

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
from typing import Tuple, Optional, Dict
from pyscf import gto, scf


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Conversion: atomic units (e*a_0) to Debye
# 1 Debye = 3.33564 x 10^-30 C*m
# 1 a_0 = 5.29177 x 10^-11 m
# 1 e = 1.60218 x 10^-19 C
# => 1 e*a_0 = 8.478 x 10^-30 C*m = 2.5418 Debye
AU_TO_DEBYE = 2.541746


# =============================================================================
# DIPOLE INTEGRALS
# =============================================================================

def get_dipole_integrals(mol: gto.Mole, origin: np.ndarray) -> np.ndarray:
    """
    Compute dipole operator integrals <mu|r - origin|nu>.

    The dipole integrals are one-electron integrals of the position
    operator relative to a chosen origin. They have shape (3, nao, nao)
    for the x, y, z components.

    In PySCF, the origin must be set before computing the integrals
    using mol.with_common_orig(origin).

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    origin : np.ndarray
        Origin for dipole calculation (in Bohr)

    Returns
    -------
    np.ndarray
        Dipole integrals, shape (3, nao, nao)
        Component 0: x, 1: y, 2: z
    """
    origin = np.asarray(origin, dtype=np.float64)

    # Set the common origin for multipole integrals
    with mol.with_common_orig(origin):
        # int1e_r gives <mu | r - origin | nu>
        # Returns shape (3, nao, nao) for x, y, z components
        dipole_ints = mol.intor_symmetric("int1e_r", comp=3)

    return dipole_ints


# =============================================================================
# DIPOLE MOMENT CALCULATION
# =============================================================================

def compute_nuclear_dipole(mol: gto.Mole, origin: np.ndarray) -> np.ndarray:
    """
    Compute nuclear contribution to dipole moment.

    mu_nuc = sum_A Z_A * (R_A - origin)

    The nuclear contribution is simply the sum of nuclear charges
    times their positions relative to the origin.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    origin : np.ndarray
        Origin for dipole calculation (in Bohr)

    Returns
    -------
    np.ndarray
        Nuclear dipole contribution (3D vector, in a.u.)
    """
    origin = np.asarray(origin, dtype=np.float64)

    # Get nuclear charges and coordinates
    charges = mol.atom_charges()       # Array of Z values
    coords = mol.atom_coords()          # Shape (natom, 3), in Bohr

    # Nuclear dipole: sum_A Z_A * (R_A - origin)
    mu_nuc = np.einsum("a,ax->x", charges, coords - origin[None, :])

    return mu_nuc


def compute_electronic_dipole(dm: np.ndarray, dipole_ints: np.ndarray) -> np.ndarray:
    """
    Compute electronic contribution to dipole moment.

    mu_elec = Tr[P * r] = sum_{uv} P_uv * r_vu

    The electronic contribution is the expectation value of the
    position operator, weighted by the electron density. Note that
    electrons have negative charge, so this contributes with opposite
    sign to the total dipole.

    Parameters
    ----------
    dm : np.ndarray
        Density matrix in AO basis, shape (nao, nao)
    dipole_ints : np.ndarray
        Dipole integrals, shape (3, nao, nao)

    Returns
    -------
    np.ndarray
        Electronic dipole contribution (3D vector, in a.u.)
    """
    # Tr[P * r] for each Cartesian component
    # einsum: sum over u,v of P_uv * r_vu = P_uv * r_uv (symmetric)
    mu_elec = np.einsum("xij,ji->x", dipole_ints, dm).real

    return mu_elec


def compute_dipole_moment(mol: gto.Mole, dm: np.ndarray,
                           origin: Optional[np.ndarray] = None) -> Dict:
    """
    Compute molecular dipole moment from density matrix.

    The total dipole is:
        mu = mu_nuc - mu_elec
           = sum_A Z_A * (R_A - O) - Tr[P * r(O)]

    The minus sign on the electronic term comes from the negative
    electron charge: the electron density creates a negative charge
    distribution.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    dm : np.ndarray
        Density matrix in AO basis
    origin : np.ndarray, optional
        Origin for dipole calculation. Default: [0, 0, 0]

    Returns
    -------
    dict
        Dictionary containing:
        - 'nuclear': nuclear contribution (a.u.)
        - 'electronic': electronic contribution (a.u.)
        - 'total': total dipole (a.u.)
        - 'magnitude_au': magnitude in a.u.
        - 'magnitude_debye': magnitude in Debye
        - 'total_debye': total dipole in Debye
    """
    if origin is None:
        origin = np.zeros(3)
    origin = np.asarray(origin, dtype=np.float64)

    # Get dipole integrals for this origin
    dipole_ints = get_dipole_integrals(mol, origin)

    # Compute nuclear and electronic contributions
    mu_nuc = compute_nuclear_dipole(mol, origin)
    mu_elec = compute_electronic_dipole(dm, dipole_ints)

    # Total dipole: nuclear - electronic
    # (electrons have negative charge, so we subtract)
    mu_total = mu_nuc - mu_elec

    # Magnitude
    magnitude_au = np.linalg.norm(mu_total)
    magnitude_debye = magnitude_au * AU_TO_DEBYE

    return {
        'origin': origin,
        'nuclear': mu_nuc,
        'electronic': mu_elec,
        'total': mu_total,
        'magnitude_au': magnitude_au,
        'magnitude_debye': magnitude_debye,
        'total_debye': mu_total * AU_TO_DEBYE
    }


# =============================================================================
# ORIGIN DEPENDENCE ANALYSIS
# =============================================================================

def analyze_origin_dependence(mol: gto.Mole, dm: np.ndarray,
                               origin1: np.ndarray, origin2: np.ndarray) -> None:
    """
    Analyze how dipole components change with origin.

    For neutral molecules, the total dipole is origin-independent because:
    - Shifting origin by d changes nuclear term by +Q*d (Q = sum Z_A)
    - Shifting origin by d changes electronic term by +N_e*d
    - For neutral: Q = N_e, so shifts cancel exactly

    For ions (Q != N_e), the dipole depends on origin choice.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    dm : np.ndarray
        Density matrix
    origin1, origin2 : np.ndarray
        Two different origins to compare
    """
    origin1 = np.asarray(origin1)
    origin2 = np.asarray(origin2)

    # Compute dipoles at both origins
    dip1 = compute_dipole_moment(mol, dm, origin1)
    dip2 = compute_dipole_moment(mol, dm, origin2)

    print("\n   Origin dependence analysis:")
    print("   " + "-" * 50)

    print(f"\n   Origin 1: {origin1}")
    print(f"      Nuclear:    {dip1['nuclear']}")
    print(f"      Electronic: {dip1['electronic']}")
    print(f"      Total:      {dip1['total']}")

    print(f"\n   Origin 2: {origin2}")
    print(f"      Nuclear:    {dip2['nuclear']}")
    print(f"      Electronic: {dip2['electronic']}")
    print(f"      Total:      {dip2['total']}")

    # Shift vector
    d = origin2 - origin1

    # Expected shift in nuclear contribution: -Q*d (Q = total nuclear charge)
    Q = float(np.sum(mol.atom_charges()))
    expected_nuc_shift = -Q * d

    # Expected shift in electronic contribution: -N_e*d
    N_e = mol.nelectron
    expected_elec_shift = -N_e * d

    print(f"\n   Shift vector d = origin2 - origin1 = {d}")
    print(f"\n   Total nuclear charge Q = {Q:.1f}")
    print(f"   Number of electrons N_e = {N_e}")

    actual_nuc_shift = dip2['nuclear'] - dip1['nuclear']
    actual_elec_shift = dip2['electronic'] - dip1['electronic']
    total_shift = dip2['total'] - dip1['total']

    print(f"\n   Nuclear shift:")
    print(f"      Expected: -Q*d = {expected_nuc_shift}")
    print(f"      Actual:   {actual_nuc_shift}")
    print(f"      Match: {np.allclose(actual_nuc_shift, expected_nuc_shift)}")

    print(f"\n   Electronic shift:")
    print(f"      Expected: -N_e*d = {expected_elec_shift}")
    print(f"      Actual:   {actual_elec_shift}")
    print(f"      Match: {np.allclose(actual_elec_shift, expected_elec_shift)}")

    print(f"\n   Total dipole shift:")
    print(f"      Shift = {total_shift}")
    print(f"      |Shift| = {np.linalg.norm(total_shift):.2e}")

    is_neutral = np.isclose(Q, N_e)
    is_origin_independent = np.linalg.norm(total_shift) < 1e-10

    print(f"\n   Molecule is neutral (Q = N_e): {is_neutral}")
    print(f"   Dipole is origin-independent: {is_origin_independent}")

    if is_neutral and is_origin_independent:
        print("\n   VERIFIED: Neutral molecule has origin-independent dipole")


# =============================================================================
# VALIDATION AGAINST PYSCF
# =============================================================================

def validate_against_pyscf(mf: scf.RHF) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate our dipole calculation against PySCF's built-in method.

    PySCF's dip_moment() method returns the dipole in Debye by default.

    Parameters
    ----------
    mf : scf.RHF
        Converged RHF object

    Returns
    -------
    our_dipole : np.ndarray
        Our calculated dipole (Debye)
    pyscf_dipole : np.ndarray
        PySCF calculated dipole (Debye)
    """
    mol = mf.mol
    dm = mf.make_rdm1()

    # Our calculation
    our_result = compute_dipole_moment(mol, dm, origin=np.zeros(3))
    our_dipole = our_result['total_debye']

    # PySCF calculation
    pyscf_dipole = mf.dip_moment(verbose=0)  # Returns in Debye

    return our_dipole, pyscf_dipole


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("Lab 3C Solution: Dipole Integrals and Dipole Moment from Density")
    print("=" * 70)

    # =========================================================================
    # SECTION 1: Build water molecule and run SCF
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. SYSTEM SETUP AND SCF CALCULATION")
    print("-" * 70)

    # Water geometry: O at origin, H atoms in xz plane
    mol = gto.M(
        atom="""
            O   0.000000   0.000000   0.000000
            H   0.758600   0.000000   0.504300
            H  -0.758600   0.000000   0.504300
        """,
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )

    print(f"\n   Molecule: H2O (water)")
    print(f"   Basis: cc-pVDZ (correlation-consistent double-zeta)")
    print(f"   Number of AOs: {mol.nao_nr()}")
    print(f"   Number of electrons: {mol.nelectron}")

    # Print geometry
    print("\n   Geometry (Bohr):")
    for i in range(mol.natm):
        symbol = mol.atom_symbol(i)
        coord = mol.atom_coord(i)
        print(f"      {symbol}: ({coord[0]:8.5f}, {coord[1]:8.5f}, {coord[2]:8.5f})")

    # Run RHF calculation
    print("\n   Running RHF calculation...")
    mf = scf.RHF(mol)
    mf.verbose = 0
    E_hf = mf.kernel()
    dm = mf.make_rdm1()

    print(f"   RHF energy: {E_hf:.10f} Hartree")
    print(f"   SCF converged: {mf.converged}")

    # =========================================================================
    # SECTION 2: Dipole integrals
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. DIPOLE INTEGRALS <mu|r|nu>")
    print("-" * 70)

    origin = np.zeros(3)
    dipole_ints = get_dipole_integrals(mol, origin)

    print(f"\n   Origin: {origin}")
    print(f"   Dipole integral shape: {dipole_ints.shape}")
    print(f"   (Components: x=0, y=1, z=2; followed by AO indices)")

    print("\n   Dipole integral norms:")
    print(f"      ||r_x|| = {np.linalg.norm(dipole_ints[0]):.6f}")
    print(f"      ||r_y|| = {np.linalg.norm(dipole_ints[1]):.6f}")
    print(f"      ||r_z|| = {np.linalg.norm(dipole_ints[2]):.6f}")

    print("\n   Physical interpretation:")
    print("   - Dipole integrals are matrix elements of position operator")
    print("   - They measure where electron density is located in space")
    print("   - Off-diagonal elements: transition dipole between AOs")

    # =========================================================================
    # SECTION 3: Compute dipole moment
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. DIPOLE MOMENT CALCULATION")
    print("-" * 70)

    result = compute_dipole_moment(mol, dm, origin)

    print(f"\n   Origin: {result['origin']}")

    print("\n   Nuclear contribution (positive charges):")
    print(f"      mu_nuc_x = {result['nuclear'][0]:12.8f} a.u.")
    print(f"      mu_nuc_y = {result['nuclear'][1]:12.8f} a.u.")
    print(f"      mu_nuc_z = {result['nuclear'][2]:12.8f} a.u.")

    print("\n   Electronic contribution (negative charges):")
    print(f"      mu_el_x  = {result['electronic'][0]:12.8f} a.u.")
    print(f"      mu_el_y  = {result['electronic'][1]:12.8f} a.u.")
    print(f"      mu_el_z  = {result['electronic'][2]:12.8f} a.u.")

    print("\n   Total dipole (nuclear - electronic):")
    print(f"      mu_x = {result['total'][0]:12.8f} a.u.")
    print(f"      mu_y = {result['total'][1]:12.8f} a.u.")
    print(f"      mu_z = {result['total'][2]:12.8f} a.u.")

    print(f"\n   Magnitude:")
    print(f"      |mu| = {result['magnitude_au']:.8f} a.u.")
    print(f"          = {result['magnitude_debye']:.6f} Debye")

    # =========================================================================
    # SECTION 4: Validation against PySCF
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. VALIDATION AGAINST PYSCF")
    print("-" * 70)

    our_dipole, pyscf_dipole = validate_against_pyscf(mf)

    print("\n   Our calculation (Debye):")
    print(f"      x: {our_dipole[0]:12.8f}")
    print(f"      y: {our_dipole[1]:12.8f}")
    print(f"      z: {our_dipole[2]:12.8f}")
    print(f"      |mu|: {np.linalg.norm(our_dipole):.6f}")

    print("\n   PySCF dip_moment() (Debye):")
    print(f"      x: {pyscf_dipole[0]:12.8f}")
    print(f"      y: {pyscf_dipole[1]:12.8f}")
    print(f"      z: {pyscf_dipole[2]:12.8f}")
    print(f"      |mu|: {np.linalg.norm(pyscf_dipole):.6f}")

    diff = np.linalg.norm(our_dipole - pyscf_dipole)
    print(f"\n   Difference: {diff:.2e} Debye")

    if diff < 1e-6:
        print("   VALIDATION PASSED: Results match PySCF!")
    else:
        print("   VALIDATION FAILED: Results differ from PySCF!")

    # =========================================================================
    # SECTION 5: Origin dependence
    # =========================================================================
    print("\n" + "-" * 70)
    print("5. ORIGIN DEPENDENCE ANALYSIS")
    print("-" * 70)

    print("\n   For neutral molecules, dipole should be origin-independent.")
    print("   This is because shifts in nuclear and electronic terms cancel.")

    # Compare dipole with origin at (0,0,0) and at oxygen nucleus
    origin1 = np.zeros(3)
    origin2 = mol.atom_coord(0)  # Oxygen position

    analyze_origin_dependence(mol, dm, origin1, origin2)

    # =========================================================================
    # SECTION 6: Direction and physical interpretation
    # =========================================================================
    print("\n" + "-" * 70)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 70)

    # Normalize dipole for direction
    mu = result['total']
    mu_norm = mu / np.linalg.norm(mu) if np.linalg.norm(mu) > 1e-10 else mu

    print(f"\n   Dipole direction (unit vector):")
    print(f"      ({mu_norm[0]:.4f}, {mu_norm[1]:.4f}, {mu_norm[2]:.4f})")

    print("""
   WATER DIPOLE MOMENT:
   --------------------
   The dipole points along the +z direction (toward the H atoms)
   because:
   1. Oxygen is more electronegative than hydrogen
   2. Electrons are pulled toward the oxygen
   3. This leaves a net positive charge on the H side
   4. Dipole points from - to + (by convention: from O toward H)

   COMPARISON WITH EXPERIMENT:
   ---------------------------
   - Experimental value: ~1.85 Debye
   - Our RHF/cc-pVDZ:    {:.2f} Debye

   The calculated value depends on:
   - Basis set quality (cc-pVDZ is moderate)
   - Level of theory (RHF neglects correlation)
   - Geometry (we used approximate bond lengths)

   ONE-ELECTRON PROPERTY:
   ----------------------
   The dipole moment is a one-electron property:
   - Requires only the density matrix P
   - Requires only one-electron integrals <mu|r|nu>
   - Does NOT require two-electron integrals
   - Computed as Tr[P * r] (simple matrix contraction)

   This is why properties like dipole moments are "cheap" compared
   to the energy, which requires ERIs.
""".format(result['magnitude_debye']))

    # =========================================================================
    # SECTION 7: Summary of key formulas
    # =========================================================================
    print("-" * 70)
    print("7. KEY FORMULAS SUMMARY")
    print("-" * 70)

    print("""
   DIPOLE MOMENT CALCULATION:
   --------------------------

   1. Nuclear contribution:
      mu_nuc = sum_A Z_A * (R_A - origin)

   2. Electronic contribution:
      mu_elec = Tr[P * r]
              = sum_{mu,nu} P_{mu,nu} * <nu|r|mu>

   3. Total dipole:
      mu = mu_nuc - mu_elec

      (The minus comes from the negative electron charge)

   4. Unit conversion:
      1 e*a_0 = 2.541746 Debye

   5. Origin independence (neutral molecules):
      For Q = N_e (neutral):
      - Shifting origin by d changes mu_nuc by -Q*d
      - Shifting origin by d changes mu_elec by -N_e*d
      - Net change: -Q*d + N_e*d = 0 when Q = N_e
    """)

    print("=" * 70)
    print("Lab 3C Solution Complete")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = main()
