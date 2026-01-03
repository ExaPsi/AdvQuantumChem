#!/usr/bin/env python3
"""
Unit Conversion Factors and Utilities

CODATA 2018 conversion factors for quantum chemistry calculations.
Essential for converting between atomic units and conventional units.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix B: Atomic Units and Conversions
"""

import numpy as np
from pyscf import gto, scf

# ================================================================
# CODATA conversion factors (2018 values)
# ================================================================

# Length
BOHR_TO_ANGSTROM = 0.529177210903      # 1 a0 -> Angstrom
ANGSTROM_TO_BOHR = 1.8897259886        # 1 Angstrom -> a0

# Energy
HARTREE_TO_EV = 27.211386245988        # 1 Eh -> eV
HARTREE_TO_KJMOL = 2625.4996394799     # 1 Eh -> kJ/mol
HARTREE_TO_KCALMOL = 627.5094740631    # 1 Eh -> kcal/mol
HARTREE_TO_CM1 = 219474.6313632        # 1 Eh -> cm^-1

# Dipole moment
AU_TO_DEBYE = 2.5417464519             # 1 e*a0 -> Debye
DEBYE_TO_AU = 0.3934303                # 1 Debye -> e*a0


# ================================================================
# Convenience functions
# ================================================================

def bohr_to_angstrom(r_bohr: float) -> float:
    """Convert distance from Bohr to Angstrom."""
    return r_bohr * BOHR_TO_ANGSTROM


def angstrom_to_bohr(r_angstrom: float) -> float:
    """Convert distance from Angstrom to Bohr."""
    return r_angstrom * ANGSTROM_TO_BOHR


def hartree_to_ev(E_hartree: float) -> float:
    """Convert energy from Hartree to electronvolts."""
    return E_hartree * HARTREE_TO_EV


def hartree_to_kcalmol(E_hartree: float) -> float:
    """Convert energy from Hartree to kcal/mol."""
    return E_hartree * HARTREE_TO_KCALMOL


def hartree_to_kjmol(E_hartree: float) -> float:
    """Convert energy from Hartree to kJ/mol."""
    return E_hartree * HARTREE_TO_KJMOL


def dipole_au_to_debye(mu_au: float) -> float:
    """Convert dipole moment from atomic units to Debye."""
    return mu_au * AU_TO_DEBYE


def main():
    """Demonstrate unit conversions with a water calculation."""

    # Water molecule
    mol = gto.M(
        atom="""
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        """,
        basis="cc-pvdz",
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    E = mf.kernel()

    print("=" * 60)
    print("Unit Conversion Demo: Water / cc-pVDZ")
    print("=" * 60)

    # Report energy in various units
    print(f"\nEnergy conversions:")
    print(f"  {E:.10f} Hartree")
    print(f"  {hartree_to_ev(E):.6f} eV")
    print(f"  {hartree_to_kjmol(E):.4f} kJ/mol")
    print(f"  {hartree_to_kcalmol(E):.4f} kcal/mol")

    # Extract and convert bond length
    coords = mol.atom_coords()  # Always in Bohr!
    R_OH_bohr = np.linalg.norm(coords[1] - coords[0])
    R_OH_ang = bohr_to_angstrom(R_OH_bohr)

    print(f"\nO-H bond length:")
    print(f"  {R_OH_bohr:.4f} Bohr")
    print(f"  {R_OH_ang:.4f} Angstrom")


if __name__ == "__main__":
    main()
