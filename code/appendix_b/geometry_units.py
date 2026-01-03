#!/usr/bin/env python3
"""
Geometry Unit Specification in PySCF

Demonstrates proper unit handling when specifying molecular geometries.
PySCF accepts geometries in either Angstrom or Bohr (atomic units).

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix B: Atomic Units and Conversions
"""

import numpy as np
from pyscf import gto, scf


def main():
    # ----------------------------------------------------------------
    # Specifying geometry in Angstrom (most common in chemistry)
    # ----------------------------------------------------------------
    # The equilibrium bond length of H2 is approximately 0.74 Angstrom
    mol_angstrom = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",  # Explicit unit specification
        verbose=0
    )

    mf_ang = scf.RHF(mol_angstrom)
    E_ang = mf_ang.kernel()

    print("=" * 60)
    print("Geometry Unit Specification Demo")
    print("=" * 60)
    print(f"\nInput: R = 0.74 Angstrom")
    print(f"Internal (Bohr): {mol_angstrom.atom_coords()[1, 2]:.6f}")
    print(f"HF Energy: {E_ang:.10f} Hartree")

    # ----------------------------------------------------------------
    # Same geometry in Bohr (atomic units)
    # ----------------------------------------------------------------
    # Convert: 0.74 Angstrom * 1.8897 = 1.398 Bohr
    R_bohr = 0.74 * 1.8897259886

    mol_bohr = gto.M(
        atom=f"H 0 0 0; H 0 0 {R_bohr}",
        basis="sto-3g",
        unit="Bohr",  # Atomic units
        verbose=0
    )

    mf_bohr = scf.RHF(mol_bohr)
    E_bohr = mf_bohr.kernel()

    print(f"\nInput: R = {R_bohr:.6f} Bohr")
    print(f"HF Energy: {E_bohr:.10f} Hartree")

    # Both specifications give identical results
    print(f"\nEnergy difference: {abs(E_ang - E_bohr):.2e} Hartree")
    print("(Should be ~0, confirming both specs are equivalent)")


if __name__ == "__main__":
    main()
