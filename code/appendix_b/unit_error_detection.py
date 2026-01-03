#!/usr/bin/env python3
"""
Detecting Unit Errors in Quantum Chemistry Calculations

Demonstrates common unit mistakes and how to detect them.
Mixing Bohr and Angstrom is one of the most frequent errors!

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix B: Atomic Units and Conversions
"""

import numpy as np
from pyscf import gto, scf


def check_geometry_sanity(mol, name="molecule"):
    """
    Check for physically unreasonable bond lengths.

    Typical covalent bonds: 0.5-3.0 Angstrom
    Anything outside this range is suspicious.
    """
    coords = mol.atom_coords()  # Always in Bohr
    n_atoms = len(coords)
    issues = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            R_bohr = np.linalg.norm(coords[i] - coords[j])
            R_ang = R_bohr * 0.5292

            if R_ang < 0.5:
                issues.append(f"Atoms {i}-{j}: R = {R_ang:.3f} A (too short!)")
            elif R_ang > 5.0:
                issues.append(f"Atoms {i}-{j}: R = {R_ang:.3f} A (too long?)")

    if issues:
        print(f"WARNING in {name}:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"{name}: geometry looks reasonable")
        return True


def main():
    R_value = 0.74  # Intended as Angstrom

    print("=" * 60)
    print("Unit Error Detection Demo")
    print("=" * 60)

    # CORRECT: R = 0.74 interpreted as Angstrom
    mol_correct = gto.M(
        atom=f"H 0 0 0; H 0 0 {R_value}",
        basis="sto-3g",
        unit="Angstrom",  # Correct!
        verbose=0
    )
    mf_correct = scf.RHF(mol_correct)
    E_correct = mf_correct.kernel()

    # WRONG: R = 0.74 interpreted as Bohr (very short bond!)
    mol_wrong = gto.M(
        atom=f"H 0 0 0; H 0 0 {R_value}",
        basis="sto-3g",
        unit="Bohr",  # Wrong! 0.74 was meant as Angstrom
        verbose=0
    )
    mf_wrong = scf.RHF(mol_wrong)
    E_wrong = mf_wrong.kernel()

    # Compare results
    print("\nBond length interpretation:")
    print(f"  Correct: {mol_correct.atom_coords()[1, 2] * 0.5292:.3f} Angstrom")
    print(f"  Wrong:   {mol_wrong.atom_coords()[1, 2] * 0.5292:.3f} Angstrom")
    print()
    print("HF Energies:")
    print(f"  Correct: {E_correct:.6f} Hartree")
    print(f"  Wrong:   {E_wrong:.6f} Hartree")
    print()
    print(f"Error: {abs(E_correct - E_wrong) * 627.5:.1f} kcal/mol!")
    print("(This is a CATASTROPHIC error - larger than any chemical energy!)")

    # Apply sanity check
    print("\n" + "-" * 40)
    print("Geometry sanity checks:")
    print("-" * 40)
    check_geometry_sanity(mol_correct, "H2 (correct)")
    check_geometry_sanity(mol_wrong, "H2 (wrong)")


if __name__ == "__main__":
    main()
