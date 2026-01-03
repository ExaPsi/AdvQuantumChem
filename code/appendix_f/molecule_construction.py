#!/usr/bin/env python3
"""
Molecule Construction in PySCF

Demonstrates all methods of specifying molecular geometry in PySCF,
including string format, list format, Z-matrix hints, and unit handling.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix F: PySCF Interface Reference
"""

import numpy as np
from pyscf import gto


def main():
    print("=" * 70)
    print("Molecule Construction in PySCF")
    print("=" * 70)

    # =========================================================================
    # Section 1: String Format Specification
    # =========================================================================
    print("\n" + "-" * 50)
    print("1. String Format Specification")
    print("-" * 50)

    # Method 1a: Semicolon-separated string
    mol_str1 = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 1a: Semicolon-separated string")
    print(f"  atom = 'H 0 0 0; H 0 0 0.74'")
    print(f"  Number of atoms: {mol_str1.natm}")
    print(f"  Number of electrons: {mol_str1.nelectron}")

    # Method 1b: Multi-line string
    mol_str2 = gto.M(
        atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  0.74
        """,
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 1b: Multi-line string")
    print("  atom = '''")
    print("      H  0.0  0.0  0.0")
    print("      H  0.0  0.0  0.74")
    print("  '''")
    print(f"  Same molecule? {np.allclose(mol_str1.atom_coords(), mol_str2.atom_coords())}")

    # Method 1c: Using newline characters
    mol_str3 = gto.M(
        atom="H 0 0 0\nH 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 1c: Newline-separated string")
    print(f"  atom = 'H 0 0 0\\nH 0 0 0.74'")

    # =========================================================================
    # Section 2: List/Tuple Format Specification
    # =========================================================================
    print("\n" + "-" * 50)
    print("2. List/Tuple Format Specification")
    print("-" * 50)

    # Method 2a: List of tuples
    mol_list1 = gto.M(
        atom=[
            ('H', (0.0, 0.0, 0.0)),
            ('H', (0.0, 0.0, 0.74))
        ],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 2a: List of (element, coords) tuples")
    print("  atom = [('H', (0, 0, 0)), ('H', (0, 0, 0.74))]")

    # Method 2b: List of lists
    mol_list2 = gto.M(
        atom=[
            ['H', [0.0, 0.0, 0.0]],
            ['H', [0.0, 0.0, 0.74]]
        ],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 2b: List of [element, coords] lists")
    print("  atom = [['H', [0, 0, 0]], ['H', [0, 0, 0.74]]]")

    # Method 2c: Using atomic numbers
    mol_list3 = gto.M(
        atom=[
            (1, (0.0, 0.0, 0.0)),
            (1, (0.0, 0.0, 0.74))
        ],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )
    print("\nMethod 2c: Using atomic numbers (Z=1 for H)")
    print("  atom = [(1, (0, 0, 0)), (1, (0, 0, 0.74))]")

    # =========================================================================
    # Section 3: Unit Specification
    # =========================================================================
    print("\n" + "-" * 50)
    print("3. Unit Specification (Angstrom vs Bohr)")
    print("-" * 50)

    # Same H2 in Angstrom
    mol_ang = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0
    )

    # Same H2 in Bohr (0.74 A = 1.3984 Bohr)
    r_bohr = 0.74 / 0.52917721092  # 1 Bohr = 0.52917721092 Angstrom
    mol_bohr = gto.M(
        atom=f"H 0 0 0; H 0 0 {r_bohr}",
        basis="sto-3g",
        unit="Bohr",
        verbose=0
    )

    print(f"\n  H-H distance in Angstrom: 0.74 A")
    print(f"  H-H distance in Bohr:     {r_bohr:.6f} a0")
    print(f"  Conversion: 1 Bohr = 0.52917721092 Angstrom")

    # Verify they give the same coordinates (stored internally in Bohr)
    coords_ang = mol_ang.atom_coords()
    coords_bohr = mol_bohr.atom_coords()
    print(f"\n  Internal coordinates (always Bohr):")
    print(f"    From Angstrom input: {coords_ang}")
    print(f"    From Bohr input:     {coords_bohr}")
    print(f"  Match? {np.allclose(coords_ang, coords_bohr)}")

    # =========================================================================
    # Section 4: Accessing Molecule Attributes
    # =========================================================================
    print("\n" + "-" * 50)
    print("4. Accessing Molecule Attributes")
    print("-" * 50)

    # Build a more interesting molecule: water
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

    print("\nWater molecule (cc-pVDZ basis):")
    print(f"  Number of atoms:          {mol_h2o.natm}")
    print(f"  Number of electrons:      {mol_h2o.nelectron}")
    print(f"  Number of AO basis funcs: {mol_h2o.nao}")
    print(f"  Total charge:             {mol_h2o.charge}")
    print(f"  Spin (2S):                {mol_h2o.spin}")

    print("\n  Atom-by-atom information:")
    for i in range(mol_h2o.natm):
        symbol = mol_h2o.atom_symbol(i)
        charge = mol_h2o.atom_charge(i)
        coords = mol_h2o.atom_coord(i)
        print(f"    Atom {i}: {symbol} (Z={charge:.0f}), coords = {coords}")

    print("\n  Basis set information:")
    print(f"    Basis functions per atom:")
    for i in range(mol_h2o.natm):
        # Count basis functions on this atom
        nao_atom = sum(1 for ao_labels in mol_h2o.ao_labels()
                       if ao_labels.startswith(f'{i} '))
        print(f"      {mol_h2o.atom_symbol(i)}: {nao_atom} AOs")

    # Shell information
    print(f"\n    Number of shells: {mol_h2o.nbas}")
    print(f"    Angular momenta present: {set(mol_h2o.bas_angular(i) for i in range(mol_h2o.nbas))}")

    # =========================================================================
    # Section 5: Charged and Open-Shell Systems
    # =========================================================================
    print("\n" + "-" * 50)
    print("5. Charged and Open-Shell Systems")
    print("-" * 50)

    # H2+ cation (charge = +1, doublet)
    mol_h2plus = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=1,
        spin=1,  # 2S = 1 for doublet
        verbose=0
    )
    print("\nH2+ cation:")
    print(f"  charge = 1, spin = 1 (doublet)")
    print(f"  Number of electrons: {mol_h2plus.nelectron}")
    print(f"  nelec tuple (alpha, beta): {mol_h2plus.nelec}")

    # O2 triplet (ground state is triplet)
    mol_o2 = gto.M(
        atom="O 0 0 0; O 0 0 1.21",
        basis="sto-3g",
        unit="Angstrom",
        spin=2,  # 2S = 2 for triplet
        verbose=0
    )
    print("\nO2 triplet:")
    print(f"  charge = 0, spin = 2 (triplet)")
    print(f"  Number of electrons: {mol_o2.nelectron}")
    print(f"  nelec tuple (alpha, beta): {mol_o2.nelec}")

    # =========================================================================
    # Section 6: Nuclear Repulsion Energy
    # =========================================================================
    print("\n" + "-" * 50)
    print("6. Nuclear Repulsion Energy")
    print("-" * 50)

    E_nuc_h2 = mol_str1.energy_nuc()
    E_nuc_h2o = mol_h2o.energy_nuc()

    print(f"\n  H2:  E_nuc = {E_nuc_h2:.10f} Hartree")
    print(f"  H2O: E_nuc = {E_nuc_h2o:.10f} Hartree")

    # Manual calculation for H2
    coords = mol_str1.atom_coords()
    R = np.linalg.norm(coords[1] - coords[0])
    Z1 = mol_str1.atom_charge(0)
    Z2 = mol_str1.atom_charge(1)
    E_nuc_manual = Z1 * Z2 / R
    print(f"\n  Manual H2: E_nuc = Z1*Z2/R = {E_nuc_manual:.10f} Hartree")
    print(f"  Match? {np.isclose(E_nuc_h2, E_nuc_manual)}")

    # =========================================================================
    # Section 7: Symmetry Handling
    # =========================================================================
    print("\n" + "-" * 50)
    print("7. Symmetry Handling")
    print("-" * 50)

    # Without symmetry (default)
    mol_nosym = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        symmetry=False,
        verbose=0
    )

    # With symmetry
    mol_sym = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        symmetry=True,
        verbose=0
    )

    print(f"\n  Without symmetry: symmetry = {mol_nosym.symmetry}")
    print(f"  With symmetry:    symmetry = {mol_sym.symmetry}")
    print(f"  Point group:      {mol_sym.groupname if mol_sym.symmetry else 'N/A'}")

    print("\n" + "=" * 70)
    print("Molecule construction demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
