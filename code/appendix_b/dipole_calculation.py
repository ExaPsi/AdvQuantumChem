#!/usr/bin/env python3
"""
Dipole Moment Calculation and Unit Conversion

Demonstrates computing dipole moments from integrals and density,
with proper conversion from atomic units to Debye.

Part of: Advanced Quantum Chemistry Lecture Notes
Appendix B: Atomic Units and Conversions
"""

import numpy as np
from pyscf import gto, scf

AU_TO_DEBYE = 2.5417464519  # 1 e*a0 = 2.5417 Debye


def compute_dipole_manual(mol, dm):
    """
    Compute dipole moment manually from density matrix and integrals.

    mu = mu_elec + mu_nuc
    mu_elec = -Tr[P * r]  (negative for electron charge)
    mu_nuc = sum_A Z_A * R_A
    """
    # Dipole integrals: <mu|r|nu> with chosen origin
    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)

    # Electronic contribution
    mu_elec = -np.einsum('xij,ji->x', ao_dip, dm)

    # Nuclear contribution
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    mu_nuc = np.einsum('i,ix->x', charges, coords)

    # Total dipole in atomic units
    mu_au = mu_elec + mu_nuc

    return mu_au


def main():
    print("=" * 60)
    print("Dipole Moment Calculation: Water")
    print("=" * 60)

    # Water molecule (experimental dipole: ~1.85 Debye)
    mol = gto.M(
        atom="""
        O  0.0000  0.0000  0.1173
        H  0.0000  0.7572 -0.4692
        H  0.0000 -0.7572 -0.4692
        """,
        basis="aug-cc-pvdz",  # Diffuse functions improve dipoles
        unit="Angstrom",
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1()

    # Method 1: Manual calculation
    mu_au = compute_dipole_manual(mol, dm)
    mu_debye = mu_au * AU_TO_DEBYE
    mu_total = np.linalg.norm(mu_debye)

    print("\nMethod 1: Manual calculation from integrals")
    print("-" * 40)
    print("Dipole moment (atomic units):")
    print(f"  mu_x = {mu_au[0]:+.6f} e*a0")
    print(f"  mu_y = {mu_au[1]:+.6f} e*a0")
    print(f"  mu_z = {mu_au[2]:+.6f} e*a0")
    print(f"  |mu| = {np.linalg.norm(mu_au):.6f} e*a0")
    print()
    print("Dipole moment (Debye):")
    print(f"  |mu| = {mu_total:.3f} D")

    # Method 2: Use PySCF's built-in method
    mu_pyscf = mf.dip_moment(verbose=0)

    print("\nMethod 2: PySCF dip_moment()")
    print("-" * 40)
    print(f"  mu_x = {mu_pyscf[0]:+.6f} D")
    print(f"  mu_y = {mu_pyscf[1]:+.6f} D")
    print(f"  mu_z = {mu_pyscf[2]:+.6f} D")
    print(f"  |mu| = {np.linalg.norm(mu_pyscf):.3f} D")

    # Verify consistency
    print("\n" + "=" * 60)
    print("Validation:")
    print(f"  Manual vs PySCF difference: {np.max(np.abs(mu_debye - mu_pyscf)):.2e} D")
    print(f"  Experimental value: 1.855 D")
    print(f"  HF overestimates by: {(mu_total / 1.855 - 1) * 100:.1f}%")
    print("  (Typical for HF - electron correlation reduces dipole)")


if __name__ == "__main__":
    main()
