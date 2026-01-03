#!/usr/bin/env python3
"""
Lab 3C: Dipole Integrals and Dipole Moment from Density

Compute the molecular dipole moment using one-electron operator integrals
contracted with the density matrix:

    mu = sum_A Z_A * (R_A - O) - Tr[P * r(O)]

where r(O) are the position integrals relative to origin O.

This lab demonstrates property calculation from:
- Density matrix P
- One-electron integrals <mu|r|nu>

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 3: One-Electron Integrals and Gaussian Product Theorem
"""

import numpy as np
from pyscf import gto, scf

# Conversion factor: atomic units to Debye
# 1 e*a0 = 2.541746 Debye
AU_TO_DEBYE = 2.541746


def compute_dipole(mol, dm, origin=None):
    """
    Compute molecular dipole moment from density matrix and integrals.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object
    dm : np.ndarray
        Density matrix in AO basis
    origin : np.ndarray, optional
        Origin for dipole calculation (default: [0, 0, 0])

    Returns
    -------
    dict
        Contains 'nuclear', 'electronic', 'total', and 'magnitude' (all in a.u.)
    """
    if origin is None:
        origin = np.zeros(3)

    origin = np.asarray(origin)

    # AO dipole integrals: <mu | r - origin | nu>, three components
    with mol.with_common_orig(origin):
        ao_r = mol.intor_symmetric("int1e_r", comp=3)  # shape (3, nao, nao)

    # Electronic contribution: Tr[P * r]
    # The minus sign comes from electron charge = -1
    el = np.einsum("xij,ji->x", ao_r, dm).real

    # Nuclear contribution: sum_A Z_A * (R_A - origin)
    charges = mol.atom_charges()
    coords = mol.atom_coords()  # in Bohr
    nucl = np.einsum("i,ix->x", charges, coords - origin[None, :])

    # Total dipole: nuclear - electronic
    total = nucl - el

    return {
        "nuclear": nucl,
        "electronic": el,
        "total": total,
        "magnitude": np.linalg.norm(total)
    }


def main():
    print("=" * 70)
    print("Lab 3C: Dipole Integrals and Dipole Moment from Density")
    print("=" * 70)

    # Build H2O molecule
    mol = gto.M(
        atom="O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043",
        basis="cc-pVDZ",
        unit="Angstrom",
        verbose=0
    )

    print(f"\nMolecule: H2O")
    print(f"Basis: cc-pVDZ")
    print(f"Number of AOs: {mol.nao_nr()}")
    print(f"Number of electrons: {mol.nelectron}")

    # Run RHF calculation
    mf = scf.RHF(mol)
    mf.verbose = 0
    E = mf.kernel()
    dm = mf.make_rdm1()

    print(f"\nRHF Energy: {E:.10f} Hartree")

    # Section 1: Dipole with origin at (0,0,0)
    print("\n" + "-" * 50)
    print("1. DIPOLE WITH ORIGIN AT (0, 0, 0)")
    print("-" * 50)

    origin1 = np.zeros(3)
    dipole1 = compute_dipole(mol, dm, origin1)

    print(f"\n   Origin: {origin1}")
    print(f"\n   Nuclear contribution (a.u.):")
    print(f"      x: {dipole1['nuclear'][0]:12.8f}")
    print(f"      y: {dipole1['nuclear'][1]:12.8f}")
    print(f"      z: {dipole1['nuclear'][2]:12.8f}")

    print(f"\n   Electronic contribution (a.u.):")
    print(f"      x: {dipole1['electronic'][0]:12.8f}")
    print(f"      y: {dipole1['electronic'][1]:12.8f}")
    print(f"      z: {dipole1['electronic'][2]:12.8f}")

    print(f"\n   Total dipole (a.u.):")
    print(f"      x: {dipole1['total'][0]:12.8f}")
    print(f"      y: {dipole1['total'][1]:12.8f}")
    print(f"      z: {dipole1['total'][2]:12.8f}")
    print(f"      |mu|: {dipole1['magnitude']:.8f} a.u.")
    print(f"            {dipole1['magnitude'] * AU_TO_DEBYE:.6f} Debye")

    # Section 2: Compare with PySCF built-in
    print("\n" + "-" * 50)
    print("2. COMPARISON WITH PYSCF BUILT-IN")
    print("-" * 50)

    pyscf_dipole = mf.dip_moment(verbose=0)  # Returns in Debye by default
    print(f"\n   PySCF dip_moment() (Debye):")
    print(f"      x: {pyscf_dipole[0]:12.8f}")
    print(f"      y: {pyscf_dipole[1]:12.8f}")
    print(f"      z: {pyscf_dipole[2]:12.8f}")
    print(f"      |mu|: {np.linalg.norm(pyscf_dipole):.6f} Debye")

    our_dipole_debye = dipole1["total"] * AU_TO_DEBYE
    diff = np.linalg.norm(our_dipole_debye - pyscf_dipole)
    print(f"\n   Difference from our calculation: {diff:.2e} Debye")
    print(f"   Match: {diff < 1e-6}")

    # Section 3: Origin at oxygen nucleus
    print("\n" + "-" * 50)
    print("3. DIPOLE WITH ORIGIN AT OXYGEN NUCLEUS")
    print("-" * 50)

    # Get oxygen coordinates (atom 0)
    origin2 = mol.atom_coord(0)  # In Bohr
    dipole2 = compute_dipole(mol, dm, origin2)

    print(f"\n   Origin: {origin2} Bohr")
    print(f"\n   Nuclear contribution (a.u.):")
    print(f"      x: {dipole2['nuclear'][0]:12.8f}")
    print(f"      y: {dipole2['nuclear'][1]:12.8f}")
    print(f"      z: {dipole2['nuclear'][2]:12.8f}")

    print(f"\n   Electronic contribution (a.u.):")
    print(f"      x: {dipole2['electronic'][0]:12.8f}")
    print(f"      y: {dipole2['electronic'][1]:12.8f}")
    print(f"      z: {dipole2['electronic'][2]:12.8f}")

    print(f"\n   Total dipole (a.u.):")
    print(f"      x: {dipole2['total'][0]:12.8f}")
    print(f"      y: {dipole2['total'][1]:12.8f}")
    print(f"      z: {dipole2['total'][2]:12.8f}")
    print(f"      |mu|: {dipole2['magnitude']:.8f} a.u.")
    print(f"            {dipole2['magnitude'] * AU_TO_DEBYE:.6f} Debye")

    # Section 4: Origin independence for neutral molecules
    print("\n" + "-" * 50)
    print("4. ORIGIN INDEPENDENCE FOR NEUTRAL MOLECULES")
    print("-" * 50)

    dipole_diff = np.linalg.norm(dipole1["total"] - dipole2["total"])
    print(f"\n   |mu(origin1) - mu(origin2)| = {dipole_diff:.2e} a.u.")
    print(f"\n   Origin independence: {dipole_diff < 1e-10}")

    print(f"""
   Physical explanation:
   - For neutral molecules, sum_A Z_A = N_electrons
   - Shifting origin by d changes nuclear term by: +Q*d (where Q = sum Z_A)
   - Shifting origin by d changes electronic term by: -(-Ne)*d = +Ne*d
   - For neutral: Q = Ne, so shifts cancel exactly
   - For ions (Q != Ne), dipole depends on origin choice
    """)

    # Section 5: Individual contributions change with origin
    print("-" * 50)
    print("5. INDIVIDUAL CONTRIBUTIONS CHANGE WITH ORIGIN")
    print("-" * 50)

    print("\n   Nuclear contribution:")
    print(f"      Origin 1: {dipole1['nuclear']}")
    print(f"      Origin 2: {dipole2['nuclear']}")
    print(f"      Difference: {dipole1['nuclear'] - dipole2['nuclear']}")

    print("\n   Electronic contribution:")
    print(f"      Origin 1: {dipole1['electronic']}")
    print(f"      Origin 2: {dipole2['electronic']}")
    print(f"      Difference: {dipole1['electronic'] - dipole2['electronic']}")

    # The differences should be equal (and cancel in total)
    nucl_shift = dipole1["nuclear"] - dipole2["nuclear"]
    elec_shift = dipole1["electronic"] - dipole2["electronic"]
    print(f"\n   Nuclear shift - Electronic shift:")
    print(f"      {nucl_shift - elec_shift}")
    print(f"      (Should be ~0 for neutral molecule)")

    # Section 6: Physical interpretation
    print("\n" + "-" * 50)
    print("6. PHYSICAL INTERPRETATION")
    print("-" * 50)

    print(f"""
   Water dipole moment:
   - Experimental: ~1.85 Debye
   - Our RHF/cc-pVDZ: {dipole1['magnitude'] * AU_TO_DEBYE:.2f} Debye

   The dipole points from negative (O) toward positive (H atoms) region.

   Formula: mu = sum_A Z_A * R_A - Tr[P * r]
            = nuclear contribution - electronic contribution

   This is a one-electron property:
   - Requires only density matrix P
   - Requires only one-electron integrals <mu|r|nu>
   - No two-electron integrals needed!

   Units:
   - 1 e*a0 = 2.541746 Debye
   - Typical molecular dipoles: 0-10 Debye
   - H2O: 1.85 D, HF: 1.83 D, NH3: 1.47 D
    """)

    print("\n" + "=" * 70)
    print("Lab 3C Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
