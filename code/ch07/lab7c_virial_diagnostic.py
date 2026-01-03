#!/usr/bin/env python3
"""
Lab 7C: Virial Theorem Diagnostic
=================================

This script demonstrates the virial theorem as a diagnostic for basis set quality
and numerical health of Hartree-Fock calculations.

The virial theorem states that for exact eigenstates of Coulombic Hamiltonians
at equilibrium geometry:
    2⟨T⟩ + ⟨V⟩ = 0

This implies the virial ratio:
    η = -⟨V⟩/⟨T⟩ = 2

For finite basis HF:
    - |η - 2| < 10^-2 indicates good basis quality at equilibrium
    - Larger deviations suggest basis incompleteness or non-equilibrium geometry
    - The virial theorem is exact only at stationary points of the energy

Key concepts from Chapter 7:
    - Virial theorem as basis quality diagnostic
    - η deviation interpretation
    - Relationship to geometry optimization

Reference: Section 7.5 (Virial Theorem as a Diagnostic)
"""

import numpy as np
from pyscf import gto, scf


def compute_energy_components(mol, mf):
    """
    Compute individual energy components from a converged HF calculation.

    Returns
    -------
    dict with keys: E_T, E_Ven, E_ee, E_nn, E_V_total, eta
    """
    dm = mf.make_rdm1()

    # One-electron integrals
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')

    # Build J and K matrices
    J, K = mf.get_jk(mol, dm)

    # Energy components
    E_T = np.einsum('ij,ji->', dm, T)           # Kinetic energy
    E_Ven = np.einsum('ij,ji->', dm, V)         # Electron-nuclear attraction
    E_ee = 0.5 * np.einsum('ij,ji->', dm, J - 0.5 * K)  # Electron-electron repulsion
    E_nn = mol.energy_nuc()                     # Nuclear repulsion

    # Total potential energy
    E_V_total = E_Ven + E_ee + E_nn

    # Virial ratio
    eta = -E_V_total / E_T if abs(E_T) > 1e-12 else np.nan

    return {
        'E_T': E_T,
        'E_Ven': E_Ven,
        'E_ee': E_ee,
        'E_nn': E_nn,
        'E_V_total': E_V_total,
        'E_total': E_T + E_V_total,
        'eta': eta,
        'eta_deviation': abs(eta - 2),
    }


def demo_virial_basis_dependence():
    """
    Demonstrate how the virial ratio improves with basis set quality.

    This corresponds to Lab 7C and Exercise 7.3 in the lecture notes.
    """
    print("=" * 70)
    print("Lab 7C: Virial Theorem Diagnostic")
    print("=" * 70)

    mol_template = '''
    O  0.0000  0.0000  0.0000
    H  0.7586  0.0000  0.5043
    H -0.7586  0.0000  0.5043
    '''

    basis_sets = ['STO-3G', '6-31G', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ']

    print("\nMolecule: H2O (equilibrium geometry)")
    print("-" * 70)
    print(f"{'Basis':<12} {'N_AO':>6} {'⟨T⟩ (Eh)':>14} {'⟨V⟩ (Eh)':>16} "
          f"{'η':>10} {'|η-2|':>12}")
    print("-" * 70)

    results = []
    for basis in basis_sets:
        mol = gto.M(atom=mol_template, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        if not mf.converged:
            print(f"{basis:<12} {'DID NOT CONVERGE':>50}")
            continue

        result = compute_energy_components(mol, mf)
        result['basis'] = basis
        result['nao'] = mol.nao
        results.append(result)

        print(f"{basis:<12} {mol.nao:>6} {result['E_T']:>14.6f} "
              f"{result['E_V_total']:>16.6f} {result['eta']:>10.6f} "
              f"{result['eta_deviation']:>12.2e}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print("""
The virial theorem 2⟨T⟩ + ⟨V⟩ = 0 (η = 2) holds exactly for:
  1. Exact eigenstates of Coulombic Hamiltonians
  2. At equilibrium geometry (energy stationary point)

For finite basis HF:
  - η approaches 2 as the basis becomes more complete
  - |η - 2| < 10^-2 is typically acceptable
  - |η - 2| > 10^-1 suggests serious basis issues

Note: Even with a complete basis, η ≠ 2 away from equilibrium geometry
because the energy is not stationary with respect to nuclear coordinates.
""")


def demo_virial_geometry_dependence():
    """
    Demonstrate how the virial ratio changes along a bond stretch.

    This corresponds to Exercise 7.10 (Research).
    """
    print("\n" + "=" * 70)
    print("Virial Ratio vs Bond Length (H2)")
    print("=" * 70)

    basis = 'cc-pVDZ'
    bond_lengths = np.linspace(0.5, 3.0, 11)  # Angstrom

    print(f"\nBasis: {basis}")
    print(f"Equilibrium bond length: ~0.74 Å")
    print("-" * 60)
    print(f"{'R (Å)':>8} {'E_total (Eh)':>16} {'⟨T⟩ (Eh)':>14} {'η':>10} {'|η-2|':>12}")
    print("-" * 60)

    results = []
    for R in bond_lengths:
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        result = compute_energy_components(mol, mf)
        result['R'] = R
        results.append(result)

        print(f"{R:>8.2f} {result['E_total']:>16.10f} {result['E_T']:>14.6f} "
              f"{result['eta']:>10.6f} {result['eta_deviation']:>12.2e}")

    # Find minimum deviation from 2
    min_idx = np.argmin([r['eta_deviation'] for r in results])
    R_best = results[min_idx]['R']
    eta_best = results[min_idx]['eta']

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print(f"η closest to 2 at R = {R_best:.2f} Å (η = {eta_best:.6f})")
    print(f"H2 equilibrium bond length: ~0.74 Å")
    print("""
The virial theorem 2⟨T⟩ + ⟨V⟩ = 0 holds at equilibrium geometry
because the energy is stationary: ∂E/∂R = 0.

The general virial theorem includes a correction term:
    2⟨T⟩ + ⟨V⟩ = -Σ_A R_A · F_A

where F_A is the force on nucleus A. At equilibrium, F_A = 0 for all A,
so the simple virial relation is recovered.
""")


def demo_energy_component_breakdown():
    """
    Show detailed energy component breakdown for a molecule.
    """
    print("\n" + "=" * 70)
    print("Energy Component Breakdown: HF Molecule")
    print("=" * 70)

    mol = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVTZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    result = compute_energy_components(mol, mf)

    print(f"\nBasis: cc-pVTZ (N_AO = {mol.nao})")
    print(f"RHF Energy: {mf.e_tot:.10f} Eh")
    print("\n" + "-" * 50)
    print("Energy Components:")
    print("-" * 50)
    print(f"  Kinetic energy (⟨T⟩):              {result['E_T']:>16.10f} Eh")
    print(f"  Electron-nuclear (⟨V_en⟩):         {result['E_Ven']:>16.10f} Eh")
    print(f"  Electron-electron (⟨V_ee⟩):        {result['E_ee']:>16.10f} Eh")
    print(f"  Nuclear repulsion (V_nn):          {result['E_nn']:>16.10f} Eh")
    print("-" * 50)
    print(f"  Total potential (⟨V⟩):             {result['E_V_total']:>16.10f} Eh")
    print(f"  Total energy (⟨T⟩ + ⟨V⟩):          {result['E_total']:>16.10f} Eh")
    print("-" * 50)
    print(f"\n  Virial ratio η = -⟨V⟩/⟨T⟩:         {result['eta']:>16.6f}")
    print(f"  Deviation |η - 2|:                 {result['eta_deviation']:>16.2e}")

    # Verify total energy
    e_diff = abs(result['E_total'] - mf.e_tot)
    print(f"\n  Energy reconstruction error:       {e_diff:>16.2e} Eh")
    assert e_diff < 1e-10, "Energy reconstruction failed!"
    print("\n[PASSED] Energy components verified!")


def demo_virial_different_molecules():
    """
    Compare virial ratios for molecules with different bonding character.
    """
    print("\n" + "=" * 70)
    print("Virial Ratio Comparison: Different Molecules")
    print("=" * 70)

    molecules = [
        ('H2', 'H 0 0 0; H 0 0 0.74', 'Covalent'),
        ('HF', 'H 0 0 0; F 0 0 0.92', 'Polar covalent'),
        ('LiH', 'Li 0 0 0; H 0 0 1.60', 'Ionic'),
        ('Ne', 'Ne 0 0 0', 'Atom'),
    ]

    basis = 'cc-pVTZ'

    print(f"\nBasis: {basis}")
    print("-" * 70)
    print(f"{'Molecule':<10} {'Type':<18} {'η':>12} {'|η-2|':>14} {'⟨T⟩ (Eh)':>14}")
    print("-" * 70)

    for name, atoms, mol_type in molecules:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        result = compute_energy_components(mol, mf)
        print(f"{name:<10} {mol_type:<18} {result['eta']:>12.6f} "
              f"{result['eta_deviation']:>14.2e} {result['E_T']:>14.6f}")


def validate_virial_calculation():
    """
    Validate energy component calculation against PySCF.
    """
    print("\n" + "=" * 70)
    print("Validation: Energy Components")
    print("=" * 70)

    mol = gto.M(
        atom='O 0 0 0; H 0.7586 0 0.5043; H -0.7586 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    result = compute_energy_components(mol, mf)

    # Check energy reconstruction
    e_diff = abs(result['E_total'] - mf.e_tot)
    print(f"Energy reconstruction error: {e_diff:.2e} Eh")
    assert e_diff < 1e-10, f"Energy reconstruction error: {e_diff}"

    # Check virial ratio is reasonable
    assert 1.5 < result['eta'] < 2.5, f"Virial ratio {result['eta']} out of expected range"

    # Check individual components have correct signs
    assert result['E_T'] > 0, "Kinetic energy should be positive"
    assert result['E_Ven'] < 0, "Electron-nuclear attraction should be negative"
    assert result['E_ee'] > 0, "Electron-electron repulsion should be positive"
    assert result['E_nn'] > 0, "Nuclear repulsion should be positive"

    print("[PASSED] All validations successful!")
    print(f"  - Energy reconstruction: OK")
    print(f"  - Virial ratio η = {result['eta']:.6f}: OK")
    print(f"  - Sign checks for all components: OK")


if __name__ == '__main__':
    demo_virial_basis_dependence()
    demo_virial_geometry_dependence()
    demo_energy_component_breakdown()
    demo_virial_different_molecules()
    validate_virial_calculation()
