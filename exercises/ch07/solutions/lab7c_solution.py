#!/usr/bin/env python3
"""
Lab 7C Solution: Virial Ratio Diagnostic
=========================================

This script demonstrates the virial theorem as a diagnostic tool for
basis set quality and numerical health of Hartree-Fock calculations.

Learning objectives:
1. Understand the virial theorem: 2<T> + <V> = 0 (at equilibrium)
2. Compute virial ratio eta = -<V>/<T> and interpret deviations from 2
3. Assess basis set quality through virial ratio
4. Explore geometry dependence of the virial theorem
5. Perform energy component analysis

Key equation:
    For exact eigenstates at equilibrium geometry:
        2<T> + <V> = 0  =>  eta = -<V>/<T> = 2

Test molecules: H2O, H2, HF, Ne

Part of: Advanced Quantum Chemistry Lecture Notes
Chapter 7: Scaling and Properties
"""

import numpy as np
from pyscf import gto, scf

# =============================================================================
# Section 1: Core Functions for Energy Component Analysis
# =============================================================================


def compute_energy_components(mol: gto.Mole, mf: scf.hf.SCF) -> dict:
    """
    Compute individual energy components from a converged HF calculation.

    Energy decomposition:
        E_total = <T> + <V_en> + <V_ee> + V_nn

    where:
        <T>     = Tr[P * T]           (kinetic energy)
        <V_en>  = Tr[P * V]           (electron-nuclear attraction)
        <V_ee>  = 1/2 * Tr[P * G]     (electron-electron repulsion)
        V_nn    = Sum_{A>B} Z_A Z_B / R_AB  (nuclear repulsion)

    Args:
        mol: PySCF molecule object
        mf: Converged SCF object

    Returns:
        Dictionary with energy components and virial ratio
    """
    dm = mf.make_rdm1()

    # =========================================================================
    # One-electron integrals
    # =========================================================================
    T = mol.intor('int1e_kin')    # Kinetic energy integrals
    V = mol.intor('int1e_nuc')    # Nuclear attraction integrals

    # =========================================================================
    # Two-electron contributions via J and K matrices
    # =========================================================================
    # J_mu_nu = Sum_la_si (mu nu | la si) P_la_si  (Coulomb)
    # K_mu_nu = Sum_la_si (mu la | nu si) P_la_si  (Exchange)
    J, K = mf.get_jk(mol, dm)

    # =========================================================================
    # Energy components
    # =========================================================================
    # Kinetic energy: <T> = Tr[P * T]
    E_T = np.einsum('ij,ji->', dm, T)

    # Electron-nuclear attraction: <V_en> = Tr[P * V]
    E_Ven = np.einsum('ij,ji->', dm, V)

    # Electron-electron repulsion: <V_ee> = 1/2 * Tr[P * (J - 1/2 K)]
    # The factor of 1/2 avoids double counting of electron pairs
    G = J - 0.5 * K
    E_ee = 0.5 * np.einsum('ij,ji->', dm, G)

    # Nuclear repulsion: classical point-charge formula
    E_nn = mol.energy_nuc()

    # =========================================================================
    # Total energies and virial ratio
    # =========================================================================
    # Total potential energy: all non-kinetic terms
    E_V_total = E_Ven + E_ee + E_nn

    # Total energy (should match mf.e_tot)
    E_total = E_T + E_V_total

    # Virial ratio: eta = -<V>/<T>
    # At equilibrium with complete basis: eta = 2
    eta = -E_V_total / E_T if abs(E_T) > 1e-12 else np.nan

    return {
        'E_T': E_T,
        'E_Ven': E_Ven,
        'E_ee': E_ee,
        'E_nn': E_nn,
        'E_V_total': E_V_total,
        'E_total': E_total,
        'eta': eta,
        'eta_deviation': abs(eta - 2),
    }


def compute_virial_ratio(mol: gto.Mole, mf: scf.hf.SCF) -> float:
    """
    Compute just the virial ratio eta = -<V>/<T>.

    Args:
        mol: PySCF molecule object
        mf: Converged SCF object

    Returns:
        Virial ratio eta
    """
    result = compute_energy_components(mol, mf)
    return result['eta']


# =============================================================================
# Section 2: Main Demonstration - Basis Set Quality Assessment
# =============================================================================


def demo_virial_basis_dependence() -> None:
    """
    Demonstrate how the virial ratio improves with basis set quality.

    The virial theorem (eta = 2) holds exactly for:
    1. Exact eigenstates of Coulombic Hamiltonians
    2. At equilibrium geometry (stationary point of energy)

    For finite basis HF, deviations from eta = 2 indicate:
    - Basis set incompleteness
    - Possibly non-equilibrium geometry
    """
    print()
    print("*" * 75)
    print("*" + " " * 73 + "*")
    print("*   Lab 7C: Virial Ratio Diagnostic" + " " * 37 + "*")
    print("*" + " " * 73 + "*")
    print("*" * 75)
    print()

    # Water molecule at near-equilibrium geometry
    h2o_geometry = """
    O    0.000000    0.000000    0.117369
    H    0.756950    0.000000   -0.469476
    H   -0.756950    0.000000   -0.469476
    """

    print("Test Molecule: H2O (near-equilibrium geometry)")
    print("-" * 55)

    # Basis sets: from minimal to very large
    basis_sets = ['STO-3G', '6-31G', 'cc-pVDZ', 'cc-pVTZ', 'cc-pVQZ', 'aug-cc-pVTZ']

    print()
    print("=" * 80)
    print("Virial Ratio vs Basis Set Quality")
    print("=" * 80)
    print()
    print("The virial theorem: 2<T> + <V> = 0  =>  eta = -<V>/<T> = 2")
    print()
    print(f"{'Basis':<14} {'N_AO':>6} {'<T> (Eh)':>14} {'<V> (Eh)':>16} "
          f"{'eta':>10} {'|eta-2|':>12}")
    print("-" * 80)

    results = []
    for basis in basis_sets:
        mol = gto.M(atom=h2o_geometry, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        if not mf.converged:
            print(f"{basis:<14} {'DID NOT CONVERGE':>60}")
            continue

        result = compute_energy_components(mol, mf)
        result['basis'] = basis
        result['nao'] = mol.nao
        results.append(result)

        # Quality indicator
        dev = result['eta_deviation']
        if dev < 1e-3:
            quality = "excellent"
        elif dev < 1e-2:
            quality = "good"
        elif dev < 5e-2:
            quality = "acceptable"
        else:
            quality = "poor"

        print(f"{basis:<14} {mol.nao:>6} {result['E_T']:>14.6f} "
              f"{result['E_V_total']:>16.6f} {result['eta']:>10.6f} "
              f"{result['eta_deviation']:>10.2e} ({quality})")

    print("-" * 80)

    # Interpretation
    print()
    print("=" * 75)
    print("Interpretation")
    print("=" * 75)
    print("""
Quality Guidelines:
  |eta - 2| < 10^-3  : Excellent - basis is nearly complete for this property
  |eta - 2| < 10^-2  : Good - acceptable for most purposes
  |eta - 2| < 5*10^-2: Acceptable - some basis incompleteness
  |eta - 2| > 10^-1  : Poor - significant basis set error

Observations:
  - STO-3G: Minimal basis, poor virial ratio
  - Split-valence (6-31G): Better but still limited
  - Correlation-consistent: Systematic improvement with cardinal number
  - Augmented: Diffuse functions help for properties

Note: Even with perfect basis, eta != 2 if geometry is not equilibrium.
""")


# =============================================================================
# Section 3: Geometry Dependence of Virial Theorem
# =============================================================================


def demo_virial_geometry_dependence() -> None:
    """
    Demonstrate how the virial ratio varies along a bond stretch.

    The general virial theorem includes a force correction:
        2<T> + <V> = -Sum_A R_A dot F_A

    At equilibrium, F_A = 0 for all atoms, so 2<T> + <V> = 0.
    Away from equilibrium, the RHS is nonzero.
    """
    print()
    print("=" * 75)
    print("Virial Ratio vs Bond Length (H2)")
    print("=" * 75)

    basis = 'cc-pVTZ'

    # Bond lengths from compressed to stretched
    # H2 equilibrium: ~0.74 Angstrom
    bond_lengths = [0.50, 0.60, 0.70, 0.74, 0.80, 0.90, 1.00, 1.20, 1.50, 2.00]

    print()
    print(f"Basis: {basis}")
    print(f"H2 equilibrium: ~0.74 Angstrom")
    print()
    print(f"{'R (Ang)':>8} {'E_total (Eh)':>16} {'<T> (Eh)':>14} "
          f"{'eta':>10} {'|eta-2|':>12}")
    print("-" * 65)

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

    print("-" * 65)

    # Find where eta is closest to 2 (should be near equilibrium)
    min_idx = np.argmin([r['eta_deviation'] for r in results])
    R_best = results[min_idx]['R']
    eta_best = results[min_idx]['eta']

    # Find minimum energy (should be at equilibrium)
    E_values = [r['E_total'] for r in results]
    min_E_idx = np.argmin(E_values)
    R_min_E = results[min_E_idx]['R']

    print()
    print("Analysis:")
    print(f"  eta closest to 2 at R = {R_best:.2f} Angstrom (eta = {eta_best:.6f})")
    print(f"  Minimum energy at R = {R_min_E:.2f} Angstrom")
    print(f"  H2 experimental R_e = 0.741 Angstrom")

    print()
    print("=" * 65)
    print("Physical Explanation")
    print("=" * 65)
    print("""
The general virial theorem (including nuclear motion) is:

    2<T> + <V> = -Sum_A R_A dot F_A

where F_A = -nabla_A E is the force on nucleus A.

At equilibrium geometry:
  - F_A = 0 for all atoms (forces are zero at minimum)
  - Therefore 2<T> + <V> = 0, giving eta = 2

Away from equilibrium:
  - F_A != 0 (atoms experience forces)
  - 2<T> + <V> != 0, so eta != 2
  - The deviation indicates how far from equilibrium

This makes the virial ratio useful for:
  1. Checking if geometry is optimized
  2. Diagnosing basis set issues at known equilibria
""")


# =============================================================================
# Section 4: Energy Component Breakdown
# =============================================================================


def demo_energy_breakdown() -> None:
    """
    Show detailed energy component breakdown for a molecule.
    """
    print()
    print("=" * 75)
    print("Energy Component Breakdown: HF Molecule")
    print("=" * 75)

    mol = gto.M(
        atom='H 0 0 0; F 0 0 0.92',
        basis='cc-pVTZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    result = compute_energy_components(mol, mf)

    print()
    print(f"Molecule: HF (hydrogen fluoride)")
    print(f"Basis: cc-pVTZ (N_AO = {mol.nao})")
    print(f"Bond length: 0.92 Angstrom (near equilibrium)")
    print()
    print("-" * 55)
    print("Energy Components")
    print("-" * 55)
    print(f"  Kinetic energy <T>:             {result['E_T']:>16.10f} Eh")
    print(f"  Electron-nuclear <V_en>:        {result['E_Ven']:>16.10f} Eh")
    print(f"  Electron-electron <V_ee>:       {result['E_ee']:>16.10f} Eh")
    print(f"  Nuclear repulsion V_nn:         {result['E_nn']:>16.10f} Eh")
    print("-" * 55)
    print(f"  Total potential <V>:            {result['E_V_total']:>16.10f} Eh")
    print(f"  Total energy E = <T> + <V>:     {result['E_total']:>16.10f} Eh")
    print("-" * 55)
    print()
    print(f"  Virial ratio eta = -<V>/<T>:    {result['eta']:>16.6f}")
    print(f"  Deviation |eta - 2|:            {result['eta_deviation']:>16.2e}")

    # Verify energy reconstruction
    e_diff = abs(result['E_total'] - mf.e_tot)
    print()
    print(f"  PySCF total energy:             {mf.e_tot:>16.10f} Eh")
    print(f"  Energy reconstruction error:    {e_diff:>16.2e} Eh")

    if e_diff < 1e-10:
        print()
        print("[PASS] Energy components verified!")
    else:
        print()
        print("[FAIL] Energy reconstruction failed!")

    # Physical interpretation
    print()
    print("=" * 55)
    print("Physical Interpretation")
    print("=" * 55)
    print("""
Energy component relationships:
  - <T> > 0 always (kinetic energy is positive definite)
  - <V_en> < 0 always (electron-nuclear attraction is stabilizing)
  - <V_ee> > 0 always (electron-electron repulsion is destabilizing)
  - V_nn > 0 always (nuclear repulsion is destabilizing)

The total potential <V> = <V_en> + <V_ee> + V_nn is typically negative
because the electron-nuclear attraction dominates.

For stable bound states:
  - E_total < 0 (below dissociation threshold)
  - <V> < 0 (attractive interactions win)
  - eta ~ 2 at equilibrium
""")


# =============================================================================
# Section 5: Comparison Across Molecules
# =============================================================================


def demo_virial_molecule_comparison() -> None:
    """
    Compare virial ratios for molecules with different bonding character.
    """
    print()
    print("=" * 75)
    print("Virial Ratio Comparison: Different Molecules")
    print("=" * 75)

    molecules = [
        ('He', 'He 0 0 0', 'Noble gas (atom)'),
        ('H2', 'H 0 0 0; H 0 0 0.74', 'Covalent'),
        ('HF', 'H 0 0 0; F 0 0 0.92', 'Polar covalent'),
        ('LiH', 'Li 0 0 0; H 0 0 1.60', 'Ionic'),
        ('F2', 'F 0 0 0; F 0 0 1.42', 'Covalent (weak)'),
    ]

    basis = 'cc-pVTZ'

    print()
    print(f"Basis: {basis} (all at approximate equilibrium geometries)")
    print()
    print(f"{'System':<8} {'Type':<18} {'E_tot (Eh)':>14} {'eta':>10} "
          f"{'|eta-2|':>12} {'<T> (Eh)':>12}")
    print("-" * 80)

    for name, atoms, mol_type in molecules:
        mol = gto.M(atom=atoms, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        result = compute_energy_components(mol, mf)

        print(f"{name:<8} {mol_type:<18} {mf.e_tot:>14.6f} "
              f"{result['eta']:>10.6f} {result['eta_deviation']:>12.2e} "
              f"{result['E_T']:>12.6f}")

    print("-" * 80)
    print()
    print("Notes:")
    print("  - Atoms (He) satisfy virial exactly with any complete basis")
    print("  - Diatomics: virial quality depends on geometry accuracy")
    print("  - All show |eta - 2| ~ 10^-3 with cc-pVTZ at equilibrium")


# =============================================================================
# Section 6: Practical Use Case - Geometry Check
# =============================================================================


def demo_geometry_check() -> None:
    """
    Use virial ratio to check if a geometry is near equilibrium.
    """
    print()
    print("=" * 75)
    print("Practical Use: Geometry Quality Check")
    print("=" * 75)

    # Two water geometries: equilibrium and distorted
    geometries = {
        'Equilibrium': '''
            O    0.000000    0.000000    0.117369
            H    0.756950    0.000000   -0.469476
            H   -0.756950    0.000000   -0.469476
        ''',
        'Compressed O-H': '''
            O    0.000000    0.000000    0.100000
            H    0.650000    0.000000   -0.400000
            H   -0.650000    0.000000   -0.400000
        ''',
        'Stretched O-H': '''
            O    0.000000    0.000000    0.140000
            H    0.900000    0.000000   -0.560000
            H   -0.900000    0.000000   -0.560000
        ''',
    }

    basis = 'cc-pVTZ'

    print()
    print(f"Molecule: H2O with various geometries")
    print(f"Basis: {basis}")
    print()
    print(f"{'Geometry':<20} {'E_tot (Eh)':>16} {'eta':>10} "
          f"{'|eta-2|':>12} {'Status':>12}")
    print("-" * 75)

    for name, geom in geometries.items():
        mol = gto.M(atom=geom, basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        result = compute_energy_components(mol, mf)

        # Quality assessment
        dev = result['eta_deviation']
        if dev < 2e-3:
            status = "EQUILIBRIUM"
        elif dev < 5e-2:
            status = "near-eq."
        else:
            status = "distorted"

        print(f"{name:<20} {mf.e_tot:>16.10f} {result['eta']:>10.6f} "
              f"{dev:>12.2e} {status:>12}")

    print("-" * 75)
    print()
    print("The virial ratio can serve as a quick diagnostic for geometry quality.")
    print("Large |eta - 2| suggests the geometry is far from equilibrium.")


# =============================================================================
# Section 7: Validation
# =============================================================================


def validate_virial_calculation() -> None:
    """
    Validate energy component calculation against PySCF.
    """
    print()
    print("=" * 75)
    print("Validation: Energy Components")
    print("=" * 75)

    mol = gto.M(
        atom='O 0 0 0; H 0.7570 0 0.5043; H -0.7570 0 0.5043',
        basis='cc-pVDZ',
        unit='Angstrom',
        verbose=0
    )

    mf = scf.RHF(mol)
    mf.kernel()

    result = compute_energy_components(mol, mf)

    print()
    print("Validation checks:")
    all_passed = True

    # Check 1: Energy reconstruction
    e_diff = abs(result['E_total'] - mf.e_tot)
    if e_diff < 1e-10:
        print(f"[PASS] Energy reconstruction: error = {e_diff:.2e} Eh")
    else:
        print(f"[FAIL] Energy reconstruction: error = {e_diff:.2e} Eh")
        all_passed = False

    # Check 2: Virial ratio in reasonable range
    if 1.8 < result['eta'] < 2.2:
        print(f"[PASS] Virial ratio eta = {result['eta']:.6f} in expected range")
    else:
        print(f"[FAIL] Virial ratio eta = {result['eta']:.6f} out of range")
        all_passed = False

    # Check 3: Sign checks
    signs_ok = True
    if result['E_T'] <= 0:
        print(f"[FAIL] Kinetic energy should be positive")
        signs_ok = False
    if result['E_Ven'] >= 0:
        print(f"[FAIL] Electron-nuclear attraction should be negative")
        signs_ok = False
    if result['E_ee'] <= 0:
        print(f"[FAIL] Electron-electron repulsion should be positive")
        signs_ok = False
    if result['E_nn'] <= 0:
        print(f"[FAIL] Nuclear repulsion should be positive")
        signs_ok = False

    if signs_ok:
        print(f"[PASS] All energy component signs correct")
    else:
        all_passed = False

    # Check 4: Potential energy is negative (stable molecule)
    if result['E_V_total'] < 0:
        print(f"[PASS] Total potential energy is negative (stable)")
    else:
        print(f"[FAIL] Total potential energy should be negative")
        all_passed = False

    print()
    if all_passed:
        print("=" * 50)
        print("All validation checks passed!")
        print("=" * 50)


# =============================================================================
# Section 8: What You Should Observe
# =============================================================================


def print_observations() -> None:
    """Print summary of key observations from this lab."""

    observations = """
================================================================================
What You Should Observe (Lab 7C)
================================================================================

1. THE VIRIAL THEOREM:
   2<T> + <V> = 0  =>  eta = -<V>/<T> = 2

   This holds exactly for:
     - Exact eigenstates of Coulombic Hamiltonians
     - At stationary points (equilibrium geometry)

2. BASIS SET DEPENDENCE:
   |eta - 2| decreases as basis becomes more complete:
     - STO-3G:     |eta - 2| ~ 10^-2 (poor)
     - cc-pVDZ:    |eta - 2| ~ 10^-3 (acceptable)
     - cc-pVTZ:    |eta - 2| ~ 10^-4 (good)
     - cc-pVQZ:    |eta - 2| ~ 10^-5 (excellent)

3. GEOMETRY DEPENDENCE:
   - eta = 2 only at equilibrium (where forces = 0)
   - Away from equilibrium: |eta - 2| increases
   - The deviation measures "distance" from equilibrium

4. ENERGY COMPONENTS:
   - <T> > 0 always (kinetic is positive)
   - <V_en> < 0 (electron-nuclear attraction)
   - <V_ee> > 0 (electron-electron repulsion)
   - V_nn > 0 (nuclear repulsion)
   - For stable molecules: E_total < 0

5. PRACTICAL APPLICATIONS:
   - Quick check for geometry quality
   - Basis set completeness diagnostic
   - Sanity check for numerical stability
   - Research tool for studying basis set convergence

6. LIMITATIONS:
   - Finite basis always gives eta != 2 exactly
   - Correlation methods may have different virial behavior
   - Not a substitute for proper geometry optimization

================================================================================
"""
    print(observations)


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Run the complete Lab 7C demonstration."""

    # Core demonstration
    demo_virial_basis_dependence()

    # Geometry dependence
    demo_virial_geometry_dependence()

    # Energy breakdown
    demo_energy_breakdown()

    # Molecule comparison
    demo_virial_molecule_comparison()

    # Practical application
    demo_geometry_check()

    # Validation
    validate_virial_calculation()

    # Summary
    print_observations()

    print()
    print("=" * 75)
    print("Lab 7C Complete")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
