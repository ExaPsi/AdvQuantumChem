#!/usr/bin/env python3
"""
Verify installation for Advanced Quantum Chemistry course.

This script checks that all required packages are installed and
validates PySCF functionality by running a simple H2/STO-3G calculation.

Usage:
    python verify_setup.py
"""
import sys


def check_packages():
    """Check all required packages are installed."""
    packages = {
        'numpy': '1.20',
        'scipy': '1.7',
        'pyscf': '2.0',
    }

    all_ok = True
    for pkg, min_ver in packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"[OK] {pkg} {ver}")
        except ImportError:
            print(f"[MISSING] {pkg} (need >= {min_ver})")
            all_ok = False
    return all_ok


def test_pyscf_integrals():
    """Test that PySCF can compute integrals."""
    import numpy as np
    from pyscf import gto, scf

    # Build H2 molecule
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
                unit='Angstrom', verbose=0)

    # Extract integrals
    S = mol.intor('int1e_ovlp')
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    eri = mol.intor('int2e')

    # Verify core Hamiltonian can be formed
    h = T + V
    assert h.shape == S.shape, "Core Hamiltonian shape error"

    # Run RHF
    mf = scf.RHF(mol).run()
    dm = mf.make_rdm1()

    # Verify electron count
    n_elec = np.trace(dm @ S)
    assert abs(n_elec - 2.0) < 1e-10, "Electron count error"

    # Verify energy
    E_ref = -1.1168  # Approximate HF/STO-3G energy for H2
    assert abs(mf.e_tot - E_ref) < 0.01, "Energy error"

    print(f"[OK] PySCF integrals: S{S.shape}, h{h.shape}, ERI{eri.shape}")
    print(f"[OK] H2 RHF energy: {mf.e_tot:.6f} Eh")
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("Advanced Quantum Chemistry - Setup Verification")
    print("=" * 50)

    if not check_packages():
        print("\nInstall missing packages with:")
        print("  pip install numpy scipy pyscf")
        sys.exit(1)

    print("\nTesting PySCF functionality...")
    try:
        test_pyscf_integrals()
        print("\n" + "=" * 50)
        print("All checks passed! You are ready for the course.")
        print("=" * 50)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
