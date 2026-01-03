# Code Examples for Advanced Quantum Chemistry

Standalone Python implementations accompanying the lecture notes for **2302638 Advanced Quantum Chemistry**, Chulalongkorn University.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
code/
├── ch01/                # Chapter 1: Electron-Integral View of QC
│   └── lab1a_integrals.py
│
├── appendix_b/          # Atomic Units and Conversions
│   ├── geometry_units.py
│   ├── unit_conversions.py
│   ├── unit_error_detection.py
│   └── dipole_calculation.py
│
├── appendix_c/          # Gaussian Integral Implementations
│   ├── gaussian_product_theorem.py
│   ├── overlap_integral.py
│   ├── kinetic_integral.py
│   ├── boys_function.py
│   ├── nuclear_attraction.py
│   ├── two_electron_integral.py
│   └── hf_energy_validation.py
│
├── appendix_d/          # Boys Functions: Stable Evaluation
│   ├── boys_function.py
│   ├── boys_validation.py
│   └── boys_stability.py
│
├── requirements.txt
└── README.md
```

## Chapter 1: Electron-Integral View of Quantum Chemistry

| File | Description |
|------|-------------|
| `lab1a_integrals.py` | AO integral extraction, symmetry verification, RHF energy validation |

**Lab 1A covers:**
- Building molecules in PySCF
- Extracting one-electron integrals (S, T, V, h)
- Extracting two-electron integrals (ERIs) with symmetry options
- Verifying 8-fold ERI symmetry
- Running RHF and validating electron count via Tr[PS]
- Reconstructing energy from integrals (Algorithm 1.1)
- Memory analysis for integral storage

**Run:**
```bash
cd ch01
python lab1a_integrals.py
```

## Appendix B: Atomic Units and Conversions

| File | Description |
|------|-------------|
| `geometry_units.py` | Demonstrates unit specification in PySCF |
| `unit_conversions.py` | CODATA conversion factors and utilities |
| `unit_error_detection.py` | Common unit mistakes and sanity checks |
| `dipole_calculation.py` | Dipole moment calculation with unit conversion |

**Run all:**
```bash
cd appendix_b
python geometry_units.py
python unit_conversions.py
python unit_error_detection.py
python dipole_calculation.py
```

## Appendix C: Gaussian Integral Formula Sheet

| File | Description |
|------|-------------|
| `gaussian_product_theorem.py` | GPT implementation and verification |
| `overlap_integral.py` | s-s overlap integral |
| `kinetic_integral.py` | s-s kinetic energy integral |
| `boys_function.py` | Boys function with series/recurrence |
| `nuclear_attraction.py` | Nuclear attraction integral using Boys function |
| `two_electron_integral.py` | (ss|ss) ERI implementation |
| `hf_energy_validation.py` | Complete HF energy from integrals |

**Run all:**
```bash
cd appendix_c
python gaussian_product_theorem.py
python overlap_integral.py
python kinetic_integral.py
python boys_function.py
python nuclear_attraction.py
python two_electron_integral.py
python hf_energy_validation.py
```

## Appendix D: Boys Functions - Stable Evaluation

| File | Description |
|------|-------------|
| `boys_function.py` | Core implementation with hybrid evaluation strategy |
| `boys_validation.py` | Validation against scipy.special.gammainc |
| `boys_stability.py` | Demonstration of stability issues and solutions |

**Run all:**
```bash
cd appendix_d
python boys_function.py
python boys_validation.py
python boys_stability.py
```

## Key Concepts

### Gaussian Product Theorem (GPT)
The product of two Gaussians centered at A and B is a Gaussian at composite center P:
```
exp(-α|r-A|²) × exp(-β|r-B|²) = exp(-p|r-P|²) × exp(-μR²_AB)
```
where `p = α+β`, `μ = αβ/(α+β)`, `P = (αA+βB)/(α+β)`

### Boys Function
```
F_n(T) = ∫₀¹ t^(2n) exp(-Tt²) dt
```
Essential for nuclear attraction and electron repulsion integrals.

### Hartree-Fock Energy
```
E_HF = ½Tr[P(h+F)] + E_nuc
```
where `P` is the density matrix, `h = T + V` is the core Hamiltonian, and `F = h + J - ½K`.

## Validation

All implementations are validated against PySCF reference calculations. Run any script to see the comparison output.

## License

Educational material for 2302638 Advanced Quantum Chemistry.
