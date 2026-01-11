# Exercise Answer Keys

This folder contains complete answer keys for all checkpoint questions, Python lab exercises, and end-of-chapter exercises from the Advanced Quantum Chemistry lecture notes (2302638).

## Folder Structure

```
exercises/
├── README.md                # This file
├── .gitignore               # Git ignore rules for LaTeX artifacts
├── solutions_style.sty      # Unified LaTeX style for all answer keys
├── ch01/solutions/          # Chapter 1: Electron-Integral View of Quantum Chemistry
│   ├── ch01_solutions.tex   #   LaTeX answer key source
│   ├── ch01_solutions.pdf   #   Compiled PDF
│   ├── exercises_ch01.py    #   Exercise Python implementations
│   └── lab1a_solution.py    #   Lab solution
├── ch02/solutions/          # Chapter 2: Gaussian Basis Sets and Orthonormalization
├── ch03/solutions/          # Chapter 3: One-Electron Integrals and GPT
├── ch04/solutions/          # Chapter 4: Two-Electron Integrals and Boys Function
├── ch05/solutions/          # Chapter 5: Rys Quadrature in Practice
├── ch06/solutions/          # Chapter 6: Hartree-Fock SCF from Integrals
└── ch07/solutions/          # Chapter 7: Scaling and Properties
```

Each chapter folder contains:
- `chXX_solutions.tex` - LaTeX answer key for checkpoint questions
- `chXX_solutions.pdf` - Compiled PDF of the answer key
- `exercises_chXX.py` - Python implementations for end-of-chapter exercises
- `labXX_solution.py` - Python solutions for hands-on labs (Lab A, B, C, etc.)

## Contents Summary

| Chapter | Checkpoints | Python Labs | Exercise Code | Key Topics |
|---------|-------------|-------------|---------------|------------|
| 1 | 3 | Lab 1A | exercises_ch01.py | AO integrals, ERI symmetry, electron count |
| 2 | 11 | Labs 2A-2C | exercises_ch02.py | Orthonormalization, conditioning, generalized eigenproblems |
| 3 | 11 | Labs 3A-3C | exercises_ch03.py | GPT, overlap/kinetic/nuclear integrals, dipole |
| 4 | 11 | Labs 4A-4C | exercises_ch04.py | Boys function, (ss|ss) ERI, quadrature |
| 5 | 12 | Labs 5A-5D | exercises_ch05.py | Rys quadrature, moment matching, J/K matrices |
| 6 | 14 | Labs 6A-6C | exercises_ch06.py | RHF SCF, DIIS, in-core vs direct |
| 7 | 12 | Labs 7A-7C | exercises_ch07.py | DF-HF, dipole moment, virial ratio |

## Style Package

All LaTeX answer keys use a shared style package for consistent formatting:

- `solutions_style.sty` - Unified styling with custom environments:
  - `solutionbox` - Blue boxes for answers
  - `warningbox` - Red boxes for common errors
  - `keyformulabox` - Green boxes for key formulas
  - `outputbox` - Gray boxes for expected program output
  - `tipbox` / `keyInsight` - Gold boxes for hints and insights

## Running the Python Solutions

All Python solutions require PySCF and NumPy. Activate the project virtual environment before running:

```bash
# From the repository root
source .venv/bin/activate

# Run individual lab solutions
python exercises/ch01/solutions/lab1a_solution.py
python exercises/ch06/solutions/lab6a_solution.py

# Run exercise implementations
python exercises/ch02/solutions/exercises_ch02.py
python exercises/ch05/solutions/exercises_ch05.py
```

## Exercise Python Implementations

In addition to lab solutions, each chapter includes Python implementations for end-of-chapter exercises that require numerical computation:

| File | Exercises | Description |
|------|-----------|-------------|
| `exercises_ch01.py` | 9 | Integral verification, ERI symmetry, density matrix |
| `exercises_ch02.py` | 7 | Orthogonalization methods, eigenvalue thresholds |
| `exercises_ch03.py` | 11 | GPT verification, overlap decay, kinetic integrals |
| `exercises_ch04.py` | 11 | Boys function stability, ERI formulas, Schwarz screening |
| `exercises_ch05.py` | 12 | Rys quadrature, moment matching, J/K build |
| `exercises_ch06.py` | 11 | RHF energy, DIIS behavior, SCF convergence |
| `exercises_ch07.py` | 12 | Scaling analysis, DF/RI concepts, virial theorem |

Each file is self-contained with PySCF validation.

## Key Formulas Verified in These Solutions

The answer keys verify understanding of these core formulas:

### RHF Conventions
- **Density matrix**: `P_uv = 2 Sum_i C_ui C_vi` (factor of 2 for closed-shell)
- **Fock matrix**: `F = h + J - 1/2 K` (note 1/2 on exchange)
- **HF energy**: `E = 1/2 Tr[P(h+F)] + E_nuc`
- **Electron count**: `N_e = Tr[PS]` (not Tr[P])

### Integral Contractions (chemist's notation)
- **Coulomb**: `J_uv = Sum_ls (uv|ls) P_ls`
- **Exchange**: `K_uv = Sum_ls (ul|vs) P_ls`

### Boys Function
- **Definition**: `F_n(T) = integral_0^1 t^(2n) exp(-Tt^2) dt`
- **Special case**: `F_n(0) = 1/(2n+1)`

### Rys Quadrature
- **Moments**: `m_k = 2F_k(T)`
- **Root count**: `n_roots = floor(L/2) + 1` where `L = l_A + l_B + l_C + l_D`

## Validation

All Python solutions validate against PySCF reference calculations with the following tolerances:
- One-electron integrals: ~10^-15 (machine precision)
- (ss|ss) ERIs: ~10^-10 Hartree
- SCF energies: ~10^-8 Hartree
- DF-HF energies: ~10^-6 to 10^-5 Hartree

## For Instructors

### Pre-compiled PDFs

Each chapter includes a pre-compiled PDF answer key:
- `ch01/solutions/ch01_solutions.pdf` through `ch07/solutions/ch07_solutions.pdf`

### Compiling LaTeX Solutions

The LaTeX solution files require the shared style package. Compile with:

```bash
cd exercises/ch01/solutions
TEXINPUTS=../../: pdflatex ch01_solutions.tex

# Or compile all chapters from exercises root:
cd exercises
for ch in ch01 ch02 ch03 ch04 ch05 ch06 ch07; do
  (cd $ch/solutions && TEXINPUTS=../../: pdflatex ${ch}_solutions.tex)
done
```

The solutions include:
- Detailed derivations for checkpoint questions
- Expected numerical outputs for labs
- Common student errors and debugging tips
- Physical interpretations of results

## License

These materials are part of the 2302638 Advanced Quantum Chemistry course at Chulalongkorn University.
