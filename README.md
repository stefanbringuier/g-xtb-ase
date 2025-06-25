# ASE Wrapper for g-xTB [![Example Runs](https://github.com/stefanbringuier/g-xtb-ase/actions/workflows/test.yml/badge.svg)](https://github.com/stefanbringuier/g-xtb-ase/actions/workflows/test.yml)

ASE calculator wrapper for the development version of [g-xTB](https://github.com/grimme-lab/g-xtb).

## Installation

```bash
git clone --recursive https://github.com/stefanbringuier/g-xtb-ase
cd g-xtb-ase
pip install .
```

> [!NOTE]
> Direct `pip install git+...` is **not supported** because pip doesn't handle git submodules. You must clone with `--recursive` to get the required g-xTB binary and parameter files.

## Usage

```python
from ase.build import molecule
from gxtb_ase import GxTB

atoms = molecule("H2O")
atoms.calc = GxTB(charge=0)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
``` 

## Attribution

S Bringuier, ASE Wrapper for g-xTB, 2025. https://github.com/stefanbringuier/g-xtb-ase

## Reference

- T. Froitzheim, M. Müller, A. Hansen, S. Grimme, g-xTB: A General-Purpose Extended Tight-Binding Electronic Structure Method For the Elements H to Lr (Z=1–103), (2025). https://doi.org/10.26434/chemrxiv-2025-bjxvt.
