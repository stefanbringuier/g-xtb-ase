# ASE Wrapper for g-xTB [![Example Runs](https://github.com/stefanbringuier/g-xtb-ase/actions/workflows/test.yml/badge.svg)](https://github.com/stefanbringuier/g-xtb-ase/actions/workflows/test.yml)

ASE calculators for [g-xTB](https://github.com/grimme-lab/g-xtb). `GxTB` uses the bundled legacy backend. `GxTB2` opts in to a separately installed g-xTB 2.x executable.

## Legacy backend: `GxTB`

```bash
git clone --recursive https://github.com/stefanbringuier/g-xtb-ase
cd g-xtb-ase
pip install .
```

> [!NOTE]
> For `GxTB`, clone with `--recursive`. An install without the submodule does not include the legacy executable and parameter files.

### Usage

```python
from ase.build import molecule
from gxtb_ase import GxTB

atoms = molecule("H2O")
atoms.calc = GxTB(charge=0)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## External backend: `GxTB2`

Get the modified `xtb` executable from the [upstream g-xTB 2.0.1 release](https://github.com/grimme-lab/g-xtb/releases/tag/v2.0.1), and verify the archive using the published SHA-256 before extracting it. An ordinary xTB executable without `--gxtb` support is not sufficient. The wrapper does not download or bundle the 2.x executable, and this path does not need the legacy parameter files or submodule.

```bash
git clone https://github.com/stefanbringuier/g-xtb-ase
cd g-xtb-ase
python -m pip install .
```

```python
from ase.build import molecule
from gxtb_ase import GxTB2

atoms = molecule("H2O")
atoms.calc = GxTB2("/path/to/xtb", charge=0)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
```

The 2.x backend supports energy and analytical forces for nonperiodic molecules only. It was tested with the Linux x86-64 g-xTB 2.0.1 release. Charges, dipole, stress, Hessian and legacy options are not exposed by `GxTB2`.

Set the total molecular `charge` as an integer. Set `uhf` to the number of unpaired electrons to request an unrestricted calculation (`uhf=0` also does this). An explicit count must have the same parity as the molecular electron count and cannot exceed it. The wrapper rejects results if the reported spin count differs from the request. Leave `uhf` unset to use xTB's default. ASE per-atom initial charges and magnetic moments are not used as xTB inputs.

Each calculation uses a fresh temporary directory and does not reuse restart files or home-directory xTB inputs. xTB warnings are treated as errors. Set `timeout=` in seconds to override the 120-second limit.

### Verify the 2.x backend

```bash
python -m unittest discover -s tests
# Optional offline comparison, using a separately verified binary:
GXTB2_TEST_EXE=/path/to/xtb OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests
```

The optional tests compare ASE energy and every force component against direct xTB output for neutral water (default and explicit `uhf=0`), its cation (`uhf=1`), H, and H₂. A GFN2 control checks that its output cannot be mistaken for g-xTB. The tests do not run optimization, MD, or spectroscopy.

## Attribution

S Bringuier, ASE Wrapper for g-xTB, 2025. https://github.com/stefanbringuier/g-xtb-ase

## Reference

- T. Froitzheim, M. Müller, A. Hansen, S. Grimme, g-xTB: A General-Purpose Extended Tight-Binding Electronic Structure Method For the Elements H to Lr (Z=1–103), (2025). https://doi.org/10.26434/chemrxiv-2025-bjxvt.
