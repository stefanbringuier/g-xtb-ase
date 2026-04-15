"""
Example: accessing electronic properties (HOMO-LUMO gap, IP, EA, spin) via atoms.info.

After a g-xTB calculation, non-standard electronic properties are attached to
atoms.info under gxtb_* keys.  Standard ASE properties (energy, forces, dipole,
charges) remain in calc.results as usual.

Run from the repo root:
    python examples/electronic_properties.py
"""

from pathlib import Path
from ase.io import read
from gxtb_ase import GxTB

SCRIPT_DIR = Path(__file__).resolve().parent

atoms = read(SCRIPT_DIR / "structures" / "Ag_min_35atoms.xyz")

# write_log=True ensures the log file is parsed so that electronic
# properties (gap, IP, EA, ...) are populated into atoms.info.
# The log is written to {directory}/{label}.log, i.e. examples/Ag35_rks.log here.
calc = GxTB(
    label="Ag35_rks",
    charge=0,
    spin=0,
    write_log=True,
    directory=str(SCRIPT_DIR),
)
atoms.calc = calc

energy = atoms.get_potential_energy()
print(f"Total energy : {energy:.6f} eV")

# Standard ASE properties live in calc.results
forces = atoms.get_forces()
print(f"Max force    : {abs(forces).max():.6f} eV/Å")

# Non-standard electronic properties are stored in atoms.info
print("\nElectronic properties (atoms.info):")
for key in sorted(k for k in atoms.info if k.startswith("gxtb_")):
    print(f"  {key}: {atoms.info[key]}")

# Access individual properties directly
gap = atoms.info.get("gxtb_gap_eV")
ip  = atoms.info.get("gxtb_IP_Janak_eV")
ea  = atoms.info.get("gxtb_EA_Janak_eV")
print(f"\nHOMO-LUMO gap : {gap} eV")
if ip is not None:
    print(f"IP (Janak)    : {ip} eV")
if ea is not None:
    print(f"EA (Janak)    : {ea} eV")

# --- UKS example: neutral doublet (charge=+1 gives 189 electrons, odd → spin=1) ---
print("\n--- UKS doublet (charge=+1, spin=1) ---")
atoms_uks = read(SCRIPT_DIR / "structures" / "Ag_min_35atoms.xyz")

# The log is written to examples/Ag35_uks.log
calc_uks = GxTB(
    label="Ag35_uks",
    charge=1,
    spin=1,
    write_log=False,
    directory=str(SCRIPT_DIR),
)
atoms_uks.calc = calc_uks

energy_uks = atoms_uks.get_potential_energy()
print(f"Total energy : {energy_uks:.6f} eV")

print("\nElectronic properties (atoms.info):")
for key in sorted(k for k in atoms_uks.info if k.startswith("gxtb_")):
    print(f"  {key}: {atoms_uks.info[key]}")

# UKS gap is a dict with alpha->alpha, beta->beta, alpha->beta channels
gap_uks = atoms_uks.info.get("gxtb_gap_eV")
sc = atoms_uks.info.get("gxtb_spin_contamination")
print(f"\nHOMO-LUMO gap : {gap_uks}")
if sc is not None:
    print(f"Spin contamination: {sc}")
