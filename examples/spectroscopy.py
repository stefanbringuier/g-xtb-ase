"""Spectroscopy calculations using g-XTB"""

import numpy as np
import shutil
from pathlib import Path
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations.infrared import Infrared
from ase.thermochemistry import IdealGasThermo
from gxtb_ase import GxTB
import matplotlib.pyplot as plt


def optimize_geometry(atoms, label="opt"):
    """Optimize molecular geometry using g-XTB."""
    calc = GxTB(label=label, charge=0, spin=0)
    atoms.calc = calc

    opt = BFGS(atoms)
    opt.run(fmax=0.001)

    return atoms


def calculate_ir_spectrum(atoms, name, ir_dir):
    """Calculate IR spectrum for neutral molecules."""

    calc = GxTB(label=f"{name}_ir", charge=0, spin=0)
    atoms.calc = calc

    # Calculate IR intensities
    ir = Infrared(atoms, name=f"{name}_ir", delta=0.01, nfree=2)
    ir.run()

    spectrum_file = ir_dir / "data" / f"{name}_spectrum.dat"
    ir.write_spectra(out=str(spectrum_file), width=20, type="Gaussian")

    plot_file = ir_dir / "plots" / f"{name}_spectrum.png"
    plot_spectrum_from_ase_data(spectrum_file, plot_file, name)

    return ir


def plot_spectrum_from_ase_data(data_file, plot_file, mol_name):
    """Plot spectrum."""
    data = np.loadtxt(data_file)
    wavenumbers = data[:, 0]  # cm^-1
    intensities = data[:, 1]  # absolute intensities

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, intensities, "b-", linewidth=2)
    plt.xlabel("Frequency (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title(f"IR Spectrum of {mol_name}")
    plt.grid(True, alpha=0.3)

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_and_save_thermochemistry(ir, mol_name, ir_dir, temperature=298.15):
    """Calculate and save comprehensive thermochemical properties."""

    freqs = ir.get_frequencies()
    freq_ev = [f.real * 1.24e-4 for f in freqs if np.isreal(f) and f > 0]

    # Symmetries
    if mol_name == "H2O":
        geometry = "nonlinear"
        symmetry_number = 2  # C2v point group
    elif mol_name == "NH3":
        geometry = "nonlinear"
        symmetry_number = 3  # C3v
    elif mol_name == "CH4":
        geometry = "nonlinear"
        symmetry_number = 12  # Td
    elif mol_name == "SiH4":
        geometry = "nonlinear"
        symmetry_number = 12  # Td
    elif mol_name == "AlCl3":
        geometry = "nonlinear"
        symmetry_number = 6  # D3h
    else:
        geometry = "nonlinear"
        symmetry_number = 1

    atoms = ir.atoms

    thermo = IdealGasThermo(
        vib_energies=freq_ev,
        geometry=geometry,
        potentialenergy=0.0,  # Relative energy
        atoms=atoms,
        symmetrynumber=symmetry_number,
        spin=0,  # Closed shell molecules
    )

    enthalpy = thermo.get_enthalpy(temperature)
    entropy = thermo.get_entropy(temperature, pressure=101325)  # 1 atm
    gibbs_energy = thermo.get_gibbs_energy(temperature, pressure=101325)

    thermo_file = ir_dir / "thermochemistry" / f"{mol_name}_thermo.txt"
    with open(thermo_file, "w") as f:
        f.write(f"Comprehensive Thermochemistry for {mol_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Temperature: {temperature} K\n")
        f.write("Pressure: 101325 Pa (1 atm)\n")
        f.write(f"Geometry: {geometry}\n")
        f.write(f"Symmetry number: {symmetry_number}\n")
        f.write("Electronic spin: 0 (closed shell)\n\n")

        f.write("THERMODYNAMIC PROPERTIES:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Enthalpy (H):      {enthalpy:.6f} eV\n")
        f.write(f"Entropy (S):       {entropy:.6f} eV/K\n")
        f.write(f"Gibbs energy (G):  {gibbs_energy:.6f} eV\n\n")

        f.write("VIBRATIONAL FREQUENCIES:\n")
        f.write("-" * 30 + "\n")
        f.write("Mode  Frequency (cm⁻¹)  Energy (eV)\n")
        for i, freq in enumerate(freqs):
            if np.isreal(freq) and freq > 0:
                energy_ev = freq.real * 1.24e-4
                f.write(f"{i+1:4d}  {freq.real:8.2f}      {energy_ev:.6f}\n")


def cleanup_calculation_files(mol_name):
    """Clean up intermediate g-XTB calculation files."""
    patterns_to_remove = [
        f"{mol_name}_opt*",
        f"{mol_name}_ir*",
        "*.log",
        "*.out",
        "gxtbrestart",
        "coord",
        "energy",
        "gradient",
        "hessian",
        "*.traj",
    ]

    # Remove files matching patterns
    for pattern in patterns_to_remove:
        for file_path in Path(".").glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except Exception:
                pass  # Ignore cleanup errors


def main():
    """Main function to demonstrate spectroscopy calculations."""

    ir_dir = Path("IR")
    ir_dir.mkdir(exist_ok=True)
    (ir_dir / "data").mkdir(exist_ok=True)
    (ir_dir / "plots").mkdir(exist_ok=True)
    (ir_dir / "thermochemistry").mkdir(exist_ok=True)

    molecules = ["H2O", "NH3", "CH4", "SiH4", "AlCl3"]

    print("=== g-XTB Spectroscopy Calculations ===")
    print(f"Calculating IR spectra for {len(molecules)} molecules")
    print(f"Output will be organized in: {ir_dir.absolute()}")

    for mol_name in molecules:
        print(f"\n{'='*60}")
        print(f"Molecule: {mol_name}")
        print(f"{'='*60}")

        atoms = molecule(mol_name)

        atoms_opt = optimize_geometry(atoms.copy(), f"{mol_name}_opt")

        ir = calculate_ir_spectrum(atoms_opt, mol_name, ir_dir)

        calculate_and_save_thermochemistry(ir, mol_name, ir_dir)

        cleanup_calculation_files(mol_name)


if __name__ == "__main__":
    main()
