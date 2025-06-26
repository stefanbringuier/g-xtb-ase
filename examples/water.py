"""Example demonstrates usage of the GxTB calculator."""

from ase.build import molecule
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import read, write
from gxtb_ase import GxTB


def main():

    atoms = molecule("H2O")

    # Attach g-xTB calculator
    calc = GxTB(
        label="water_neutral", charge=0, spin=0, write_log=True
    )  # Singlet state (closed shell)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    print(f"Water molecule energy: {energy:.6f} eV")

    forces = atoms.get_forces()
    print("Forces (eV/Å):")
    for i, (symbol, force) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
        print(
            f"  {i}: {symbol} " f"[{force[0]:8.5f}, {force[1]:8.5f}, {force[2]:8.5f}]"
        )

    atoms_cation = molecule("H2O")
    calc_cation = GxTB(
        label="water_cation", charge=+1, spin=1  # Doublet state (one unpaired electron)
    )
    atoms_cation.calc = calc_cation

    energy_cation = atoms_cation.get_potential_energy()
    print(f"Water cation energy: {energy_cation:.6f} eV")

    # Slightly distorted geometry
    atoms_opt = molecule("H2O")
    atoms_opt.positions[1] += [0.1, 0.0, 0.0]

    calc_opt = GxTB(label="water_opt", charge=0)
    atoms_opt.calc = calc_opt

    initial_energy = atoms_opt.get_potential_energy()
    print(f"Initial energy (distorted): {initial_energy:.6f} eV")

    opt = BFGS(atoms_opt)
    opt.run(fmax=0.05)

    final_energy = atoms_opt.get_potential_energy()
    print(f"Final energy (optimized):   {final_energy:.6f} eV")
    print(f"Energy lowering:            " f"{initial_energy - final_energy:.6f} eV")

    print("\nRunning 500-step MD simulation...")
    atoms_md = atoms_opt.copy()  # Use optimized geometry

    calc_md = GxTB(label="water_md", charge=0, write_log=True)
    atoms_md.calc = calc_md

    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
    )

    MaxwellBoltzmannDistribution(atoms_md, temperature_K=300)
    Stationary(atoms_md)
    ZeroRotation(atoms_md)

    dyn = VelocityVerlet(atoms_md, timestep=1.0 * units.fs, trajectory="water_md.traj")
    dyn.run(500)

    print("MD simulation complete. Trajectory saved to water_md.traj")
    print("View with: ase gui water_md.traj")

    traj = read("water_md.traj", index=":")
    write("water_md.extxyz", traj)


if __name__ == "__main__":
    main()
