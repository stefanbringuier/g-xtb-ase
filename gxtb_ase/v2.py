"""Opt-in ASE calculator for the standalone g-xTB 2.x release."""

import math
import os
from numbers import Integral, Real
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    CalculationFailed,
    PropertyNotImplementedError,
    all_changes,
)
from ase.units import Bohr, Hartree


class GxTB2(Calculator):
    """Run an explicit g-xTB 2.x ``xtb`` executable for energy and forces.

    ``charge`` is the molecular charge. ``uhf`` is the number of unpaired
    electrons, or None to let xTB select its default. Explicit zero is not
    equivalent to None: it requests an unrestricted calculation with zero
    unpaired electrons. ``timeout`` is the subprocess time limit in seconds.
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {"charge": 0, "uhf": None, "timeout": 120.0}
    discard_results_on_any_change = True

    def __init__(self, executable, *, charge=0, uhf=None, timeout=120.0):
        self._executable = Path(executable).expanduser().resolve()
        if not self._executable.is_file() or not os.access(self._executable, os.X_OK):
            raise ValueError(f"Not an executable g-xTB 2.x file: {self._executable}")
        super().__init__(charge=charge, uhf=uhf, timeout=timeout)

    def set(self, **kwargs):
        unknown = sorted(kwargs.keys() - self.default_parameters.keys())
        if unknown:
            raise TypeError(f"Unsupported g-xTB 2.x parameters: {', '.join(unknown)}")
        for key in ("charge", "uhf"):
            if key not in kwargs:
                continue
            value = kwargs[key]
            if key == "uhf" and value is None:
                continue
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer" + (" or None" if key == "uhf" else ""))
            if key == "uhf" and value < 0:
                raise ValueError("uhf must be nonnegative")
            kwargs[key] = int(value)
        if "timeout" in kwargs:
            timeout = kwargs["timeout"]
            if (isinstance(timeout, bool) or not isinstance(timeout, Real)
                    or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("timeout must be a positive finite number of seconds")
            kwargs["timeout"] = float(timeout)
        return super().set(**kwargs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        unsupported = set(properties).difference(self.implemented_properties)
        if unsupported:
            raise PropertyNotImplementedError(
                f"Unsupported g-xTB 2.x properties: {', '.join(sorted(unsupported))}"
            )
        if atoms is None:
            atoms = self.atoms
        self.results = {}
        if atoms is None or not len(atoms):
            raise CalculationFailed("g-xTB 2.x needs a nonempty molecule")
        if atoms.pbc.any():
            raise CalculationFailed("g-xTB 2.x only supports nonperiodic molecules")
        if ((atoms.numbers < 1) | (atoms.numbers > 103)).any():
            raise CalculationFailed("g-xTB 2.x supports atomic numbers 1 through 103")
        if not np.isfinite(atoms.positions).all():
            raise CalculationFailed("g-xTB 2.x requires finite atomic positions")

        parameters = self.parameters
        electrons = int(atoms.numbers.sum()) - parameters.charge
        if electrons < 0:
            raise CalculationFailed("Molecular charge would give a negative number of electrons")
        if parameters.uhf is not None and (
            parameters.uhf > electrons or (electrons - parameters.uhf) % 2
        ):
            raise CalculationFailed(f"uhf={parameters.uhf} is incompatible with {electrons} electrons")
        super().calculate(atoms, properties, system_changes)
        command = [str(self._executable), "molecule.xyz", "--gxtb", "--grad",
                   "--norestart", "--strict", "--chrg", str(parameters.charge)]
        if parameters.uhf is not None:
            command.extend(("--uhf", str(parameters.uhf)))

        with tempfile.TemporaryDirectory(prefix="gxtb-ase-v2-") as work:
            workdir = Path(work)
            # Fresh working directory and no inherited xTB input/parameter lookup.
            env = dict(os.environ, HOME=work, XTBHOME=work, XTBPATH=work)
            rows = [f"{symbol} {x:.17g} {y:.17g} {z:.17g}"
                    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)]
            (workdir / "molecule.xyz").write_text(
                "\n".join((str(len(atoms)), "gxtb-ase", *rows)) + "\n", encoding="ascii"
            )
            try:
                run = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                     text=True, errors="replace", timeout=parameters.timeout)
            except subprocess.TimeoutExpired as exc:
                raise CalculationFailed(f"g-xTB 2.x timed out after {parameters.timeout:g} s") from exc
            except OSError as exc:
                raise CalculationFailed(f"Could not start g-xTB 2.x: {exc}") from exc
            if run.returncode:
                output = "\n".join((run.stdout.strip(), run.stderr.strip())).strip()
                raise CalculationFailed(f"g-xTB 2.x exited with status {run.returncode}: {output[-2000:]}")
            if (workdir / "NOT_CONVERGED").exists():
                raise CalculationFailed("g-xTB 2.x SCF did not converge")
            _check_gxtb_method(run.stdout)
            if parameters.uhf is not None:
                actual = _result_value(run.stdout, "number of unpaired electrons", "e")
                if not math.isclose(actual, parameters.uhf, rel_tol=0, abs_tol=1e-6):
                    raise CalculationFailed(f"Requested uhf={parameters.uhf}; g-xTB reported {actual:g}")
            self.results = _read_engrad(workdir / "molecule.engrad", atoms.numbers)


def _check_gxtb_method(output):
    # 2.0.1 prints g-xTB citations even for GFN2, and light systems do
    # not have a core increment. Its g-xTB gradient instead reports the
    # timing for the atom-centered potential (ACP) gradient calculation.
    for line in output.splitlines():
        fields = line.split()
        if fields[:3] == ["-", "acp", "gradient"] and fields[4:5] == ["sec"]:
            return
    raise CalculationFailed("Missing g-xTB ACP gradient diagnostic; use the g-xTB 2.x release")


def _result_value(output, label, unit):
    """Read a finite scalar from a g-xTB calculation diagnostic, not its bibliography."""
    for line in output.splitlines():
        fields = line.split()
        if fields[:-2] == label.split() and fields[-1] == unit:
            try:
                value = _number(fields[-2])
            except ValueError:
                break
            if math.isfinite(value):
                return value
    raise CalculationFailed(f"Missing or invalid g-xTB {label}; use the g-xTB 2.x release")


def _read_engrad(path, atomic_numbers):
    """Read the one ORCA-format energy/gradient record written by xTB --grad."""
    try:
        rows = [line.strip() for line in path.read_text(encoding="ascii").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        n = len(atomic_numbers)
        if len(rows) != 2 + 4 * n or int(rows[0]) != n:
            raise ValueError("wrong atom/gradient count")
        energy = Hartree * _number(rows[1])
        gradient = np.array([_number(row) for row in rows[2:2 + 3 * n]]).reshape(n, 3)
        geometry = [row.split() for row in rows[2 + 3 * n:]]
        if any(len(row) != 4 for row in geometry):
            raise ValueError("invalid geometry records")
        if any(int(row[0]) != z for row, z in zip(geometry, atomic_numbers)):
            raise ValueError("atom order in gradient output does not match input")
        coordinates = [_number(field) for row in geometry for field in row[1:]]
        with np.errstate(over="ignore", invalid="ignore"):
            forces = (-Hartree / Bohr) * gradient
        if not (math.isfinite(energy) and np.isfinite(forces).all()
                and all(math.isfinite(x) for x in coordinates)):
            raise ValueError("non-finite energy, gradient, or coordinates")
    except (OSError, ValueError) as exc:
        raise CalculationFailed(f"Invalid g-xTB 2.x gradient output ({path.name}): {exc}") from exc
    return {"energy": energy, "forces": forces}


def _number(field):
    return float(field.replace("D", "E").replace("d", "e"))
