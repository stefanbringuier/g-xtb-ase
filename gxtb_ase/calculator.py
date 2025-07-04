"""ASE calculator wrapper for g-xTB."""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.calculators.calculator import CalculatorSetupError
from ase.io import write
from ase.units import Hartree, Bohr


class GxTB(FileIOCalculator):
    """ASE calculator for g-xTB.

    This calculator wraps the g-xTB binary, which approximates
    ωB97M-V/def2-TZVPPD properties. The calculator handles g-xTB's
    requirement for parameter files in the home directory by using
    a temporary HOME environment during execution.

    Units: g-xTB outputs energies in Hartree and gradients in Hartree/Bohr.
    These are converted to ASE's expected eV and eV/Å units.

    File Management: By default, temporary calculation files are cleaned up
    after each calculation. The gxtbrestart file is preserved between
    calculations for efficiency but removed when parameters change or the
    calculator is reset. Set keep_files=True to disable cleanup entirely,
    or write_log=True to keep just the log file.

    Parameters
    ----------
    charge : int
        Molecular charge (default: 0)
    spin : int
        Number of unpaired electrons, 2S (default: 0)
    numerical_grad : bool
        Use numerical gradients (default: True)
    numerical_hess : bool
        Calculate numerical Hessian (default: False)
    write_molden : bool
        Write molden file with orbitals (default: False)
    write_log : bool
        Write and keep calculation log file (default: False)
    tmpdir : str or None
        Directory for temporary files. If None, uses system temp.
    keep_files : bool
        Keep calculation files after completion (default: False)
    """

    name = "gxtb"
    implemented_properties = ["energy", "forces", "dipole", "charges"]
    command = None  # Set dynamically

    default_parameters = {
        "charge": 0,
        "spin": 0,  # Number of unpaired electrons (2S)
        "numerical_grad": True,
        "numerical_hess": False,
        "write_molden": False,
        "write_log": False,  # Write calculation log file
    }

    def __init__(
        self,
        restart=None,
        label="gxtb",
        atoms=None,
        directory=".",
        command=None,
        profile=None,
        tmpdir=None,
        keep_files=False,
        **kwargs,
    ):
        """Initialize g-xTB calculator.

        Parameters
        ----------
        restart : str
            Ignored, for consistency with FileIOCalculator.
        label : str
            Name for output files.
        atoms : Atoms
            Atoms object to attach calculator to.
        directory : str
            Working directory for calculation.
        command : str
            Command to execute g-xTB. If None, uses the bundled binary.
        profile : None
            Not used, for API consistency.
        tmpdir : str or None
            Directory for temporary files. If None, uses system temp.
        keep_files : bool
            Keep calculation files after completion.
        **kwargs : dict
            Additional parameters: charge, spin, numerical_grad, etc.
        """
        # g-xTB config for files
        self.tmpdir = tmpdir
        self.keep_files = keep_files
        self._temp_home = None

        # Find vendor directory in multiple possible locations
        try:
            import gxtb_ase

            pkg_path = Path(gxtb_ase.__file__).parent

            # Search for vendor files in multiple locations
            binary_candidates = [
                # Installed package data (preferred)
                pkg_path / "vendor" / "g-xtb" / "binary" / "gxtb",
                # Source tree (development)
                pkg_path.parent / "vendor" / "g-xtb" / "binary" / "gxtb",
                # Site-packages level fallback
                pkg_path.parent / "vendor" / "g-xtb" / "binary" / "gxtb",
            ]

            param_candidates = [
                # Installed package data (preferred)
                pkg_path / "vendor" / "g-xtb" / "parameters",
                # Source tree (development)
                pkg_path.parent / "vendor" / "g-xtb" / "parameters",
                # Site-packages level fallback
                pkg_path.parent / "vendor" / "g-xtb" / "parameters",
            ]

            # Find first existing binary
            self.binary_path = None
            for candidate in binary_candidates:
                if candidate.exists():
                    self.binary_path = candidate.resolve()
                    break

            # Find first existing parameter directory
            self.parameter_path = None
            for candidate in param_candidates:
                if candidate.exists():
                    self.parameter_path = candidate.resolve()
                    break

        except ImportError:
            # Final fallback to git source
            pkg_dir = Path(__file__).parent.parent
            self.binary_path = pkg_dir / "vendor" / "g-xtb" / "binary" / "gxtb"
            self.parameter_path = pkg_dir / "vendor" / "g-xtb" / "parameters"

        if not self.binary_path or not self.binary_path.exists():
            # Provide helpful error message for debugging
            try:
                import gxtb_ase

                pkg_path = Path(gxtb_ase.__file__).parent
                searched_paths = [
                    pkg_path / "vendor" / "g-xtb" / "binary" / "gxtb",
                    pkg_path.parent / "vendor" / "g-xtb" / "binary" / "gxtb",
                ]
                error_msg = "g-xTB binary not found. Searched locations:\n" + "\n".join(
                    f"  - {path} (exists: {path.exists()})" for path in searched_paths
                )
            except ImportError:
                error_msg = "g-xTB binary not found and package import failed."

            raise CalculatorSetupError(
                f"g-xTB binary not found at {self.binary_path}. "
                "Please ensure the package is properly installed."
            )

        # Make binary executable
        if not os.access(self.binary_path, os.X_OK):
            self.binary_path.chmod(0o755)

        # Verify param files
        if not self.parameter_path or not self.parameter_path.exists():
            raise CalculatorSetupError(
                f"Parameter directory not found. Expected at: {self.parameter_path}\n"
                "Please ensure the package is properly installed with: pip install ."
            )

        for param_file in [".gxtb", ".eeq", ".basisq"]:
            param_path = self.parameter_path / param_file
            if not param_path.exists():
                alt_path = self.binary_path.parent / param_file
                if alt_path.exists():
                    self.parameter_path = self.binary_path.parent
                else:
                    raise CalculatorSetupError(
                        f"Parameter file {param_file} not found in "
                        f"{self.parameter_path} or {self.binary_path.parent}\n"
                        "Please ensure the package is properly installed with: pip install ."
                    )

        # Base command
        if command is None:
            command = f"{self.binary_path} -c coord"

        FileIOCalculator.__init__(
            self,
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            command=command,
            profile=profile,
            **kwargs,
        )

    def _setup_temp_home(self):
        """Set up temporary HOME directory with parameter files."""
        if self.tmpdir:
            self._temp_home = tempfile.mkdtemp(dir=self.tmpdir)
        else:
            self._temp_home = tempfile.mkdtemp()

        # Must copy parameter files to temp home
        for param_file in [".gxtb", ".eeq", ".basisq"]:
            src = self.parameter_path / param_file
            dst = Path(self._temp_home) / param_file
            shutil.copy2(src, dst)

        return self._temp_home

    def _cleanup_temp_home(self):
        """Clean up temporary HOME directory."""
        if self._temp_home and not self.keep_files:
            shutil.rmtree(self._temp_home, ignore_errors=True)
            self._temp_home = None

    def write_input(self, atoms, properties=None, system_changes=None):
        """
        Write input files for g-xTB calculation.

        g-xTB looks for the following files:
        - coord (TURBOMOLE format)
        - .CHRG
        - .UHF
        - .GRAD
        - .HESS
        - .MOLDEN

        """

        coord_file = Path(self.directory) / "coord"

        write(coord_file, atoms, format="turbomole")

        # .CHRG
        charge = self.parameters.get("charge", 0)
        if charge != 0:
            with open(Path(self.directory) / ".CHRG", "w") as f:
                f.write(f"{charge}\n")

        # .UHF
        spin = self.parameters.get("spin", 0)
        if spin != 0:
            with open(Path(self.directory) / ".UHF", "w") as f:
                f.write(f"{spin}\n")

        # .GRAD - only create if forces are requested
        if (
            properties
            and "forces" in properties
            and self.parameters.get("numerical_grad")
        ):
            grad_file = Path(self.directory) / ".GRAD"
            grad_file.touch()

    def _build_command(self, properties):
        """Build g-xTB command based on requested properties.

        See g-xTB README for details.

        """
        cmd_parts = [str(self.binary_path)]

        cmd_parts.extend(["-c", "coord"])

        if "forces" in properties and self.parameters.get("numerical_grad"):
            cmd_parts.append("-grad")

        if self.parameters.get("numerical_hess"):
            cmd_parts.append("-hess")

        if self.parameters.get("write_molden"):
            cmd_parts.append("-molden")

        # NOTE: Don't redirect, FileIOCalculator should handle?
        return " ".join(cmd_parts)

    def execute(self):
        """Execute calculation with proper output capture.

        This overrides FileIOCalculator.execute() to properly handle
        subprocess output according to ASE best practices.
        """
        temp_home = self._setup_temp_home()
        old_home = os.environ.get("HOME")

        try:
            # Set temporary HOME environment
            os.environ["HOME"] = temp_home

            # Build command
            command = self.command
            if isinstance(command, str):
                command = command.split()

            # Execute with proper output capture
            log_file = None
            if self.keep_files or self.parameters.get("write_log", False):
                log_file = Path(self.directory) / f"{self.label}.log"

            self._run_subprocess(command, log_file)

        finally:
            # Restore HOME environment
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                if "HOME" in os.environ:
                    del os.environ["HOME"]

            if not self.keep_files:
                self._cleanup_temp_home()

    def _run_subprocess(self, command, log_file=None):
        """Run subprocess with proper output handling."""
        try:
            with subprocess.Popen(
                command,
                cwd=self.directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:

                output_lines = []
                log_handle = None

                if log_file:
                    log_handle = open(log_file, "w", encoding="utf-8")

                try:
                    # Read output
                    for line in proc.stdout:
                        output_lines.append(line.rstrip())

                        if log_handle:
                            log_handle.write(line)
                            log_handle.flush()

                    proc.wait()

                finally:
                    if log_handle:
                        log_handle.close()

                # Check return code
                if proc.returncode != 0:
                    cmd_str = " ".join(command)

                    raise subprocess.CalledProcessError(
                        proc.returncode, cmd_str, output="\n".join(output_lines)
                    )

        except FileNotFoundError:
            raise CalculatorSetupError(f"Could not find g-xTB executable: {command[0]}")

    def _extract_error_summary(self, output_lines, max_lines=3):
        """Extract concise error information from output lines."""
        if not output_lines:
            return "Unknown error (no output)"

        # Some errors
        error_keywords = [
            "error",
            "failed",
            "cannot",
            "unable",
            "invalid",
            "not found",
            "permission denied",
            "terminated",
        ]

        # Look for error patterns
        start_idx = max(0, len(output_lines) - max_lines)

        for i in range(len(output_lines) - 1, start_idx - 1, -1):
            line = output_lines[i].strip().lower()
            if any(keyword in line for keyword in error_keywords):
                if i > 0:
                    return f"{output_lines[i-1].strip()}\n" f"{output_lines[i].strip()}"
                return output_lines[i].strip()

        # return none empty line
        for line in reversed(output_lines):
            if line.strip():
                return line.strip()

        return "Unknown error (no specific information found)"

    def _cleanup_calculation_files(self):
        """Clean up temporary calculation files, but keep restart file."""
        if self.keep_files:
            return

        files_to_remove = [
            "coord",
            "energy",
            "gradient",
            "hessian",
            ".CHRG",
            ".UHF",
            ".GRAD",
            ".HESS",
            ".MOLDEN",
            ".data",
            f"{self.label}.out",
        ]

        # Remove log file if not requested
        if not self.parameters.get("write_log", False):
            files_to_remove.append(f"{self.label}.log")

        for filename in files_to_remove:
            filepath = Path(self.directory) / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                except OSError:
                    pass

    def _cleanup_restart_file(self):
        """Clean up restart file specifically."""
        if self.keep_files:
            return

        restart_file = Path(self.directory) / "gxtbrestart"
        if restart_file.exists():
            try:
                restart_file.unlink()
            except OSError:
                pass

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        """Perform g-xTB calculation with modified HOME environment."""

        # Safe cleanup
        if system_changes and system_changes != set():
            restart_file = Path(self.directory) / "gxtbrestart"
            if restart_file.exists():
                restart_file.unlink()

        # Force log if dipole or charges
        need_dipole = "dipole" in properties
        need_charges = "charges" in properties
        if need_dipole or need_charges:
            self.parameters["write_log"] = True

        # Build command based on properties
        force_grad = "forces" in properties and self.parameters.get("numerical_grad")

        command_parts = [str(self.binary_path), "-c", "coord"]

        if force_grad:
            command_parts.append("-grad")

        if self.parameters.get("numerical_hess"):
            command_parts.append("-hess")

        if self.parameters.get("write_molden"):
            command_parts.append("-molden")

        # Set command
        self.command = " ".join(command_parts)

        # Call parent's calculate method
        FileIOCalculator.calculate(self, atoms, properties, system_changes)

        self.read(self.label)

        self._cleanup_calculation_files()

    def read(self, label):
        """Read results from g-xTB output files."""

        self._parse_output()

        gradient_file = Path(self.directory) / "gradient"
        if gradient_file.exists():
            self._parse_gradient(gradient_file)

        hessian_file = Path(self.directory) / "hessian"
        if hessian_file.exists():
            self._parse_hessian(hessian_file)

    def _parse_output(self):
        """Parse main g-xTB output files."""
        energy = None
        dipole = None
        charges = None

        # First try to parse from energy file
        energy_file = Path(self.directory) / "energy"
        if energy_file.exists():
            energy = self._parse_energy_file(energy_file)

        # Parse dipole and charges from log file
        log_file = Path(self.directory) / f"{self.label}.log"
        if log_file.exists():
            result = self._parse_output_file(log_file)
            if isinstance(result, tuple) and len(result) == 3:
                log_energy, parsed_dipole, parsed_charges = result
                if energy is None:
                    energy = log_energy
                if parsed_dipole is not None:
                    dipole = parsed_dipole
                if parsed_charges is not None:
                    charges = parsed_charges
            elif isinstance(result, tuple) and len(result) == 2:
                log_energy, parsed_dipole = result
                if energy is None:
                    energy = log_energy
                if parsed_dipole is not None:
                    dipole = parsed_dipole
            else:
                if energy is None:
                    energy = result

        # Fallback to .out file if it exists
        if energy is None:
            output_file = Path(self.directory) / f"{self.label}.out"
            if output_file.exists():
                result = self._parse_output_file(output_file)
                if isinstance(result, tuple) and len(result) == 3:
                    energy, parsed_dipole, parsed_charges = result
                    if parsed_dipole is not None:
                        dipole = parsed_dipole
                    if parsed_charges is not None:
                        charges = parsed_charges
                elif isinstance(result, tuple) and len(result) == 2:
                    energy, parsed_dipole = result
                    if parsed_dipole is not None:
                        dipole = parsed_dipole
                else:
                    energy = result

        if energy is None:
            error_msg = "Could not parse energy from g-xTB calculation."
            files = list(Path(self.directory).glob("*"))
            error_msg += f"\nFiles in directory: {files}"

            # Show energy file content if it exists
            energy_file = Path(self.directory) / "energy"
            if energy_file.exists():
                with open(energy_file, "r") as f:
                    content = f.read()
                error_msg += f"\nEnergy file content:\n{content}"

            raise RuntimeError(error_msg)

        # Store results in eV and eÅ
        self.results["energy"] = energy * Hartree
        if dipole is not None:
            self.results["dipole"] = dipole
        if charges is not None:
            self.results["charges"] = charges

    def _parse_energy_file(self, energy_file):
        """Parse energy from TURBOMOLE-style energy file."""
        with open(energy_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("$"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # Second column contains the energy in Hartree
                            return float(parts[1])
                        except (ValueError, IndexError):
                            continue
        return None

    def _parse_output_file(self, output_file):
        """Parse energy, dipole and charges from g-xTB output file."""
        energy = None
        dipole = None
        charges = None

        with open(output_file, "r") as f:
            lines = f.readlines()

        # Parse backwards to get final values
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]

            # Parse total energy final line
            if line.strip().startswith("total") and len(line.split()) >= 2:
                if energy is None:  # Get the last occurrence
                    try:
                        energy = float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue

            # FIXME: Brittle parsing of dipole
            if "dipole moment  X         Y          Z" in line and i + 1 < len(lines):
                try:
                    dipole_line = lines[i + 1].strip()
                    parts = dipole_line.split()
                    if len(parts) >= 3:
                        # Get X, Y, Z components in atomic units
                        dipole_x = float(parts[0])
                        dipole_y = float(parts[1])
                        dipole_z = float(parts[2])

                        # Convert from atomic units to eÅ
                        # 1 au = 2.541746473 eÅ
                        conversion = 2.541746473
                        dipole = np.array([dipole_x, dipole_y, dipole_z]) * conversion
                        break
                except (ValueError, IndexError):
                    continue

        # FIXME: Brittle parsing of charges
        in_charge_section = False
        charge_lines = []

        for line in lines:
            if "E E Q (BC)  c h a r g e s" in line:
                in_charge_section = True
                continue
            elif in_charge_section and (
                "g - x T B   e n e r g y" in line
                or line.strip().startswith("---")
                and "g - x T B" in line
            ):
                break
            elif in_charge_section and line.strip():
                # Look for charge lines: "    1 O    1.1266  -0.3510   0.7795  -0.2783"
                parts = line.split()
                if (
                    len(parts) >= 6 and parts[0].isdigit() and len(parts[1]) <= 2
                ):  # Element symbol
                    try:
                        # Last column is the charge
                        charge = float(parts[-1])
                        charge_lines.append(charge)
                    except (ValueError, IndexError):
                        continue

        if charge_lines:
            charges = np.array(charge_lines)
        else:
            print("No charges found")

        if charges is not None and dipole is not None:
            return energy, dipole, charges
        elif dipole is not None:
            return energy, dipole
        else:
            return energy

    def _parse_gradient(self, gradient_file):
        """Parse forces from TURBOMOLE gradient file."""
        with open(gradient_file, "r") as f:
            lines = f.readlines()

        # Find gradient section
        grad_start = None
        for i, line in enumerate(lines):
            if "$grad" in line:
                grad_start = i + 1
                break

        if grad_start is None:
            return

        # Parse the TURBOMOLE gradient format
        gradients = []
        coordinates_section = True
        n_atoms = 0

        i = grad_start
        while i < len(lines) and "$end" not in lines[i]:
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if "cycle =" in line:
                i += 1
                continue

            if coordinates_section:
                parts = line.split()
                if len(parts) >= 4 and parts[-1].isalpha():
                    n_atoms += 1
                elif len(parts) == 3:
                    coordinates_section = False
                    continue
                i += 1
                continue

            if len(line.split()) == 3:
                try:
                    # D notation to E notation for Python
                    grad_line = line.replace("D", "E")
                    grad = [float(x) for x in grad_line.split()]
                    gradients.append(grad)
                except ValueError:
                    pass

            i += 1

        if len(gradients) == n_atoms and n_atoms > 0:
            # Convert gradient to forces (F = -grad)
            # and  Hartree/Bohr to eV/Å
            gradients = np.array(gradients)
            forces = -gradients * Hartree / Bohr
            self.results["forces"] = forces

    def _parse_hessian(self, hessian_file):
        """Parse Hessian matrix from file."""
        import warnings

        warnings.warn("Hessian parsing not yet implemented", UserWarning)
        return None

    def reset(self):
        """Reset calculator and clean up all files including restart."""
        self._cleanup_calculation_files()
        self._cleanup_restart_file()
        FileIOCalculator.reset(self)

    def __del__(self):
        """Clean up all files when calculator is destroyed."""
        try:
            self._cleanup_calculation_files()
            self._cleanup_restart_file()
            self._cleanup_temp_home()
        except (AttributeError, OSError):
            pass  # Ignore errors during cleanup in destructor

    def get_dipole_moment(self, atoms=None):
        """Get dipole moment for ASE infrared calculations."""
        if atoms is not None:
            self.calculate(atoms, ["dipole"])

        if "dipole" not in self.results:
            raise RuntimeError(
                "Dipole moment not available. Ensure calculator has been run."
            )

        return self.results["dipole"]

    def set(self, **kwargs):
        """Set parameters."""
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            # Clean up when parameters change since restart may be invalid
            self._cleanup_restart_file()
            self.reset()
        return changed_parameters
