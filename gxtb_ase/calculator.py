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
    implemented_properties = ["energy", "forces", "dipole"]
    command = None  # Set dynamically based on properties

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

        # Find packaged files
        try:
            import gxtb_ase

            pkg_path = Path(gxtb_ase.__file__).parent

            # Look in relative path
            binary_rel_path = "../vendor/g-xtb/binary/gxtb"
            self.binary_path = (pkg_path / binary_rel_path).resolve()
            param_rel_path = "../vendor/g-xtb/parameters"
            self.parameter_path = (pkg_path / param_rel_path).resolve()

            # Check if they're in package dir
            if not self.binary_path.exists():
                for possible_binary in [
                    pkg_path / "gxtb",
                    pkg_path / "binary" / "gxtb",
                    pkg_path.parent / "gxtb",
                ]:
                    if possible_binary.exists():
                        self.binary_path = possible_binary
                        break

                # Look for parameter files
                if self.binary_path.exists():
                    param_path = self.binary_path.parent.parent / "parameters"
                    self.parameter_path = param_path
                    if not self.parameter_path.exists():
                        self.parameter_path = self.binary_path.parent
        except ImportError:
            # Fallback to source tree location
            pkg_dir = Path(__file__).parent.parent
            self.binary_path = pkg_dir / "vendor" / "g-xtb" / "binary" / "gxtb"
            self.parameter_path = pkg_dir / "vendor" / "g-xtb" / "parameters"

        if not self.binary_path.exists():
            raise CalculatorSetupError(
                f"g-xTB binary not found at {self.binary_path}. "
                "Please ensure the package is properly installed."
            )

        # Make binary executable
        if not os.access(self.binary_path, os.X_OK):
            self.binary_path.chmod(0o755)

        # Verify param files
        for param_file in [".gxtb", ".eeq", ".basisq"]:
            param_path = self.parameter_path / param_file
            if not param_path.exists():
                alt_path = self.binary_path.parent / param_file
                if alt_path.exists():
                    self.parameter_path = self.binary_path.parent
                else:
                    raise CalculatorSetupError(
                        f"Parameter file {param_file} not found in "
                        f"{self.parameter_path} or {self.binary_path.parent}"
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

        energy_file = Path(self.directory) / "energy"

        if energy_file.exists():
            energy = self._parse_energy_file(energy_file)

        if energy is None:
            output_file = Path(self.directory) / f"{self.label}.out"
            if output_file.exists():
                energy = self._parse_output_file(output_file)

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

        # Store results eV
        self.results["energy"] = energy * Hartree
        if dipole is not None:
            self.results["dipole"] = dipole

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
        """Parse energy from g-xTB output file."""
        energy = None
        with open(output_file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # Parse total energy - look for the final total energy line
            if line.strip().startswith("total") and len(line.split()) >= 2:
                try:
                    energy = float(line.split()[-1])
                except (ValueError, IndexError):
                    continue

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

    def set(self, **kwargs):
        """Set parameters."""
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            # Clean up when parameters change since restart may be invalid
            self._cleanup_restart_file()
            self.reset()
        return changed_parameters
