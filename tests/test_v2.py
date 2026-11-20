import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import CalculationFailed, PropertyNotImplementedError
from ase.units import Bohr, Hartree

from gxtb_ase import GxTB, GxTB2
from gxtb_ase.calculator import GxTB as LegacyGxTB
from gxtb_ase.v2 import _check_gxtb_method


GXTB_RESULT = " - acp gradient                           0.000 sec (  7%)\n"


class V2Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        self.bin = root / "external program"
        self.bin.touch(mode=0o700)
        self.workdirs = []
        self.calc = GxTB2(self.bin)
        self.atoms = Atoms("HO", positions=[[0, 0, 0], [0, 0, 1]])
        self.atoms.calc = self.calc

    def test_legacy_is_still_the_default_import(self):
        self.assertIs(GxTB, LegacyGxTB)
        self.assertIsNot(GxTB, GxTB2)

    def run_xtb(self, command, **kwargs):
        work = Path(kwargs["cwd"])
        self.workdirs.append(work)
        self.assertEqual(command[0], str(self.bin))
        self.assertEqual(command[1:7],
                         ["molecule.xyz", "--gxtb", "--grad", "--norestart", "--strict", "--chrg"])
        self.assertEqual(kwargs["env"]["HOME"], str(work))
        self.assertEqual(kwargs["env"]["XTBPATH"], str(work))
        self.assertEqual(kwargs["env"]["XTBHOME"], str(work))
        self.assertFalse((work / "molecule.engrad").exists())
        xyz = (work / command[1]).read_text().splitlines()
        self.assertEqual(xyz[0], "2")
        self.assertEqual([line.split()[0] for line in xyz[2:]], ["H", "O"])
        result = "2\n-1.25\n0.1\n0.2\n0.3\n-0.1D0\n-0.2\n-0.3\n1 0 0 0\n8 0 0 1.89\n"
        (work / "molecule.engrad").write_text(result, encoding="ascii")
        output = GXTB_RESULT
        if "--uhf" in command:
            output += f"number of unpaired electrons   {command[command.index('--uhf') + 1]} e\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    def test_method_citation_is_not_an_active_method(self):
        def bibliography_only(cmd, **kwargs):
            run = self.run_xtb(cmd, **kwargs)  # otherwise valid gradient
            run.stdout = "   for g-xTB:\n   for GFN2-xTB:\n"
            return run

        with patch("gxtb_ase.v2.subprocess.run", side_effect=bibliography_only):
            with self.assertRaisesRegex(CalculationFailed, "ACP gradient"):
                self.atoms.get_forces()
        self.assertFalse(self.calc.results)

    def test_dispatch_units_cache_and_option_changes(self):
        with patch("gxtb_ase.v2.subprocess.run", side_effect=self.run_xtb) as run:
            self.assertEqual(self.atoms.get_potential_energy(), -1.25 * Hartree)
            np.testing.assert_allclose(self.atoms.get_forces(),
                                      -Hartree / Bohr * np.array([[.1, .2, .3], [-.1, -.2, -.3]]))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args[0][0][7:], ["0"])
            self.calc.set(charge=1, uhf=0, timeout=2.5)
            self.atoms.get_forces()
            self.assertEqual(run.call_args[0][0][7:], ["1", "--uhf", "0"])
            self.assertEqual(run.call_args[1]["timeout"], 2.5)
            self.atoms.positions[1, 2] += .01
            self.atoms.get_forces()
            self.assertEqual(run.call_count, 3)
        self.assertEqual(len(set(self.workdirs)), 3)
        self.assertTrue(all(not work.exists() for work in self.workdirs))

    def test_bad_input_refused_without_starting_xtb(self):
        with patch("gxtb_ase.v2.subprocess.run") as run:
            with self.assertRaises(PropertyNotImplementedError):
                self.atoms.get_charges()
            with self.assertRaises(TypeError):
                self.calc.set(numerical_grad=True)
            for kwargs in ({"charge": 0.2}, {"uhf": -1}, {"timeout": float("inf")}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.calc.set(**kwargs)
            periodic = self.atoms.copy()
            periodic.pbc = True
            with self.assertRaisesRegex(CalculationFailed, "nonperiodic"):
                self.calc.get_forces(periodic)
            with self.assertRaisesRegex(CalculationFailed, "nonempty"):
                self.calc.get_forces(Atoms())
            with self.assertRaisesRegex(CalculationFailed, "atomic numbers"):
                self.calc.get_forces(Atoms("X"))
            self.atoms.positions[0, 0] = np.nan
            with self.assertRaisesRegex(CalculationFailed, "finite"):
                self.atoms.get_forces()
            run.assert_not_called()

    def test_reject_incompatible_spin_or_charge_before_execution(self):
        water = Atoms("OH2")
        for charge, uhf in [(1, 0), (0, 12), (11, None)]:
            with self.subTest(charge=charge, uhf=uhf), patch("gxtb_ase.v2.subprocess.run") as run:
                self.calc.set(charge=charge, uhf=uhf)
                with self.assertRaisesRegex(CalculationFailed, "electrons"):
                    self.calc.get_forces(water)
                run.assert_not_called()

    def test_reject_backend_spin_override(self):
        self.calc.set(charge=1, uhf=0)

        def wrong_spin(cmd, **kwargs):
            run = self.run_xtb(cmd, **kwargs)
            run.stdout = GXTB_RESULT + "number of unpaired electrons   2 e\n"
            return run

        with patch("gxtb_ase.v2.subprocess.run", side_effect=wrong_spin):
            with self.assertRaisesRegex(CalculationFailed, "Requested uhf=0; g-xTB reported 2"):
                self.atoms.get_forces()
        self.assertFalse(self.calc.results)

    def test_failed_process_or_invalid_output_clear_results(self):
        failures = {
            "exited with status 1": lambda cmd, work: (1, "", "failed"),
            "ACP gradient": lambda cmd, work: (0, "GFN2-xTB", ""),
            "Invalid .* gradient output": lambda cmd, work: (0, GXTB_RESULT, ""),
        }
        for message, result in failures.items():
            def fail(cmd, **kwargs):
                self.workdirs.append(Path(kwargs["cwd"]))
                return subprocess.CompletedProcess(cmd, *result(cmd, Path(kwargs["cwd"])))

            with self.subTest(message=message):
                self.calc.results = {"energy": -15.0}
                with patch("gxtb_ase.v2.subprocess.run", side_effect=fail):
                    with self.assertRaisesRegex(CalculationFailed, message):
                        self.calc.calculate(self.atoms)
                self.assertEqual(self.calc.results, {})
        with patch("gxtb_ase.v2.subprocess.run", side_effect=subprocess.TimeoutExpired("xtb", 1)):
            with self.assertRaisesRegex(CalculationFailed, "timed out"):
                self.calc.calculate(self.atoms)
        self.assertTrue(all(not work.exists() for work in self.workdirs))

    def test_malformed_output_and_nonconvergence(self):
        for source, replacement, message in [
            ("-1.25", "NaN", "non-finite"),
            ("8 0 0 1.89", "1 0 0 1.89", "atom order"),
            ("-0.3\n", "", "wrong atom/gradient count"),
        ]:
            def malformed(cmd, **kwargs):
                run = self.run_xtb(cmd, **kwargs)
                path = Path(kwargs["cwd"]) / "molecule.engrad"
                path.write_text(path.read_text().replace(source, replacement))
                return run

            with self.subTest(message=message):
                with patch("gxtb_ase.v2.subprocess.run", side_effect=malformed):
                    with self.assertRaisesRegex(CalculationFailed, message):
                        self.calc.calculate(self.atoms)
                self.assertFalse(self.calc.results)

        def unconverged(cmd, **kwargs):
            run = self.run_xtb(cmd, **kwargs)
            (Path(kwargs["cwd"]) / "NOT_CONVERGED").touch()
            return run

        with patch("gxtb_ase.v2.subprocess.run", side_effect=unconverged):
            with self.assertRaisesRegex(CalculationFailed, "SCF did not converge"):
                self.calc.calculate(self.atoms)


@unittest.skipUnless(os.environ.get("GXTB2_TEST_EXE"), "set GXTB2_TEST_EXE to verified g-xTB 2.0.1")
class RealV2Test(unittest.TestCase):
    def test_energy_and_all_forces_against_direct_xtb(self):
        executable = Path(os.environ["GXTB2_TEST_EXE"]).resolve()
        water = Atoms("OH2", positions=[[0, 0, 0], [.957, 0, 0], [-.240, .927, 0]])
        water_xyz = "3\nwater\nO 0 0 0\nH .957 0 0\nH -.240 .927 0\n"
        cases = [(water, water_xyz, 0, None), (water, water_xyz, 0, 0), (water, water_xyz, 1, 1),
                 (Atoms("H"), "1\nH\nH 0 0 0\n", 0, 1),
                 (Atoms("HH", positions=[[0, 0, 0], [0, 0, .74]]), "2\nH2\nH 0 0 0\nH 0 0 .74\n", 0, 0)]
        for atoms, xyz, charge, uhf in cases:
            with self.subTest(atoms=atoms.symbols, charge=charge, uhf=uhf), tempfile.TemporaryDirectory() as work:
                root = Path(work)
                (root / "direct.xyz").write_text(xyz)
                command = [str(executable), "direct.xyz", "--gxtb", "--grad", "--chrg", str(charge)]
                if uhf is not None:
                    command.extend(("--uhf", str(uhf)))
                env = dict(os.environ, HOME=work, XTBHOME=work, XTBPATH=work)
                subprocess.run(command, cwd=work, env=env, check=True, capture_output=True, timeout=15)
                # Independent xTB output files, not the wrapper's .engrad parser.
                energy_hartree = float((root / "energy").read_text().splitlines()[1].split()[1])
                records = (root / "gradient").read_text().splitlines()
                gradient = np.loadtxt(records[-(len(atoms) + 1):-1], ndmin=2)
                atoms.calc = GxTB2(executable, charge=charge, uhf=uhf, timeout=15)
                self.assertAlmostEqual(atoms.get_potential_energy(), energy_hartree * Hartree, places=8)
                np.testing.assert_allclose(atoms.get_forces(), -Hartree / Bohr * gradient,
                                           rtol=0, atol=3e-11)

    def test_gfn2_control_is_not_gxtb(self):
        executable = Path(os.environ["GXTB2_TEST_EXE"]).resolve()
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "h2.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 .74\n")
            env = dict(os.environ, HOME=work, XTBHOME=work, XTBPATH=work)
            run = subprocess.run([str(executable), "h2.xyz", "--gfn2", "--grad"],
                                 cwd=work, env=env, check=True, capture_output=True,
                                 text=True, timeout=15)
            self.assertTrue((Path(work) / "h2.engrad").is_file())
            self.assertIn("for g-xTB:", run.stdout)
            with self.assertRaisesRegex(CalculationFailed, "ACP gradient"):
                _check_gxtb_method(run.stdout)
