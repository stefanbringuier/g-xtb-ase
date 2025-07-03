#!/usr/bin/env python3
"""Setup script for gxtb-ase with proper vendor file handling."""

import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithVendor(build_py):
    """Custom build command that copies vendor files to the package."""

    def run(self):
        # Run standard build first
        super().run()

        # Important copy vendor files into python build env
        # This is important because if cloned repo deleted,
        # the vendor files would not be available, so copy.
        vendor_src = Path("vendor")
        if vendor_src.exists():
            gxtb_build_dir = Path(self.build_lib) / "gxtb_ase"
            vendor_dst = gxtb_build_dir / "vendor"

            if vendor_dst.exists():
                shutil.rmtree(vendor_dst)

            print(f"Copying vendor files from {vendor_src} to {vendor_dst}")
            shutil.copytree(vendor_src, vendor_dst)

            # Make binary executable
            binary_path = vendor_dst / "g-xtb" / "binary" / "gxtb"
            if binary_path.exists():
                binary_path.chmod(0o755)
                print(f"Made {binary_path} executable")


setup(
    packages=["gxtb_ase"],
    package_data={"gxtb_ase": ["vendor/g-xtb/binary/*", "vendor/g-xtb/parameters/*"]},
    include_package_data=True,
    cmdclass={
        "build_py": BuildPyWithVendor,
    },
)
