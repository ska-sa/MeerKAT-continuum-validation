import os
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

cmdclass = {}


def run_download_script():
    """Execute the shell script to download reference catalogues."""
    script_path = os.path.join(
        os.path.dirname(__file__), "src", "continuum_validation", "download_catalogues.sh"
    )
    if os.path.isfile(script_path):
        print(f"Running catalogue download script: {script_path}")
        subprocess.check_call(["sh", script_path])


class CustomBuildPy(build_py):
    """Triggered on standard `pip install .` and wheel builds."""

    def run(self):
        run_download_script()
        super().run()


class CustomDevelop(develop):
    """Triggered on legacy editable installs `python setup.py develop`."""

    def run(self):
        run_download_script()
        super().run()


cmdclass["build_py"] = CustomBuildPy
cmdclass["develop"] = CustomDevelop

try:
    from setuptools.command.editable_wheel import editable_wheel

    class CustomEditableWheel(editable_wheel):
        """Triggered on PEP 660 editable installs `pip install -e .`."""

        def run(self):
            run_download_script()
            super().run()

    cmdclass["editable_wheel"] = CustomEditableWheel
except ImportError:
    pass

setup(cmdclass=cmdclass)
