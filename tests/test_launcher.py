"""The bootstrap Python override must not bypass installed runtime packages."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_runtime_uses_project_venv_even_with_bootstrap_override(self):
        with tempfile.TemporaryDirectory(prefix="case500 launcher ") as temporary:
            bundle = Path(temporary)
            shutil.copyfile(ROOT / "run.sh", bundle / "run.sh")
            (bundle / "scripts").mkdir()
            (bundle / "scripts/run_pipeline.py").write_text("")
            executable = bundle / ".venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/usr/bin/env bash\nprintf "VENV\\n"\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            result = subprocess.run(
                ["bash", str(bundle / "run.sh"), "status", "/path with spaces/data"],
                env={**os.environ, "PYTHON_BIN": "/nonexistent/bootstrap/python"},
                text=True, capture_output=True, check=True,
            )
            self.assertEqual(result.stdout.splitlines(), [
                "VENV", str(bundle / "scripts/run_pipeline.py"),
                "status", "/path with spaces/data",
            ])

    def test_missing_venv_requests_setup_without_using_base_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            shutil.copyfile(ROOT / "run.sh", bundle / "run.sh")
            result = subprocess.run(["bash", str(bundle / "run.sh"), "plan"],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("setup_env.sh", result.stderr)


if __name__ == "__main__":
    unittest.main()
