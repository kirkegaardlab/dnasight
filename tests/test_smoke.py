import subprocess
import sys
import pytest
from pathlib import Path

@pytest.mark.parametrize("command", [
    [
        sys.executable,               # Ensures the current Python interpreter is used
        "dnasight-cmd.py",
        "run",
        "--dna_segmentation",
        "--folder", "data/example_data_linear_-hsp",
        "--output", "output"
    ]
])
def test_dnasight_run_command(command):
    """
    Test that the dnasight-cmd.py run command executes successfully.
    """
    # Ensure paths exist
    assert Path("dnasight-cmd.py").exists(), "dnasight-cmd.py not found in current directory"
    assert Path("data/example_data_linear_-hsp").exists(), "Input folder does not exist"

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Log output in case of failure
    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

    # Check it executed successfully
    assert result.returncode == 0, "dnasight-cmd.py run command failed"
