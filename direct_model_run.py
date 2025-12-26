#!/usr/bin/env python3
"""
Direct notebook execution using nbclient instead of nbconvert CLI
"""

import json
import sys
import os
from pathlib import Path
from nbclient import NotebookClient
import nbformat

def run_notebook_direct(nb_path, option="B"):
    """Execute notebook directly with nbclient"""

    nb_path = Path(nb_path).resolve()
    print(f"\nLoading notebook: {nb_path}")

    # Change to notebook directory so relative paths work
    notebook_dir = nb_path.parent
    original_cwd = os.getcwd()
    os.chdir(notebook_dir)
    print(f"Changed working directory to: {notebook_dir}")

    # Read notebook
    with open(nb_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Modify AlphaEarth option in Cell 2
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if 'ALPHA_EARTH_OPTION' in source:
                cell['source'] = source.replace(
                    "ALPHA_EARTH_OPTION = 'B'",
                    f"ALPHA_EARTH_OPTION = '{option}'"
                )
                print(f"✓ Modified ALPHA_EARTH_OPTION to '{option}'")
                break

    # Execute notebook
    print(f"Executing notebook...")
    client = NotebookClient(
        nb,
        timeout=14400,
        kernel_name='python3'
    )

    try:
        client.execute()
        print(f"✓ Notebook executed successfully")
        os.chdir(original_cwd)
        return True
    except Exception as e:
        import traceback
        print(f"✗ Notebook execution failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)[:1000]}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        os.chdir(original_cwd)
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 direct_model_run.py <notebook_path> [option]")
        sys.exit(1)

    nb_path = Path(sys.argv[1])
    option = sys.argv[2] if len(sys.argv) > 2 else "B"

    if not nb_path.exists():
        print(f"✗ Notebook not found: {nb_path}")
        sys.exit(1)

    success = run_notebook_direct(nb_path, option)
    sys.exit(0 if success else 1)
