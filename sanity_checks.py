import inspect
import os, sys, pathlib

# Ensure the src directory is in the Python path
root = pathlib.Path().resolve()
print(f"Root path: {root}")

src_path = root / "src"
print(f"Src path: {src_path}")

if src_path not in sys.path:
    print("Src path not in sys.path.")
