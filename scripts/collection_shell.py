"""Resolve the Windows child shell without depending on Isaac's isolated PATH."""
import os
from pathlib import Path


def powershell_executable():
    path=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    if not path.is_file():raise FileNotFoundError(f'Windows PowerShell is unavailable: {path}')
    return path
