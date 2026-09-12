#!/usr/bin/env python3
"""Stable training entrypoint; original scripts/run_eval.py is untouched."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cold_ghost.cli import main
if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
