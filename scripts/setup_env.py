#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / ".env.example"
dst = root / ".env"
if dst.exists():
    print(f"{dst} already exists; leaving it untouched")
else:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"created {dst}")
    print("fill in DEEPSEEK_API_KEY before running dataset generation")
