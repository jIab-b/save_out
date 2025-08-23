#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
import os
import tempfile


def test_diffuse_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    bin_path = root / "build/diffuse"
    if not bin_path.exists():
        # Skip if binary not built
        print("skip: diffuse binary not found; build with ./build.sh")
        return
    out_dir = Path(tempfile.mkdtemp(prefix="diffuse_test_"))
    out_path = out_dir / "out.png"
    cmd = [str(bin_path), str(root / "diffusion/sdxl.profile.json"), "--steps", "1", "--prompt", "a test", "--out", str(out_path)]
    env = os.environ.copy()
    r = subprocess.run(cmd, cwd=str(root), env=env)
    assert r.returncode == 0, f"diffuse exited with {r.returncode}"
    assert out_path.exists(), f"no output image: {out_path}"
    # basic size check (> PNG header)
    assert out_path.stat().st_size > 100, f"output PNG too small: {out_path.stat().st_size} bytes"
    print(f"OK: diffuse smoke ran and wrote {out_path}")


if __name__ == "__main__":
    test_diffuse_smoke()
