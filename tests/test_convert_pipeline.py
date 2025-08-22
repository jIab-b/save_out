import json
import subprocess
from pathlib import Path


def test_download_and_convert(tmp_path: Path):
    repo = "stabilityai/stable-diffusion-xl-base-1.0"
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "gguf"

    subprocess.check_call([
        "python",
        "-m",
        "download_sdxl",
        "--repo",
        repo,
        "--outdir",
        str(raw_dir),
    ])

    subprocess.check_call([
        "python",
        "-m",
        "convert_sdxl",
        str(raw_dir),
        "--outdir",
        str(out_dir),
    ])

    manifest = out_dir / "manifest.json"
    assert manifest.exists(), "manifest.json missing"
    data = json.load(open(manifest))
    assert data.get("files"), "No gguf files listed"
    for fp in data["files"]:
        assert Path(fp).exists(), f"GGUF file {fp} missing"
