import json
import subprocess
from pathlib import Path


def test_convert_sdxl_snapshot(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "gguf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download via diffusers (skipped if not available)
    try:
        print(f"Downloading SDXL snapshot to: {raw_dir}")
        subprocess.check_call([
            "python",
            "/home/beed1089/save_out/download_sdxl.py",
            "--repo",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "--outdir",
            str(raw_dir),
        ])
        print("Download complete")
    except Exception:
        print("Download skipped (diffusers or network unavailable)")
        return

    # Convert
    print(f"Converting HF snapshot → GGUF: out={out_dir}")
    subprocess.check_call([
        "python",
        "/home/beed1089/save_out/convert_hf_diff.py",
        str(raw_dir),
        "--outdir",
        str(out_dir),
    ])
    print("Conversion complete")

    # Expect standard component files
    expect = [
        out_dir / "text_encoder.gguf",
        out_dir / "unet.gguf",
        out_dir / "vae.gguf",
    ]
    # Optional second text encoder
    opt = out_dir / "text_encoder2.gguf"
    for p in expect:
        assert p.exists(), f"missing: {p}"
        assert p.stat().st_size > 0, f"empty: {p}"
        print(f"OK: {p.name} size={p.stat().st_size}")
    if (raw_dir / "text_encoder_2").exists():
        assert opt.exists(), "missing: text_encoder2.gguf"
        assert opt.stat().st_size > 0, "empty: text_encoder2.gguf"
        print(f"OK: {opt.name} size={opt.stat().st_size}")
