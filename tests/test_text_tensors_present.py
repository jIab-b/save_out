#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def load_profile(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_tensor(reader, name: str) -> None:
    t = next((t for t in reader.tensors if t.name == name), None)
    assert t is not None, f"missing tensor: {name}"


def test_text_tensors_present() -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = root / "diffusion/sdxl.profile.json"
    assert profile_path.exists(), f"profile not found: {profile_path}"
    prof = load_profile(profile_path)

    model_dir = Path(prof["model_dir"]).resolve()
    text1 = model_dir / "text_encoder.gguf"
    text2 = model_dir / "text_encoder2.gguf"
    assert text1.exists(), f"missing text1 gguf: {text1}"

    sys.path.insert(0, str((root / "gguf-py").resolve()))
    from gguf.gguf_reader import GGUFReader  # type: ignore

    def check(reader: GGUFReader, prefix: str, hidden: int, n_layers: int, max_pos: int, projection_dim: int | None) -> None:
        # embeddings
        assert_tensor(reader, f"text_model.embeddings.token_embedding.weight")
        assert_tensor(reader, f"text_model.embeddings.position_embedding.weight")
        # final ln
        assert_tensor(reader, f"text_model.final_layer_norm.weight")
        assert_tensor(reader, f"text_model.final_layer_norm.bias")
        # layers
        for i in range(n_layers):
            base = f"text_model.encoder.layers.{i}"
            for nm in ("layer_norm1", "layer_norm2"):
                assert_tensor(reader, f"{base}.{nm}.weight")
                assert_tensor(reader, f"{base}.{nm}.bias")
            for nm in ("q_proj", "k_proj", "v_proj", "out_proj"):
                assert_tensor(reader, f"{base}.self_attn.{nm}.weight")
                assert_tensor(reader, f"{base}.self_attn.{nm}.bias")
            for nm in ("fc1", "fc2"):
                assert_tensor(reader, f"{base}.mlp.{nm}.weight")
                assert_tensor(reader, f"{base}.mlp.{nm}.bias")
        # optional projection
        if projection_dim is not None:
            assert_tensor(reader, "text_projection.weight")

    # text1
    r1 = GGUFReader(str(text1))
    check(
        r1,
        "text",
        hidden=int(prof.get("text_hidden_size") or 0),
        n_layers=int(prof.get("text_num_layers") or 0),
        max_pos=int(prof.get("text_max_pos") or 0),
        projection_dim=None,
    )

    # text2 (if present)
    if text2.exists():
        r2 = GGUFReader(str(text2))
        proj_dim = prof.get("text2_projection_dim")
        check(
            r2,
            "text2",
            hidden=int(prof.get("text2_hidden_size") or 0),
            n_layers=int(prof.get("text2_num_layers") or 0),
            max_pos=int(prof.get("text2_max_pos") or 0),
            projection_dim=int(proj_dim) if proj_dim is not None else None,
        )

    print("OK: required text encoder tensors present")


if __name__ == "__main__":
    test_text_tensors_present()
