#!/usr/bin/env python3
"""
Test that GGUF text encoder files contain tokenizer data and that the
generated profile JSON includes tokenizer IDs. Uses gguf-py in-repo.

Usage:
  python tests/test_gguf_tokenizer_check.py [--model-dir DIR] [--profile PATH]

Defaults:
  model_dir = gguf_out/sdxl_from_hf
  profile   = diffusion/sdxl.profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def check_tokenizer(reader, label: str) -> None:
    # keys to verify
    need_keys = [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ]
    missing = []
    for k in need_keys:
        if reader.get_field(k) is None:
            missing.append(k)
    assert not missing, f"[{label}] missing fields: {missing}"

    # arrays: tokens, merges
    f_tok = reader.get_field("tokenizer.ggml.tokens")
    f_mrg = reader.get_field("tokenizer.ggml.merges")
    n_tok = len(f_tok.contents()) if f_tok else 0
    n_mrg = len(f_mrg.contents()) if f_mrg else 0
    assert n_tok > 0, f"[{label}] tokenizer.ggml.tokens is empty"
    assert n_mrg > 0, f"[{label}] tokenizer.ggml.merges is empty"

    # scalars: ids
    bos = reader.get_field("tokenizer.ggml.bos_token_id").contents()
    eos = reader.get_field("tokenizer.ggml.eos_token_id").contents()
    pad = reader.get_field("tokenizer.ggml.padding_token_id").contents()
    assert isinstance(bos, int) and bos >= 0, f"[{label}] invalid BOS id: {bos}"
    assert isinstance(eos, int) and eos >= 0, f"[{label}] invalid EOS id: {eos}"
    assert isinstance(pad, int) and pad >= 0, f"[{label}] invalid PAD id: {pad}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="gguf_out/sdxl_from_hf")
    ap.add_argument("--profile", default="diffusion/sdxl.profile.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    profile_path = Path(args.profile).resolve()
    assert model_dir.is_dir(), f"model dir not found: {model_dir}"

    # import gguf reader from vendor dir
    sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "gguf-py").resolve()))
    from gguf.gguf_reader import GGUFReader  # type: ignore

    # text encoders
    paths = [
        model_dir / "text_encoder.gguf",
        model_dir / "text_encoder2.gguf",
    ]
    for p in paths:
        assert p.exists(), f"missing GGUF: {p}"
        r = GGUFReader(str(p))
        check_tokenizer(r, p.name)

    # profile ids populated
    assert profile_path.exists(), f"profile not found: {profile_path}"
    prof = json.loads(profile_path.read_text())
    for key in ("text_bos_id", "text_eos_id", "text2_bos_id", "text2_eos_id"):
        v = prof.get(key)
        assert isinstance(v, int), f"profile missing/invalid {key}: {v}"

    print("OK: tokenizer fields present in GGUFs and profile IDs populated")


if __name__ == "__main__":
    main()
