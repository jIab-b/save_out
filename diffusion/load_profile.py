#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf.gguf_reader import GGUFReader


def get_scalar(reader: GGUFReader, key: str, default: Optional[Any] = None) -> Any:
    f = reader.get_field(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:
        return default


def read_text_metadata(path: Path, prefix: str) -> dict[str, Any]:
    r = GGUFReader(str(path))
    d: dict[str, Any] = {
        f"{prefix}_hidden_size":       get_scalar(r, f"diffusion.{prefix}.hidden_size"),
        f"{prefix}_num_layers":        get_scalar(r, f"diffusion.{prefix}.num_hidden_layers"),
        f"{prefix}_num_heads":         get_scalar(r, f"diffusion.{prefix}.num_attention_heads"),
        f"{prefix}_max_pos":           get_scalar(r, f"diffusion.{prefix}.max_position_embeddings"),
        f"{prefix}_layer_norm_eps":    get_scalar(r, f"diffusion.{prefix}.layer_norm_eps"),
    }
    # tokenizer ids (per-encoder)
    d[f"{prefix}_bos_id"]  = get_scalar(r, "tokenizer.ggml.bos_token_id")
    d[f"{prefix}_eos_id"]  = get_scalar(r, "tokenizer.ggml.eos_token_id")
    d[f"{prefix}_unk_id"]  = get_scalar(r, "tokenizer.ggml.unknown_token_id")
    d[f"{prefix}_pad_id"]  = get_scalar(r, "tokenizer.ggml.padding_token_id")
    d[f"{prefix}_sep_id"]  = get_scalar(r, "tokenizer.ggml.seperator_token_id")
    d[f"{prefix}_mask_id"] = get_scalar(r, "tokenizer.ggml.mask_token_id")
    return d


def read_unet_metadata(path: Path) -> dict[str, Any]:
    r = GGUFReader(str(path))
    d: dict[str, Any] = {
        "unet_in_channels":  get_scalar(r, "diffusion.unet.in_channels"),
        "unet_out_channels": get_scalar(r, "diffusion.unet.out_channels"),
        "unet_sample_size":  get_scalar(r, "diffusion.unet.sample_size"),
        "unet_cross_attention_dim": get_scalar(r, "diffusion.unet.cross_attention_dim"),
        # arrays
        "unet_block_out_channels":   r.get_field("diffusion.unet.block_out_channels").contents() if r.get_field("diffusion.unet.block_out_channels") else None,
        "unet_attention_head_dim":   r.get_field("diffusion.unet.attention_head_dim").contents() if r.get_field("diffusion.unet.attention_head_dim") else None,
        "unet_down_block_types":     r.get_field("diffusion.unet.down_block_types").contents() if r.get_field("diffusion.unet.down_block_types") else None,
        "unet_up_block_types":       r.get_field("diffusion.unet.up_block_types").contents() if r.get_field("diffusion.unet.up_block_types") else None,
        "unet_layers_per_block":     r.get_field("diffusion.unet.layers_per_block").contents() if r.get_field("diffusion.unet.layers_per_block") else None,
        # strings
        "unet_mid_block_type":       get_scalar(r, "diffusion.unet.mid_block_type"),
        "unet_time_embedding_type":  get_scalar(r, "diffusion.unet.time_embedding_type"),
        "unet_resnet_time_scale_shift": get_scalar(r, "diffusion.unet.resnet_time_scale_shift"),
        # scheduler (if present)
        "sched_prediction_type":     get_scalar(r, "diffusion.scheduler.prediction_type"),
        "sched_num_train_timesteps": get_scalar(r, "diffusion.scheduler.num_train_timesteps"),
        "sched_beta_start":          get_scalar(r, "diffusion.scheduler.beta_start"),
        "sched_beta_end":            get_scalar(r, "diffusion.scheduler.beta_end"),
        "sched_beta_schedule":       get_scalar(r, "diffusion.scheduler.beta_schedule"),
        "sched_timestep_spacing":    get_scalar(r, "diffusion.scheduler.timestep_spacing"),
    }
    return d


def read_vae_metadata(path: Path) -> dict[str, Any]:
    r = GGUFReader(str(path))
    d: dict[str, Any] = {
        "vae_in_channels":     get_scalar(r, "diffusion.vae.in_channels"),
        "vae_out_channels":    get_scalar(r, "diffusion.vae.out_channels"),
        "vae_sample_size":     get_scalar(r, "diffusion.vae.sample_size"),
        "vae_latent_channels": get_scalar(r, "diffusion.vae.latent_channels"),
        "vae_scaling_factor":  get_scalar(r, "diffusion.vae.scaling_factor"),
        "vae_block_out_channels": r.get_field("diffusion.vae.block_out_channels").contents() if r.get_field("diffusion.vae.block_out_channels") else None,
        "vae_down_block_types":   r.get_field("diffusion.vae.down_block_types").contents() if r.get_field("diffusion.vae.down_block_types") else None,
        "vae_up_block_types":     r.get_field("diffusion.vae.up_block_types").contents() if r.get_field("diffusion.vae.up_block_types") else None,
    }
    return d


def build_profile(model_dir: Path) -> dict[str, Any]:
    text = model_dir / "text_encoder.gguf"
    text2 = model_dir / "text_encoder2.gguf"
    unet = model_dir / "unet.gguf"
    vae = model_dir / "vae.gguf"

    if not unet.exists() or not vae.exists() or not text.exists():
        raise SystemExit(f"missing required GGUF(s) in {model_dir}")

    prof: dict[str, Any] = {
        "model_dir": str(model_dir),
        "text_path": str(text),
        "unet_path": str(unet),
        "vae_path":  str(vae),
    }

    prof.update(read_text_metadata(text, "text"))
    prof.update(read_unet_metadata(unet))
    prof.update(read_vae_metadata(vae))

    if text2.exists():
        prof["text2_path"] = str(text2)
        prof.update(read_text_metadata(text2, "text2"))

    # prefer scheduler from UNet; if missing there, try text
    if prof.get("sched_prediction_type") is None:
        r_text = GGUFReader(str(text))
        prof["sched_prediction_type"] = get_scalar(r_text, "diffusion.scheduler.prediction_type")
        prof["sched_num_train_timesteps"] = get_scalar(r_text, "diffusion.scheduler.num_train_timesteps")

    return prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory with text_encoder(.gguf), unet.gguf, vae.gguf")
    ap.add_argument("--out", default=None, help="Output JSON path (default: diffusion/sdxl.profile.json)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (Path(__file__).resolve().parent / "sdxl.profile.json")

    profile = build_profile(model_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
