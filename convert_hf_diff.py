#!/usr/bin/env python3
"""convert_hf_diff.py
Diffusion-pipeline → GGUF converter built on top of convert_hf_to_gguf.py.
Writes three GGUFs: text encoder, UNet and VAE. Keeps original convert_hf_to_gguf.py intact.
Run:
    python convert_hf_diff.py path/to/hf_pipeline --outdir out/
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any
import json
import numpy as np
import torch

from convert_hf_to_gguf import gguf
from safetensors import safe_open

############################
# CLI driver
############################

def _add_kv_int(writer: gguf.GGUFWriter, key: str, val) -> None:
    if isinstance(val, bool):
        writer.add_bool(key, val)
    elif isinstance(val, int):
        writer.add_uint32(key, val)

def _add_kv_float(writer: gguf.GGUFWriter, key: str, val) -> None:
    if isinstance(val, (float, int)):
        writer.add_float32(key, float(val))

def _add_kv_str(writer: gguf.GGUFWriter, key: str, val) -> None:
    if isinstance(val, str):
        writer.add_string(key, val)

def _add_kv_array(writer: gguf.GGUFWriter, key: str, val) -> None:
    if isinstance(val, (list, tuple)) and len(val) > 0:
        writer.add_array(key, list(val))

def _get_int(d: dict, key: str) -> int | None:
    v = d.get(key)
    return int(v) if isinstance(v, (int,)) else None

def _get_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None

def write_unet_metadata(writer: gguf.GGUFWriter, cfg: dict) -> None:
    p = "diffusion.unet."
    _add_kv_int(writer, p + "in_channels", cfg.get("in_channels"))
    _add_kv_int(writer, p + "out_channels", cfg.get("out_channels"))
    _add_kv_int(writer, p + "sample_size", cfg.get("sample_size"))
    ca = cfg.get("cross_attention_dim")
    if isinstance(ca, int):
        _add_kv_int(writer, p + "cross_attention_dim", ca)
    _add_kv_array(writer, p + "block_out_channels", cfg.get("block_out_channels"))
    ahd = cfg.get("attention_head_dim")
    if isinstance(ahd, int):
        _add_kv_int(writer, p + "attention_head_dim", ahd)
    elif isinstance(ahd, list):
        _add_kv_array(writer, p + "attention_head_dim", ahd)
    _add_kv_array(writer, p + "down_block_types", cfg.get("down_block_types"))
    _add_kv_array(writer, p + "up_block_types", cfg.get("up_block_types"))
    lpb = cfg.get("layers_per_block")
    if isinstance(lpb, int):
        _add_kv_int(writer, p + "layers_per_block", lpb)
    elif isinstance(lpb, list):
        _add_kv_array(writer, p + "layers_per_block", lpb)
    _add_kv_str(writer, p + "mid_block_type", cfg.get("mid_block_type"))
    _add_kv_str(writer, p + "time_embedding_type", cfg.get("time_embedding_type"))
    _add_kv_str(writer, p + "resnet_time_scale_shift", cfg.get("resnet_time_scale_shift"))
    # SDXL-specific additions if present
    _add_kv_str(writer, p + "addition_embed_type", cfg.get("addition_embed_type"))
    _add_kv_int(writer, p + "addition_time_embed_dim", cfg.get("addition_time_embed_dim"))
    _add_kv_int(writer, p + "time_embedding_dim", cfg.get("time_embedding_dim"))
    # Common extra projection/class/time cond fields
    _add_kv_int(writer, p + "time_cond_proj_dim", cfg.get("time_cond_proj_dim"))
    _add_kv_int(writer, p + "class_embeddings_input_dim", cfg.get("class_embeddings_input_dim"))
    _add_kv_int(writer, p + "projection_class_embeddings_input_dim", cfg.get("projection_class_embeddings_input_dim"))

def write_vae_metadata(writer: gguf.GGUFWriter, cfg: dict) -> None:
    p = "diffusion.vae."
    _add_kv_int(writer, p + "in_channels", cfg.get("in_channels"))
    _add_kv_int(writer, p + "out_channels", cfg.get("out_channels"))
    _add_kv_int(writer, p + "sample_size", cfg.get("sample_size"))
    _add_kv_int(writer, p + "latent_channels", cfg.get("latent_channels"))
    _add_kv_float(writer, p + "scaling_factor", cfg.get("scaling_factor"))
    _add_kv_array(writer, p + "block_out_channels", cfg.get("block_out_channels"))
    _add_kv_array(writer, p + "down_block_types", cfg.get("down_block_types"))
    _add_kv_array(writer, p + "up_block_types", cfg.get("up_block_types"))

def write_text_metadata(writer: gguf.GGUFWriter, cfg: dict, prefix: str) -> None:
    p = prefix
    _add_kv_int(writer, p + "hidden_size", cfg.get("hidden_size"))
    _add_kv_int(writer, p + "num_hidden_layers", cfg.get("num_hidden_layers"))
    _add_kv_int(writer, p + "num_attention_heads", cfg.get("num_attention_heads"))
    _add_kv_int(writer, p + "max_position_embeddings", cfg.get("max_position_embeddings"))
    _add_kv_float(writer, p + "layer_norm_eps", cfg.get("layer_norm_eps"))
    # Additional helpful fields
    _add_kv_int(writer, p + "intermediate_size", cfg.get("intermediate_size"))
    _add_kv_str(writer, p + "hidden_act", cfg.get("hidden_act"))
    _add_kv_int(writer, p + "vocab_size", cfg.get("vocab_size"))
    # For CLIPTextModelWithProjection
    _add_kv_int(writer, p + "projection_dim", cfg.get("projection_dim"))

def write_scheduler_metadata(writer: gguf.GGUFWriter, sched_cfg: dict) -> None:
    p = "diffusion.scheduler."
    _add_kv_str(writer, p + "prediction_type", sched_cfg.get("prediction_type"))
    _add_kv_str(writer, p + "timestep_spacing", sched_cfg.get("timestep_spacing"))
    _add_kv_int(writer, p + "num_train_timesteps", sched_cfg.get("num_train_timesteps"))
    _add_kv_float(writer, p + "beta_start", sched_cfg.get("beta_start"))
    _add_kv_float(writer, p + "beta_end", sched_cfg.get("beta_end"))
    _add_kv_str(writer, p + "beta_schedule", sched_cfg.get("beta_schedule"))
    # Optional extras for various solvers
    _add_kv_int(writer, p + "steps_offset", sched_cfg.get("steps_offset"))
    _add_kv_float(writer, p + "sigma_min", sched_cfg.get("sigma_min"))
    _add_kv_float(writer, p + "sigma_max", sched_cfg.get("sigma_max"))
    _add_kv_float(writer, p + "sigma_data", sched_cfg.get("sigma_data"))
    _add_kv_float(writer, p + "variance_type", sched_cfg.get("variance_type"))
    _add_kv_int(writer, p + "use_karras_sigmas", sched_cfg.get("use_karras_sigmas"))
    _add_kv_int(writer, p + "rescale_betas_zero_snr", sched_cfg.get("rescale_betas_zero_snr"))
    _add_kv_int(writer, p + "clip_sample", sched_cfg.get("clip_sample"))
    _add_kv_float(writer, p + "clip_sample_range", sched_cfg.get("clip_sample_range"))

def find_components(pipeline_dir: Path) -> Iterable[Tuple[str, Path, list[Path]]]:
    """Return tuples (component_name, component_dir, list_of_weight_files) discovered from configs."""
    def classify(dir_path: Path) -> str | None:
        cfg = dir_path / "config.json"
        if not cfg.exists():
            return None
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            return None
        cls = data.get("_class_name") or data.get("architectures") or data.get("model_type")
        if isinstance(cls, list):
            cls = cls[0] if cls else None
        if not isinstance(cls, str):
            return None
        if cls == "UNet2DConditionModel":
            return "unet"
        if cls == "AutoencoderKL":
            return "vae"
        if cls.startswith("CLIPTextModel") or cls == "CLIPTextModel" or cls == "CLIPTextModelWithProjection":
            return "text"
        return None

    def shard_files(dir_path: Path, base: str) -> list[Path]:
        f = dir_path / base
        if f.exists():
            return [f]
        idx = dir_path / f"{base}.index.json"
        if idx.exists():
            try:
                m = json.loads(idx.read_text()).get("weight_map") or {}
                shards = sorted(set(m.values()))
                paths = [dir_path / s for s in shards if (dir_path / s).exists()]
                if paths:
                    return paths
            except Exception:
                pass
        # fallback: any *.safetensors
        sts = sorted(dir_path.glob("*.safetensors"))
        return sts

    cand_dirs = {p for p in pipeline_dir.iterdir() if p.is_dir()}
    for n in ("unet", "vae", "vae_1_0", "text_encoder", "text_encoder_2"):
        d = pipeline_dir / n
        if d.is_dir():
            cand_dirs.add(d)

    seen_text = 0
    for d in sorted(cand_dirs):
        role = classify(d)
        if not role:
            continue
        bases = [
            "diffusion_pytorch_model.safetensors" if role in ("unet", "vae") else "model.safetensors",
            "model.safetensors",
        ]
        weights: list[Path] = []
        for b in bases:
            weights = shard_files(d, b)
            if weights:
                break
        if not weights:
            continue
        name = role
        if role == "text":
            seen_text += 1
            if seen_text > 1:
                name = "text2"
        yield name, d, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", help="Path to diffusers pipeline directory or HF repo id (snapshot)" )
    ap.add_argument("--outdir", default="gguf_out", help="Directory to write gguf files")
    args = ap.parse_args()

    p_dir = Path(args.pipeline)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, comp_dir, weight_paths in find_components(p_dir):
        tensors = {}
        for weight_path in weight_paths:
            with safe_open(str(weight_path), framework="pt", device="cpu") as f:
                for tname in f.keys():
                    tensors[tname] = f.get_tensor(tname)
        if name == "text":
            out_name = "text_encoder.gguf"
        elif name == "text2":
            out_name = "text_encoder2.gguf"
        elif name == "unet":
            out_name = "unet.gguf"
        elif name == "vae":
            out_name = "vae.gguf"
        else:
            out_name = f"{Path(weight_paths[0]).stem}.gguf"
        gguf_path = out_dir / out_name
        print(f"Writing {name} → {gguf_path}")
        writer = gguf.GGUFWriter(str(gguf_path), arch=f"sd_{'text' if name in ('text','text2') else name}")
        # tokenizer for text encoder
        if name in ("text", "text2"):
            tok_dir = p_dir / ("tokenizer_2" if comp_dir.name.endswith("_2") else "tokenizer")
            tok_json = tok_dir / "tokenizer.json"
            merges_txt = tok_dir / "merges.txt"
            vocab_txt = tok_dir / "vocab.json"
            token_to_id: Dict[str, int] = {}
            if tok_json.exists():
                try:
                    tj = json.loads(tok_json.read_text())
                    model_type = (tj.get("model", {}).get("type") or "bpe").lower()
                    writer.add_tokenizer_model(model_type)
                    vocab = tj.get("model", {}).get("vocab") or {}
                    if isinstance(vocab, dict):
                        items = sorted(vocab.items(), key=lambda kv: kv[1])
                        tokens = [t for t, _ in items]
                        token_to_id = {t: i for t, i in items}
                        writer.add_token_list(tokens)
                    merges = tj.get("model", {}).get("merges")
                    if isinstance(merges, list) and merges:
                        writer.add_token_merges(merges)
                except Exception:
                    pass
            else:
                if vocab_txt.exists():
                    try:
                        vocab = json.loads(Path(vocab_txt).read_text())
                        items = sorted(vocab.items(), key=lambda kv: kv[1])
                        tokens = [t for t, _ in items]
                        token_to_id = {t: i for t, i in items}
                        writer.add_tokenizer_model("bpe")
                        writer.add_token_list(tokens)
                    except Exception:
                        pass
                if merges_txt.exists():
                    try:
                        merges = [ln.strip() for ln in merges_txt.read_text().splitlines() if ln and not ln.startswith("#")]
                        writer.add_token_merges(merges)
                    except Exception:
                        pass
            tok_cfg = tok_dir / "tokenizer_config.json"
            if tok_cfg.exists():
                try:
                    cfg = json.loads(tok_cfg.read_text())
                    def resolve_id(field: str) -> int | None:
                        obj = cfg.get(field)
                        # Prefer explicit numeric id
                        if isinstance(obj, dict) and isinstance(obj.get("id"), int):
                            return int(obj["id"])
                        # Else try by content string
                        content = None
                        if isinstance(obj, dict):
                            content = obj.get("content") or obj.get("piece") or obj.get("token")
                        elif isinstance(obj, str):
                            content = obj
                        if isinstance(content, str) and token_to_id:
                            return token_to_id.get(content)
                        return None
                    v = resolve_id("bos_token");  v is not None and writer.add_bos_token_id(v)
                    v = resolve_id("eos_token");  v is not None and writer.add_eos_token_id(v)
                    v = resolve_id("unk_token");  v is not None and writer.add_unk_token_id(v)
                    v = resolve_id("sep_token");  v is not None and writer.add_sep_token_id(v)
                    v = resolve_id("pad_token");  v is not None and writer.add_pad_token_id(v)
                    v = resolve_id("mask_token"); v is not None and writer.add_mask_token_id(v)
                except Exception:
                    pass
            # Also inspect special_tokens_map.json as a fallback
            stm = tok_dir / "special_tokens_map.json"
            if stm.exists():
                try:
                    sm = json.loads(stm.read_text())
                    def add_if(name: str, fn):
                        obj = sm.get(name)
                        tid = None
                        if isinstance(obj, dict) and isinstance(obj.get("id"), int):
                            tid = obj["id"]
                        elif isinstance(obj, str) and token_to_id:
                            tid = token_to_id.get(obj)
                        if isinstance(tid, int):
                            fn(tid)
                    add_if("bos_token", writer.add_bos_token_id)
                    add_if("eos_token", writer.add_eos_token_id)
                    add_if("unk_token", writer.add_unk_token_id)
                    add_if("sep_token", writer.add_sep_token_id)
                    add_if("pad_token", writer.add_pad_token_id)
                    add_if("mask_token", writer.add_mask_token_id)
                except Exception:
                    pass
        mi = p_dir / "model_index.json"
        if mi.exists():
            try:
                writer.add_string("diffusion.hf.model_index_json", mi.read_text())
            except Exception:
                pass
        # add component-specific metadata from config.json
        comp_cfg = comp_dir / "config.json"
        if comp_cfg.exists():
            try:
                cfg = json.loads(comp_cfg.read_text())
                if name == "unet":
                    write_unet_metadata(writer, cfg)
                elif name == "vae":
                    write_vae_metadata(writer, cfg)
                elif name in ("text", "text2"):
                    prefix = f"diffusion.{name}."
                    write_text_metadata(writer, cfg, prefix)
            except Exception:
                pass

        sch = p_dir / "scheduler" / "scheduler_config.json"
        if sch.exists():
            try:
                s_cfg = json.loads(sch.read_text())
                write_scheduler_metadata(writer, s_cfg)
                writer.add_string("diffusion.hf.scheduler_config_json", json.dumps(s_cfg))
            except Exception:
                pass
        # keep raw jsons for full fidelity
        if comp_cfg.exists():
            try:
                writer.add_string(f"diffusion.hf.{name}_config_json", comp_cfg.read_text())
            except Exception:
                pass
        for tn, tv in tensors.items():
            np_arr = tv.detach().cpu().numpy()
            if np_arr.dtype == np.float32 or np_arr.dtype == np.float64:
                np_arr = np_arr.astype(np.float16)
            writer.add_tensor(tn, np_arr)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

if __name__ == "__main__":
    main()
