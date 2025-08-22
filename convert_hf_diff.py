#!/usr/bin/env python3
"""convert_hf_diff.py
Diffusion-pipeline → GGUF converter built on top of convert_hf_to_gguf.py.
Writes three GGUFs: text encoder, UNet and VAE. Keeps original convert_hf_to_gguf.py intact.
Run:
    python convert_hf_diff.py path/to/hf_pipeline --outdir out/
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple
import json
import numpy as np
import torch

from convert_hf_to_gguf import gguf, ModelBase
from safetensors import safe_open
from parser import _index_safetensors_file

############################
# Tensor-name maps (minimal)
############################

sdxl_unet_map = gguf.TensorNameMap({})  # TODO: fill as needed
sdxl_vae_map = gguf.TensorNameMap({})
sdxl_text_map = gguf.TensorNameMap({})

############################
# Sub-models
############################

@ModelBase.register("UNet2DConditionModel")
class SDUnet(ModelBase):
    model_arch = gguf.MODEL_ARCH.SD_UNET  # requires enum patch in core file
    tensor_map = sdxl_unet_map
    block_count = 0

    def set_vocab(self):
        pass  # UNet has no tokenizer

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_image_channels(4)
        self.gguf_writer.add_block_count(0)


@ModelBase.register("AutoencoderKL")
class SDVAE(ModelBase):
    model_arch = gguf.MODEL_ARCH.SD_VAE
    tensor_map = sdxl_vae_map
    block_count = 0

    def set_vocab(self):
        pass

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_block_count(0)


@ModelBase.register("CLIPTextModel")
class SDText(ModelBase):
    model_arch = gguf.MODEL_ARCH.SD_TEXT
    tensor_map = sdxl_text_map
    block_count = 0

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

############################
# CLI driver
############################

def find_components(pipeline_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Return tuples (component_name, path_to_weights)"""
    # Diffusers layout
    mapping = {
        "text": pipeline_dir / "text_encoder/model.safetensors",
        "unet": pipeline_dir / "unet/diffusion_pytorch_model.safetensors",
        "vae": pipeline_dir / "vae/diffusion_pytorch_model.safetensors",
    }
    for k, v in mapping.items():
        if v.exists():
            yield k, v
        else:
            raise SystemExit(f"{v} not found – unsupported pipeline layout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", help="Path to diffusers pipeline directory or HF repo id (snapshot)" )
    ap.add_argument("--outdir", default="gguf_out", help="Directory to write gguf files")
    args = ap.parse_args()

    p_dir = Path(args.pipeline)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, weight_path in find_components(p_dir):
        # trivial loader: read all tensors into memory for now
        offs, _, _, _, _ = _index_safetensors_file(str(weight_path))
        tensors = {}
        with safe_open(str(weight_path), framework="pt", device="cpu") as f:
            for tname in offs.keys():
                tensors[tname] = f.get_tensor(tname)
        tmp = Path(weight_path).stem
        gguf_path = out_dir / f"{tmp}.gguf"
        print(f"Writing {name} → {gguf_path}")
        writer = gguf.GGUFWriter(str(gguf_path), arch=f"sd_{name}")
        # tokenizer for text encoder
        if name == "text":
            tok_dir = p_dir / "tokenizer"
            tok_json = tok_dir / "tokenizer.json"
            merges_txt = tok_dir / "merges.txt"
            vocab_txt = tok_dir / "vocab.json"
            if tok_json.exists():
                try:
                    from tokenizers import Tokenizer
                    tok = Tokenizer.from_file(str(tok_json))
                    writer.add_tokenizer_model(tok.get_vocab().get("model", "bpe"))
                    tokens = [t for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1])]
                    writer.add_token_list(tokens)
                except Exception:
                    pass
            else:
                if vocab_txt.exists():
                    try:
                        vocab = json.loads(Path(vocab_txt).read_text())
                        tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
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
        for tn, tv in tensors.items():
            np_arr = tv.detach().cpu().numpy()
            writer.add_tensor(tn, np_arr)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

if __name__ == "__main__":
    main()
