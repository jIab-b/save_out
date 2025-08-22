## Diffusion → GGUF conversion: current state and changes

### Overview
This repository now includes a modular converter to export Diffusers pipelines (e.g., SDXL) to GGUF in a llama.cpp-style layout. The converter writes separate GGUFs for text encoder(s), UNet, and VAE, and embeds tokenizer and configuration metadata for reproducible reconstruction.

### What the converter does
- Component discovery
  - Scans the pipeline directory for component subfolders by reading each `config.json` and classifying:
    - `UNet2DConditionModel` → `unet`
    - `AutoencoderKL` → `vae`
    - `CLIPTextModel*` → `text` (supports dual text encoders as `text` and `text2`)
  - Supports both single-file and sharded `.safetensors` via local `*.index.json`, and falls back to any `*.safetensors` in the folder.

- Tokenizer export (text encoder)
  - Writes tokenizer model type, tokens list (ordered by id), merges, and special token IDs (bos/eos/unk/sep/pad/mask) when present.
  - Uses `tokenizer/` for `text` and `tokenizer_2/` for `text2` automatically.

- Metadata export
  - Writes structured diffusion metadata keys under:
    - `diffusion.unet.*`: in/out channels, sample size, block layout, attention dims, time embedding details.
    - `diffusion.vae.*`: sample/latent channels, scaling factor, block layout.
    - `diffusion.text.*` and `diffusion.text2.*`: hidden size, layers, heads, positions, eps.
    - `diffusion.scheduler.*`: selected scheduler parameters.
  - Retains raw HF JSONs for full fidelity:
    - `diffusion.hf.model_index_json`
    - `diffusion.hf.scheduler_config_json`
    - `diffusion.hf.{component}_config_json`

- Output files
  - Stable, non-overwriting names in the output directory:
    - `text_encoder.gguf`
    - `text_encoder2.gguf` (if applicable)
    - `unet.gguf`
    - `vae.gguf`

### Files changed
- convert_hf_diff.py
  - Rewrote component discovery to be config-driven and shard-aware.
  - Added tokenizer serialization (tokens, merges, special IDs) and raw JSON embedding.
  - Added structured diffusion metadata (UNet/VAE/Text/Scheduler) as GGUF KVs.
  - Standardized output filenames to prevent overwrites.

- gguf-py/gguf/constants.py  (new diff system)
  - Added new model architectures:
    - `MODEL_ARCH.SD_UNET`, `MODEL_ARCH.SD_VAE`, `MODEL_ARCH.SD_TEXT`
  - Added corresponding string names in `MODEL_ARCH_NAMES`:
    - `sd_unet`, `sd_vae`, `sd_text`
  - Note: these changes are labeled with a comment `# new diff system`.

- tests/test_convert_pipeline.py
  - End-to-end test that downloads SDXL (via `download_sdxl.py`), runs the converter, and asserts presence and size of `text_encoder.gguf`, `unet.gguf`, `vae.gguf` (and `text_encoder2.gguf` when present).
  - Added print statements to describe progress and outcomes.

### How to run
- Convert a local snapshot directory:
```
python /home/beed1089/save_out/convert_hf_diff.py \
  /path/to/pipeline/snapshot \
  --outdir /path/to/out
```
- Example (HF cache snapshot):
```
python /home/beed1089/save_out/convert_hf_diff.py \
  ~/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/<SHA> \
  --outdir models/sdxl
```

### What’s not included (runtime)
- No C/C++ loader or kernels for diffusion are included. For inference, a runtime must:
  - Recognize `sd_unet`, `sd_vae`, `sd_text` architectures.
  - Read the `diffusion.*` KV metadata and bind canonical tensor names to UNet/VAE/Text kernels.
  - Implement/port a scheduler using the stored parameters.

### Extensibility
- Other Diffusers-based models (SD 1.5, SD 2.1, FLUX/flow models) can be supported by:
  - Extending the `classify` mapping for their component class names if needed.
  - Optionally adding family-specific KV fields for full topology/scheduler coverage.
  - Reusing the same shard-aware loading and tokenizer export.

### Notes
- The GGUF files today include sufficient data and metadata for lossless archiving and future runtime consumption.
- For llama.cpp-style deterministic loading, define canonical tensor name mappings per diffusion architecture in the runtime and wire them to kernels.
