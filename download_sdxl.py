import argparse
from pathlib import Path
from typing import List

from diffusers import StableDiffusionXLPipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Download Stable Diffusion XL safetensors without instantiating the model")
    p.add_argument("--repo", default="stabilityai/stable-diffusion-xl-base-1.0", help="HuggingFace repo id")
    p.add_argument("--revision", default=None, help="Git revision / branch / tag")
    p.add_argument("--outdir", default="models/sdxl_raw", help="Output directory for downloaded files")
    args = p.parse_args()

    out_dir = Path(args.outdir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # We only need to fetch and save model locally; diffusers handles caching
    print(f"Downloading {args.repo} (revision={args.revision}) via diffusers to {out_dir} …")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.repo,
        revision=args.revision,
        torch_dtype="float16",
        variant="fp16",
    )
    pipe.save_pretrained(out_dir)
    print("✓ download completed. Exiting before inference.")


if __name__ == "__main__":
    main()
