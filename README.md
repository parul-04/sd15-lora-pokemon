# Stable Diffusion 1.5 + LoRA Fine-Tuning on Pokemon

Fine-tunes Stable Diffusion 1.5 with a LoRA adapter (via PEFT) on the Pokemon BLIP-captions dataset to generate custom Pokemon-style creature art. Trains only rank-8 LoRA weights injected into UNet attention layers while the base model stays frozen. Includes a before/after baseline comparison and training loss curve.

## Overview

This project uses **Low-Rank Adaptation (LoRA)** to efficiently fine-tune Stable Diffusion 1.5 on a Pokemon image/caption dataset. Instead of updating all ~860M UNet parameters, small low-rank adapter matrices are injected into the attention projection layers (`to_k`, `to_q`, `to_v`, `to_out.0`), and only those are trained. This keeps memory usage low enough to train on a single Colab T4 GPU while still noticeably shifting the model's output style toward Pokemon-style creature art.

## How it works

1. **Baseline generation** — the untouched base model generates one reference image, so you have a genuine before/after comparison once LoRA is trained.
2. **Freeze + inject** — VAE, text encoder, and UNet are frozen; a fresh LoRA adapter (`init_lora_weights="gaussian"`, so it starts as a no-op) is injected into the UNet's cross-/self-attention projections.
3. **Dataset** — [`svjack/pokemon-blip-captions-en-zh`](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh), an open, no-login-required mirror of the Pokemon BLIP-captions dataset (image + caption pairs), resized/cropped to 512×512.
4. **Training loop** — standard SD denoising-diffusion objective: encode images to latents, add noise, predict the noise with the UNet (LoRA-adapted), and minimize MSE against the true noise. Mixed precision (fp16 autocast + `GradScaler`) with LoRA params kept in fp32 for stable optimization.
5. **Generation** — after training, the same pipeline (now carrying the trained LoRA weights) generates new images from a text prompt.

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| LoRA rank | 8 (configurable: 4 / 8 / 16) |
| LoRA alpha | = rank (scaling factor of 1.0) |
| LoRA target modules | `to_k`, `to_q`, `to_v`, `to_out.0` |
| Trainable params | 1,594,368 |
| Optimizer | AdamW, lr = 1e-4 |
| Precision | fp16 autocast, `GradScaler` |
| Training steps | 100 |
| Batch size | 1 |
| Resolution | 512×512 |
| GPU used | Tesla T4 (Colab) |

## Results

- Training loss (MSE on predicted noise) drops from **0.0194** at step 1 to **0.0088** by step 100, with the usual noisy-but-decreasing trend typical of diffusion training on a small dataset (833 image/caption pairs) run for only 100 steps.
- The fine-tuned model, prompted with `"a monster pokemon creature, sci-fi style"`, produces stylized creature illustrations with the flat colors, bold outlines, and creature-design language typical of Pokemon artwork — a clear shift from the base model's photorealistic default style.

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA GPU with ≥15 GB VRAM (developed on a Colab T4)
- Optional: a Hugging Face account/token (read scope) for higher Hub rate limits

### Installation

```bash
git clone https://github.com/<your-username>/sd15-lora-pokemon.git
cd sd15-lora-pokemon
pip install -r requirements.txt
```

### Usage

Open and run `SDLoraPokemon.ipynb` in Jupyter, JupyterLab, or Google Colab (a T4 GPU runtime is recommended):

```bash
jupyter notebook SDLoraPokemon.ipynb
```

Run the cells in order:

1. Check GPU / install dependencies / verify versions
2. Load base SD 1.5 and generate a baseline image
3. (Optional) Log in to Hugging Face
4. Freeze the base model, inject a LoRA adapter, load the dataset
5. Run the training loop
6. Plot the training loss curve
7. Generate new images with a custom prompt

To sweep the LoRA rank, change `RANK` in the "freeze + inject" cell and re-run from that cell onward — no kernel restart needed.

## Project Structure

```
sd15-lora-pokemon/
├── SDLoraPokemon.ipynb   # Main notebook: setup, LoRA fine-tuning, generation
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Notes

- The original `runwayml/stable-diffusion-v1-5` repo was taken down; this notebook uses the `stable-diffusion-v1-5/stable-diffusion-v1-5` community mirror instead.
- `lambdalabs/pokemon-blip-captions` returns a `DatasetNotFoundError` (removed/gated on the Hub); this notebook uses the open `svjack/pokemon-blip-captions-en-zh` mirror instead.
- The safety checker is disabled (`safety_checker=None`) to avoid false-positive blanking of legitimate training/generation images — re-enable it if you plan to serve outputs publicly.
- If your Colab runtime restarts mid-session, reload the trained adapter with `pipe.load_lora_weights(f"lora-r{RANK}")` instead of retraining, provided you saved it to disk.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Note that the base Stable Diffusion 1.5 weights and the Pokemon dataset carry their own separate licenses on the Hugging Face Hub — review those before any commercial use.
