# Stable Diffusion PyTorch Implementation

A lightweight PyTorch implementation of Stable Diffusion for text-to-image and image-to-image generation. This implementation focuses on inference and is designed to be easy to use while maintaining high-quality results.

## Features

* Text-to-Image generation
* Image-to-Image generation
* Support for CPU, CUDA, and MPS (Apple Silicon) devices
* Configurable inference parameters
* Built-in prompt templates for better results

## Requirements

* Python 3.8+
* PyTorch 2.0+
* Transformers
* Pillow
* Other dependencies in `requirements.txt`

1. Clone the repository:
```bash
git clone https://github.com/StarAtNyte/Stable-Diffusion-Pytorch.git
cd Stable-Diffusion-Pytorch
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required model files:

* Download `vocab.json` and `merges.txt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the `data` folder
* Download `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the `data` folder

## Directory Structure
```
Stable-Diffusion-Pytorch/
├── data/
│   ├── v1-5-pruned-emaonly.ckpt
│   ├── vocab.json
│   └── merges.txt
├── images/                    # For input/output images
├── src/
│   ├── inference.py     # Main inference script
│   ├── model_loader.py
│   └── attention.py
│   ├── clip.py
│   └── ddpm.py
│   ├── demo.ipynb
│   └── diffusion.py
│   ├── encoder.py
│   └── model_converter.py
│   └── pipeline.py
└── README.md
```

## Usage

You can use the demo.ipynb for inference or use inference.py from the command line.

### Basic Text-to-Image Generation

```bash
python src/inference.py \
    --prompt "A stunning mountain landscape at sunset, photorealistic, 8k resolution" \
    --output output.png
```


### Image-to-Image Generation

```bash
python src/inference.py \
    --prompt "A dog with sunglasses" \
    --image-path input.jpg \
    --strength 0.9 \
    --output output.png
```

### Advanced Options

```bash
python src/inference.py \
    --prompt "Your detailed prompt here" \
    --negative-prompt "Low quality, blurry, bad anatomy" \
    --cfg-scale 7.5 \
    --steps 50 \
    --seed 42 \
    --output custom_output.png
```

### Available Arguments

* `--prompt`: Text description of the desired image (required)
* `--negative-prompt`: Things to avoid in the generation
* `--image-path`: Input image for image-to-image generation
* `--strength`: Transformation strength for image-to-image (0.0 to 1.0)
* `--cfg-scale`: Classifier-free guidance scale (1.0 to 14.0)
* `--steps`: Number of inference steps (default: 50)
* `--seed`: Random seed for reproducibility
* `--output`: Output image path
* `--no-cuda`: Disable CUDA even if available
* `--allow-mps`: Allow MPS (Apple Silicon) acceleration


Example:
```bash
python src/inference.py \
    --prompt "portrait of a young woman with long flowing red hair, freckles, emerald green eyes, soft smile, natural lighting, shallow depth of field, shot on Canon EOS R5, 85mm f/1.2 lens, professional photography, 8k resolution, hyperrealistic detail" \
    --negative-prompt "blurry, low quality, distorted, deformed" \
    --cfg-scale 7.5 \
    --steps 50 \
    --output portrait.png
```



