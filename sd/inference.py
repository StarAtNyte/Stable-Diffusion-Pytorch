import argparse
import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import model_loader
import pipeline

MODEL_PATH = "../data/v1-5-pruned-emaonly.ckpt"
VOCAB_PATH = "../data/vocab.json"
MERGES_PATH = "../data/merges.txt"

def get_device(allow_cuda=True, allow_mps=False):
    if torch.cuda.is_available() and allow_cuda:
        return "cuda"
    elif (hasattr(torch, 'has_mps') or torch.backends.mps.is_available()) and allow_mps:
        return "mps"
    return "cpu"

def load_models(device):
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)
    
    print("Loading model...")
    models = model_loader.preload_models_from_standard_weights(MODEL_PATH, device)
    
    return tokenizer, models

def generate_image(args, models, tokenizer, device):
    # Load input image if path is provided
    input_image = None
    if args.image_path:
        try:
            input_image = Image.open(args.image_path)
            print(f"Loaded input image from {args.image_path}")
        except Exception as e:
            print(f"Error loading input image: {e}")
            return None

    # Generate the image
    output_image = pipeline.generate(
        prompt=args.prompt,
        uncond_prompt=args.negative_prompt,
        input_image=input_image,
        strength=args.strength,
        do_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler,
        n_inference_steps=args.steps,
        seed=args.seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    return Image.fromarray(output_image)

def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inference Script")
    
    # Device settings
    parser.add_argument("--no-cuda", action="store_true",
                      help="Disable CUDA even if available")
    parser.add_argument("--allow-mps", action="store_true",
                      help="Allow MPS (Apple Silicon) acceleration")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                      help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default="",
                      help="Negative prompt for generation")
    parser.add_argument("--image-path", type=str,
                      help="Path to input image for img2img generation")
    parser.add_argument("--strength", type=float, default=0.9,
                      help="Strength for img2img generation (0.0 to 1.0)")
    parser.add_argument("--use-cfg", action="store_true", default=True,
                      help="Use classifier-free guidance")
    parser.add_argument("--cfg-scale", type=float, default=8.0,
                      help="Classifier-free guidance scale (1.0 to 14.0)")
    parser.add_argument("--sampler", type=str, default="ddpm",
                      help="Sampler to use for generation")
    parser.add_argument("--steps", type=int, default=50,
                      help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for generation")
    parser.add_argument("--output", type=str, default="output.png",
                      help="Output image path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up device
    device = get_device(allow_cuda=not args.no_cuda, allow_mps=args.allow_mps)
    print(f"Using device: {device}")
    
    # Load models and tokenizer
    tokenizer, models = load_models(device)
    
    # Generate image
    output_image = generate_image(args, models, tokenizer, device)
    
    if output_image:
        output_image.save(args.output)
        print(f"Generated image saved to {args.output}")
    else:
        print("Failed to generate image")

if __name__ == "__main__":
    main()