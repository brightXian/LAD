#!/usr/bin/env python
import argparse
import os
import pickle
import random
from collections.abc import Iterable

import clip  # type: ignore
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_image_paths(
    image_dir: str, extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".webp")
) -> list[str]:
    """Return sorted list of image file paths under ``image_dir`` (non-recursive)."""
    exts = tuple(e.lower() for e in extensions)
    names = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))
    return [os.path.join(image_dir, f) for f in names]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and cache CLIP features for images",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/val2014",
        help="Directory containing images",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default="data/annotations_trainval/file_to_caption.pkl",
        help="Path to caption file (pkl format)",
    )
    parser.add_argument(
        "--cache_output",
        type=str,
        default="data/clip_cache.pkl",
        help="Output path for CLIP cache",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        help="CLIP model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for CLIP encoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    return parser.parse_args()


def load_captions(caption_file: str) -> dict:
    """Load captions from pickle file."""
    with open(caption_file, "rb") as f:
        return pickle.load(f)


def prepare_clip_index(
    image_dir: str,
    caption_file: str,
    cache_output: str = "data/clip_cache.pkl",
    clip_model: str = "ViT-L/14",
    batch_size: int = 32,
) -> None:
    """
    Pre-process dataset: extract CLIP features.
    Run once and cache results.
    """
    # Check required files exist
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file not found: {caption_file}")

    print("Loading CLIP model ...")
    # Check for MPS (Apple Silicon), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model, preprocess = clip.load(clip_model, device=device)

    image_paths = load_image_paths(image_dir)

    # Load captions
    print("Loading captions ...")
    file_to_caption = load_captions(caption_file)

    # Filter image_paths that have captions
    valid_paths = [p for p in image_paths if os.path.basename(p) in file_to_caption]
    print(f"Found {len(valid_paths)} images with captions (total: {len(image_paths)})")

    # Extract CLIP features in batches
    print("Extracting CLIP image features ...")
    path_to_embeddings = {}

    for i in tqdm(range(0, len(valid_paths), batch_size), desc="Extracting batches"):
        batch_paths = valid_paths[i : i + batch_size]

        # Prepare batch images
        batch_images = []
        successful_paths = []
        for path in batch_paths:
            try:
                image = preprocess(Image.open(path))
                batch_images.append(image)
                successful_paths.append(path)
            except OSError as e:
                print(f"Failed to load {path}: {e}")
                continue

        if not batch_images:
            continue

        # Stack images into batch
        batch_tensor = torch.stack(batch_images).to(device)

        # Prepare batch captions
        batch_captions = [file_to_caption[os.path.basename(p)] for p in successful_paths]
        batch_text = clip.tokenize(batch_captions).to(device)

        # Encode batch
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor).cpu().numpy()
            text_features = model.encode_text(batch_text).cpu().numpy()

        # Normalize embeddings
        image_emb = image_features / np.linalg.norm(
            image_features,
            axis=-1,
            keepdims=True,
        )
        text_emb = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

        # Store results
        for j, path in enumerate(successful_paths):
            path_to_embeddings[path] = {
                "image_emb": image_emb[j],
                "text_emb": text_emb[j],
            }

    # Save cache
    print("Saving cache ...")
    cache_data = {"path_to_embeddings": path_to_embeddings}
    with open(cache_output, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Cache saved to {cache_output}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    prepare_clip_index(
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        cache_output=args.cache_output,
        clip_model=args.clip_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
