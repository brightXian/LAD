#!/usr/bin/env python
import argparse
import json
import os
import pickle
import random

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build evaluation dataset for position bias",
    )
    parser.add_argument("--num_views", type=int, default=4, help="Number of images per sample")
    parser.add_argument(
        "--n_seq",
        type=int,
        default=300,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--raw_img_dir",
        type=str,
        default="data/val2014",
        help="Directory containing raw images (must exist)",
    )
    parser.add_argument(
        "--clip_pkl",
        type=str,
        default="data/clip_cache.pkl",
        help="Path to CLIP embeddings cache",
    )
    parser.add_argument(
        "--cat_pkl",
        type=str,
        default="data/annotations_trainval/file_to_cat.pkl",
        help="Path to category pickle",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/eval",
        help="Output directory",
    )
    parser.add_argument(
        "--caption_pkl",
        type=str,
        default="data/annotations_trainval/file_to_caption.pkl",
        help="Path to caption pickle",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=20000,
        help="Pool size for sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat))


def sample_random(anchor_name: str, k: int, pool: list) -> list:
    """Random: sample k random images from pool excluding anchor."""
    candidates = [p for p in pool if p != anchor_name]
    if len(candidates) < k:
        raise ValueError(f"Pool too small: need {k} candidates, got {len(candidates)}")
    return random.sample(candidates, k)


def sample_adversarial(
    anchor_name: str,
    k: int,
    pool: list,
    path2emb: dict,
    file_to_cat_ids: dict,
    basename_to_path: dict,
    text_sim_threshold: float = 0.9,
) -> list:
    """Adversarial: sample images with high visual similarity but low text similarity."""
    anchor_path = basename_to_path[anchor_name]
    anchor_img_emb = path2emb[anchor_path]["image_emb"]
    anchor_text_emb = path2emb[anchor_path]["text_emb"]
    anchor_cats = file_to_cat_ids[anchor_name]

    candidates = []
    for p in pool:
        if p == anchor_name:
            continue
        p_cats = file_to_cat_ids[p]
        # Skip if categories are identical
        if not (anchor_cats - p_cats):
            continue

        p_path = basename_to_path[p]
        p_text_emb = path2emb[p_path]["text_emb"]
        p_image_emb = path2emb[p_path]["image_emb"]
        text_sim = cosine_sim(anchor_text_emb, p_text_emb)
        image_sim = cosine_sim(anchor_img_emb, p_image_emb)

        if text_sim >= text_sim_threshold:
            continue
        candidates.append((p, image_sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in candidates[:k]]


def main():
    args = parse_args()

    if args.num_views < 2:
        raise ValueError("num_views must be >= 2")

    if not os.path.isdir(args.raw_img_dir):
        raise FileNotFoundError(
            f"Image directory not found: {args.raw_img_dir}. Download COCO val2014 images first.",
        )

    image_per_sample = args.num_views

    # Check required files exist
    for f in [args.clip_pkl, args.caption_pkl, args.cat_pkl]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # Load pickle files
    print("Loading pickle files...")
    with open(args.clip_pkl, "rb") as f:
        path2emb = pickle.load(f)["path_to_embeddings"]

    with open(args.caption_pkl, "rb") as f:
        file_to_caption = pickle.load(f)

    with open(args.cat_pkl, "rb") as f:
        cat_data = pickle.load(f)
    file_to_cat_ids = cat_data["file_to_catIds"]

    # Filter valid paths - use path2emb keys directly (may be absolute paths)
    valid_paths = set(path2emb.keys())
    print(f"Valid images: {len(valid_paths)}")

    # Create mapping from basename to full path
    basename_to_path = {os.path.basename(p): p for p in valid_paths}

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # Prepare data - use basenames that have category info
    img_names = sorted(n for n in basename_to_path if n in file_to_cat_ids)
    print(f"Images with category info: {len(img_names)}")

    # Build pool
    pool_size = min(args.pool_size, len(img_names))
    pool = random.sample(img_names, pool_size)
    print(f"Pool size: {len(pool)}")

    # Build sequences for each mode
    for mode, sampler in [
        ("random", sample_random),
        ("adversarial", sample_adversarial),
    ]:
        sequences = []

        for seq_id in tqdm(range(args.n_seq), desc=f"Building {mode} sequences"):
            selected_names = random.sample(pool, image_per_sample)
            anchor_name = selected_names[0]
            anchor_path = basename_to_path[anchor_name]

            if mode == "random":
                negatives = sampler(anchor_name, k=image_per_sample - 1, pool=pool)
            else:
                negatives = sampler(
                    anchor_name,
                    k=image_per_sample - 1,
                    pool=pool,
                    path2emb=path2emb,
                    file_to_cat_ids=file_to_cat_ids,
                    basename_to_path=basename_to_path,
                )

            assert len(negatives) == image_per_sample - 1, (
                f"Expected {image_per_sample - 1} negatives, got {len(negatives)}"
            )

            negative_paths = [basename_to_path[p] for p in negatives]
            final_seq = [anchor_path, *negative_paths]
            random.shuffle(final_seq)
            true_idx_final = final_seq.index(anchor_path)
            caption = file_to_caption[anchor_name]

            sequences.append(
                {
                    "id": seq_id,
                    "image_ids": final_seq,
                    "index": true_idx_final,
                    "target": os.path.basename(anchor_path),
                    "caption": caption,
                },
            )

        out_json = os.path.join(
            args.out_dir,
            f"annotations_{args.num_views}_1_{mode}.json",
        )
        with open(out_json, "w") as f:
            json.dump(sequences, f, indent=2)
        print(f"{mode} sequences saved -> {out_json} ({len(sequences)} items)")

    print("All modes completed!")


if __name__ == "__main__":
    main()
