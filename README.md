# LAD

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Download Models

```bash
hf download lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --local-dir models/LLaVA-OneVision-1.5-8B-Instruct

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2_5-VL-3B-Instruct

hf download OpenGVLab/InternVL3-8B-HF \
  --local-dir models/InternVL3-8B-HF
```

## Download & Preprocess Data

### 1. Download COCO val2014 images

Caption and category pickles are already in `data/annotations_trainval/`.

```bash
curl -L -o data/val2014.zip http://images.cocodataset.org/zips/val2014.zip
unzip -q data/val2014.zip -d data
```

### 2. Extract CLIP features

```bash
uv run python data/extract_clip_features.py \
  --image_dir data/val2014 \
  --caption_file data/annotations_trainval/file_to_caption.pkl \
  --cache_output data/clip_cache.pkl
```

### 3. Build evaluation dataset

```bash
uv run python data/build_eval_dataset.py \
  --raw_img_dir data/val2014 \
  --clip_pkl data/clip_cache.pkl \
  --caption_pkl data/annotations_trainval/file_to_caption.pkl \
  --cat_pkl data/annotations_trainval/file_to_cat.pkl \
  --out_dir data/eval \
  --num_views 4 \
  --n_seq 1000
```

## Run

```bash
CUDA_VISIBLE_DEVICES=7 uv run python -m src.evaluate \
  --model-path models/Qwen2_5-VL-3B-Instruct \
  --method agd \
  --mode random \
  --num-views 4 \
  --num-shuffles 5 \
  --max-samples 1000
```

## Metrics

```bash
uv run python -m src.metric \
  --result results/eval_LLaVA-OneVision-1.5-8B-Instruct_agd_random_views4_shuffles5_samples1000.json
```
