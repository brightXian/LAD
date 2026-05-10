# Logit-Attention Divergence: Mitigating Position Bias in Multi-Image Retrieval via Attention-Guided Calibration

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)]()

---

## 🔥 News

- **2026-05-10** — 🚀 Code released
- **2026-05-08** — 📁 Repository created  
- **2026-04-30** — 🎉 Accepted to **ICML 2026 Main Track**

---

## 📖 Abstract

**Multimodal Large Language Models (MLLMs) have shown strong performance in multi-image cross-modal retrieval, yet suffer from severe position bias, where predictions are dominated by input order rather than semantic relevance. Through empirical analysis, we identify a phenomenon termed Logit-Attention Divergence, in which output logits are heavily biased while internal attention maps remain well-aligned with relevant visual evidence. This observation reveals a fundamental limitation of existing logit-level calibration methods such as PriDe. Based on this insight, we propose a training-free, attention-guided debiasing framework that leverages intrinsic attention signals for instance-level correction at inference time, requiring only a minimal calibration set with negligible computational overhead. Experiments on MS-COCO-based benchmarks show that our method substantially improves permutation invariance and achieves state-of-the-art performance, enhancing accuracy by over 40% compared to baselines.**

## ⚒️ TODO

- [x] Release code
- [ ] Release arXiv paper

---

## 🛠️ Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

## 📥 Download Models

```bash
hf download lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --local-dir models/LLaVA-OneVision-1.5-8B-Instruct

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2_5-VL-3B-Instruct

hf download OpenGVLab/InternVL3-8B-HF \
  --local-dir models/InternVL3-8B-HF
```

---

## 📦 Data Preparation

```bash
# 1. Download COCO val2014 images
curl -L -o data/val2014.zip http://images.cocodataset.org/zips/val2014.zip
unzip -q data/val2014.zip -d data

# 2. Extract CLIP features
uv run python data/extract_clip_features.py \
  --image_dir data/val2014 \
  --caption_file data/annotations_trainval/file_to_caption.pkl \
  --cache_output data/clip_cache.pkl

# 3. Build evaluation dataset
uv run python data/build_eval_dataset.py \
  --raw_img_dir data/val2014 \
  --clip_pkl data/clip_cache.pkl \
  --caption_pkl data/annotations_trainval/file_to_caption.pkl \
  --cat_pkl data/annotations_trainval/file_to_cat.pkl \
  --out_dir data/eval \
  --num_views 4 \
  --n_seq 1000
```

---

## 🚀 Evaluation

```bash
uv run python -m src.evaluate \
  --model-path models/Qwen2_5-VL-3B-Instruct \
  --method agd \
  --mode random \
  --num-views 4 \
  --num-shuffles 5 \
  --max-samples 1000

uv run python -m src.metric \
  --result results/eval_LLaVA-OneVision-1.5-8B-Instruct_agd_random_views4_shuffles5_samples1000.json
```
