# Florence-2 Picture Detection Training Pipeline (DocLayNet)

## Architecture

```
scripts/dataset/     → build_doclaynet.py, validate, inspect
scripts/training/    → train_florence.py, overfit_test.py
scripts/evaluation/  → evaluate_florence.py, visualize_predictions.py
scripts/inference/   → detect_figures.py
src/layout/          → core library (bbox, model, metrics)
configs/             → florence_layout.yaml
model/florence_layout/ → trained checkpoints (best/, latest/, epoch_NNN/)
```

**Training objective:** Picture-only detection via Florence-2 `<OD>` task on **DocLayNet**.

## Dataset

| Source | [ds4sd/DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet) on Hugging Face |
|--------|-------------------------------------------------------------------------------------|
| Pages  | 80,863 real document pages (financial, scientific, manuals, patents, …)             |
| Labels | 11 layout classes — we keep only **Picture**                                        |
| Splits | Official train / val / test (no synthetic leakage)                                  |

Old AI2D + SyntheticLayout data is retired. Do not re-extract `training_data.zip`.

## Prerequisites

```powershell
cd "c:\coding\fun project\file reader work"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Needs internet access the first time (Hugging Face download). A GPU (Colab T4+) is strongly recommended for full training; CPU is fine for `--pilot` smoke tests only.

## Step-by-step

### 1. Load DocLayNet and convert to our manifest format

**Pilot (recommended first — ~500 Picture pages):**

```powershell
.\venv\Scripts\python.exe scripts\dataset\build_doclaynet.py --pilot --clear
```

**Full training set (default caps: 30,000 train / 10,000 val / 5,000 test):**

```powershell
.\venv\Scripts\python.exe scripts\dataset\build_doclaynet.py --custom-split --include-negative --clear --max-train 30000 --max-val 10000 --max-test 5000
```

This writes:

- `data/training_data/images/doclaynet_*.png`
- `data/training_data/dataset_manifest.jsonl`

### 2. Validate (required)

```powershell
.\venv\Scripts\python.exe scripts\dataset\validate_dataset.py
```

Must print `STATUS: PASS`. Training refuses to start otherwise.

### 3. Inspect composition

```powershell
.\venv\Scripts\python.exe scripts\dataset\inspect_dataset.py
```

### 4. Model preflight

```powershell
.\venv\Scripts\python.exe scripts\training\train_florence.py --preflight
```

### 5. Tiny overfit test (required before full training)

```powershell
.\venv\Scripts\python.exe scripts\training\overfit_test.py --fast
```

Loss must drop and localization must pass. If FAIL, do not start full training.

### 6. Train

**Google Colab / GPU:**

```python
!pip install -q pyyaml peft timm einops datasets accelerate torchvision
!python scripts/dataset/build_doclaynet.py --custom-split --include-negative --clear --max-train 30000 --max-val 10000 --max-test 5000
!python scripts/dataset/validate_dataset.py
!python scripts/training/train_florence.py --config configs/florence_layout.yaml
```

**Local CPU (pilot only — very slow):**

```powershell
.\venv\Scripts\python.exe scripts\training\train_florence.py --config configs\florence_layout.yaml
```

### 7. Evaluate + visualize

```powershell
.\venv\Scripts\python.exe scripts\evaluation\evaluate_florence.py
.\venv\Scripts\python.exe scripts\evaluation\visualize_predictions.py
```

### 8. Run detection on one page

```powershell
.\venv\Scripts\python.exe scripts\inference\detect_figures.py --image path\to\page.png
```

## Configuration

Edit `configs/florence_layout.yaml`:

- `data.manifest: data/training_data/dataset_manifest.jsonl`
- `learning_rate: 1.0e-6`
- `batch_size: 2`, `gradient_accumulation: 4` → effective batch 8
- `epochs: 2` for first GPU experiment

## Output locations

| Artifact | Path |
|----------|------|
| Best model | `model/florence_layout/best/` |
| Latest model | `model/florence_layout/latest/` |
| Training runs | `runs/YYYYMMDD_HHMMSS/` |
| Eval metrics | `evaluation/metrics.json` |
| Visualizations | `evaluation/visualizations/` |
