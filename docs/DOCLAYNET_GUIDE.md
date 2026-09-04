# DocLayNet → Florence-2 Picture Detection

## Why switch from AI2D + Synthetic?

| Old (AI2D + Synthetic) | New (DocLayNet) |
|------------------------|-----------------|
| Fake grey text rectangles | Real document pages |
| Diagrams pasted on blank pages | Human-annotated layouts |
| Source leakage / path bugs | Official train/val/test splits |
| Weak generalization to textbooks | Much closer to real books/manuals |

DocLayNet: **~80,863** human-annotated pages. Classes include **Picture**, Caption, Table, Text, etc.  
We train Florence on **Picture only** first.

Hugging Face: https://huggingface.co/datasets/docling-project/DocLayNet

---

## Phase A — Clean your project (run in PowerShell)

```powershell
cd "c:\coding\fun project\file reader work"

# 1) Remove failed / obsolete training zips (frees disk)
Remove-Item -Force training_data.zip, training_data_pilot.zip, training_data_clean.zip -ErrorAction SilentlyContinue
Remove-Item -Force colab_code.zip, fine_tuned_layout.zip -ErrorAction SilentlyContinue

# 2) Remove old AI2D/synthetic training folder if present
Remove-Item -Recurse -Force data\training_data -ErrorAction SilentlyContinue

# 3) Remove old LoRA experiments (optional — keeps BGE search model)
Remove-Item -Recurse -Force model\fine_tuned_layout -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force model\florence_layout -ErrorAction SilentlyContinue

# 4) Clear Python caches
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# KEEP these:
#   app.py, src\, scripts\, configs\, model\bge-large-en-v1.5\
#   data\uploads\, data\images\, data\extracted\
#   venv\, archive\  (old scripts for reference)
```

**Do NOT delete:** `data\uploads` (your PDFs), `model\bge-large-en-v1.5`, `venv`.

---

## Phase B — Install dependency (if needed)

```powershell
.\venv\Scripts\Activate.ps1
pip install -q datasets huggingface_hub pyyaml peft timm einops
```

---

## Phase C — Download & convert DocLayNet (on your PC)

### Pilot (recommended first — ~500 pages, ~15–30 min)

```powershell
.\venv\Scripts\python.exe scripts\dataset\build_doclaynet.py --pilot --clear
```

### Full serious set (~4000 train + 800 val)

```powershell
.\venv\Scripts\python.exe scripts\dataset\build_doclaynet.py --max-train 4000 --max-val 800 --clear
```

This writes:

```
data/training_data/images/doclaynet_*.png
data/training_data/dataset_manifest.jsonl
```

Only pages that contain a **Picture** box are kept.

---

## Phase D — Validate

```powershell
.\venv\Scripts\python.exe scripts\dataset\validate_dataset.py
```

Must print: `STATUS: PASS`

```powershell
.\venv\Scripts\python.exe scripts\dataset\inspect_dataset.py
```

---

## Phase E — Overfit test (CPU or Colab)

```powershell
.\venv\Scripts\python.exe scripts\training\overfit_test.py --fast
```

Want: `STATUS: PASS`

---

## Phase F — Train (Colab T4 — direct upload, no Drive)

### On PC — pack for Colab

```powershell
# After DocLayNet pilot or full build:
Compress-Archive -Path data\training_data -DestinationPath doclaynet_training.zip -Force
Compress-Archive -Path scripts,src,configs,requirements.txt -DestinationPath colab_code.zip -Force
```

If the overfit test ends with `AttributeError: 'NoneType' object has no attribute 'shape'`,
make a fresh `colab_code.zip` after the generation compatibility fix in
`src/layout/florence_model.py`, then replace the code archive in Kaggle. The loss values
printed before that error are still valid; the failure is in evaluation generation.

### In Kaggle (GPU T4)

**Cell 1 — GPU check**
```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

**Cell 2 — Upload code**
```python
print("Upload colab_code.zip with Kaggle's file upload control, then continue")
!mkdir -p /kaggle/working/florence_project
!unzip -q -o colab_code.zip -d /kaggle/working/florence_project/
```

**Cell 3 — Upload data**
```python
print("Upload doclaynet_training.zip with Kaggle's file upload control, then continue")
!mkdir -p /kaggle/working/florence_project/data
!unzip -q -o doclaynet_training.zip -d /kaggle/working/florence_project/data/
# If zip contained training_data/ at top level:
!ls /kaggle/working/florence_project/data/training_data/dataset_manifest.jsonl
```

If unzip nests oddly:
```python
!find /kaggle/working/florence_project -name dataset_manifest.jsonl
# then:
!mkdir -p /kaggle/working/florence_project/data && ln -sfn "$(dirname $(find /kaggle/working/florence_project -name dataset_manifest.jsonl | head -1))" /kaggle/working/florence_project/data/training_data
```

**Cell 4 — Install + validate + train**
```python
!pip install -q -r /kaggle/working/florence_project/requirements.txt
%cd /kaggle/working/florence_project
!python scripts/dataset/validate_dataset.py
!python scripts/training/overfit_test.py --fast
!python scripts/training/train_florence.py --config configs/florence_layout.yaml
```

Do not start full training unless the overfit test prints `STATUS: PASS`.

**Cell 5 — Download model**
```python
!zip -r florence_layout_best.zip model/florence_layout/best/
from IPython.display import FileLink
FileLink("florence_layout_best.zip")
```

Extract on PC to:
```
model\florence_layout\best\
```

---

## Phase G — Use in Smart Book Reader

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

The app loads `model/florence_layout/best/` automatically.

---

## Honest expectations

- DocLayNet has **scientific articles / manuals / reports**, not only school textbooks.
- It is still **far better** than synthetic grey bars for figure detection.
- For medical textbooks specifically, later add 200–500 of **your own annotated pages**.

## Commands summary

```text
1. Clean zips + old training_data
2. build_doclaynet.py --pilot --clear
3. validate_dataset.py
4. overfit_test.py --fast
5. Colab: upload code + data → train_florence.py
6. Download adapter → model/florence_layout/best/
7. streamlit run app.py
```
