"""Florence-2 training dataset and inference helpers."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.layout.bbox_utils import TASK_PROMPT, build_od_target
from src.layout.paths import resolve_image_path


class PictureDetectionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor: Any, manifest_dir):
        self.records = records
        self.processor = processor
        self.manifest_dir = manifest_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = resolve_image_path(rec["image_path"], self.manifest_dir)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image for record id={rec.get('id')} "
                f"path={rec.get('image_path')} dataset={rec.get('dataset')} "
                f"source={rec.get('source_id')}: {e}"
            ) from e

        w, h = image.size
        target_text = build_od_target(rec.get("annotations", []), w, h)
        inputs = self.processor(text=TASK_PROMPT, images=image, return_tensors="pt")
        labels = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
            return_token_type_ids=False,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0),
            "record_id": rec.get("id", str(idx)),
        }
        if "attention_mask" in inputs:
            item["attention_mask"] = inputs["attention_mask"].squeeze(0)
        return item


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    if "attention_mask" in batch[0]:
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    return out


@torch.no_grad()
def generate_detections(
    model: Any,
    processor: Any,
    image: Image.Image,
    device: str,
    max_new_tokens: int = 128,
    num_beams: int = 2,
) -> Dict[str, Any]:
    model.eval()
    inputs = processor(text=TASK_PROMPT, images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    # Florence-2 remote generate() breaks on newer transformers Cache/past_key_values.
    # Greedy decode with use_cache=False is the stable path for eval/overfit.
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": False,
        "num_beams": 1,
    }
    gen_ids = model.generate(**gen_kwargs)
    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        gen_text, task=TASK_PROMPT, image_size=image.size
    )
    od = parsed.get(TASK_PROMPT, {})
    return {
        "raw_text": gen_text,
        "bboxes": od.get("bboxes", []),
        "labels": od.get("labels", []),
    }
