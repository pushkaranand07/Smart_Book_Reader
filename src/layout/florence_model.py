"""Florence-2 model loading, preflight checks, and LoRA configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.layout.bbox_utils import TASK_PROMPT
from src.layout.environment import collect_environment
from src.layout.paths import FLORENCE_LAYOUT_DIR, MODEL_DIR

BASE_MODEL_ID = "microsoft/Florence-2-base"


def discover_lora_targets(model: Any) -> List[str]:
    """Find attention projection modules in Florence language model."""
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred:
            found.add(leaf)
    return sorted(found) if found else preferred[:4]


def load_processor(model_id: str = BASE_MODEL_ID) -> Any:
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def _patch_florence2_attn_flags() -> None:
    """Florence-2 remote code lacks SDPA flags expected by newer transformers."""
    import sys

    for name, mod in list(sys.modules.items()):
        if "modeling_florence2" not in name:
            continue
        cls = getattr(mod, "Florence2ForConditionalGeneration", None)
        if cls is None:
            continue
        if not hasattr(cls, "_supports_sdpa"):
            cls._supports_sdpa = False
        if not hasattr(cls, "_supports_flash_attn_2"):
            cls._supports_flash_attn_2 = False


def _patch_florence2_generation_compatibility() -> None:
    """Handle Florence remote code that dereferences an empty generation cache."""
    import sys

    for name, mod in list(sys.modules.items()):
        if "modeling_florence2" not in name:
            continue
        cls = getattr(mod, "Florence2ForConditionalGeneration", None)
        if cls is None or getattr(cls, "_smart_book_reader_generation_patch", False):
            continue

        original = cls.prepare_inputs_for_generation

        def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
        ):
            if past_key_values is not None:
                return original(
                    self,
                    decoder_input_ids=decoder_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    use_cache=use_cache,
                    encoder_outputs=encoder_outputs,
                    **kwargs,
                )
            return {
                "input_ids": None,
                "encoder_outputs": encoder_outputs,
                "past_key_values": None,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,
            }

        cls.prepare_inputs_for_generation = prepare_inputs_for_generation
        cls._smart_book_reader_generation_patch = True


def load_base_model(model_id: str = BASE_MODEL_ID, device: str | None = None) -> Tuple[Any, str]:
    from transformers import AutoConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load remote config/code first so we can patch class attrs before model init.
    AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    _patch_florence2_attn_flags()
    _patch_florence2_generation_compatibility()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    _patch_florence2_attn_flags()
    _patch_florence2_generation_compatibility()
    model.to(device)
    return model, device


def attach_lora(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] | None = None,
) -> Any:
    from peft import LoraConfig, get_peft_model
    import inspect

    if target_modules is None:
        target_modules = discover_lora_targets(model)

    config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
        config_kwargs["ensure_weight_tying"] = True

    config = LoraConfig(
        **config_kwargs,
    )
    model = get_peft_model(model, config)

    # Freeze vision encoder
    for name, param in model.named_parameters():
        if "vision" in name.lower() or "image" in name.lower():
            param.requires_grad = False

    return model


def load_trained_model(
    adapter_dir: Path | str | None = None,
    base_model_id: str = BASE_MODEL_ID,
) -> Tuple[Any, Any, str]:
    adapter_dir = Path(adapter_dir or FLORENCE_LAYOUT_DIR / "best")
    processor = load_processor(base_model_id)
    model, device = load_base_model(base_model_id, None)

    if adapter_dir.is_dir() and (adapter_dir / "adapter_model.safetensors").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, processor, device


def count_parameters(model: Any) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "trainable_pct": round(100 * trainable / max(1, total), 4),
    }


def run_preflight(
    model_id: str = BASE_MODEL_ID,
    sample_image: Image.Image | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": "FAIL", "environment": collect_environment()}

    try:
        processor = load_processor(model_id)
        result["processor"] = "OK"

        model, device = load_base_model(model_id)
        result["model"] = "OK"
        result["device"] = device
        result["parameters"] = count_parameters(model)

        if sample_image is None:
            sample_image = Image.new("RGB", (400, 300), (240, 240, 240))

        inputs = processor(text=TASK_PROMPT, images=sample_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        dtype = next(model.parameters()).dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                use_cache=False,
            )

        gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            gen_text, task=TASK_PROMPT, image_size=sample_image.size
        )
        result["generation"] = "OK"
        result["parsing"] = "OK"
        result["sample_output_keys"] = list(parsed.keys())
        result["status"] = "PASS"
    except Exception as e:
        result["error"] = str(e)

    return result


def print_preflight(result: Dict[str, Any]) -> None:
    print("MODEL PREFLIGHT")
    print("=" * 40)
    for k in ("processor", "model", "device", "generation", "parsing", "status"):
        if k in result:
            print(f"  {k}: {result[k]}")
    if "parameters" in result:
        p = result["parameters"]
        print(f"  trainable: {p['trainable']:,} ({p['trainable_pct']}%)")
    if "error" in result:
        print(f"  error: {result['error']}")
    print("=" * 40)
