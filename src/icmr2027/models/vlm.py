"""Qwen2.5-VL inference with an interface ready for optional textual evidence."""
from typing import Protocol

from PIL import Image

from icmr2027.config import ModelConfig


class VLM(Protocol):
    def generate(self, image: Image.Image, question: str, context: str | None = None) -> str:
        ...


def build_prompt(question: str, context: str | None = None) -> str:
    if context is None:
        return f"Answer the question based on the image.\n\nQuestion: {question}\n\nGive a concise answer."
    return (
        "Use the image and the provided evidence to answer the question.\n\n"
        f"Evidence:\n{context}\n\nQuestion:\n{question}\n\nGive a concise answer."
    )


class HuggingFaceVLM:
    """Single-device Qwen2.5-VL adapter; checkpoint ID lives only in config."""

    def __init__(self, config: ModelConfig):
        # Lazy imports keep CPU unit tests independent of torch/Transformers.
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.config = config
        self.device = torch.device(config.device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Batch 1 supports only cpu or cuda devices")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable: install CUDA-enabled PyTorch on a GPU server")
            if config.dtype == "bfloat16":
                with torch.cuda.device(self.device):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("This GPU does not support bfloat16; set model.dtype: float16")
        options = {"revision": config.revision, "local_files_only": config.local_files_only}
        # PIL image processing avoids qwen-vl-utils. Transformers still requires
        # torchvision to initialize the Qwen2.5-VL video processor, even for images.
        image_options = {key: getattr(config, key) for key in ("min_pixels", "max_pixels")
                         if getattr(config, key) is not None}
        self.processor = AutoProcessor.from_pretrained(config.name, use_fast=False, **image_options, **options)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.name,
            torch_dtype=getattr(torch, config.dtype),
            device_map={"": str(self.device)},
            attn_implementation="sdpa",
            **options,
        )
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def generate(self, image: Image.Image, question: str, context: str | None = None) -> str:
        import torch

        prompt = build_prompt(question, context)
        if self.config.prompt_format == "ravenea_cvqa":
            from icmr2027.evaluation.ravenea import build_cvqa_prompt
            prompt = build_cvqa_prompt(question, context)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=True
        ).to(self.device)
        if self.config.max_input_tokens is not None and (
            inputs["input_ids"].shape[1] + self.config.max_new_tokens > self.config.max_input_tokens
        ):
            raise ValueError("Prompt plus generation budget exceeds max_input_tokens; no silent truncation")
        sampling = self.config.temperature > 0
        generation = {"max_new_tokens": self.config.max_new_tokens, "do_sample": sampling}
        if sampling:
            generation["temperature"] = self.config.temperature
        # Greedy generation at temperature=0; never pass temperature=0 to sampling.
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        continuation = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
