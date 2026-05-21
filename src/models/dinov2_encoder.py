"""Frozen DINOv2 encoder: a thin wrapper over the HF AutoModel / AutoImageProcessor.

v0 uses DINOv2 strictly as a fixed feature extractor -- all parameters are frozen
and inference runs under ``torch.inference_mode`` with fp16 autocast on GPU.
"""

import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv2Encoder:
    """Frozen DINOv2 backbone producing one pooled embedding per image."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: str | None = None,
        autocast_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = autocast_dtype
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed batch ``(B,3,H,W)`` -> ``(B, embedding_dim)`` float32 CPU."""
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        on_gpu = self.device == "cuda"
        with torch.autocast(
            device_type="cuda" if on_gpu else "cpu",
            dtype=self.autocast_dtype,
            enabled=on_gpu,
        ):
            out = self.model(pixel_values=pixel_values)
        # transformers >=5 keeps pooler_output; fall back to the CLS token otherwise.
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled.float().cpu()
