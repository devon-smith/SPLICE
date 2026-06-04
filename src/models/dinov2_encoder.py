# thin wrapper over HF AutoModel / AutoImageProcessor for DINOv2 -- used by v0/v1
# all params frozen, fp16 autocast on GPU

import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv2Encoder:
    def __init__(self, model_id="facebook/dinov2-base", device=None,
                 autocast_dtype=torch.float16):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = autocast_dtype
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, pixel_values):
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        on_gpu = self.device == "cuda"
        with torch.autocast(device_type="cuda" if on_gpu else "cpu",
                            dtype=self.autocast_dtype, enabled=on_gpu):
            out = self.model(pixel_values=pixel_values)
        # transformers >=5 keeps pooler_output; fall back to CLS token
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]
        return pooled.float().cpu()
