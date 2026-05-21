"""Smoke test: load DINOv2, run a dummy forward pass, print shapes."""

import torch
from transformers import AutoImageProcessor, AutoModel

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model_id = "facebook/dinov2-base"
print(f"\nloading {model_id}...")
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

x = torch.randn(2, 3, 224, 224)
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()

with torch.no_grad():
    out = model(pixel_values=x)

print(f"\nCLS (pooler) shape: {out.pooler_output.shape}")
print(f"patch tokens shape: {out.last_hidden_state.shape}")
print("\nsmoke test passed")
