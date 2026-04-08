import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.yolo1d import yolo_1d_v11_n
from src.training.train import yolo_loss

print("Instantiating generic YOLO1D structure...")
model = yolo_1d_v11_n(in_channels=18)

# Fake Tensor input identically matching X: [Batch, Channels, Length]
dummy_x = torch.randn(2, 18, 5024)
# Fake Tensor Target identically matching Y: [Batch, S, 1, 5]
dummy_y = torch.zeros(2, 100, 1, 5)

# Insert a dummy spatial object logically
dummy_y[0, 50, 0, 0] = 1.0 # obj
dummy_y[0, 50, 0, 1] = 0.5 # tx
dummy_y[0, 50, 0, 3] = 1.0 # cls

print(f"Feeding shape X: {dummy_x.shape}")
preds = model(dummy_x)
print(f"Network cleanly yielded spatial mapping output preds: {preds.shape}")

print("Testing boundary masking YOLO losses...")
l_obj, l_box, l_cls = yolo_loss(preds, dummy_y)

print(f"Loss Obj: {l_obj.item():.4f}")
print(f"Loss Box: {l_box.item():.4f}")
print(f"Loss Cls: {l_cls.item():.4f}")
print("Compile success!")
