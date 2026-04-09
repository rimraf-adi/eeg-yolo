import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.yolo1d import yolo_1d_v11_n
from src.model.yolo2d import yolo_2d_v11_n
from src.training.train import yolo_loss

print("Instantiating generic YOLO1D structure...")
model_1d = yolo_1d_v11_n(in_channels=18, S=200)

# Fake Tensor input identically matching X: [Batch, Channels, Length]
dummy_x_1d = torch.randn(2, 18, 5000)
# Fake Tensor Target identically matching Y: [Batch, S, 5]
dummy_y = torch.zeros(2, 200, 5)

# Insert a dummy spatial object logically
dummy_y[0, 100, 0] = 1.0 # obj
dummy_y[0, 100, 1] = 0.5 # tx
dummy_y[0, 100, 3] = 1.0 # cls

print(f"Feeding shape X (1D): {dummy_x_1d.shape}")
preds_1d = model_1d(dummy_x_1d)
print(f"1D network cleanly yielded spatial mapping output preds: {preds_1d.shape}")

print("Testing boundary masking YOLO losses...")
l_obj, l_box, l_cls = yolo_loss(preds_1d, dummy_y)

print(f"Loss Obj: {l_obj.item():.4f}")
print(f"Loss Box: {l_box.item():.4f}")
print(f"Loss Cls: {l_cls.item():.4f}")

print("Instantiating generic YOLO2D structure...")
model_2d = yolo_2d_v11_n(in_channels=1, S=200, num_classes=3)
dummy_x_2d = torch.randn(2, 1, 18, 5000)
print(f"Feeding shape X (2D): {dummy_x_2d.shape}")
preds_2d = model_2d(dummy_x_2d)
print(f"2D network cleanly yielded spatial mapping output preds: {preds_2d.shape}")

print("Testing 2D boundary masking YOLO losses...")
l_obj_2d, l_box_2d, l_cls_2d = yolo_loss(preds_2d, dummy_y)
print(f"2D Loss Obj: {l_obj_2d.item():.4f}")
print(f"2D Loss Box: {l_box_2d.item():.4f}")
print(f"2D Loss Cls: {l_cls_2d.item():.4f}")
print("Compile success!")
