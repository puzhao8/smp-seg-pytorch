import torch
from segformer import SegFormer_B0

model = SegFormer_B0(num_classes=2)
tensor = torch.randn(1, 3, 256, 256)
out = model(tensor)
print(out[0].shape)