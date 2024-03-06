import torch
from vision_mamba.model import Vim

# Create a random tensor
x = torch.randn(1, 3, 224, 224)

# Create an instance of the Vim model
model = Vim(
    dim=256,
    heads=8,
    dt_rank=32,
    dim_inner=256,
    d_state=256,
    num_classes=1000,
    image_size=224,
    patch_size=16,
    channels=3,
    dropout=0.1,
    depth=12,
)

# Perform a forward pass through the model
out = model(x)

# Print the shape and output of the forward pass
print(out.shape)
print(out)
