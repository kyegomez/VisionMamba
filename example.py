<<<<<<< HEAD
"""Example of using the VisionMambaBlock model"""
# Import the necessary libraries
import torch
from vision_mamba.model import VisionMambaBlock
=======
import torch 
from vision_mamba import Vim
# Forward pass
x = torch.randn(1, 3, 224, 224)
>>>>>>> 28048af ([VIM])

# Model
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

# Forward pass
out = model(x)
print(out.shape)
print(out)
