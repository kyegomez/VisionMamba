"""Example of using the VisionMambaBlock model"""
# Import the necessary libraries
import torch
from vision_mamba.model import VisionMambaBlock

# Create a random tensor of shape (1, 512, 512)
x = torch.randn(1, 512, 1024)

# Create an instance of the VisionMambaBlock model
# Parameters:
# - dim: The input dimension
# - heads: The number of attention heads
# - dt_rank: The rank of the dynamic tensor
# - dim_inner: The inner dimension of the model
# - d_state: The state dimension of the model
model = VisionMambaBlock(
    dim=1024, heads=8, dt_rank=32, dim_inner=512, d_state=256
)

# Pass the input tensor through the model
out = model(x)

# Print the shape of the output tensor
print(out)
