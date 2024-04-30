import torch
<<<<<<< HEAD
from vision_mamba.model import Vim

# Create a random tensor
x = torch.randn(1, 3, 224, 224)
=======
from vision_mamba import Vim

# Forward pass
x = torch.randn(1, 3, 224, 224)  # Input tensor with shape (batch_size, channels, height, width)
>>>>>>> 9d8b929 ([CLEANUP])

# Create an instance of the Vim model
model = Vim(
<<<<<<< HEAD
    dim=256,  # Dimension of the model
    heads=8,  # Number of attention heads
    dt_rank=32,  # Rank of the dynamic routing tensor
    dim_inner=256,  # Inner dimension of the model
    d_state=256,  # State dimension of the model
    num_classes=1000,  # Number of output classes
    image_size=224,  # Size of the input image
    patch_size=16,  # Size of the image patch
    channels=3,  # Number of input channels
    dropout=0.1,  # Dropout rate
    depth=12,  # Depth of the model
)

# Perform a forward pass through the model
out = model(x)

# Print the shape and output of the forward pass
print(out.shape)
print(out)
=======
    dim=256,  # Dimension of the transformer model
    heads=8,  # Number of attention heads
    dt_rank=32,  # Rank of the dynamic routing matrix
    dim_inner=256,  # Inner dimension of the transformer model
    d_state=256,  # Dimension of the state vector
    num_classes=1000,  # Number of output classes
    image_size=224,  # Size of the input image
    patch_size=16,  # Size of each image patch
    channels=3,  # Number of input channels
    dropout=0.1,  # Dropout rate
    depth=12,  # Depth of the transformer model
)

# Forward pass
out = model(x)  # Output tensor from the model
print(out.shape)  # Print the shape of the output tensor
print(out)  # Print the output tensor
>>>>>>> 9d8b929 ([CLEANUP])
