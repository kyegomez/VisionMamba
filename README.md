[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vision Mamba
Implementation of Vision Mamba from the paper: "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" It's 2.8x faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on high-res images. 

[PAPER LINK](https://arxiv.org/abs/2401.09417)

## Installation

```bash
pip install vision-mamba
```

# Usage
```python
import torch
from vision_mamba import Vim

# Forward pass
x = torch.randn(1, 3, 224, 224)  # Input tensor with shape (batch_size, channels, height, width)

# Model
model = Vim(
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



```



## Citation
```bibtex
@misc{zhu2024vision,
    title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model}, 
    author={Lianghui Zhu and Bencheng Liao and Qian Zhang and Xinlong Wang and Wenyu Liu and Xinggang Wang},
    year={2024},
    eprint={2401.09417},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

# License
MIT



# Todo
- [ ] Create training script for imagenet
- [ ] Create a visual mamba for facial recognition