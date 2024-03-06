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
- [ ] Fix the encoder block with the forward and backward convolutions
- [ ] Make a training script for imagenet