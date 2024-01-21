[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vision Mamba
Implementation of Vision Mamba from the paper: "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" It's 2.8x faster than DeiT and saves 86.8% GPU memory when performing batch inference to extract features on high-res images



## Installation

You can install the package using pip

```bash
pip install vision-mamba
```

# Usage
```python

# Import the necessary libraries
import torch
from vision_mamba.model import VisionMambaBlock

# Create a random tensor of shape (1, 512, 512)
x = torch.randn(1, 512, 512)

# Create an instance of the VisionMambaBlock model
# Parameters:
# - dim: The input dimension
# - heads: The number of attention heads
# - dt_rank: The rank of the dynamic tensor
# - dim_inner: The inner dimension of the model
# - d_state: The state dimension of the model
model = VisionMambaBlock(
    dim=512, heads=8, dt_rank=32, dim_inner=512, d_state=256
)

# Pass the input tensor through the model
out = model(x)

# Print the shape of the output tensor
print(out)

```



### Code Quality 🧹

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`

### Tests 🧪

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.


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


