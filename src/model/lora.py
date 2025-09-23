from torch import nn
import torch.nn.init as init
# from transformers.modeling_utils import Conv1D

class LoRA(nn.Module):
    def __init__(self, orig_module, rank, alpha, dropout_p, init_std=0.01, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        
        if isinstance(orig_module, nn.Linear):
            in_features = orig_module.in_features
            out_features = orig_module.out_features
        elif isinstance(orig_module, nn.Conv1d):
            # For nn.Conv1d, weight is (out_channels, in_channels, kernel_size)
            # For kernel_size=1, this is equivalent to Linear(in_channels, out_channels)
            in_features = orig_module.weight.shape[1]  # in_channels
            out_features = orig_module.weight.shape[0]  # out_channels
        else:
            raise TypeError(f"LoRA is not supported for module of type {type(orig_module)}")

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        init.normal_(self.lora_down.weight, mean=0, std=init_std)

        self.lora_up = nn.Linear(rank, out_features, bias=False)
        init.zeros_(self.lora_up.weight)
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.lora_down(x)
        x = self.lora_up(x)
        x = (self.alpha / self.rank) * x
        return x
