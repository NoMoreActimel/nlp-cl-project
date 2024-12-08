from torch import nn

from src.model.t5_adapter_model_base import T5AdapterBase

class LoRA(nn.Module):
    def __init__(self, orig_module, rank, alpha, dropout_p, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        self.rank = rank
        self.alpha = alpha
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.lora_down(x)
        x = self.lora_up(x)
        x = (self.alpha / self.rank) * x
        return x

class T5LoRA(T5AdapterBase):
    def __init__(self, t5_config, lora_config, **cfg):
        super().__init__(**t5_config)
        
        for p in self.parameters():
            p.requires_grad = False
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = LoRA(module, **lora_config)
                module.forward = self.add_lora_forward(module)
        
    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.lora_config['target_layers']
