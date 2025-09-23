# from transformers.modeling_utils import Conv1D
from torch import nn

from src.model.lora_model_base import LoRAModelBase
from src.model.lora_sequential_model_base import LoRASequentialModelBase
from src.model.svd_lora_model_base import SVDLoRAModelBase
from src.model.svd_lora_sequential_model_base import SVDLoRASequentialModelBase
from src.model.ignoretopk_model_base import IgnoreTopKModelBase
from src.model.gpt2model import GPT2forGeneration


class GPT2LoRA(GPT2forGeneration, LoRAModelBase):
    def __init__(self, gpt2_config, lora_config, **cfg):
        super().__init__(**gpt2_config)
        self.init_lora(lora_config, **cfg)
        
    def check_module(self, name, module):
        return isinstance(module, nn.Conv1d) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2LoRASequential(GPT2forGeneration, LoRASequentialModelBase):
    def __init__(self, gpt2_config, lora_config, **cfg):
        super().__init__(**gpt2_config)
        self.init_lora_sequential(lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Conv1d) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2SVDLoRA(GPT2forGeneration, SVDLoRAModelBase):
    def __init__(self, gpt2_config, svd_lora_config, **cfg):
        super().__init__(**gpt2_config)
        self.init_svd_lora(svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Conv1d) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2SVDLoRASequential(GPT2forGeneration, SVDLoRASequentialModelBase):
    def __init__(self, gpt2_config, svd_lora_config, **cfg):
        super().__init__(**gpt2_config)
        self.init_svd_lora_sequential(svd_lora_config, **cfg)

    def check_module(self, name, module):
        return isinstance(module, nn.Conv1d) and name.split('.')[-1] in self.lora_config['target_layers']


class GPT2IgnoreTopK(GPT2forGeneration, IgnoreTopKModelBase):
    def __init__(self, gpt2_config, ignoretopk_config, output_hidden_states=True, **cfg):
        super().__init__(output_hidden_states=output_hidden_states, **gpt2_config)
        self.init_ignoretopk(**ignoretopk_config, **cfg)
        
    def forward(self, batch):
        outputs = GPT2forGeneration.forward(self, batch)
        return self.ignoretopk_on_forward(outputs)
