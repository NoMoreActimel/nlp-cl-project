from src.model.adapter_model_base import AdapterModelBase
from src.model.lora import LoRA

class LoRAModelBase(AdapterModelBase):
    def init_lora(self, lora_config, **cfg):
        for p in self.parameters():
            p.requires_grad = False
        
        self.lora_config = lora_config
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = LoRA(module, **lora_config)
                module.forward = self.add_lora_forward(module)
