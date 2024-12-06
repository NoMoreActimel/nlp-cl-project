import torch
from torch import nn

from src.model.t5model import T5forSummarization

class SVDLoRA(nn.Module):
    def __init__(self, orig_module, calc_extra_loss, rank, **kwargs):
        super().__init__()
        # self.dropout = nn.Dropout(dropout_p)
        # self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        # self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        # self.rank = rank
        # self.alpha = alpha
        
        # weight - (out_features, in_features)
        self.u, self.s, self.vt = torch.linalg.svd(orig_module.weight.T, full_matrices=False) 
        # u - out_features, k
        # s - k
        # vt - k, in_features
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features
        self.k = self.s.shape[0]

        self.lora_u = nn.Parameter(self.u[:, -rank:], requires_grad=True) # out_features, rank
        self.lora_s = nn.Parameter(self.s[-rank:], requires_grad=True)  # rank, 
        self.lora_vt = nn.Parameter(self.vt[-rank:, :], requires_grad=True) # rank, in_features
        
        self.u = nn.Parameter(self.u[:, :-rank], requires_grad=False)
        self.s = nn.Parameter(self.s[:-rank], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-rank, :], requires_grad=False)

        self.register_buffer('orig_weight', self.u @ torch.diag(self.s) @ self.vt)
        # self.orig_weight = self.u @ torch.diag(self.s) @ self.vt

        self.extra_loss = torch.tensor(0., device=self.u.device)
        self.calc_extra_loss = calc_extra_loss
    
    def forward(self, x):
        # print('=='*10)
        # print("x.shape", x.shape)
        # print("self.u", self.u.shape)
        # print("self.s", self.s.shape)
        # print("self.vt", self.vt.shape)

        orig_pass = x @ self.orig_weight
        lora_pass = x @ self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt

        if self.calc_extra_loss:
            u_cat = torch.cat([self.u, self.lora_u], 1).to(self.u.device)
            vt_cat = torch.cat([self.vt, self.lora_vt], 0).to(self.u.device)
            u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(self.k, device=self.u.device, requires_grad=False))
            vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(self.k, device=self.u.device, requires_grad=False))
            self.extra_loss = u_norm + vt_norm

        return orig_pass + lora_pass


class T5SVDLoRA(T5forSummarization):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)
        
        for p in self.parameters():
            p.requires_grad = False

        self.count_adaptable_weights = 0
        self.svd_lora_config = svd_lora_config

        self.svd_loras = []
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = SVDLoRA(module, calc_extra_loss=True, **svd_lora_config)
                module.forward = self.add_lora_forward(module)
                self.svd_loras.append(module.lora)
                self.count_adaptable_weights += 2
        
        self.extra_loss = torch.tensor(0., device=self.model.device)
        print("Init loss:", self.calc_extra_loss())
        
    @staticmethod 
    def add_lora_forward(module):
        def new_forward(x):
            return module.lora(x)
        module.original_forward = module.forward
        return new_forward 
    
    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']

    def enable_extra_loss(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora.calc_extra_loss = True
    
    def disable_extra_loss(self):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora.calc_extra_loss = False
            
    def collect_extra_loss(self):
        extra_loss = torch.tensor(0., device=self.model.device)
        for lora in self.svd_loras:
            extra_loss = extra_loss + lora.extra_loss
            lora.extra_loss = torch.tensor(0., device=self.model.device)
        
        extra_loss = extra_loss / self.count_adaptable_weights
        return extra_loss

    def calc_extra_loss(self):
        res = 0
        cnt = 0
        for name, module in self.named_modules():
            if hasattr(module, 'lora'):
                mod = getattr(module, 'lora')
                u_cat = torch.cat([mod.u, mod.lora_u], 1)
                vt_cat = torch.cat([mod.vt, mod.lora_vt], 0)
                u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(mod.k, device=u_cat.device)) 
                vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(mod.k, device=u_cat.device)) 
                res += (u_norm + vt_norm)
                cnt += 2
        return (res / cnt)


class SVDLoRASequential(nn.Module):
    def __init__(self, orig_module, rank, n_adapters, **kwargs):
        super().__init__()
        # self.dropout = nn.Dropout(dropout_p)
        # self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        # self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        # self.rank = rank
        # self.alpha = alpha
        
        # weight - (out_features, in_features)
        self.u, self.s, self.vt = torch.linalg.svd(orig_module.weight.T, full_matrices=False) 
        # u - out_features, k
        # s - k
        # vt - k, in_features
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features
        self.k = self.s.shape[0]

        self.loras_u = []
        self.loras_s = []
        self.loras_vt = []

        for k in range(n_adapters):
            self.loras_u.append(nn.Parameter(self.u[:, -rank * (k + 1) : -rank * k], requires_grad=True)) # out_features, rank
            self.loras_s.append(nn.Parameter(self.s[-rank * (k + 1) : -rank * k], requires_grad=True))  # rank, 
            self.loras_vt.append(nn.Parameter(self.vt[-rank * (k + 1) : -rank * k, :], requires_grad=True)) # rank, in_features
        
        self.u = nn.Parameter(self.u[:, :-rank * n_adapters], requires_grad=False)
        self.s = nn.Parameter(self.s[:-rank * n_adapters], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-rank * n_adapters, :], requires_grad=False)

        # SELECT ADAPTER FOR THE CURRENT TASK
    
    def forward(self, x):
        # print('=='*10)
        # print("x.shape", x.shape)
        # print("self.u", self.u.shape)
        # print("self.s", self.s.shape)
        # print("self.vt", self.vt.shape)
        orig_pass = x @ self.u @ torch.diag(self.s) @ self.vt
        lora_pass = x @ self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt
        return orig_pass + lora_pass


class T5SVDLoRASequential(T5forSummarization):
    def __init__(self, t5_config, svd_lora_config, **cfg):
        super().__init__(**t5_config)

        self.current_adapter_idx = 0
        self.n_adapters = svd_lora_config['n_adapters']
        self.svd_lora_config = svd_lora_config
        
        for p in self.parameters():
            p.requires_grad = False

        self.svd_loras = [{} for _ in range(self.n_adapters)]
        
        for name, module in self.named_modules():
            if self.check_module(name, module):
                for i in range(self.n_adapters):
                    self.svd_loras[i][name] = SVDLoRA(module, **svd_lora_config)
        
        self.update_adapter(adapter_index=0)
    
        print("Init loss:", self.calc_extra_loss())


        # WRONG

    def update_adapter(self, adapter_index):
        for name, module in self.named_modules():
            if self.check_module(name, module):
                module.lora = self.svd_loras[adapter_index][name]
                module.forward = self.add_lora_forward(module)

    def check_module(self, name, module):
        return isinstance(module, nn.Linear) and name.split('.')[-1] in self.svd_lora_config['target_layers']

    def update_adapter_index(self, adapter_idx):
        if self.current_adapter_idx != adapter_idx:
            print(f"Changing to {adapter_idx+1}-th adapter!")
            self.update_adapter(adapter_idx)
        self.current_adapter_idx = adapter_idx

    @staticmethod
    def add_lora_forward(module):
        def new_forward(x):
            return module.original_forward(x) + module.lora(x)
        module.original_forward = module.forward
        return new_forward 

    def calc_extra_loss(self, ):
        res = 0
        cnt = 0
        for name, module in self.named_modules():
            if hasattr(module, 'lora'):
                mod = getattr(module, 'lora')
                u_cat = torch.cat([mod.u, mod.lora_u], 1)
                vt_cat = torch.cat([mod.vt, mod.lora_vt], 0)
                u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(mod.k, device=u_cat.device)) 
                vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(mod.k, device=u_cat.device)) 
                res += (u_norm + vt_norm)
                cnt += 2
        return (res / cnt)
