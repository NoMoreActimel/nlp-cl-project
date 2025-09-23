import torch
from torch import nn
# from transformers.modeling_utils import Conv1D


class SVDLoRA(nn.Module):
    def __init__(
            self,
            orig_module,
            enable_extra_loss,
            rank,
            reinit_lora=False,
            reinit_singular_values=False,
            reinit_std=1.0,
            reinit_singular_values_value=1.0,
            reinit_use_qr=True,
            reinit_ortho_project=True,
            **kwargs
    ):
        super().__init__()
        # self.dropout = nn.Dropout(dropout_p)
        # self.lora_down = nn.Linear(orig_module.in_features, rank, bias=False)
        # self.lora_up = nn.Linear(rank, orig_module.out_features, bias=False)
        # self.rank = rank
        # self.alpha = alpha

        if isinstance(orig_module, nn.Linear):
            self.in_features = orig_module.in_features
            self.out_features = orig_module.out_features
        elif isinstance(orig_module, nn.Conv1d):
            # For nn.Conv1d, weight is (out_channels, in_channels, kernel_size)
            # For kernel_size=1, this is equivalent to Linear(in_channels, out_channels)
            self.in_features = orig_module.weight.shape[1]  # in_channels
            self.out_features = orig_module.weight.shape[0]  # out_channels
        else:
            raise TypeError(f"SVDLoRA is not supported for module of type {type(orig_module)}")

        self.rank = rank
        self.reinit_lora = reinit_lora
        self.reinit_singular_values = reinit_singular_values
        self.reinit_std = reinit_std
        self.reinit_singular_values_value = reinit_singular_values_value
        self.reinit_use_qr = reinit_use_qr
        self.reinit_ortho_project = reinit_ortho_project

        self.init_with_weight(orig_module.weight)
        
        self.extra_loss = torch.tensor(0., device=self.u.device)
        self.enable_extra_loss = enable_extra_loss
    
    def calc_extra_loss(self):
        u_cat = torch.cat([self.u, self.lora_u], 1)
        vt_cat = torch.cat([self.vt, self.lora_vt], 0)
        u_norm = torch.norm(u_cat.T @ u_cat - torch.eye(self.k, device=self.u.device, requires_grad=False))
        vt_norm = torch.norm(vt_cat @ vt_cat.T - torch.eye(self.k, device=self.u.device, requires_grad=False))
        self.extra_loss = u_norm + vt_norm
    
    def clean_extra_loss(self):
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def lora_orthogonal_reinit(self):
        # Reinitialize with random gaussian weights
        with torch.no_grad():
            nn.init.normal_(self.lora_u, mean=0.0, std=self.reinit_std)
            nn.init.normal_(self.lora_vt, mean=0.0, std=self.reinit_std)
        
        if self.reinit_singular_values:
            nn.init.constant_(self.lora_s, self.reinit_singular_values_value)
        
        lora_u = self.lora_u
        lora_v = self.lora_vt.T
        
        # Project so that it's orthogonal to U and Vt
        if self.reinit_ortho_project:
            Iu = torch.eye(self.u.shape[0], dtype=self.u.dtype, device=self.u.device)
            Iv = torch.eye(self.vt.shape[1], dtype=self.vt.dtype, device=self.vt.device)
            lora_u = (Iu - self.u @ self.u.T) @ self.lora_u
            lora_v = (Iv - self.vt.T @ self.vt) @ self.lora_vt.T
        
        # QR decomposition so that vectors inside lora parts are orthogonal
        if self.reinit_use_qr:
            lora_u, _ = torch.qr(lora_u)
            lora_v, _ = torch.qr(lora_v)

        with torch.no_grad():
            self.lora_u.copy_(lora_u)
            self.lora_vt.copy_(lora_v.T)

    def init_with_weight(self, prev_weight):
        # weight - (out_features, in_features)
        self.u, self.s, self.vt = torch.linalg.svd(prev_weight.T, full_matrices=False) 
        # u - out_features, k
        # s - k
        # vt - k, in_features
        self.k = self.s.shape[0]

        self.lora_u = nn.Parameter(self.u[:, -self.rank:], requires_grad=True) # out_features, rank
        self.lora_s = nn.Parameter(self.s[-self.rank:], requires_grad=True)  # rank, 
        self.lora_vt = nn.Parameter(self.vt[-self.rank:, :], requires_grad=True) # rank, in_features
        
        self.u = nn.Parameter(self.u[:, :-self.rank], requires_grad=False)
        self.s = nn.Parameter(self.s[:-self.rank], requires_grad=False)
        self.vt = nn.Parameter(self.vt[:-self.rank, :], requires_grad=False)

        self.register_buffer('orig_weight', self.u @ torch.diag(self.s) @ self.vt)
        # self.orig_weight = self.u @ torch.diag(self.s) @ self.vt

        if self.reinit_lora:
            self.lora_orthogonal_reinit()

    def reinit_self(self):
        """ Recomputes rank least significant singular components """
        weight = self.orig_weight + self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt
        self.init_with_weight(weight)
        self.extra_loss = torch.tensor(0., device=self.u.device)
    
    def forward(self, x):
        # print('=='*10)
        # print("x.shape", x.shape)
        # print("self.u", self.u.shape)
        # print("self.s", self.s.shape)
        # print("self.vt", self.vt.shape)

        orig_pass = x @ self.orig_weight
        lora_pass = x @ self.lora_u @ torch.diag(self.lora_s) @ self.lora_vt

        if self.enable_extra_loss:
            self.calc_extra_loss()
        
        return orig_pass + lora_pass