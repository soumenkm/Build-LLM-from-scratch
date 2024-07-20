import torch, tqdm, tiktoken, sys, random, json, os, torchinfo
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent)) # Adds instruct_fine_tuning in the search list
sys.path.append(str(Path(__file__).parent.parent)) # Adds build-llm in the search list

from gpt_model.gpt import GPTModel
from gpt_model.load_openai_gpt2 import get_openai_gpt2_parameters
from typing import Tuple, List

class LoRALayer(torch.nn.Module):
    
    def __init__(
        self,
        rank: int,
        alpha: float,
        d_in: int,
        d_out: int
    ) -> None:
        
        super(LoRALayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."

        delta_W = torch.matmul(self.A, self.B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        
        return z

class LinearWithLoRA(torch.nn.Module):
    
    def __init__(
        self,
        linear: torch.nn.Linear,
        rank: int,
        alpha: float
    ) -> None:
        
        super(LinearWithLoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.linear = linear
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        
        self.lora = LoRALayer(rank=self.rank, 
                              alpha=self.alpha, 
                              d_in=self.d_in,
                              d_out=self.d_out)
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."

        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
    
        return z

class GPTModelWithLoRA(torch.nn.Module):
    
    def __init__(
        self,
        config: dict,
        pretrain_params: dict,
        lora_rank: int,
        lora_alpha: float
    ) -> None:
        
        super(GPTModelWithLoRA, self).__init__()
        self.config = config
        self.model = GPTModel(config=self.config)
        openai_gpt2_state_dict = get_openai_gpt2_parameters(self.model, pretrain_params)
        self.model.load_state_dict(state_dict=openai_gpt2_state_dict, strict=False)
    
        # Freeze the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace all the linear layers of model with LinearWithLora layers
        self.rank = lora_rank
        self.alpha = lora_alpha
        GPTModelWithLoRA.replace_linear_with_lora(model=self.model, 
                                                  rank=self.rank, 
                                                  alpha=self.alpha)
    
    @staticmethod
    def replace_linear_with_lora(
        model: torch.nn.Module,
        rank: int,
        alpha: float
    ) -> None:
        
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                linear_lora = LinearWithLoRA(module, rank, alpha)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                GPTModelWithLoRA.replace_linear_with_lora(module, rank, alpha)
       
    def calc_num_lora_params(self) -> None:
        
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
        
    def forward(
        self, 
        inputs: torch.tensor, 
        is_logits: bool=True
    ) -> torch.tensor:
        
        assert list(inputs.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        z = self.model(inputs, is_logits)
        
        return z
  