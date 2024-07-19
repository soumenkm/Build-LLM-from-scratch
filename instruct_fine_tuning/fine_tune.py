import torch, tqdm, tiktoken, sys, random, json, os
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent)) # Adds instruct_fine_tuning in the search list
sys.path.append(str(Path(__file__).parent.parent)) # Adds build-llm in the search list

from gpt_model.gpt import GPTModel
from gpt_model.load_openai_gpt2 import get_openai_gpt2_parameters
from gpt_model.gpt_download import download_and_load_gpt2
from typing import Tuple, List
from instruct_dataset import InstructDataset

class FineTuner:
    
    def __init__(
        self,
        device: torch.device,
        model: GPTModel,
        optimizer: torch.optim.Optimizer,
        tokenizer: "tiktoken.tokenizer",
        dataset: InstructDataset,
        batch_size: int
    ) -> None:
        
        super(FineTuner, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.train_dl, self.val_dl = self.dataset.prepare_dataloader(batch_size=batch_size)
           
    def generate_text(
        self,
        context: dict,
        temperature: float=0.7,
        max_num_tokens: int=50,
        top_k: int=50
    ) -> str:
        """context = Dict{instruction: str, input: "" or str}"""
        
        input_text = self.dataset._apply_alphaca_format(example=context, 
                                                        is_only_input=True)
        inputs = torch.tensor([self.tokenizer.encode(input_text)],
                              dtype=torch.int64,
                              device=self.device) # (1, T)
        if inputs.shape[1] > self.model.Tmax:
            inputs = inputs[1, :self.model.Tmax]
            print(f"Input token ids length exceeded max context length of {self.model.Tmax} tokens. " +
                  "Please interpret the output carefully.")
        
        T = inputs.shape[-1] # T
        self.model.eval()
        for i in range(max_num_tokens):
            with torch.no_grad():
                inputs = inputs[:, -self.model.Tmax:] # (1, T) but at max (1, Tmax) (discard the first few tokens)
                outputs = self.model(inputs, is_logits=True) # (1, T, V)
                outputs = outputs[0, -1, :] # (V,)
            
            top_k_limit = torch.sort(outputs, descending=True)[0][:top_k][-1] # scalar
            mask = torch.full_like(outputs, fill_value=-torch.inf) # (V,)
            outputs_mask = torch.where(condition=outputs<top_k_limit, input=mask, other=outputs) # (V,)
            outputs_mask = outputs_mask / temperature
            outputs_mask_prob = torch.nn.functional.softmax(input=outputs_mask, dim=-1) # (V,)
            
            last_multnom_index = torch.multinomial(input=outputs_mask_prob, num_samples=1) # (1,)
            last_token_pred = last_multnom_index.unsqueeze(-1) # (1, 1)  
            if last_token_pred[0,0] == self.tokenizer.eot_token:
                break
            
            inputs = torch.cat([inputs, last_token_pred], dim=-1) # (1, T+1)
        
        generated_token_ids = inputs[0, T:] # (Ti,)
        generated_text = self.tokenizer.decode(generated_token_ids.tolist())
        
        return generated_text

def main():
    
    current_dir = Path.cwd()
    settings, params = download_and_load_gpt2(model_size="1558M", 
                                              models_dir=Path(current_dir,"gpt_model","openai_gpt2"), 
                                              is_download=False)

    CONFIG = {
        "vocab_size": settings["n_vocab"],
        "max_context_length": settings["n_ctx"],
        "emb_dim": settings["n_embd"],
        "n_heads": settings["n_head"],
        "n_layers": settings["n_layer"],
        "drop_rate": 0.1,
        "qkv_bias": True
    }
    
    model = GPTModel(config=CONFIG)
    openai_gpt2_state_dict = get_openai_gpt2_parameters(model, params)
    model.load_state_dict(state_dict=openai_gpt2_state_dict, strict=False)
    
    # Freeze the parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the output layer, last norm layer and last transformer layer
    for param in model.output_layer.parameters():
        param.requires_grad = True
    for param in model.layernorm.parameters():
        param.requires_grad = True
    for param in model.transformer_layers[-1].parameters():
        param.requires_grad = True  
        
    # Get the dataloader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}...")
    device = torch.device(device=device)   
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructDataset(dataset_path=Path(current_dir, "instruct_fine_tuning/data/instruct_data.json"),
                              tokenizer=tokenizer,
                              device=device,
                              max_context_length=model.Tmax)

    # Train the model
    optimizer = torch.optim.AdamW(params=model.parameters())
    
    # Get the finetuner
    finetuner = FineTuner(device=device,
                          model=model,
                          optimizer=optimizer,
                          tokenizer=tokenizer,
                          dataset=dataset,
                          batch_size=16)
    
    # Generate text
    context = {
        "instruction": "Who is prime minister of India?",
        "input": "" 
    }
    gen_text = finetuner.generate_text(context=context)
    print(gen_text)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
    main()
