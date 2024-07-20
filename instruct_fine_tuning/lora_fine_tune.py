import torch, tqdm, tiktoken, sys, random, json, os, torchinfo
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent)) # Adds instruct_fine_tuning in the search list
sys.path.append(str(Path(__file__).parent.parent)) # Adds build-llm in the search list

from lora_model import GPTModelWithLoRA
from gpt_model.gpt_download import download_and_load_gpt2
from typing import Tuple, List
from instruct_dataset import InstructDataset
           
class LoRAFineTuner:
    
    def __init__(
        self,
        device: torch.device,
        model: GPTModelWithLoRA,
        optimizer: torch.optim.Optimizer,
        tokenizer: "tiktoken.tokenizer",
        dataset: InstructDataset,
        batch_size: int,
        num_epochs: int
    ) -> None:
        
        super(LoRAFineTuner, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_dl, self.val_dl = self.dataset.prepare_dataloader(batch_size=self.batch_size)
        self.num_epochs = num_epochs
           
    def generate_text(
        self,
        context: dict,
        temperature: float=0.7,
        max_num_tokens: int=50,
        top_k: int=50
    ) -> str:
        """context = Dict{instruction: str, input: "" or str}"""
        
        Tmax = self.model.model.Tmax
        input_text = self.dataset._apply_alphaca_format(example=context, 
                                                        is_only_input=True)
        inputs = torch.tensor([self.tokenizer.encode(input_text)],
                              dtype=torch.int64,
                              device=self.device) # (1, T)
        if inputs.shape[1] > Tmax:
            inputs = inputs[1, :Tmax]
            print(f"Input token ids length exceeded max context length of {Tmax} tokens. " +
                  "Please interpret the output carefully.")
        
        T = inputs.shape[-1] # T
        self.model.eval()
        for i in range(max_num_tokens):
            with torch.no_grad():
                inputs = inputs[:, -Tmax:] # (1, T) but at max (1, Tmax) (discard the first few tokens)
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

    def _forward_batch(
        self,
        batch_x: torch.tensor # (b, T)
    ) -> torch.tensor:
        
        pred_y = self.model(batch_x) # (b, T, d)
        
        return pred_y 
    
    def _calc_loss(
        self,
        pred_y: torch.tensor, # (b, T, d)
        true_y: torch.tensor # (b, T)
    ) -> torch.tensor:
        
        pred_y = pred_y.flatten(start_dim=0, end_dim=1) # (b*T, d)
        true_y = true_y.flatten(start_dim=0, end_dim=1) # (b*T,)
        loss = torch.nn.functional.cross_entropy(input=pred_y, target=true_y, ignore_index=-100)
        
        return loss # returns the entire computational graph!
    
    def _train_batch(
        self,
        batch: Tuple[torch.tensor, torch.tensor]
    ) -> float:
        
        train_x, train_y = batch
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)
        
        pred_y = self._forward_batch(batch_x=train_x)
        loss = self._calc_loss(pred_y=pred_y, true_y=train_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self
    ) -> None:
        
        self.model.train()
        train_loss = []
        for ep in range(self.num_epochs):
            with tqdm.tqdm(iterable=self.train_dl, 
                        desc=f"Training Epoch: {ep}/{self.num_epochs-1}",
                        total=len(self.train_dl),
                        unit=" batch",
                        colour="green") as pbar:
                
                for batch in pbar:
                    loss = self._train_batch(batch=batch)
                    train_loss.append(loss)
                    pbar.set_postfix(loss=f"{loss:.3f}")
            
            # Generate text
            context = {
                "instruction": "What will be the result of 32 + 42?",
                "input": "" 
            }
            gen_text = self.generate_text(context=context)
            print(gen_text)
                    
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
    
    model = GPTModelWithLoRA(config=CONFIG, pretrain_params=params, lora_alpha=16, lora_rank=8)
    model.calc_num_lora_params()
        
    # Get the dataloader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}...")
    device = torch.device(device=device)   
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructDataset(dataset_path=Path(current_dir, "instruct_fine_tuning/data/instruct_data.json"),
                              tokenizer=tokenizer,
                              device=device,
                              max_context_length=model.model.Tmax)

    # Train the model
    optimizer = torch.optim.AdamW(params=model.parameters())
    
    # Get the finetuner
    finetuner = LoRAFineTuner(device=device,
                              model=model,
                              optimizer=optimizer,
                              tokenizer=tokenizer,
                              dataset=dataset,
                              batch_size=16,
                              num_epochs=5)
    
    finetuner.train()
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
    main()
