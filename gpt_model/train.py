import torch, torchinfo, json, sys, tiktoken
from gpt import GPTModel
from dataset import prepare_dataloader, train_val_split
from typing import List, Tuple

def text_to_token_ids(text: str, tokenizer: "tiktoken.tokenizer") -> torch.tensor:
    
    token_ids_list = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_ids_tensor = torch.tensor(token_ids_list).unsqueeze(0) # (1, T)
    
    return token_ids_tensor

def token_ids_to_text(token_ids: torch.tensor, tokenizer: "tiktoken.tokenizer") -> str:
    
    assert token_ids.shape[0] == 1, "token_ids.shape = (1, T)"
    assert list(token_ids.shape).__len__() == 2, "token_ids rank must be 2"
    
    token_ids = token_ids.squeeze(0) # (T,)
    text = tokenizer.decode(token_ids.tolist())
    
    return text
    
def generate_token_ids(start_context: str, model: GPTModel, tokenizer: "tiktoken.tokenizer", max_num_tokens: int=50) -> torch.tensor:

    inputs = text_to_token_ids(text=start_context, tokenizer=tokenizer) # (1, T)
    
    model.eval()
    for i in range(max_num_tokens):
        inputs = inputs[:, :model.Tmax] # (1, T) but at max (1, Tmax)
        
        with torch.no_grad():
            outputs = model(inputs, is_logits=False) # (1, T, V)
        
        max_prob_index = outputs.argmax(dim=-1, keepdim=False) # (1, T)
        last_token_pred = max_prob_index[:, -1].unsqueeze(-1) # (1, 1)  
        if last_token_pred[0,0] == tokenizer.eot_token:
            break
        
        inputs = torch.cat([inputs, last_token_pred], dim=-1) # (1, T+1)
    
    generated_token_ids = inputs # (1, T+Ti)
    
    return generated_token_ids

def calculate_loss_batch(batch: Tuple[torch.tensor, torch.tensor], model: GPTModel, device: "torch.device") -> torch.tensor:
    
    inputs, targets = batch # (b, T)
    inputs, targets = inputs.to(device), targets.to(device)
    model = model.to(device)
    
    pred_logits = model(inputs) # (b, T, V)
    pred_logits = pred_logits.flatten(start_dim=0, end_dim=1)
    targets = targets.flatten(start_dim=0, end_dim=1)
    
    loss = torch.nn.functional.cross_entropy(input=pred_logits, target=targets)
    
    return loss

def calculate_loss_dataloader(dataloader: "torch.dataloader", model: GPTModel, device: "torch.device", batch_fraction: float=0.2) -> float:
    
    num_batches = int(len(dataloader) * batch_fraction) # a small num of batches for speed
    
    loss_list = []
    for i, batch in enumerate(dataloader):
        
        loss_batch = calculate_loss_batch(batch=batch, model=model, device=device)
        loss_list.append(loss_batch.item())
        
        if i >= num_batches:
            break
    
    loss_dl = sum(loss_list)/len(loss_list)
    
    return loss_dl
        
# Set the device
if torch.cuda.is_available():
    device_type = "cuda"
    print("Using GPU...")
    print(f"Total # of GPU: {torch.cuda.device_count()}")
    print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
else:
    device_type = "cpu"
    print("Using CPU...")

device = torch.device(device_type)

BATCH_SIZE = 2
CONFIG = {
        "vocab_size": 50257,
        "max_context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1
}
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(config=CONFIG)

file_list = ["/Users/soumen/Desktop/IITB/sem2/LLM/Build-LLM-from-scratch/tokenizer/corpus/the-verdict.txt"]*2
train_text, val_text = train_val_split(file_list=file_list,
                                       train_val_split_ratio=0.8)

train_dl = prepare_dataloader(text=train_text, 
                              tokenizer=tokenizer,
                              max_context_length=CONFIG["max_context_length"],
                              batch_size=BATCH_SIZE,
                              is_train=True)

val_dl = prepare_dataloader(text=val_text, 
                            tokenizer=tokenizer,
                            max_context_length=CONFIG["max_context_length"],
                            batch_size=BATCH_SIZE,
                            is_train=False)

loss = calculate_loss_batch(batch=train_dl.__iter__().__next__(), model=model, device=device)
print(loss)
loss1 = calculate_loss_dataloader(dataloader=val_dl, model=model, device=device)
print(loss1)