import torch, torchinfo, json, sys, tiktoken, pathlib, random, math
from gpt import GPTModel
from dataset import prepare_dataloader, train_val_split
from typing import List, Tuple
from tqdm import tqdm

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
    
def generate_text(start_context: str, model: GPTModel, tokenizer: "tiktoken.tokenizer", device: "torch.device", max_num_tokens: int=50, temperature: float=0.7, top_k: int=50) -> str:

    inputs = text_to_token_ids(text=start_context, tokenizer=tokenizer) # (1, T)
    model = model.to(device)
    
    model.eval()
    for i in range(max_num_tokens):
        inputs = inputs[:, :model.Tmax] # (1, T) but at max (1, Tmax)
        
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs, is_logits=True) # (1, T, V)
        
        top_k_limit = torch.sort(outputs, dim=-1, descending=True)[0][:,:,-1].unsqueeze(-1) # (1, T, 1)
        mask = torch.full_like(outputs, fill_value=-torch.inf) # (1, T, V)
        outputs_mask = torch.where(condition=outputs<top_k_limit, input=mask, other=outputs) # (1, T, V)
        outputs_mask = outputs_mask / temperature
        outputs_mask_prob = torch.nn.functional.softmax(input=outputs_mask, dim=-1) # (1, T, V)
        
        last_multnom_index = torch.multinomial(input=outputs_mask_prob[0,-1,:], num_samples=1) # (1,)
        last_token_pred = last_multnom_index.unsqueeze(-1) # (1, 1)  
        if last_token_pred[0,0] == tokenizer.eot_token:
            break
        
        inputs = torch.cat([inputs, last_token_pred], dim=-1) # (1, T+1)
    
    generated_token_ids = inputs # (1, T+Ti)
    generated_text = token_ids_to_text(token_ids=generated_token_ids, tokenizer=tokenizer)
    
    return generated_text

def evaluate_batch(batch: Tuple[torch.tensor, torch.tensor], model: GPTModel, device: "torch.device", is_train: bool) -> Tuple["torch.tensor", "torch.tensor"]:
    
    inputs, targets = batch # (b, T)
    inputs, targets = inputs.to(device), targets.to(device) # (b, T)
    model = model.to(device)
    
    if is_train:
        model.train()
        pred_logits = model(inputs) # (b, T, V)
    else:
        model.eval()
        with torch.no_grad():
            pred_logits = model(inputs) # (b, T, V)
        
    pred_logits = pred_logits.flatten(start_dim=0, end_dim=1) # (b*T, V)
    targets = targets.flatten(start_dim=0, end_dim=1) # (b*T,)
    
    loss_metric = torch.nn.functional.cross_entropy(input=pred_logits, target=targets)
    acc_metric = (pred_logits.argmax(dim=-1) == targets).to(torch.float32).mean()

    return loss_metric, acc_metric

def evaluate_dataloader(dataloader: "torch.dataloader", model: GPTModel, device: "torch.device", is_train: bool, batch_fraction: float) -> Tuple[float, float]:
    
    if int(batch_fraction) < 1:
        num_batches = len(dataloader)
        num_examples = len(dataloader.dataset)
        num_batches_to_eval = int(num_batches * batch_fraction) # for speed
        num_examples_to_eval = num_batches_to_eval * dataloader.batch_size
        
        example_indices = list(range(num_examples))
        eval_example_indices = random.sample(example_indices, k=num_examples_to_eval) # without replacement
        
        random_sampler = torch.utils.data.SubsetRandomSampler(eval_example_indices)
        random_dataloader = torch.utils.data.DataLoader(dataset=dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False, drop_last=True, sampler=random_sampler)
        dataloader = random_dataloader
    
    loss_metric_list = []
    acc_metric_list = []
    
    for batch in dataloader:
        loss_metric_batch, acc_metric_batch = evaluate_batch(batch=batch, model=model, device=device, is_train=is_train)
        loss_metric_list.append(loss_metric_batch.item())
        acc_metric_list.append(acc_metric_batch.item())
    
    loss_metric_dl = sum(loss_metric_list)/len(loss_metric_list)
    acc_metric_dl = sum(acc_metric_list)/len(acc_metric_list)
    
    return loss_metric_dl, acc_metric_dl

def train(num_epochs: int, model: GPTModel, optimizer: "torch.optimizer", train_dl: "torch.dataloader", val_dl: "torch.dataloader", device: "torch.device") -> dict:
    
    model = model.to(device)
    
    initial_lr = 0.0001
    peak_lr = 0.01
    min_lr = 0.1 * initial_lr
    warmup_steps = 20
    total_steps = len(train_dl) * num_epochs

    train_loss_ep = []
    train_acc_ep = []
    val_loss_ep = []
    val_acc_ep = []
    
    current_step = 0
    lr_list = []
    
    for ep in range(num_epochs):
        train_loss_list = []
        train_acc_list = []
        
        with tqdm(train_dl, desc=f"Epoch: {ep+1}/{num_epochs}", postfix={"train_batch_loss": 0, "train_batch_acc": 0}, colour="green") as pbar:
        
            for i, train_batch in enumerate(pbar):           
                optimizer.zero_grad() # resets the gradients to zero for this batch
                
                if current_step < warmup_steps:
                    lr = initial_lr + current_step * (peak_lr - initial_lr) / warmup_steps
                else:
                    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
                
                lr_list.append(lr)
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]["lr"] = lr
                
                loss, acc = evaluate_batch(batch=train_batch, model=model, device=device, is_train=True)
                loss.backward() # calculates the gradients of the loss w.r.t. model parameters
                
                if current_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2.0)
                
                optimizer.step() # performs gradient descent step on model paramters
                
                train_loss_list.append(loss.item())
                train_acc_list.append(acc.item())
            
                pbar.set_postfix({"train_batch_loss": f"{loss:.3f}", "train_batch_acc": f"{acc:.3f}"})
                current_step += 1
            
        train_loss = sum(train_loss_list)/len(train_loss_list)
        train_acc = sum(train_acc_list)/len(train_acc_list)
            
        val_loss, val_acc = evaluate_dataloader(dataloader=val_dl, model=model, device=device, is_train=False, batch_fraction=0.25)

        train_loss_ep.append(train_loss)
        train_acc_ep.append(train_acc)
        val_loss_ep.append(val_loss)
        val_acc_ep.append(val_acc)
        
        print(f"Epoch: {ep+1}/{num_epochs}, train_epoch_loss: {train_loss:.3f}, train_epoch_acc: {train_acc:.3f}, val_epoch_loss: {val_loss:.3f}, val_epoch_acc: {val_acc:.3f}")
    
    history = {"train": {"loss": train_loss_ep, "acc": train_acc_ep},
            "val": {"loss": val_loss_ep, "acc": val_acc_ep},
            "lr": lr_list}

    return history
    
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Using GPU...")
        print(f"Total # of GPU: {torch.cuda.device_count()}")
        print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
    else:
        device_type = "cpu"
        print("Using CPU...")

    device = torch.device(device_type)

    BATCH_SIZE = 16
    CONFIG = {
        "vocab_size": 50257,
        "max_context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True
    }
    IS_TRAIN = False
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(config=CONFIG)

    corpus_path = pathlib.Path(pathlib.Path.cwd(),"tokenizer","corpus")
    file_list = [str(i) for i in corpus_path.iterdir()]

    train_text, val_text = train_val_split(file_list=file_list, train_val_split_ratio=0.8)
    train_dl = prepare_dataloader(text=train_text, tokenizer=tokenizer, max_context_length=CONFIG["max_context_length"], batch_size=BATCH_SIZE, is_train=True)
    val_dl = prepare_dataloader(text=val_text, tokenizer=tokenizer, max_context_length=CONFIG["max_context_length"], batch_size=BATCH_SIZE, is_train=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0, betas=(0.9, 0.999), weight_decay=0.1)
    
    if IS_TRAIN:
        history = train(num_epochs=1, model=model, optimizer=optimizer, train_dl=train_dl, val_dl=val_dl, device=device)
        torch.save(model.state_dict(), pathlib.Path(pathlib.Path.cwd(), "model.pth"))
    else:
        model.load_state_dict(torch.load(pathlib.Path(pathlib.Path.cwd(), "model.pth")))
        text = generate_text(start_context="Hello, I am not a", model=model, tokenizer=tokenizer, device=device, max_num_tokens=20)
        print(text)
        