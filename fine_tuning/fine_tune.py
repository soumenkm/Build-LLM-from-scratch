import torch, tqdm, tiktoken, pathlib, sys, random
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).parent)) # Adds fine_tuning in the search list
sys.path.append(str(pathlib.Path(__file__).parent.parent)) # Adds build-llm in the search list

from gpt_model.gpt import GPTModel
from gpt_model.load_openai_gpt2 import get_openai_gpt2_parameters
from gpt_model.gpt_download import download_and_load_gpt2
from spam_dataset import prepare_dataloader
from typing import Tuple, List

def classify_text(text: str, model: GPTModel, tokenizer: "tiktoken.tokenizer", device: "torch.device") -> str:
    
    token_ids_list = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_ids_tensor = torch.tensor(token_ids_list).unsqueeze(0) # (1, T)
    model = model.to(device)
    
    model.eval()
    inputs = token_ids_tensor[:, :model.Tmax] # (1, T) but at max (1, Tmax)
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs, is_logits=True) # (1, T, c)
    
    class_label = int(outputs[0, -1, :].argmax(dim=-1).item())
    
    return "Spam" if class_label == 1 else "Not Spam"

def calculate_metric_batch(batch: Tuple[torch.tensor, torch.tensor], model: GPTModel, device: "torch.device", is_train: bool) -> Tuple[torch.tensor, torch.tensor]:
    
    input_x, target_y = batch # (b, T), (b,)
    input_x, target_y = input_x.to(device), target_y.to(device)
    model = model.to(device)
    
    if is_train:
        model.train()
        pred_y = model(input_x) # (b, T, c)
    else:
        model.eval()
        with torch.no_grad():
            pred_y = model(input_x) # (b, T, c)
    
    pred_y_last = pred_y[:, -1, :] # (b, c)
        
    loss_metric = torch.nn.functional.cross_entropy(pred_y_last, target_y)
    acc_metric = (pred_y_last.argmax(dim=-1) == target_y).to(torch.float32).mean()
    
    return loss_metric, acc_metric # remember that it sends whole computational graph for future use, use .item() to break it

def calculate_metric_dataloader(dataloader: "torch.dataloader", model: GPTModel, device: "torch.device", is_train: bool, batch_fraction: float) -> Tuple[float, float]:
    
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
    
    with tqdm.tqdm(iterable=dataloader, desc=f"Evaluation for {'Train' if is_train else 'Val'}: ", total=len(dataloader), unit=" batch", colour="cyan") as pbar:
        for batch in pbar:
            loss_metric_batch, acc_metric_batch = calculate_metric_batch(batch=batch, model=model, device=device, is_train=is_train)
            loss_metric_list.append(loss_metric_batch.item())
            acc_metric_list.append(acc_metric_batch.item())
    
    loss_metric_dl = sum(loss_metric_list)/len(loss_metric_list)
    acc_metric_dl = sum(acc_metric_list)/len(acc_metric_list)
    
    return loss_metric_dl, acc_metric_dl

def fine_tune_train(num_epochs: int, model: GPTModel, optimizer: "torch.optimizer", train_dl: "torch.dataloader", val_dl: "torch.dataloader", device: "torch.device") -> dict:
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    for ep in range(num_epochs):
        
        train_loss = []
        train_acc = []
        
        with tqdm.tqdm(iterable=train_dl, desc=f"Epoch: {ep+1}/{num_epochs}", total=len(train_dl), unit=" step", colour="green") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                loss, acc = calculate_metric_batch(batch=batch, model=model, device=device, is_train=True)
                loss.backward()
                optimizer.step()
                
                train_loss.append(loss.item())
                train_acc.append(acc.item())
                pbar.set_postfix(train_loss=f"{loss.item():.3f}", train_acc=f"{acc.item():.3f}")
        
        val_loss_dl, val_acc_dl = calculate_metric_dataloader(dataloader=train_dl, model=model, device=device, is_train=False, batch_fraction=1.0)     
        train_loss_dl = sum(train_loss)/len(train_loss)
        train_acc_dl = sum(train_acc)/len(train_acc)
        
        train_loss_list.append(train_loss_dl)
        train_acc_list.append(train_acc_dl)
        val_loss_list.append(val_loss_dl)
        val_acc_list.append(val_acc_dl)
        
        print(f"Epoch: {ep+1}/{num_epochs}, train_epoch_loss: {train_loss_dl:.3f}, train_epoch_acc: {train_acc_dl:.3f}, val_epoch_loss: {val_loss_dl:.3f}, val_epoch_acc: {val_acc_dl:.3f}")
  
    history = {"train": {"loss": train_loss_list, "acc": train_acc_list},
            "val": {"loss": val_loss_list, "acc": val_acc_list}}
    
    return history

def plot_history(history: dict, save_dir: pathlib.Path) -> None:

    train_loss = history["train"]["loss"]
    train_acc = history["train"]["acc"]
    val_loss = history["val"]["loss"]
    val_acc = history["val"]["acc"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(16, 9))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save the figure
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'training_validation_metrics.png')
    plt.show()

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
    tokenizer = tiktoken.get_encoding("gpt2")
    
    current_dir = pathlib.Path.cwd()
    settings, params = download_and_load_gpt2(model_size="124M", models_dir=pathlib.Path(current_dir,"gpt_model","openai_gpt2"), is_download=False)

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
    
    # Replace the classification head
    model.output_layer = torch.nn.Linear(in_features=model.d, out_features=2, bias=False)

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
    train_dl, val_dl = prepare_dataloader(is_download=False, tokenizer=tokenizer, batch_size=32, max_context_length=model.Tmax, is_var_batch_length=False)

    # Train the model
    optimizer = torch.optim.AdamW(params=model.parameters())
    IS_TRAIN = False
    
    if IS_TRAIN:
        history = fine_tune_train(num_epochs=5, model=model, optimizer=optimizer, train_dl=train_dl, val_dl=val_dl, device=device)
        torch.save(model.state_dict(), pathlib.Path(pathlib.Path.cwd(), "fine_tuning/outputs/model.pth"))
        plot_history(history=history, save_dir=pathlib.Path(pathlib.Path.cwd(), "fine_tuning/outputs"))
    else:
        model.load_state_dict(torch.load(pathlib.Path(pathlib.Path.cwd(), "fine_tuning/outputs/model.pth")))
        msg1 = "You have won lottery of 100 Cr. Come to Kolkata airport to collect"
        res1 = classify_text(text=msg1, model=model, tokenizer=tokenizer, device=device)
        msg2 = f"Never fall into this kind of messages which says '{msg1}'"
        res2 = classify_text(text=msg2, model=model, tokenizer=tokenizer, device=device)
        
        print(f"Message: {msg1}\nResult: {res1}") # shows spam
        print(f"Message: {msg2}\nResult: {res2}") # shows not spam