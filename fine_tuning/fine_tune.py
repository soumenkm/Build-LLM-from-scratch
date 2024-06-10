import torch, tqdm, tiktoken, pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent)) # Adds fine_tuning in the search list
sys.path.append(str(pathlib.Path(__file__).parent.parent)) # Adds build-llm in the search list

from gpt_model.gpt import GPTModel
from gpt_model.load_openai_gpt2 import get_openai_gpt2_parameters
from gpt_model.gpt_download import download_and_load_gpt2
from gpt_model.train import generate_text

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
    
    model = model.to(device)
    text = generate_text(start_context="Every effort moves you", model=model, tokenizer=tokenizer, device=device, max_num_tokens=50, temperature=0.7)
    print(text) 