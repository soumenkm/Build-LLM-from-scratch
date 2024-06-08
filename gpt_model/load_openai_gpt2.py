from gpt_download import download_and_load_gpt2
from gpt import GPTModel
import pathlib, torch, tiktoken
from train import generate_text

def assign_parameter(old_param: torch.tensor, new_param: "Any Tensor", qkv_type: "str: q, k, v"=None, is_bias: bool=None) -> torch.tensor:
    
    new_param = torch.tensor(new_param)
    
    if qkv_type is None:
        if torch.equal(torch.tensor(old_param.shape),torch.tensor(new_param.shape)):
            return torch.nn.Parameter(new_param)
        else:
            raise ValueError(f"Shape mismatch! old_param.shape={old_param.shape} doesn't match with new_param.shape={new_param.shape}")
    else:
        c1 = old_param.shape[-1]
        c2 = new_param.shape[-1]
        
        if 3 * c1 == c2:
            if qkv_type == "q":
                output = torch.nn.Parameter(new_param[0:c1]) if is_bias else torch.nn.Parameter(new_param[:, 0:c1].transpose(0,-1))
            elif qkv_type == "k":
                output = torch.nn.Parameter(new_param[c1:2*c1]) if is_bias else torch.nn.Parameter(new_param[:, c1:2*c1].transpose(0,-1))
            elif qkv_type == "v":
                output = torch.nn.Parameter(new_param[2*c1:3*c1]) if is_bias else torch.nn.Parameter(new_param[:, 2*c1:3*c1].transpose(0,-1))
            else:
                raise ValueError("Invalid qkv_type!")
            return output
        else:
            raise ValueError(f"Shape incompatible! (1, 3) * old_param.shape={old_param.shape} doesn't match with new_param.shape={new_param.shape}")

def get_openai_gpt2_parameters(model: GPTModel) -> dict:
         
    openai_gpt2_state_dict = {}

    for param_name, param_val in model.state_dict().items():
        
        if param_name == "token_embed_layer.weight":
            openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["wte"])
        
        elif param_name == "pos_embed_layer.weight":
            openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["wpe"])
        
        elif "transformer_layers" in param_name:
            layer_index = int(param_name.split(".")[1])
            
            if param_name == f"transformer_layers.{layer_index}.layernorm1.shift":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["ln_1"]["b"])
            
            elif param_name == f"transformer_layers.{layer_index}.layernorm1.scale":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["ln_1"]["g"])
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.query_layer.weight":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["w"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="q")
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.query_layer.bias":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["b"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="q", is_bias=True)
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.key_layer.weight":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["w"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="k")
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.key_layer.bias":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["b"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="k", is_bias=True)
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.value_layer.weight":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["w"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="v")
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.value_layer.bias":
                new_param_comb = params["blocks"][layer_index]["attn"]["c_attn"]["b"]
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, new_param_comb, qkv_type="v", is_bias=True)
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.linear.weight":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["attn"]["c_proj"]["w"].T)
            
            elif param_name == f"transformer_layers.{layer_index}.mhcsa.linear.bias":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["attn"]["c_proj"]["b"])
            
            elif param_name == f"transformer_layers.{layer_index}.layernorm2.shift":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["ln_2"]["b"])
            
            elif param_name == f"transformer_layers.{layer_index}.layernorm2.scale":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["ln_2"]["g"])
            
            elif param_name == f"transformer_layers.{layer_index}.ff.linear1.weight":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["mlp"]["c_fc"]["w"].T)
            
            elif param_name == f"transformer_layers.{layer_index}.ff.linear1.bias":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["mlp"]["c_fc"]["b"])
            
            elif param_name == f"transformer_layers.{layer_index}.ff.linear2.weight":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["mlp"]["c_proj"]["w"].T)
            
            elif param_name == f"transformer_layers.{layer_index}.ff.linear2.bias":
                openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["blocks"][layer_index]["mlp"]["c_proj"]["b"])
            
            elif "mask" in param_name:
                pass
            
            else:
                print(f"{param_name} is not assigned from OpenAI GPT2 model's state")
        
        elif param_name == "layernorm.shift":
            openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["b"])
            
        elif param_name == "layernorm.scale":
            openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["g"])
            
        elif param_name == "output_layer.weight":
            openai_gpt2_state_dict[param_name] = assign_parameter(param_val, params["wte"])
            
        else:
            print(f"{param_name} is not assigned from OpenAI GPT2 model's state")
    
    return openai_gpt2_state_dict
    
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
    settings, params = download_and_load_gpt2(model_size="1558M", models_dir=pathlib.Path(current_dir,"gpt_model","openai_gpt2"))

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
    openai_gpt2_state_dict = get_openai_gpt2_parameters(model)
    model.load_state_dict(state_dict=openai_gpt2_state_dict, strict=False)  
    
    model = model.to(device)
    text = generate_text(start_context="The quick sort algorithm is", model=model, tokenizer=tokenizer, device=device, max_num_tokens=250, temperature=0.7)
    print(text) 
