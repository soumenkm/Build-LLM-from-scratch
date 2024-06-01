import torch, torchinfo, tiktoken
import json
import sys
sys.path.append("/Users/soumen/Desktop/IITB/sem2/LLM/Build-LLM-from-scratch/attention")
from multihead_self_attention_parallel import MultiHeadCausalSelfAttention

class LayerNormalization(torch.nn.Module):
    
    def __init__(self, num_features: int):
        
        super(LayerNormalization, self).__init__()
        self.d = num_features
        self.eps = 1e-5
        self.shift = torch.nn.Parameter(torch.zeros(size=(self.d,))) # (d,)
        self.scale = torch.nn.Parameter(torch.ones(size=(self.d,))) # (d,)
        
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (b, T, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        mean = inputs.mean(dim=-1, keepdim=True) # (b, T, 1)
        var = inputs.var(dim=-1, keepdim=True, unbiased=False) # (b, T, 1)
        x_hat = (inputs-mean)/torch.sqrt(var+self.eps) # (b, T, d)
        z = x_hat * self.scale + self.shift # (b, T, d)
        
        return z

class GELU(torch.nn.Module):
    
    def __init__(self):
        
        super(GELU, self).__init__()
        
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3, inputs.shape = (b, T, d)"
        outputs = 0.5 * inputs * (1 + torch.tanh(torch.sqrt(torch.tensor([2/torch.pi])) 
                                                 * (inputs + 0.044715 * (inputs ** 3)))) # (b, T, d)

        return outputs

class FeedForward(torch.nn.Module):
    
    def __init__(self, num_features: int, dropout_prob: float=0.2):
        
        super(FeedForward, self).__init__()
        self.d = num_features
        self.linear1 = torch.nn.Linear(in_features=self.d,
                                       out_features=4*self.d,
                                       bias=True)
        self.gelu = GELU()
        self.linear2 = torch.nn.Linear(in_features=4*self.d,
                                       out_features=self.d,
                                       bias=True)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (b, T, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        x = self.linear1(inputs) # (b, T, 4d)
        x = self.gelu(x) # (b, T, 4d)
        x = self.linear2(x) # (b, T, d)
        z = self.dropout(x) # (b, T, d)
        
        return z

class Transformer(torch.nn.Module):
    
    def __init__(self, num_features: int, max_sequence_length: int, num_heads: int, dropout_prob: float):
        
        super(Transformer, self).__init__()
        self.d = num_features
        self.Tmax = max_sequence_length
        self.h = num_heads
        self.p = dropout_prob
        
        self.layernorm1 = LayerNormalization(num_features=self.d)
        self.mhcsa = MultiHeadCausalSelfAttention(in_embedding_dim=self.d,
                                                  out_embedding_dim=self.d,
                                                  max_sequence_length=self.Tmax,
                                                  num_heads=self.h,
                                                  dropout_prob=self.p)
        
        self.layernorm2 = LayerNormalization(num_features=self.d)
        self.ff = FeedForward(num_features=self.d,
                              dropout_prob=self.p)

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (b, T, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        x = inputs # (b, T, d)
        x1 = self.layernorm1(x) # (b, T, d)
        x2 = self.mhcsa(x1)[0] # (b, T, d), attention weights are not sought
        y = x2 + x # (b, T, d)
        
        y1 = self.layernorm2(y) # (b, T, d)
        y2 = self.ff(y1) # (b, T, d)
        z = y2 + y # (b, T, d)
        
        return z

class GPTModel(torch.nn.Module):
    
    def __init__(self, config: dict):
        
        super(GPTModel, self).__init__()
        self.V = config["vocab_size"]
        self.Tmax = config["max_context_length"]
        self.d = config["emb_dim"]
        self.h = config["n_heads"]
        self.L = config["n_layers"]
        self.p = config["drop_rate"]
        
        self.token_embed_layer = torch.nn.Embedding(num_embeddings=self.V,
                                                    embedding_dim=self.d)
        self.pos_embed_layer = torch.nn.Embedding(num_embeddings=self.Tmax,
                                                  embedding_dim=self.d)
        self.drop_emb_layer = torch.nn.Dropout(self.p)
        
        self.transformer_layers = torch.nn.Sequential(*[Transformer(num_features=self.d,
                                                                    max_sequence_length=self.Tmax,
                                                                    num_heads=self.h,
                                                                    dropout_prob=self.p) for _ in range(self.L)])
        self.layernorm = LayerNormalization(num_features=self.d)
        
        self.output_layer = torch.nn.Linear(in_features=self.d, 
                                            out_features=self.V, 
                                            bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
         
    def forward(self, inputs: torch.tensor, is_logits: bool=False) -> torch.tensor:
        
        assert list(inputs.shape).__len__() == 2, "inputs rank must be 2 and inputs.shape = (b, T)"
        
        b, T = inputs.shape
        
        x1 = self.token_embed_layer(inputs) # (b, T, d)
        pos_tokens = torch.arange(T).unsqueeze(0).repeat(b, 1) # (b, T)
        x2 = self.pos_embed_layer(pos_tokens) # (b, T, d)
        x = x1 + x2 # (b, T, d)
        x = self.drop_emb_layer(x) # (b, T, d)
        
        x = self.transformer_layers(x) # (b, T, d)
        
        x = self.layernorm(x) # (b, T, d)
        z = self.output_layer(x) # (b, T, V)
        p = self.softmax(z) # (b, T, V)
        
        return z if is_logits else p

def generate_text(start_context: list, model: GPTModel, max_num_tokens: int=50) -> str:
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    inputs = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0) # (1, T)
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
    
    decoded_text = tokenizer.decode(inputs.squeeze(0).tolist())
    
    return decoded_text
          
if __name__ == "__main__":
    
    CONFIG = {
        "vocab_size": 50257,
        "max_context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1
    }
    
    model = GPTModel(config=CONFIG)
    text = generate_text(start_context="Hello, I am", model=model, max_num_tokens=200)
    print(text)