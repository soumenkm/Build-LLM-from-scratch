import torch
import json
import sys
sys.path.append("/raid/speech/soumen/build-llm")

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class GPTModel(torch.nn.Module):
    
    def __init__(self, config):
        
        super(GPTModel, self).__init__()
        self.V = config["vocab_size"]
        self.T = config["context_length"]
        self.d = config["emb_dim"]
        self.h = config["n_heads"]
        self.L = config["n_layers"]
        self.drop_rate = config["drop_rate"]
        self.qkv_bias = config["qkv_bias"]
        
        self.token_embed_layer = torch.nn.Embedding(num_embeddings=self.V,
                                                    embedding_dim=self.d)
        self.pos_embed_layer = torch.nn.Embedding(num_embeddings=self.T,
                                                  embedding_dim=self.d)
        self.drop_emb_layer = torch.nn.Dropout(self.drop_rate)
        
        self.output_layer = torch.nn.Linear(in_features=self.d, 
                                            out_features=self.V, 
                                            bias=False)
        
        self.layernorm = LayerNormalization(num_features=self.d)
         
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.T, "Invalid context length"
        assert list(inputs.shape).__len__() == 2, "Invalid rank"
        
        x1 = self.token_embed_layer(inputs)
        pos_tokens = torch.arange(self.T).unsqueeze(0).repeat(inputs.shape[0], 1)
        x2 = self.pos_embed_layer(pos_tokens)
        x = x1 + x2
        x = self.drop_emb_layer(x)
        x = self.layernorm(x)
        
        z = self.output_layer(x)
        
        return z

class LayerNormalization(torch.nn.Module):
    
    def __init__(self, num_features):
        
        super(LayerNormalization, self).__init__()
        self.d = num_features
        self.eps = 1e-5
        self.shift = torch.nn.Parameter(torch.zeros(size=(self.d,)))
        self.scale = torch.nn.Parameter(torch.ones(size=(self.d,)))
        
    def forward(self, inputs):
        
        assert list(inputs.shape).__len__() >= 2, "inputs rank should be at least 2"
        
        mean = inputs.mean(dim=-1, keepdim=True)
        var = inputs.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (inputs-mean)/torch.sqrt(var+self.eps)
        z = x_hat * self.scale + self.shift
        
        return z
        
if __name__ == "__main__":
    
    inputs = torch.randint(low=0, 
                           high=CONFIG["vocab_size"]+1, 
                           size=(2, CONFIG["context_length"]))
    
    gpt_model = GPTModel(CONFIG)
    outputs = gpt_model(inputs)
    print(outputs.shape)
    print(outputs.mean(dim=-1))
    print(outputs.var(dim=-1))
    