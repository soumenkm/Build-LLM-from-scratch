import torch, torchinfo
import json
import sys
sys.path.append("/Users/soumen/Desktop/IITB/sem2/LLM/Build-LLM-from-scratch/attention")
from multihead_self_attention_parallel import MultiHeadCausalSelfAttention

CONFIG = {
    "vocab_size": 50257,
    "context_length": 4, # 1024
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1
}

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

class GELU(torch.nn.Module):
    
    def __init__(self):
        
        super(GELU, self).__init__()
        
    def forward(self, inputs):
        
        assert list(inputs.shape).__len__() >= 2, "inputs rank should be at least 2"
        outputs = 0.5 * inputs * (1 + torch.tanh(torch.sqrt(torch.tensor([2/torch.pi])) 
                                                 * (inputs + 0.044715 * (inputs ** 3))))

        return outputs

class FeedForward(torch.nn.Module):
    
    def __init__(self, num_features, dropout_prob=0.2):
        
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
        
    def forward(self, inputs):
        
        assert list(inputs.shape).__len__() >= 2, "inputs rank should be at least 2"
        x = self.linear1(inputs)
        x = self.gelu(x)
        x = self.linear2(x)
        z = self.dropout(x)
        
        return z

class Transformer(torch.nn.Module):
    
    def __init__(self, num_features, sequence_length, num_heads, dropout_prob):
        
        super(Transformer, self).__init__()
        self.d = num_features
        self.T = sequence_length
        self.h = num_heads
        self.p = dropout_prob
        
        self.layernorm1 = LayerNormalization(num_features=self.d)
        self.mhcsa = MultiHeadCausalSelfAttention(in_embedding_dim=self.d,
                                                  out_embedding_dim=self.d,
                                                  sequence_length=self.T,
                                                  num_heads=self.h,
                                                  dropout_prob=self.p)
        
        self.layernorm2 = LayerNormalization(num_features=self.d)
        self.ff = FeedForward(num_features=self.d,
                              dropout_prob=self.p)

    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (None, {self.T}, {self.d})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be (None, {self.T}, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        x = inputs
        x1 = self.layernorm1(x)
        x2 = self.mhcsa(x1)[0]
        y = x2 + x
        
        y1 = self.layernorm2(y)
        y2 = self.ff(y1)
        z = y2 + y
        
        return z

class GPTModel(torch.nn.Module):
    
    def __init__(self, config):
        
        super(GPTModel, self).__init__()
        self.V = config["vocab_size"]
        self.T = config["context_length"]
        self.d = config["emb_dim"]
        self.h = config["n_heads"]
        self.L = config["n_layers"]
        self.p = config["drop_rate"]
        
        self.token_embed_layer = torch.nn.Embedding(num_embeddings=self.V,
                                                    embedding_dim=self.d)
        self.pos_embed_layer = torch.nn.Embedding(num_embeddings=self.T,
                                                  embedding_dim=self.d)
        self.drop_emb_layer = torch.nn.Dropout(self.p)
        
        self.transformer_layers = torch.nn.Sequential(*([Transformer(num_features=self.d,
                                                                   sequence_length=self.T,
                                                                   num_heads=self.h,
                                                                   dropout_prob=self.p) for _ in range(self.L)]))
        self.layernorm = LayerNormalization(num_features=self.d)
        
        self.output_layer = torch.nn.Linear(in_features=self.d, 
                                            out_features=self.V, 
                                            bias=False)
         
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.T, f"inputs.shape must be (None, {self.T})"
        assert list(inputs.shape).__len__() == 2, "inputs rank must be 2"
        
        x1 = self.token_embed_layer(inputs)
        pos_tokens = torch.arange(self.T).unsqueeze(0).repeat(inputs.shape[0], 1)
        x2 = self.pos_embed_layer(pos_tokens)
        x = x1 + x2
        x = self.drop_emb_layer(x)
        
        x = self.transformer_layers(x)
        
        x = self.layernorm(x)
        z = self.output_layer(x)
        
        return z
        
if __name__ == "__main__":
    
    inputs = torch.randint(low=0, 
                           high=CONFIG["vocab_size"], 
                           size=(32, CONFIG["context_length"]))
    
    gpt = GPTModel(config=CONFIG)
    torchinfo.summary(gpt)
    
    out = gpt(inputs)
    print(out.shape)