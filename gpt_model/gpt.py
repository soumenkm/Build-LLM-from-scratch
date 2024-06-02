import torch, torchinfo, tiktoken, json, sys

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

class MultiHeadCausalSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, max_sequence_length, num_heads, dropout_prob=0.2):
        
        super(MultiHeadCausalSelfAttention, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.Tmax = max_sequence_length
        self.h = num_heads
        
        if self.d_out % self.h == 0:
            self.d_h = self.d_out // self.h
        else:
            raise ValueError("out_embedding_dim must be divisible by num_heads!")
        
        self.query_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)
        self.key_layer = torch.nn.Linear(in_features=self.d_in,
                                         out_features=self.d_out,
                                         bias=False)
        self.value_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)
        self.linear = torch.nn.Linear(in_features=self.d_out,
                                      out_features=self.d_out,
                                      bias=True) # Optional
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        mask = torch.full(size=(self.Tmax, self.Tmax), fill_value=-torch.inf) # (Tmax, Tmax)
        mask = torch.triu(mask, diagonal=1) # (Tmax, Tmax)
        self.register_buffer("mask", mask)
        
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be (b, T, {self.d_in})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        b, T = inputs.shape[0:2]
        
        Q = self.query_layer(inputs) # (b, T, d_out)
        K = self.key_layer(inputs) # (b, T, d_out)
        V = self.value_layer(inputs) # (b, T, d_out)
        
        Q = Q.reshape(b, T, self.h, self.d_h) # (b, T, h, d_h)
        K = K.reshape(b, T, self.h, self.d_h) # (b, T, h, d_h)
        V = V.reshape(b, T, self.h, self.d_h) # (b, T, h, d_h)
        
        Q = Q.transpose(1, 2) # (b, h, T, d_h)
        K = K.transpose(1, 2) # (b, h, T, d_h)
        V = V.transpose(1, 2) # (b, h, T, d_h)
        
        W = torch.matmul(Q, K.transpose(2, 3)) # (b, h, T, T)
        scaled_W = W / torch.sqrt(torch.tensor([self.d_h])) # (b, h, T, T)
        scaled_W = torch.tril(scaled_W, diagonal=0) # (b, h, T, T) (last 2 dim is taken)   
        scaled_W = scaled_W + self.mask[:T, :T] # (b, h, T, T) (mask is broadcasted and sliced for dynamic seq length)
         
        A = torch.nn.functional.softmax(scaled_W, dim=-1) # (b, h, T, T)
        A = self.dropout(A) # (b, h, T, T)
        
        Z = torch.matmul(A, V) # (b, h, T, d_h)
        Z = Z.transpose(1, 2) # (b, T, h, d_h)
        Z = Z.contiguous() # to get rid of unsafe gradient
        Z = Z.reshape(b, T, self.d_out) # (b, T, d_out)
        Z = self.linear(Z) # (b, T, d_out) (optional)
        
        assert Z.shape[-1] == self.d_out, f"contexts.shape must be (b, {T}, {self.d_out})"
        assert Z.shape[-2] == T, f"contexts.shape must be (b, {T}, {self.d_out})"
        assert list(Z.shape).__len__() == 3, "contexts rank must be 3"
        
        return Z

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
        x2 = self.mhcsa(x1) # (b, T, d)
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
         
    def forward(self, inputs: torch.tensor, is_logits: bool=True) -> torch.tensor:
        
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
    torchinfo.summary(model)