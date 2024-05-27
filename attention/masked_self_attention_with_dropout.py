import torch

class MaskedSelfAttentionWithDropout(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, sequence_length, dropout_prob=0.2):
        
        super(MaskedSelfAttentionWithDropout, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.T = sequence_length
        self.query_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)
        self.key_layer = torch.nn.Linear(in_features=self.d_in, 
                                         out_features=self.d_out, 
                                         bias=False)
        self.value_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.register_buffer("mask", torch.triu(torch.full(size=(self.T, self.T), 
                                                           fill_value=-torch.inf), 
                                                diagonal=1))

    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be ({self.T}, {self.d_in})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be ({self.T}, {self.d_in})"
        
        Qm = self.query_layer(inputs) # (T, d_out)
        Km = self.key_layer(inputs) # (T, d_out)
        Vm = self.value_layer(inputs) # (T, d_out)
        
        W = torch.matmul(Qm, Km.T) # (T, T)
        scaled_W = W/torch.sqrt(torch.tensor([self.d_out])) # (T, T)
        
        scaled_W = torch.tril(scaled_W)  
        masked_W = scaled_W + self.mask
        
        A = torch.nn.functional.softmax(masked_W, dim=-1) # (T, T)
        A = self.dropout(A)
        
        contexts = torch.matmul(A, Vm) # (T, d_out)
        assert contexts.shape[-1] == self.d_out, f"contexts.shape must be ({self.T}, {self.d_out})"
        
        return contexts, A

if __name__ == "__main__":
    
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your (x^1)
        [0.55, 0.87, 0.66], # journey (x^2)
        [0.57, 0.85, 0.64], # starts (x^3)
        [0.22, 0.58, 0.33], # with (x^4)
        [0.77, 0.25, 0.10], # one (x^5)
        [0.05, 0.80, 0.55]] # step (x^6)
    )
    msa_layer = MaskedSelfAttentionWithDropout(in_embedding_dim=3, 
                                               out_embedding_dim=2, 
                                               sequence_length=6,
                                               dropout_prob=0.5)
    outputs, A = msa_layer(inputs)
    
    print(outputs)
    print(A)