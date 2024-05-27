import torch

class MaskedSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, sequence_length):
        
        super(MaskedSelfAttention, self).__init__()
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
        self.register_buffer("mask", torch.triu(torch.full(size=(self.T, self.T), 
                                                           fill_value=-torch.inf), 
                                                diagonal=1))

    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be ({self.T}, {self.d_in})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be ({self.T}, {self.d_in})"
        
        Qm = self.query_layer(inputs) # (T, d_out)
        Km = self.key_layer(inputs) # (T, d_out)
        Vm = self.value_layer(inputs) # (T, d_out)
        
        W = torch.matmul(Qm, Km.transpose(-1, -2)) # (T, T)
        scaled_W = W/torch.sqrt(torch.tensor([self.d_out])) # (T, T)
        
        scaled_W = torch.tril(scaled_W)  
        masked_W = scaled_W + self.mask
        
        A = torch.nn.functional.softmax(masked_W, dim=-1) # (T, T)
        
        contexts = torch.matmul(A, Vm) # (T, d_out)
        assert contexts.shape[-1] == self.d_out, f"contexts.shape must be ({self.T}, {self.d_out})"
        assert contexts.shape[-2] == self.T, f"contexts.shape must be ({self.T}, {self.d_out})"

        return contexts, A

if __name__ == "__main__":
    
    inputs = torch.randn(size=(2,6,3))
    msa_layer = MaskedSelfAttention(in_embedding_dim=3, out_embedding_dim=2, sequence_length=6)
    outputs, A = msa_layer(inputs)
    
    print(outputs)
    print(A)