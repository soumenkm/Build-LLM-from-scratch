import torch

class MaskedSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim):
        
        super(MaskedSelfAttention, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.query_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)
        self.key_layer = torch.nn.Linear(in_features=self.d_in, 
                                         out_features=self.d_out, 
                                         bias=False)
        self.value_layer = torch.nn.Linear(in_features=self.d_in, 
                                           out_features=self.d_out, 
                                           bias=False)

    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, "inputs.shape must be (T, d_in)"
        
        Qm = self.query_layer(inputs) # (T, d_out)
        Km = self.key_layer(inputs) # (T, d_out)
        Vm = self.value_layer(inputs) # (T, d_out)
        
        W = torch.matmul(Qm, Km.T) # (T, T)
        scaled_W = W/torch.sqrt(torch.tensor([self.d_out])) # (T, T)
        
        scaled_W = torch.tril(scaled_W)  
        mask = torch.full(size=(tuple(scaled_W.shape)), fill_value=-torch.inf)
        mask = torch.triu(mask, diagonal=1)
        masked_W = scaled_W + mask
        
        A = torch.nn.functional.softmax(masked_W, dim=-1) # (T, T)
        
        contexts = torch.matmul(A, Vm) # (T, d_out)
        assert contexts.shape[-1] == self.d_out, "contexts.shape must be (T, d_out)"
        
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
    msa_layer = MaskedSelfAttention(in_embedding_dim=3, out_embedding_dim=2)
    outputs, A = msa_layer(inputs)
    
    print(outputs)
    print(A)