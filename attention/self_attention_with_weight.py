import torch

class SelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim):
        
        super(SelfAttention, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.query_parameter = torch.nn.Parameter(torch.randn(size=(self.d_out, self.d_in)))
        self.key_parameter = torch.nn.Parameter(torch.randn(size=(self.d_out, self.d_in)))
        self.value_parameter = torch.nn.Parameter(torch.randn(size=(self.d_out, self.d_in)))
        
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, "inputs.shape must be (T, d_in)"
        
        Q = torch.matmul(self.query_parameter, inputs.T) # (d_out, T)
        K = torch.matmul(self.key_parameter, inputs.T) # (d_out, T)
        V = torch.matmul(self.value_parameter, inputs.T) # (d_out, T)
        
        W = torch.matmul(Q.T, K) # (T, T)
        scaled_W = W/torch.sqrt(torch.tensor([self.d_out])) # (T, T)
        A = torch.nn.functional.softmax(scaled_W, dim=-1) # (T, T)
        
        contexts = torch.matmul(A, V.T) # (T, d_out)
        assert contexts.shape[-1] == self.d_out, "contexts.shape must be (T, d_out)"
        
        return contexts, A

class SelfAttentionWithLinear(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim):
        
        super(SelfAttentionWithLinear, self).__init__()
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
        A = torch.nn.functional.softmax(scaled_W, dim=-1) # (T, T)
        
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
    sa_layer = SelfAttention(in_embedding_dim=3, out_embedding_dim=2)
    Wq = sa_layer.query_parameter
    Wk = sa_layer.key_parameter
    Wv = sa_layer.value_parameter
    outputs_1, A_1 = sa_layer(inputs)
    
    sal_layer = SelfAttentionWithLinear(in_embedding_dim=3, out_embedding_dim=2)
    with torch.no_grad():
        sal_layer.query_layer.weight.copy_(Wq)
        sal_layer.key_layer.weight.copy_(Wk)
        sal_layer.value_layer.weight.copy_(Wv)
    outputs_2, A_2 = sa_layer(inputs)
    
    print((outputs_1-outputs_2).abs().sum())
    print((A_1-A_2).abs().sum())