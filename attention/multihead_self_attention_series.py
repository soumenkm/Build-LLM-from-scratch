import torch
from masked_self_attention import MaskedSelfAttention

class MultiHeadMaskedSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, sequence_length, num_heads):
        
        super(MultiHeadMaskedSelfAttention, self).__init__()
        self.h = num_heads
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim * num_heads
        self.T = sequence_length
        
        self.sa_layers = torch.nn.ModuleList([
            MaskedSelfAttention(in_embedding_dim=3, out_embedding_dim=2, sequence_length=6) 
            for _ in range(num_heads)])
    
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be ({self.T}, {self.d_in})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be ({self.T}, {self.d_in})"
        
        contexts_list = []
        attn_wts_list = []
        for sa_layer in self.sa_layers:
            Z, A = sa_layer(inputs)
            contexts_list.append(Z)
            attn_wts_list.append(A)
        
        contexts = torch.cat(contexts_list, dim=-1)
        assert contexts.shape[-1] == self.d_out, f"contexts.shape must be ({self.T}, {self.d_out})"
        assert contexts.shape[-2] == self.T, f"contexts.shape must be ({self.T}, {self.d_out})"

        return contexts, attn_wts_list

if __name__ == "__main__":
    
    inputs = torch.randn(size=(2,6,3))
    mhsa_layer = MultiHeadMaskedSelfAttention(in_embedding_dim=3, 
                                              out_embedding_dim=2, 
                                              sequence_length=6,
                                              num_heads=2)
    outputs, A = mhsa_layer(inputs)
    
    print(outputs)
    print(A)