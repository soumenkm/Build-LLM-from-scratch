import torch

class MultiHeadSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, sequence_length, num_heads, dropout_prob=0.2):
        
        super(MultiHeadSelfAttention, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.T = sequence_length
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
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be (None, {self.T}, {self.d_in})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be (None, {self.T}, {self.d_in})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        Q = self.query_layer(inputs) # (b, T, d_out)
        K = self.key_layer(inputs) # (b, T, d_out)
        V = self.value_layer(inputs) # (b, T, d_out)
        
        b = Q.shape[0]
        
        Q = Q.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        K = K.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        V = V.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        
        Q = Q.transpose(1, 2) # (b, h, T, d_h)
        K = K.transpose(1, 2) # (b, h, T, d_h)
        V = V.transpose(1, 2) # (b, h, T, d_h)
        
        W = torch.matmul(Q, K.transpose(2, 3)) # (b, h, T, T)
        scaled_W = W / torch.sqrt(torch.tensor([self.d_h])) # (b, h, T, T)
        
        A = torch.nn.functional.softmax(scaled_W, dim=-1) # (b, h, T, T)
        A = self.dropout(A) # (b, h, T, T)
        
        Z = torch.matmul(A, V) # (b, h, T, d_h)
        Z = Z.transpose(1, 2) # (b, T, h, d_h)
        Z = Z.contiguous() # to get rid of unsafe gradient
        Z = Z.reshape(b, self.T, self.d_out) # (b, T, d_out)
        
        assert Z.shape[-1] == self.d_out, f"contexts.shape must be (None, {self.T}, {self.d_out})"
        assert Z.shape[-2] == self.T, f"contexts.shape must be (None, {self.T}, {self.d_out})"
        assert list(Z.shape).__len__() == 3, "contexts rank must be 3"
        
        return Z, A

class MultiHeadCausalSelfAttention(torch.nn.Module):
    
    def __init__(self, in_embedding_dim, out_embedding_dim, sequence_length, num_heads, dropout_prob=0.2):
        
        super(MultiHeadCausalSelfAttention, self).__init__()
        self.d_in = in_embedding_dim
        self.d_out = out_embedding_dim
        self.T = sequence_length
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
        self.dropout = torch.nn.Dropout(dropout_prob)
        
        mask = torch.full(size=(self.T, self.T), fill_value=-torch.inf)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)
        
    def forward(self, inputs):
        
        assert inputs.shape[-1] == self.d_in, f"inputs.shape must be (None, {self.T}, {self.d_in})"
        assert inputs.shape[-2] == self.T, f"inputs.shape must be (None, {self.T}, {self.d_in})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        Q = self.query_layer(inputs) # (b, T, d_out)
        K = self.key_layer(inputs) # (b, T, d_out)
        V = self.value_layer(inputs) # (b, T, d_out)
        
        b = Q.shape[0]
        
        Q = Q.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        K = K.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        V = V.reshape(b, self.T, self.h, self.d_h) # (b, T, h, d_h)
        
        Q = Q.transpose(1, 2) # (b, h, T, d_h)
        K = K.transpose(1, 2) # (b, h, T, d_h)
        V = V.transpose(1, 2) # (b, h, T, d_h)
        
        W = torch.matmul(Q, K.transpose(2, 3)) # (b, h, T, T)
        scaled_W = W / torch.sqrt(torch.tensor([self.d_h])) # (b, h, T, T)
        scaled_W = torch.tril(scaled_W, diagonal=0) # (b, h, T, T) (last 2 dim is taken)   
        scaled_W = scaled_W + self.mask # (b, h, T, T) (mask is broadcasted)
         
        A = torch.nn.functional.softmax(scaled_W, dim=-1) # (b, h, T, T)
        A = self.dropout(A) # (b, h, T, T)
        
        Z = torch.matmul(A, V) # (b, h, T, d_h)
        Z = Z.transpose(1, 2) # (b, T, h, d_h)
        Z = Z.contiguous() # to get rid of unsafe gradient
        Z = Z.reshape(b, self.T, self.d_out) # (b, T, d_out)
        
        assert Z.shape[-1] == self.d_out, f"contexts.shape must be (None, {self.T}, {self.d_out})"
        assert Z.shape[-2] == self.T, f"contexts.shape must be (None, {self.T}, {self.d_out})"
        assert list(Z.shape).__len__() == 3, "contexts rank must be 3"
        
        return Z, A
             
if __name__ == "__main__":
    
    inputs = torch.randn(size=(3, 6, 8))
    mhcsa_layer = MultiHeadCausalSelfAttention(in_embedding_dim=8,
                                               out_embedding_dim=8,
                                               sequence_length=6,
                                               num_heads=2, 
                                               dropout_prob=0.2)
    
    contexts, attn_weights = mhcsa_layer(inputs)
    print(contexts)
    print(attn_weights)
          