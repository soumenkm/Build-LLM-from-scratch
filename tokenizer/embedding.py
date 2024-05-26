import torch, tiktoken
from data_sampling import GPTDataset

token_emb_layer = torch.nn.Embedding(num_embeddings=50257, embedding_dim=256)
# print(token_emb_layer.weight)

input_token_ids = torch.tensor([1, 3, 5, 2])
# print(token_emb_layer(input_token_ids))

with open("/raid/speech/soumen/build-llm/tokenizer/corpus/the-verdict.txt", "r") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

gpt_ds = GPTDataset(raw_text=raw_text, tokenizer=tokenizer, stride=4, max_length=4)
gpt_dl = torch.utils.data.DataLoader(gpt_ds, batch_size=8, shuffle=False, drop_last=True)

sample_x, sample_y = gpt_dl.__iter__().__next__()
sample_x_emb = token_emb_layer(sample_x)
print("Token embedding shape: ", sample_x_emb.shape)

context_length = 4
pos_emb_layer = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=246)

pos_emd = pos_emb_layer(torch.arange(context_length).unsqueeze(0).repeat(8, 1))
print("Position embedding shape: ", pos_emd.shape)