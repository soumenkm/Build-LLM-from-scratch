import tiktoken
import torch
import json

with open("/raid/speech/soumen/build-llm/corpus/the-verdict.txt", "r") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

class GPTDataset(torch.utils.data.Dataset):
    
    def __init__(self, raw_text, tokenizer, max_length=4):
        
        super(GPTDataset, self).__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.raw_text = raw_text
        self.enc_text = self.tokenizer.encode(self.raw_text)   
        
    def __len__(self):
        
        length = torch.floor(torch.tensor((len(self.enc_text)-1)/self.max_length))
        return int(length.numpy())

    def __getitem__(self, index, is_text=False):
        
        start = index * self.max_length
        end = start + self.max_length
        input_x = torch.tensor(self.enc_text[start: end], 
                               dtype=torch.int32)
        target_y = torch.tensor(self.enc_text[start+1: end+1], 
                                dtype=torch.int32)
        
        input_x_text = self.tokenizer.decode(self.enc_text[start: end])
        target_y_text = self.tokenizer.decode(self.enc_text[start+1: end+1])
        
        if is_text:
            return (input_x_text, target_y_text)
        else:
            return (input_x, target_y)

gpt_ds = GPTDataset(raw_text=raw_text, tokenizer=tokenizer)
print(len(gpt_ds))
print(gpt_ds[0])
print(gpt_ds.__getitem__(0, True))

gpt_dl = torch.utils.data.DataLoader(gpt_ds, batch_size=4, shuffle=False, drop_last=True)
sample_batch = gpt_dl.__iter__().__next__()
print(sample_batch[0])
print(sample_batch[1])