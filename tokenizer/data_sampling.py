import torch, tiktoken
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple

class RomeoDataset(Dataset):
    def __init__(self, 
                 corpus_file_path: Path, 
                 tokenizer: "tiktoken.tokenizer", 
                 max_context_len: int) -> None:   
        super(RomeoDataset, self).__init__()
        self.tokenizer = tokenizer
        self.Tmax = max_context_len
        with open(corpus_file_path, "r") as f:
            raw_text = f.read()
            self.enc_text = self.tokenizer.encode(raw_text, 
                                                  allowed_special={"<|endoftext|>"})
            self.enc_text.append(self.tokenizer.eot_token) # eot token is needed for last example
        
    def __len__(self) -> int:
        length = torch.floor(torch.tensor((len(self.enc_text)-1)/self.Tmax)) # excludes eot token
        return int(length.item())

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        start = index * self.Tmax
        end = start + self.Tmax
        input_x = torch.tensor(self.enc_text[start: end])
        target_y = torch.tensor(self.enc_text[start+1: end+1])
        return (input_x, target_y)

if __name__ == "__main__":
    
    cwd = Path.cwd()
    corpus_file_path = Path(cwd, "tokenizer/corpus/pg1513.txt")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = RomeoDataset(corpus_file_path, tokenizer, 5)
    
    print("Length of the dataset: ", len(dataset))
    print("Example 1: ", dataset[0])
    print("Example 2: ", dataset[1])
    print("Example 3: ", dataset[2])

    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, drop_last=True)
    print("Number of batches: ", len(dataloader))
    batch = next(iter(dataloader))
    print("input_x: \n", batch[0])
    print("target_y: \n", batch[1])