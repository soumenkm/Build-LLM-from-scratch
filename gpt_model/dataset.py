import torch, tiktoken, random
from typing import List, Tuple

class GPTDataset(torch.utils.data.Dataset):
    
    def __init__(self, raw_text: str, tokenizer: "tiktoken.tokenizer", stride: int, max_length: int):
        
        super(GPTDataset, self).__init__()
        
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tokenizer
        self.raw_text = raw_text
        self.token_ids = self.tokenizer.encode(self.raw_text, allowed_special={"<|endoftext|>"})   
        
    def __len__(self) -> int:
        
        length = torch.floor(torch.tensor((len(self.token_ids)-1)/self.max_length))
        return int(length.numpy())

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        
        length = self.__len__()
        if index >= length or index < -length:
            raise ValueError(f"The maximum/minimum possible index is {length-1}/{-length}")
        
        start = index * self.stride
        end = start + self.max_length
        input_x = torch.tensor(self.token_ids[start: end], 
                               dtype=torch.int64)
        target_y = torch.tensor(self.token_ids[start+1: end+1], 
                                dtype=torch.int64)
        
        return (input_x, target_y)

def prepare_dataloader(text: str, tokenizer: "tiktoken.tokenizer", max_context_length: int, batch_size: int, is_train: bool) -> "torch.dataloader":

    dataset = GPTDataset(raw_text=text, tokenizer=tokenizer, stride=max_context_length, max_length=max_context_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=True)
    
    return dataloader

def train_val_split(file_list: List[str], train_val_split_ratio: float) -> Tuple[str, str]:

    random.shuffle(file_list)
    
    raw_text = ""
    for file in file_list:
        with open(file, "r") as f:
            raw_text += f.read()
    
    length = len(raw_text)
    train_text = raw_text[:int(length*train_val_split_ratio)]
    val_text = raw_text[int(length*train_val_split_ratio):]
    
    return (train_text, val_text)
   
if __name__ == "__main__":
    
    raw_text = "Hello there, how are you doing today? I am fine, and you?"

    tokenizer = tiktoken.get_encoding("gpt2")
    
    dl = prepare_dataloader(raw_text, tokenizer, 4, 2, True)
    
    a = dl.__iter__()
    print(a.__next__())