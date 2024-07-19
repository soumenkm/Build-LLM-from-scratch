import torch, tqdm, tiktoken, sys, json, os, random
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
sys.path.append(str(Path(__file__))) # instruct_fine_tuning is in search list now

from download import download_file, uncompress_file
import pandas as pd
from typing import Tuple, List

class InstructDataset(Dataset):
    
    def __init__(
        self, 
        dataset_path: Path, 
        tokenizer: "tiktoken.tokenizer",
        device: torch.device,
        max_context_length: int
    ) -> None:
        """dataset must be in json format 
        List[Dict{instruction: str, input: str, output: str}]"""
        
        super(InstructDataset, self).__init__()
        self.dataset_path = dataset_path
        self.raw_dataset = json.load(open(self.dataset_path, "r"))
        self.max_context_length = max_context_length
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = self._preprocess_dataset()

    def _preprocess_dataset(self) -> List[dict]:
        
        dataset = []
        eot_token_id = torch.tensor([self.tokenizer.eot_token], 
                                        device=self.device) # (1,)
        for index in range(len(self.raw_dataset)):
            example = self.raw_dataset[index]
            input_text = self._apply_alphaca_format(example=example)
            input_token_ids = torch.tensor(self.tokenizer.encode(text=input_text), 
                                           device=self.device) # (T,)
            target_token_ids = torch.concat([input_token_ids[1:], eot_token_id], dim=-1) # (T,)
            seq_len = input_token_ids.shape[0]
            
            output_dict = {
                "input_ids": input_token_ids, # (T,)
                "target_ids": target_token_ids, # (T,)
                "seq_len": seq_len # scalar
            }
            if seq_len < self.max_context_length:
                dataset.append(output_dict)
        
        return dataset
    
    def __len__(self) -> int:
        
        return len(self.dataset)
    
    def _apply_alphaca_format(
        self,
        example: dict,
        is_only_input: bool=False
    ) -> str:
        
        input_ph = f"\n\n### Input:\n{example['input']}" if example['input'] else ""
        output_ph = f"\n\n### Response:\n" if is_only_input else f"\n\n### Response:\n{example['output']}"
        instruction_text = (
            f"Below is an instruction that describes a task. " +
            f"Write a response that appropriately completes the request." +
            f"\n\n### Instruction:\n{example['instruction']}" + 
            input_ph + 
            output_ph
        )
        
        return instruction_text
    
    def __getitem__(
        self,
        index: int
    ) -> dict:
        
        if index > self.__len__() - 1 or index < 0:
            raise IndexError(f"Invalid index, index must be between 0 and {self.__len__()-1}")
         
        return self.dataset[index]
    
    def collate_function(
        self,
        data: List[torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        
        # TODO: Instead of storing the entire tokenized dataset in a 
        # variable, consider accessing them using __getitem__() as tokenized dataset
        # but here, we can drop them if it exceeds max length so (b, T) -> (b', T) 
        # where b' < b which is okay but make sure that b' != 0 (it is efficient method)
        
        x = []
        y = []
        length = []
        for example in data:
            x.append(example["input_ids"])
            y.append(example["target_ids"])
            length.append(example["seq_len"])
        
        max_len = max(length)
        pad_token_id = [self.tokenizer.eot_token]
        input_list = []
        target_list = []
        
        for i in range(len(data)):
            padded_seq = torch.tensor(pad_token_id * (max_len - length[i]),
                                      device=self.device) 
            masked_seq = torch.tensor([-100] * (max_len - length[i]),
                                      device=self.device)

            input_list.append(torch.concat([x[i], padded_seq], dim=-1)) # List[b x (T,)]
            target_list.append(torch.concat([y[i], masked_seq], dim=-1)) # List[b x (T,)]
        
        input_x = torch.stack(input_list, dim=0).to(torch.int64) # (b, T)
        target_y = torch.stack(target_list, dim=0).to(torch.int64) # (b, T)
        
        return (input_x, target_y)
    
    def prepare_dataloader(
        self,
        batch_size: int=16,
        train_val_split: float=0.8,
    ) -> Tuple[DataLoader, DataLoader]:
        
        dataset_len = self.__len__()
        train_ds_len = int(dataset_len * train_val_split)
        
        indices = random.sample(range(dataset_len), k=dataset_len)
        train_ds = Subset(dataset=self, indices=indices[:train_ds_len])
        val_ds = Subset(dataset=self, indices=indices[train_ds_len:])
        
        train_dl = DataLoader(dataset=train_ds, 
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=self.collate_function,
                              drop_last=True)
        val_dl = DataLoader(dataset=val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=self.collate_function,
                            drop_last=True)
        
        return (train_dl, val_dl)
              
def main():
    
    src_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    dataset_path = Path(Path(__file__).parent, "data/instruct_data.json")
    Path.mkdir(dataset_path.parent, exist_ok=True)
    download_file(src_url=src_url, destination_file_path=dataset_path, is_text=False) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device=device)   
    
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructDataset(dataset_path=dataset_path,
                              tokenizer=tokenizer,
                              device=device,
                              max_context_length=1024)
    
    train_dl, val_dl = dataset.prepare_dataloader()
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main()