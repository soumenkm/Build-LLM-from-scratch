import torch, tqdm, pathlib, tiktoken, sys

sys.path.append(str(pathlib.Path(__file__))) # fine_tuning is in search list now

from download import download_file, uncompress_file
import pandas as pd
from typing import Tuple, List

def train_val_split(text_file_path: pathlib.Path, delimiter: str, split_ratio: float) -> Tuple[pathlib.Path, pathlib.Path]:
    
    data_df = pd.read_csv(filepath_or_buffer=text_file_path, 
                          delimiter=delimiter,
                          on_bad_lines="skip",
                          skip_blank_lines=True,
                          names=["label","text"],
                          header=None).dropna(axis=0, how="any", ignore_index=True)
    data_df.loc[:,"label"] = data_df.loc[:,"label"].map({"ham": 0, "spam": 1})
    
    split = int(data_df.shape[0] * split_ratio)
    train_df = data_df[:split]
    train_df.columns = data_df.columns
    val_df = data_df[split:]
    val_df.columns = data_df.columns
    
    train_csv_path = pathlib.Path(text_file_path.parent, text_file_path.stem + "_train.csv")
    val_csv_path = pathlib.Path(text_file_path.parent, text_file_path.stem + "_val.csv")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    return (train_csv_path, val_csv_path)

class SpamDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_path: pathlib.Path, tokenizer: "tiktoken.tokenizer", max_model_context_length: int):
        
        super(SpamDataset, self).__init__()
        self.csv_path = csv_path
        self.data_df = pd.read_csv(filepath_or_buffer=self.csv_path, sep=",", header=0)
        self.max_model_context_length = max_model_context_length
        self.tokenizer = tokenizer
        self.enc_list = []
        self.max_dataset_seq_length = self.get_max_dataset_seq_length()
        
    def __len__(self) -> int:
        
        return self.data_df.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        
        label, enc = self.data_df.loc[index,"label"], self.enc_list[index]
        
        if len(enc) > self.max_model_context_length:
            enc = enc[:self.max_model_context_length]
        
        if len(enc) < self.max_dataset_seq_length:
            eot_token_id_list = self.tokenizer.encode(text="<|endoftext|>", allowed_special={"<|endoftext|>"})
            enc.extend(eot_token_id_list * (self.max_dataset_seq_length - len(enc)))
        
        x = torch.tensor(data=enc)
        y = torch.tensor(data=label)
        
        return (x, y)
    
    def get_max_dataset_seq_length(self) -> int:
        
        length_list = []
        for i, text in enumerate(self.data_df.loc[:,"text"]):
            enc = self.tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
            self.enc_list.append(enc)
            length_list.append(len(enc))
        
        return min(max(length_list), self.max_model_context_length)

def collate_fn_for_var_seq_len(samples_list: List[Tuple[torch.tensor, torch.tensor]]) -> Tuple[torch.tensor, torch.tensor]:
    
    last_token_list = []
    x_list = []
    y_list = []
    stop_index_list = []
    
    for sample in samples_list:
        x, y = sample
        last_token_list.append(x[-1].item())
        x_list.append(x)
        y_list.append(y)
        
        length_x = len(x)
        for i in range(length_x-1, -1, -1):
            if x[i] != x[i-1]:
                stop_index = i-1
                break
        else:
            stop_index = 0
        
        stop_index_list.append(stop_index)
                 
    if len(set(last_token_list)) == 1:
        eot_token_id = last_token_list[0] # makes sure all the list elements are same
        max_stop_index = max(stop_index_list)
        x_list_mod = []
        
        for x in x_list:
            x_list_mod.append(x[:max_stop_index])
        
        batch_x = torch.stack(tensors=x_list_mod, dim=0)
        batch_y = torch.stack(tensors=y_list, dim=0)
        
    else:
        batch_x = torch.stack(tensors=x_list, dim=0)
        batch_y = torch.stack(tensors=y_list, dim=0)
    
    return (batch_x, batch_y)
                   
def prepare_dataloader(is_download: bool, tokenizer: "tiktoken.tokenizer", batch_size: int, max_context_length: int, is_var_batch_length: bool) -> Tuple["torch.dataloader", "torch.dataloader"]:
    
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    cwd = pathlib.Path.cwd()
    zip_file_path = pathlib.Path(cwd,"fine_tuning/datasets/sms_spam_collection.zip")
    if is_download:
        download_file(src_url=url, destination_file_path=zip_file_path)
    
    dataset_dir = pathlib.Path(zip_file_path.parent, zip_file_path.stem)
    if is_download:
        uncompress_file(src_file_path=zip_file_path, dest_dir_path=dataset_dir)
    
    text_file_path = pathlib.Path(dataset_dir, "SMSSpamCollection")
    train_csv_path, val_csv_path = train_val_split(text_file_path=text_file_path, delimiter="\t", split_ratio=0.8)

    train_ds = SpamDataset(csv_path=train_csv_path, tokenizer=tokenizer, max_model_context_length=max_context_length)
    val_ds = SpamDataset(csv_path=val_csv_path, tokenizer=tokenizer, max_model_context_length=max_context_length)
    
    if is_var_batch_length:
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_for_var_seq_len)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_for_var_seq_len)
    else:
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dl, val_dl

if __name__ == "__main__":
    
    train_dl, val_dl = prepare_dataloader(is_download=False,
                                          tokenizer=tiktoken.get_encoding("gpt2"),
                                          batch_size=32,
                                          max_context_length=1024,
                                          is_var_batch_length=False)
    train_dl_1, val_dl_1 = prepare_dataloader(is_download=False,
                                          tokenizer=tiktoken.get_encoding("gpt2"),
                                          batch_size=32,
                                          max_context_length=1024,
                                          is_var_batch_length=True)
    a = train_dl.__iter__()
    b = train_dl_1.__iter__()
    
    print(a.__next__()[0].shape)
    print(a.__next__()[0].shape)
    print(a.__next__()[0].shape)
    
    print(b.__next__()[0].shape)
    print(b.__next__()[0].shape)
    print(b.__next__()[0].shape)