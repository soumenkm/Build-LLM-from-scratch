import torch, os, pathlib
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from typing import List, Tuple, Union
from torchvision import datasets
from torchvision.transforms import ToTensor

class FeedForward(torch.nn.Module):
    
    def __init__(self, num_features: int, num_classes: int):
        
        super(FeedForward, self).__init__()
        self.d = num_features
        self.c = num_classes
        
        self.linear1 = torch.nn.Linear(in_features=self.d,
                                       out_features=4*self.d,
                                       bias=True)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=4*self.d,
                                       out_features=self.d,
                                       bias=True)
        self.linear3 = torch.nn.Linear(in_features=self.d,
                                       out_features=self.c,
                                       bias=True)
        
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        
        assert inputs.shape[-1] == self.d, f"inputs.shape = {inputs.shape} must be (b, {self.d})"
        assert list(inputs.shape).__len__() == 2, "inputs rank must be 2"
        
        x = self.linear1(inputs) # (b, 4d)
        x = self.relu(x) # (b, 4d)
        x = self.linear2(x) # (b, d)
        x = self.relu(x) # (b, d)
        x = self.linear3(x) # (b, c)
        
        return x   

class LossFunction(torch.nn.Module):
    
    def __init__(self):
        
        super(LossFunction, self).__init__()
        
    def forward(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        assert true_outputs.dim() == 1, "Targets must be 1D tensor"
        loss = torch.nn.functional.cross_entropy(input=pred_outputs, target=true_outputs)
        return loss

class Trainer:
    
    def __init__(self, 
                 model: "torch.nn.Module", 
                 device: Union["torch.device", int],
                 is_ddp: bool,
                 optimizer: "torch.optim.Optimizer", 
                 loss_fn: LossFunction,
                 train_ds: "torch.utils.data.Dataset",
                 val_ds: "torch.utils.data.Dataset",
                 batch_size: int,
                 num_epochs: int, 
                 collate_fn: "function" = None,
                 val_fraction: float = 1.0,
                 checkpoint_dir: pathlib.Path = None):
        
        self.is_ddp = is_ddp
        self.device = device
        self.model = DDP(model.to(self.device), device_ids=[self.device]) if self.is_ddp else model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.collate_fn = collate_fn
        self.val_fraction = val_fraction
        self.checkpoint_dir = checkpoint_dir
        
        self.train_dl = self._create_dataloader(is_train=True)
        self.val_dl = self._create_dataloader(is_train=False)
        
        self.start_epoch = 0 # to keep track of resuming from checkpoint
        
    def _create_dataloader(self, is_train: bool) -> "torch.utils.data.DataLoader":
        
        if is_train:
            if self.is_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.train_ds, shuffle=True, drop_last=True)
                train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.collate_fn, sampler=sampler)
            else:
                train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True, collate_fn=self.collate_fn, drop_last=True)
        else:
            len_val_ds = len(self.val_ds)
            indices = torch.randperm(n=len_val_ds)[:int(len_val_ds * self.val_fraction)]
            val_ds = torch.utils.data.Subset(dataset=self.val_ds, indices=indices)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, num_workers=4, shuffle=False, collate_fn=self.collate_fn, drop_last=True)

        return train_dl if is_train else val_dl
    
    def _forward_batch(self, batch: Tuple[torch.tensor, torch.tensor], is_train: bool) -> torch.tensor:
        
        inputs_x, targets_y = batch
        inputs_x, targets_y = inputs_x.to(self.device), targets_y.to(self.device)
        self.model = self.model.to(self.device)
        
        if is_train:
            self.model.train()
            pred_y = self.model(inputs_x)
        else:
            self.model.eval()
            with torch.no_grad():
                pred_y = self.model(inputs_x)
        
        return pred_y
    
    def _calc_loss_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        
        pred_outputs, true_outputs = pred_outputs.to(self.device), true_outputs.to(self.device)
        
        loss = self.loss_fn(pred_outputs=pred_outputs, true_outputs=true_outputs)
        
        return loss
    
    def _calc_accuracy_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        
        pred_outputs, true_outputs = pred_outputs.to(self.device), true_outputs.to(self.device)
        
        pred_outputs = pred_outputs.argmax(dim=-1)
        acc = (pred_outputs == true_outputs).to(torch.float32).mean()
        
        return acc
    
    def _optimize_batch(self, batch: Tuple[torch.tensor, torch.tensor]) -> Tuple[float, float]:
        
        pred_y = self._forward_batch(batch=batch, is_train=True)
        self.optimizer.zero_grad(set_to_none=True)
        
        loss = self._calc_loss_batch(pred_outputs=pred_y, true_outputs=batch[1])
        acc = self._calc_accuracy_batch(pred_outputs=pred_y, true_outputs=batch[1])
        loss.backward()
        self.optimizer.step()
        
        return loss, acc
                   
    def _optimize_dataloader(self, dataloader: "torch.utils.data.DataLoader") -> Tuple[float, float]:
        
        num_steps = len(self.train_dl)
        loss_list = []
        acc_list = []
        
        for i, batch in enumerate(self.train_dl):
            loss, acc = self._optimize_batch(batch=batch)
            loss_list.append(loss.item())
            acc_list.append(acc.item())
            # print(f"Step: {i+1}/{num_steps}, Train loss: {loss:.3f}, Train acc: {acc:.3f}")
        
        return sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)
    
    def _validate_dataloader(self, dataloader: "torch.utils.data.DataLoader") -> Tuple[float, float]:
        
        loss_list = []
        acc_list = []
        
        for i, batch in enumerate(self.val_dl):  
            pred_y = self._forward_batch(batch=batch, is_train=False)
            loss = self._calc_loss_batch(pred_outputs=pred_y, true_outputs=batch[1])
            acc = self._calc_accuracy_batch(pred_outputs=pred_y, true_outputs=batch[1])
            
            loss_list.append(loss.item())
            acc_list.append(acc.item())
        
        return sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)
    
    def _save_checkpoint(self, epoch: int, loss: float, acc: float) -> None:
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "opt_state": self.optimizer.state_dict(),
            "loss": loss,
            "acc": acc,
        }
        if not pathlib.Path.exists(self.checkpoint_dir):
            pathlib.Path.mkdir(self.checkpoint_dir, parents=True, exist_ok=True)
            
        checkpoint_path = pathlib.Path(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_dir: pathlib.Path) -> int:
        
        checkpoint_files = []
        for i in pathlib.Path.iterdir(checkpoint_dir):
            if i.suffix == ".pth":
                checkpoint_files.append((i, i.stat().st_mtime))
        
        # If no checkpoints found, return 0 to start from scratch
        if not checkpoint_files:
            return 0
        
        # Find the latest checkpoint file
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
        map_loc = torch.device(f"cuda:{self.device}") if self.is_ddp else self.device
        checkpoint = torch.load(latest_checkpoint, map_location=map_loc)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch

        if self.is_ddp:
            if self.device == 0:
                print(f"Checkpoint loaded from {latest_checkpoint}")
        else:
            print(f"Checkpoint loaded from {latest_checkpoint}")
        
        return start_epoch

    def train(self, is_load_checkpoint: bool = True) -> None:
        
        if is_load_checkpoint:
            self.start_epoch = self._load_checkpoint(checkpoint_dir=self.checkpoint_dir)
        else:
            self.start_epoch = 0
            
        for ep in range(self.start_epoch, self.num_epochs):
            if self.is_ddp:
                self.train_dl.sampler.set_epoch(ep)
                
            loss, acc = self._optimize_dataloader(dataloader=self.train_dl)
            loss_val, acc_val = self._validate_dataloader(dataloader=self.val_dl)
            msg = f"Epoch: {ep+1}/{self.num_epochs}, Train loss: {loss:.3f}, Train acc: {acc:.3f}, Val loss: {loss_val:.3f}, Val acc: {acc_val:.3f}"
            
            if self.is_ddp:
                if self.device == 0:
                    print(f"Device : DDP_GPU_RANK_{self.device}, {msg}")
                    self._save_checkpoint(epoch=ep, loss=loss, acc=acc)
            else:
                print(f"Device: {str(self.device).upper()}, {msg}")
                self._save_checkpoint(epoch=ep, loss=loss, acc=acc)
            
def prepare_datasets() -> Tuple["torch.utils.data.Dataset", "torch.utils.data.Dataset"]:
    
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    val_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, val_data
    
def collate_fn(batch: List[torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
    # batch is a list of items returned by __getitem__
    collate_batch_x = []
    collate_batch_y = []
    for example in batch:
        collate_batch_x.append(example[0].flatten())
        collate_batch_y.append(example[1])
    
    collate_batch_x = torch.stack(collate_batch_x, dim=0)
    collate_batch_y = torch.tensor(collate_batch_y)
    
    return (collate_batch_x, collate_batch_y)    

def ddp_setup(world_size: int, rank: int) -> None:
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device=rank)
    
def ddp_cleanup() -> None:
    
    torch.distributed.destroy_process_group()

def main(rank: int, is_ddp: bool, world_size: int, num_epochs: int, batch_size: int) -> None:
    
    if is_ddp:
        if torch.cuda.is_available():
            ddp_setup(world_size=world_size, rank=rank)
            device = rank
        else:
            raise ValueError("Cuda is not available, set is_ddp=False and run again")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = FeedForward(num_features=784, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = LossFunction()
    
    training_data, val_data = prepare_datasets()
    trainer = Trainer(model=model, 
                      device=device, 
                      is_ddp=is_ddp,
                      optimizer=optimizer, 
                      loss_fn=loss_fn, 
                      train_ds=training_data, 
                      val_ds=val_data, 
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      val_fraction=0.5,
                      checkpoint_dir=pathlib.Path(pathlib.Path.cwd(), "ckpt"))
    
    trainer.train(is_load_checkpoint=True)
    
    if is_ddp:
        ddp_cleanup()
    
if __name__ == "__main__":
    cuda_ids = [6,7]
    cvd = ""
    for i in cuda_ids:
        cvd += str(i) + ","
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    num_epochs = 10
    batch_size = 64
    is_ddp = True if len(cuda_ids) > 1 else False
    
    if is_ddp:
        world_size = len(cuda_ids)
        mp.spawn(fn=main, args=(is_ddp, world_size, num_epochs, batch_size), nprocs=world_size)
    else:
        main(rank=None, is_ddp=False, world_size=None, num_epochs=num_epochs, batch_size=batch_size)