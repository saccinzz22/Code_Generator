import torch # type: ignore
from tqdm import tqdm # type: ignore

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.device=device
        self.scheduler=scheduler
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss=0
        pbar=tqdm(dataloader, desc="Training")
        for batch in pbar:
            src=batch["src"].to(self.device)
            decoder_input=batch["decoder_input"].to(self.device)
            labels=batch["labels"].to(self.device)
            logits=self.model(src, decoder_input)
            vocab_size=logits.shape[-1]
            loss=self.criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss+=loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss=total_loss/len(dataloader)
        return avg_loss
    @torch.no_grad()
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss=0
        pbar=tqdm(dataloader, desc="Validation")
        for batch in pbar:
            src=batch["src"].to(self.device)
            decoder_input=batch["decoder_input"].to(self.device)
            labels=batch["labels"].to(self.device)
            logits=self.model(src, decoder_input)
            vocab_size=logits.shape[-1]
            loss=self.criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))
            total_loss+=loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss=total_loss/len(dataloader)
        return avg_loss
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        torch.save({"epoch": epoch, 
                    "train_loss": train_loss, 
                    "val_loss": val_loss, 
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                    path)
        
    def load_checkpoint(self, path):
        checkpoint=torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch=checkpoint["epoch"]
        train_loss=checkpoint["train_loss"]
        val_loss=checkpoint["val_loss"]
        return (epoch, train_loss, val_loss)