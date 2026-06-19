import os
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader # type: ignore
from model.dataset import CodingDataset
from model.transformer import Transformer
from model.trainer import Trainer
from configs import *

def main():
    device=torch.device(DEVICE)
    print()
    print("Using device:")
    print(device)
    print()
    train_dataset=CodingDataset(
        path="preprocessing/train.jsonl", 
        src_max_len=SRC_MAX_LEN, 
        tgt_max_len=TGT_MAX_LEN, 
        pad_idx=PAD_IDX)
    
    val_dataset=CodingDataset(
        path="preprocessing/val.jsonl", 
        src_max_len=SRC_MAX_LEN, 
        tgt_max_len=TGT_MAX_LEN, 
        pad_idx=PAD_IDX)
    
    train_loader=DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True)
    val_loader=DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False)
    
    model=Transformer(
        src_vocab_size=VOCAB_SIZE, 
        trg_vocab_size=VOCAB_SIZE, 
        src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, 
        embed_size=EMBED_SIZE, head=HEADS, 
        forward_expansion=FORWARD_EXPANSION, 
        dropout=DROPOUT, num_layers=NUM_LAYERS, 
        src_max_length=SRC_MAX_LEN, 
        trg_max_length=TGT_MAX_LEN, 
        device=device).to(device)
    
    optimizer=torch.optim.AdamW(
        model.parameters(), 
        lr=LR, betas=(0.9,0.98), 
        weight_decay=WEIGHT_DECAY)
    
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion=nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    trainer=Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, scheduler=scheduler)
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss=float("inf")
    
    for epoch in range(EPOCHS):
        print()
        print("="*50)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("="*50)
        train_loss=trainer.train_epoch(train_loader)
        val_loss=trainer.validate(val_loader)
        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        trainer.save_checkpoint(path=f"checkpoints/epoch_{epoch+1}.pt", epoch=epoch+1, train_loss=train_loss, val_loss=val_loss)
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            trainer.save_checkpoint(path="checkpoints/best.pt", epoch=epoch+1, train_loss=train_loss, val_loss=val_loss)
            print()
            print("Best model saved.")
        scheduler.step()
    print()
    print("Training completed.")
if __name__=="__main__":
    main()