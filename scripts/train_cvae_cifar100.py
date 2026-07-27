import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from models.cvae import ConditionalVAE

# ==========================================
# HIGH-CAPACITY PATCHED ARCHITECTURE
# ==========================================
class PatchedConditionalVAE(ConditionalVAE):
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        bottleneck_channels = self.decoder_input.out_features // 4
        result = result.view(-1, bottleneck_channels, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Overrides the default MSE loss with L1 Loss (Mean Absolute Error)
        to force the decoder to generate sharper edges.
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] 

        # Using L1 Loss instead of MSE
        recons_loss = F.l1_loss(recons, input, reduction='mean')

        # Standard KL Divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

# ==========================================
# MAXIMIZED CONFIGURATION
# ==========================================
CONFIG = {
    "dataset_path": "/public_datasets/PublicDatasets/cifar-100/",
    "out_dir": "../../checkpoints/vae",
    
    # Upgraded Architecture Capacity
    "img_size": 32,
    "in_channels": 3,
    "num_classes": 100,
    "latent_dim": 512,                           # Quadrupled latent bandwidth
    "hidden_dims": [64, 128, 256, 512],          # Doubled channel width
    
    # Training
    "batch_size": 128,               
    "lr": 0.002,                                 # Slightly lower base LR for the larger network
    "scheduler_gamma": 0.95,         
    "kld_weight": 0.0001,                        # Relaxed KL penalty to prioritize reconstruction
    "max_epochs": 1000,
    "seed": 1265,                    
    
    # Validation & Early Stopping
    "val_split": 0.10,             
    "early_stopping_patience": 30, 
    "save_interval": 50            
}
# ==========================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_stratified_dataloaders(dataset, val_split, batch_size):
    targets = np.array(dataset.targets)
    train_idx, val_idx = [], []
    
    for class_id in range(CONFIG["num_classes"]):
        idx = np.where(targets == class_id)[0]
        np.random.shuffle(idx)
        split_point = int(len(idx) * (1.0 - val_split))
        
        train_idx.extend(idx[:split_point])
        val_idx.extend(idx[split_point:])
        
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, len(train_subset), len(val_subset)

def train():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = PatchedConditionalVAE(
        in_channels=CONFIG["in_channels"], 
        num_classes=CONFIG["num_classes"], 
        latent_dim=CONFIG["latent_dim"], 
        img_size=CONFIG["img_size"], 
        hidden_dims=CONFIG["hidden_dims"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = ExponentialLR(optimizer, gamma=CONFIG["scheduler_gamma"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = datasets.CIFAR100(root=CONFIG["dataset_path"], train=True, transform=transform)
    
    train_loader, val_loader, train_size, val_size = get_stratified_dataloaders(
        full_dataset, 
        CONFIG["val_split"], 
        CONFIG["batch_size"]
    )

    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    kld_weight = CONFIG["kld_weight"]

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        total_train_loss = 0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels_onehot = F.one_hot(labels, num_classes=CONFIG["num_classes"]).float().to(device)
            
            optimizer.zero_grad()
            results = model(imgs, labels=labels_onehot)
            loss_dict = model.loss_function(*results, M_N=kld_weight)
            
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels_onehot = F.one_hot(labels, num_classes=CONFIG["num_classes"]).float().to(device)
                
                results = model(imgs, labels=labels_onehot)
                loss_dict = model.loss_function(*results, M_N=kld_weight) 
                total_val_loss += loss_dict['loss'].item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}/{CONFIG['max_epochs']} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % CONFIG["save_interval"] == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["out_dir"], f"cvae_cifar100_epoch_{epoch+1}.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["out_dir"], "cvae_cifar100_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs! Best Val Loss: {best_val_loss:.4f}")
                break

if __name__ == '__main__':
    train()