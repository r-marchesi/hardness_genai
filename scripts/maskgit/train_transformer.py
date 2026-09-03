import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from accelerate import Accelerator
from torch.optim import Adam
from transformers import PreTrainedTokenizerBase

PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

def main():
    accelerator = Accelerator(mixed_precision="bf16")
    dataset_root = "../../data"
    results_dir = "../../checkpoints/maskgit/transformer"
    os.makedirs(results_dir, exist_ok=True)
    
    vae = VQGanVAE(dim=128, channels=3, layers=2, discr_layers=2, codebook_size=256)
    
    ema_path = "../../checkpoints/maskgit/vqgan/vae.49000.ema.pt"
    if not os.path.exists(ema_path):
        ema_path = "../../checkpoints/maskgit/vqgan/vae.50000.ema.pt"
        
    ema_state_dict = torch.load(ema_path, map_location="cpu")
    clean_state_dict = {k.replace("ema_model.", ""): v for k, v in ema_state_dict.items() if k.startswith("ema_model.")}
    
    vae.load_state_dict(clean_state_dict)
    vae.requires_grad_(False)
    vae.eval()
    
    transformer = MaskGitTransformer(
        num_tokens = 256,
        seq_len = 64,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = False
    )
    
    # THE FIX: Higher dropout forces the model to stop memorizing the text prompts
    maskgit = MaskGit(
        vae=vae, 
        transformer=transformer, 
        image_size=32, 
        cond_drop_prob=0.25  
    )
    
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = datasets.CIFAR100(root=dataset_root, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
    
    # THE FIX: Higher weight decay to prevent mathematical overconfidence
    optim = Adam(maskgit.parameters(), lr=1e-4, weight_decay=0.05)
    maskgit, optim, dataloader = accelerator.prepare(maskgit, optim, dataloader)
    
    epochs = 250  
    global_step = 0
    
    print("Starting MaskGIT Stage 2 (Transformer) Training...")
    for epoch in range(epochs):
        maskgit.train()
        for images, labels in dataloader:
            optim.zero_grad()
            texts = [f"class {label.item()}" for label in labels]
            loss = maskgit(images, texts=texts)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(maskgit.parameters(), max_norm=1.0)
            optim.step()
            
            if global_step % 100 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
            global_step += 1
            
        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            save_path = os.path.join(results_dir, f"maskgit_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(maskgit).state_dict(), save_path)

if __name__ == "__main__":
    main()