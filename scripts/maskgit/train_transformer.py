import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from accelerate import Accelerator
from torch.optim import Adam

# Monkey-patch HuggingFace Transformers v5 compatibility
from transformers import PreTrainedTokenizerBase
PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__

def main():
    # Use BF16 to prevent T5 NaNs
    accelerator = Accelerator(mixed_precision="bf16")
    
    dataset_root = "../../data"
    results_dir = "../../checkpoints/maskgit/transformer"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load the frozen Stage 1 VQGAN Tokenizer
    # THE FIX: num_tokens is 65536, NOT 8192!
    vae = VQGanVAE(dim=64, channels=3, layers=2, discr_layers=2, num_tokens=65536, temperature=0.9)
    
    print("Loading EMA weights...")
    ema_state_dict = torch.load("../../checkpoints/maskgit/vqgan/vae.49000.ema.pt", map_location="cpu")
    clean_state_dict = {
        k.replace("ema_model.", ""): v 
        for k, v in ema_state_dict.items() 
        if k.startswith("ema_model.")
    }
    
    vae.load_state_dict(clean_state_dict)
    vae.requires_grad_(False)
    vae.eval()
    
    # 2. Create the Bidirectional Transformer
    transformer = MaskGitTransformer(
        num_tokens = 65536,  # THE FIX: Matches the true VAE vocabulary!
        seq_len = 64,        # THE FIX: Matches the true 8x8 visual grid!
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = False
    )
    
    # 3. Create the MaskGit Wrapper (No hacks needed anymore!)
    maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        image_size = 32,
        cond_drop_prob = 0.1,  
    )
    
    # 4. Standard CIFAR-100 Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = datasets.CIFAR100(root=dataset_root, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
    
    # 5. Optimizer
    optim = Adam(maskgit.parameters(), lr=3e-4)
    
    # 6. Wrap everything with Accelerate
    maskgit, optim, dataloader = accelerator.prepare(maskgit, optim, dataloader)
    
    # 7. The Training Loop
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
            optim.step()
            
            if global_step % 100 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                
            global_step += 1
            
        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            save_path = os.path.join(results_dir, f"maskgit_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(maskgit).state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    print("Stage 2 Training Complete!")

if __name__ == "__main__":
    main()