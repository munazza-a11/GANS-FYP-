"""
Training script for Autoencoder on CLEAN data only
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import AdversarialAutoencoder

def get_clean_dataloader(data_path, batch_size=32, shuffle=True):
    """Load clean images only"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    
    return dataloader

def train_autoencoder(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train autoencoder on clean data"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, _ in train_bar:
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, _ in val_bar:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'models/autoencoder_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'models/autoencoder_clean_final.pth')
    print("\n✅ Training complete! Model saved.")
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/autoencoder_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_reconstructions(model, dataloader, device, num_samples=5):
    """Visualize original vs reconstructed images"""
    model.eval()
    
    images, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # Move to CPU for plotting
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/autoencoder_reconstruction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load data (CLEAN ONLY)
    print("📂 Loading clean datasets...")
    train_loader = get_clean_dataloader('data/clean', batch_size=32, shuffle=True)
    val_loader = get_clean_dataloader('data/val_clean', batch_size=32, shuffle=False)
    
    # Initialize model
    print("🏗️  Initializing Autoencoder...")
    model = AdversarialAutoencoder(input_channels=3, latent_dim=128)
    
    # Train
    print("🚀 Starting training...\n")
    train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader, device, epochs=20, lr=1e-3
    )
    
    # Plot results
    print("📊 Plotting training curves...")
    plot_training_curves(train_losses, val_losses)
    
    # Visualize reconstructions
    print("🖼️  Visualizing reconstructions...")
    visualize_reconstructions(model, val_loader, device)
    
    print("\n✨ All done! Check 'models/' and 'results/' directories.")
