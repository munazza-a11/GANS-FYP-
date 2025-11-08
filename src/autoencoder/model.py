import torch
import torch.nn as nn

class AdversarialAutoencoder(nn.Module):
  
    def __init__(self, input_channels=3, latent_dim=128):
        super(AdversarialAutoencoder, self).__init__()
        
        # Encoder: Compress image to latent representation
        self.encoder = nn.Sequential(
            # Input: 3x224x224
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256x14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, latent_dim)
        )
        
        # Decoder: Reconstruct image from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 14, 14)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),  # 3x224x224
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        """Forward pass through encoder and decoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_reconstruction_error(self, x):
        """
        Calculate reconstruction error (MSE) for anomaly detection.
        Higher error = likely adversarial
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Per-sample MSE
            error = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
        return error

    def detect_adversarial(self, x, threshold=0.05):
        """
        Detect if image is adversarial based on reconstruction error
        
        Args:
            x: Input image tensor
            threshold: Error threshold (tune based on validation)
        
        Returns:
            is_adversarial (bool), reconstruction_error (float)
        """
        error = self.get_reconstruction_error(x)
        is_adversarial = error > threshold
        return is_adversarial, error
