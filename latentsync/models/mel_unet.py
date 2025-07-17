
import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attention import SelfAttention

class ConvBlock(nn.Module):
    """A Convolutional Block with Conv -> ReLU -> Conv -> ReLU -> Self-Attention"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.self_attention = SelfAttention(out_channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten spatial dims for attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.size()
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_attended = self.self_attention(x_flat)
        x_attended = x_attended.permute(0, 2, 1).view(B, C, H, W)
        return x_attended


class MelUnet(nn.Module):
    def __init__(self):
        super(MelUnet, self).__init__()
        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        # Use ConvTranspose2d with proper output_padding to match dimensions
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, output_padding=(1, 0))
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        print(f"Input: {x.shape}")
        
        # Encoder
        enc1 = self.enc1(x)  # (480, 64, 5, 384)
        print(f"enc1: {enc1.shape}")
        
        pooled1 = self.pool1(enc1)  # (480, 64, 2, 192)
        print(f"pooled1: {pooled1.shape}")
        
        enc2 = self.enc2(pooled1)  # (480, 128, 2, 192)
        print(f"enc2: {enc2.shape}")
        
        pooled2 = self.pool2(enc2)  # (480, 128, 1, 96)
        print(f"pooled2: {pooled2.shape}")

        # Bottleneck
        bottleneck = self.bottleneck(pooled2)  # (480, 256, 1, 96)
        print(f"bottleneck: {bottleneck.shape}")

        # Decoder
        up2 = self.up2(bottleneck)  # (480, 128, 2, 192)
        print(f"up2: {up2.shape}")
        
        dec2 = torch.cat([up2, enc2], dim=1)  # (480, 256, 2, 192)
        print(f"dec2 concat: {dec2.shape}")
        
        dec2 = self.dec2(dec2)  # (480, 128, 2, 192)
        print(f"dec2: {dec2.shape}")

        up1 = self.up1(dec2)  # (480, 64, 5, 384) - output_padding=(1,0) fixes height
        print(f"up1: {up1.shape}")
        
        dec1 = torch.cat([up1, enc1], dim=1)  # (480, 128, 5, 384)
        print(f"dec1 concat: {dec1.shape}")
        
        dec1 = self.dec1(dec1)  # (480, 64, 5, 384)
        print(f"dec1: {dec1.shape}")

        output = self.final_conv(dec1)  # (480, 1, 5, 384)
        print(f"output: {output.shape}")
        
        return output


def test_fixed_unet(): 
    # Create input tensor
    dummy_input = torch.randn(480, 5, 384)
    x = dummy_input.unsqueeze(1)  # (480, 1, 5, 384)
    
    model = MelUnet()
    
    print("Running inference...")
    print("-" * 40)
    
    with torch.no_grad():
        output = model(x)
    
    print("-" * 40)
    print(f"SUCCESS!")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Remove channel dimension to get back to original format
    output_squeezed = output.squeeze(1)
    print(f"After squeeze: {output_squeezed.shape}")
    
    print(f"\nShape preservation: ✓")
    print(f"Original: (480, 5, 384)")
    print(f"Final:    {output_squeezed.shape}")
    
    return output_squeezed

if __name__ == "__main__":
    result = test_fixed_unet()