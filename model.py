import torch
import torch.nn as nn
from mamba_ssm import Mamba 

class VisionSSMBlock(nn.Module):
    """
    State-Space Model block adapting 2D feature maps to 1D continuous sequence processing.
    Achieves global receptive field with O(N) complexity.
    """
    def __init__(self, dim):
        super(VisionSSMBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        # Mamba operates as a selective state-space model
        self.mamba = Mamba(
            d_model=dim, # Model dimension (channels)
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # 2. Apply Normalization and SSM 
        # The SSM implicitly calculates the global atmospheric light dependencies
        x_mamba = self.mamba(self.norm(x_flat))
        
        # 3. Residual connection and reshape back to 2D spatial format
        out = (x_flat + x_mamba).transpose(1, 2).view(B, C, H, W)
        return out


class LFD_Net(nn.Module):
    def __init__(self):
        super(LFD_Net, self).__init__()

        # Feature Extraction Architecture
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv_layer1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        self.conv_layer3 = nn.Conv2d(32, 32, 7, 1, 3, bias=True)
        
        self.conv_layer5 = nn.Conv2d(64, 16, 3, 1, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(16, 3, 1, 1, 0, bias=True)

        # Gated Fusion
        self.gate = nn.Conv2d(32 * 3, 3, 3, 1, 1, bias=True)
        
        # Global Feature Interaction via SSM (Replaces CALayer and PALayer)
        # 64 represents the channel dimension of the concatenated tensor x7
        self.ssm_layer = VisionSSMBlock(dim=64)

    def forward(self, img):
        # Multi-scale feature extraction
        x1 = self.relu(self.conv_layer1(img))
        x2 = self.relu(self.conv_layer2(x1))
        x3 = self.relu(self.conv_layer3(x2))
        x4 = x1 + x3
        
        # Local Gated Fusion
        gates = self.gate(torch.cat((x1, x2, x4), 1))
        x6 = x1 * gates[:, [0], :, :] + x2 * gates[:, [1], :, :] + x4 * gates[:, [2], :, :]
        
        # Concatenate gated fusion output with deep features
        x7 = torch.cat((x6, x3), 1)
        
        # --- SSM Integration ---
        # x7 is passed through the SSM. 
        # This acts as a spatial-spectral global routing mechanism.
        x_global = self.ssm_layer(x7)
        # -----------------------

        # High-resolution reconstruction stage
        x10 = self.relu(self.conv_layer5(x_global))
        x11 = self.conv_layer6(x10)

        # Reformulated Atmospheric Scattering Model (ASM) output projection
        dehaze_image = self.relu((x11 * img) - x11 + 1)

        return dehaze_image