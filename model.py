import torch
import torch.nn as nn
from mamba_ssm import Mamba 

class VisionSSMBlock(nn.Module):
    """
    State-Space Model block adapting 2D feature maps to 1D continuous sequence processing.
    Uses Patch Embedding to compress the sequence length and prevent OOM errors.
    """
    def __init__(self, dim, patch_size=8):
        super(VisionSSMBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = dim * 2  # Expand channels slightly to retain information lost spatially
        
        # 1. Patch Embedding (Downsampling spatial dims, expanding channels)
        self.patch_embed = nn.Conv2d(dim, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 2. Mamba Block (Now operates on a tiny, manageable sequence)
        self.mamba = Mamba(
            d_model=self.embed_dim, 
            d_state=16,  
            d_conv=4,    
            expand=2,    
        )
        
        # 3. Patch Reconstruction (Upsampling back to original spatial resolution)
        self.patch_unembed = nn.ConvTranspose2d(self.embed_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- 1. Patchify ---
        # Shape: (B, embed_dim, H/P, W/P)
        x_patched = self.patch_embed(x)
        _, _, H_p, W_p = x_patched.shape
        
        # Flatten: (B, embed_dim, H_p * W_p) -> Transpose: (B, L, embed_dim)
        x_flat = x_patched.flatten(2).transpose(1, 2)
        
        # --- 2. State-Space Sequence Modeling ---
        x_mamba = self.mamba(self.norm(x_flat))
        
        # --- 3. Unpatchify ---
        # Add residual in the latent space, then reshape to 2D
        x_res = (x_flat + x_mamba).transpose(1, 2).view(B, self.embed_dim, H_p, W_p)
        
        # Project back to original spatial dimensions
        out = self.patch_unembed(x_res)
        
        # Final global residual connection 
        return out + x


class LFD_Net_SSM(nn.Module):
    def __init__(self):
        super(LFD_Net_SSM, self).__init__()

        # Feature Extraction Architecture
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv_layer1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv_layer2 = nn.Conv2d(32, 32, 5, 1, 2, bias=True)
        self.conv_layer3 = nn.Conv2d(32, 32, 7, 1, 3, bias=True)
        
        self.conv_layer5 = nn.Conv2d(64, 16, 3, 1, 1, bias=True)
        self.conv_layer6 = nn.Conv2d(16, 3, 1, 1, 0, bias=True)

        # Gated Fusion
        self.gate = nn.Conv2d(32 * 3, 3, 3, 1, 1, bias=True)
        
        # Global Feature Interaction via Patch-SSM
        # We pass the channel dimension (64) and set patch_size=8
        self.ssm_layer = VisionSSMBlock(dim=64, patch_size=8)

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
        # The SSM now safely processes the features without memory explosion
        x_global = self.ssm_layer(x7)
        # -----------------------

        # High-resolution reconstruction stage
        x10 = self.relu(self.conv_layer5(x_global))
        x11 = self.conv_layer6(x10)

        # Reformulated Atmospheric Scattering Model (ASM) output projection
        dehaze_image = self.relu((x11 * img) - x11 + 1)

        return dehaze_image