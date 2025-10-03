import torch.nn as nn
import torch
import torch.nn.functional as F

class FeatureClassifier(nn.Module):
    def __init__(self, unet_model):
        super(FeatureClassifier, self).__init__()
        
        # Frozen U-Netu sed purely as a feature extractor.
        self.unet_feature_extractor = unet_model
        
         # Freeze ALL parameters of the backbone so only the head learns.
        for param in self.unet_feature_extractor.parameters():
            param.requires_grad = False 
        
        # Number of channels produced by the backbone's feature map.
        FEATURE_CHANNELS = 512 
        
        # Classification head
        self.classifier_head = nn.Sequential(
            # Global Average Pooling: riduce (512, H/32, W/32) a (512, 1, 1)
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(FEATURE_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # Output for Binary classification
            #nn.Sigmoid() # Sigmoid is intentionally omitted so we can train with BCEWithLogitsLoss.
        )

    def forward(self, x):
        # Extract deep features from the frozen U-Net backbone.
        x = self.unet_feature_extractor.extract_features(x)
        
        # Classification head to get a single logit per sample.
        x = self.classifier_head(x)
        return x


class SimpleSegNet(nn.Module):
    """
    A compact SegNet-style encoder–decoder for semantic segmentation.

    Notes:
    - This model returns **logits** by default (no final activation). 
      Apply `torch.sigmoid` (binary) or `torch.softmax` (multi-class) outside.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleSegNet, self).__init__()
        
        # --- ENCODER (reduced channels) ---
        self.enc1 = self._make_encoder_block(in_channels, 32)
        self.enc2 = self._make_encoder_block(32, 64)
        self.enc3 = self._make_encoder_block(64, 128)
        
         # Block 4 (bottleneck): 128 -> 128, no pooling here
        self.enc4 = self._make_encoder_block(128, 128, final_block=True) 
        
        
        # --- DECODER (mirrored, reduced channels) ---
        self.dec4 = self._make_decoder_block(128, 64) 
        self.dec3 = self._make_decoder_block(64, 32)
        self.dec2 = self._make_decoder_block(32, 16)
        
        # Final conv: 16 -> out_channels (no BN/ReLU)
        self.dec1 = self._make_decoder_block(16, out_channels, final_layer=True)


    def _make_encoder_block(self, in_c, out_c, kernel_size=3, padding=1, final_block=False):
        """Standard encoder block: (Conv-BN-ReLU) x 2."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_c, out_c, kernel_size=3, padding=1, final_layer=False):
        if final_layer:
            return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
        else:
            # Blocco di decodifica intermedio
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=kernel_size, padding=padding), 
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        
        # --- ENCODER ---
        # Block 1
        size1_pre_pool = x.size() 
        x = self.enc1(x)
        x, indices1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Block 2
        size2_pre_pool = x.size()
        x = self.enc2(x)
        x, indices2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Block 3
        size3_pre_pool = x.size()
        x = self.enc3(x)
        x, indices3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        # Bottleneck (no pooling)
        x = self.enc4(x)
        
        
        # --- DECODER ---
        # Decode 4
        # x e indices3 hanno entrambi 256 canali. output_size è la dimensione PRIMA del pool3.
        x = F.max_unpool2d(x, indices3, kernel_size=2, stride=2, output_size=size3_pre_pool)
        x = self.dec4(x) 
        
        # Decode 3
        x = F.max_unpool2d(x, indices2, kernel_size=2, stride=2, output_size=size2_pre_pool)
        x = self.dec3(x)
        
        # Decode 2
        x = F.max_unpool2d(x, indices1, kernel_size=2, stride=2, output_size=size1_pre_pool)
        x = self.dec2(x)
        
        # Final projection to output classes/channels (logits)
        x = self.dec1(x)
        
        return x


class CustomUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(CustomUNet, self).__init__()
        
        # ENCODER: feature extraction with downsampling via MaxPool
        # Block 1: 3 -> 16
        self.c1 = ConvBlock(in_channels, 16, 0.1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2: 16 -> 32
        self.c2 = ConvBlock(16, 32, 0.1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 3: 32 -> 64
        self.c3 = ConvBlock(32, 64, 0.2) # Dropout 0.2
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 4: 64 -> 128
        self.c4 = ConvBlock(64, 128, 0.2) # Dropout 0.2
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 5: 128 -> 256
        self.c5 = ConvBlock(128, 256, 0.3) # Dropout 0.3
        self.p5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # BOTTLENECK: deepest features (no further pooling)
        # Block 6: 256 -> 512
        self.c6 = ConvBlock(256, 512, 0.3) # Dropout 0.3


        # DECODER: upsample (transpose conv), concat skip, refine with ConvBlock
        # Block 7 (Back 1): 512 -> 256
        self.u7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = ConvBlock(512, 256, 0.3) # 512 = 256 (UpConv) + 256 (c5 Skip)
        # Block 8 (Back 2): 256 -> 128
        self.u8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = ConvBlock(256, 128, 0.2) # 256 = 128 (UpConv) + 128 (c4 Skip)
        # Block 9 (Back 3): 128 -> 64
        self.u9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = ConvBlock(128, 64, 0.2) # 128 = 64 (UpConv) + 64 (c3 Skip)
        # Block 10 (Back 4): 64 -> 32
        self.u10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c10 = ConvBlock(64, 32, 0.1) # 64 = 32 (UpConv) + 32 (c2 Skip)
        # Block 11 (Back 5): 32 -> 16
        self.u11 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c11 = ConvBlock(32, 16, 0.1) # 32 = 16 (UpConv) + 16 (c1 Skip)

        # 1×1 convolution to map features to desired output channels (e.g., 1 for binary mask)
        self.outputs = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # Apply He/Kaiming initialization to conv / deconv layers
        self.apply(weights_init)


    def forward(self, x):
        # x: (N, C, H, W)
        # ENCODER (save outputs for skip connections)
        c1 = self.c1(x) # c1 (skip)
        p1 = self.p1(c1)
        c2 = self.c2(p1) # c2 (skip)
        p2 = self.p2(c2)
        c3 = self.c3(p2) # c3 (skip)
        p3 = self.p3(c3)
        c4 = self.c4(p3) # c4 (skip)
        p4 = self.p4(c4)
        c5 = self.c5(p4) # c5 (skip)
        p5 = self.p5(c5)
        
        # BOTTLENECK
        c6 = self.c6(p5) 
        
        # DECODER: upsample -> concat skip -> conv block
        u7 = self.u7(c6)
        u7 = torch.cat([u7, c5], dim=1) 
        c7 = self.c7(u7)
        # Block 8 (Back 2): UpConv + c4
        u8 = self.u8(c7)
        u8 = torch.cat([u8, c4], dim=1) 
        c8 = self.c8(u8)
        # Block 9 (Back 3): UpConv + c3
        u9 = self.u9(c8)
        u9 = torch.cat([u9, c3], dim=1) 
        c9 = self.c9(u9)
        # Block 10 (Back 4): UpConv + c2
        u10 = self.u10(c9)
        u10 = torch.cat([u10, c2], dim=1)
        c10 = self.c10(u10)
        # Block 11 (Back 5): UpConv + c1
        u11 = self.u11(c10)
        u11 = torch.cat([u11, c1], dim=1)
        c11 = self.c11(u11)
        
        # OUTPUT
        # Produce logits and squash with sigmoid for probabilities in [0,1]
        # Last conv layer (kernel 1x1, 1 canale di output)
        logits = self.outputs(c11)
        output = torch.sigmoid(logits) 
        return output

    def extract_features(self, x):
        """Extract compressed features from the bottleneck for the classification task of Bonus A"""
        # ENCODER
        c1 = self.c1(x) 
        p1 = self.p1(c1)
        
        c2 = self.c2(p1) 
        p2 = self.p2(c2)
        
        c3 = self.c3(p2) 
        p3 = self.p3(c3)
        
        c4 = self.c4(p3) 
        p4 = self.p4(c4)
        
        c5 = self.c5(p4) 
        p5 = self.p5(c5)
        
        # Bottleneck
        c6 = self.c6(p5) # Output compressed features (N, 512, H/32, W/32)
        return c6

# === Initialization function 'he_normal' ===
def weights_init(m):
    if isinstance(m, nn.Conv2d):
         # Apply Kaiming/He initialization to Conv2D weights.
        # In Keras, 'he_normal' corresponds to Kaiming initialization for ReLU activations.
        # Here we use 'kaiming_uniform_' as the closest match (uniform instead of normal).
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        # Initialize biases to 0 (if present).
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.ConvTranspose2d):
        # Same initialization strategy for ConvTranspose2d (used in decoders/upsampling).
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# === Convolutional block for Encoder and Decoder ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ConvBlock, self).__init__()
        
        # Sequential container of layers (executed in order).
        self.conv_seq = nn.Sequential(
             # Conv 1: 3x3 kernel, padding=1 to keep same H/W size, followed by ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv 2: another 3x3 + ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            #  Dropout for regularization (spatial dropout: drops entire channels at once)
            nn.Dropout2d(dropout_rate),
            
            # Conv 3: another 3x3 + ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Pass input through the sequence of conv + activation + dropout layers
        return self.conv_seq(x)
