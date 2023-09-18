
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path

# Directory containing the images
image_folder_base = '/training_data/'

folders = os.listdir(image_folder_base)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding

        x = self.transformer(x,x)

        latent_representation = x[:, 0]  # The first token represents the aggregated information

        output = self.fc(latent_representation)
        return output

# Hyperparameters
image_size = 224
patch_size = 16
num_classes = 10
dim = 256
num_heads = 8
num_layers = 6


# Instantiate the ViT model
vit_model = ViT(image_size, patch_size, num_classes, dim, num_heads, num_layers)


# Loop through each image file
for folder_name in folders:
    image_folder = os.path.join(image_folder_base, folder_name)

    image_filenames = Path(image_folder).glob('*.png')
    for image_filename in image_filenames:
        image_path = os.path.join(image_folder_base, image_filename)

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor()])
        example_image = preprocess(image).unsqueeze(0)

        # Get the latent space representation
        latent_representation = vit_model(example_image)
        
        print(latent_representation.shape)
        print(latent_representation) #Print the shape of the latent space representation
        
