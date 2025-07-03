import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import shutil

# Constants
ROOT_DIR = ""
FIRST_MODEL_PATH = ""
METADATA_CSV = os.path.join(ROOT_DIR, "data/metadata/clean_metadata.csv")
ATTN_NOISE_DATA_DIR_NAME = ""
ATTN_NOISE_DATA_DIR = os.path.join(ROOT_DIR, ATTN_NOISE_DATA_DIR_NAME)
ATTN_NOISE_METADATA_CSV = os.path.join(ROOT_DIR, "")

IMAGE_SIZE = 224
PATCH_SIZE = 16

LABELS_MAP = {
    "melanoma": 0,
    "nevus": 1,
    "basal cell carcinoma": 2,
    "squamous cell carcinoma": 3,
    "dermatofibroma": 4,
    "benign keratosis": 5,
    "actinic keratosis": 6,
    "vascular lesion": 7
}

labels_map_inv = {v: k for k, v in LABELS_MAP.items()}

# Class for generating noise based on attention maps
class AttentionNoiseGenerator:
    def __init__(self, attention_model_instance, device, attention_threshold=0.6, noise_probability=0.6, image_target_size=224, patch_size=16):
        self.attention_model_instance = attention_model_instance
        self.device = device
        self.attention_threshold = attention_threshold
        self.noise_probability = noise_probability
        self.image_target_size = image_target_size
        self.patch_size = patch_size
        self.preprocess_for_attention = transforms.Compose([
            transforms.Resize((self.image_target_size, self.image_target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def get_attention_map(self, pil_image):
        # Prepare image and model
        img_tensor = self.preprocess_for_attention(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.attention_model_instance(img_tensor)
            attentions = outputs.attentions
        
        # Use last layer's to look at the multihead attention
        last_layer_attentions = attentions[-1]

        # Average the attention weights across all heads
        avg_attentions = last_layer_attentions.mean(dim=1).squeeze(0)

        num_patches_side = self.image_target_size // self.patch_size
        num_patches_total = num_patches_side * num_patches_side
        actual_num_patches = attentions[-1].shape[-1] - 1 

        # Remove the cls token
        cls_attention_to_patches = avg_attentions[0, 1:actual_num_patches+1] 

        # retrive the width and height of image in terms of patches
        sqrt_actual_num_patches = np.sqrt(actual_num_patches)
        num_patches_side_actual = int(sqrt_actual_num_patches)

        # Reshape the attention map to match the patch grid
        attention_map_patches = cls_attention_to_patches.reshape(num_patches_side_actual, num_patches_side_actual)
        
        # unsample the attention map to the original image size
        attention_map_resized = F.interpolate(
            attention_map_patches.unsqueeze(0).unsqueeze(0),
            size=(self.image_target_size, self.image_target_size),
            mode='bilinear', align_corners=False
        ).squeeze()
        
        # normalize the attention map to [0, 1] with a small epsilon to avoid division by zero
        attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / \
                                   (attention_map_resized.max() - attention_map_resized.min() + 1e-6)

        return attention_map_normalized.cpu().numpy()

    def apply_noise_to_image(self, pil_image_original_size):
        # Determine if noise should be applied based on the probability
        if np.random.rand() >= self.noise_probability:
            return pil_image_original_size 

        # Get attention map
        attention_map = self.get_attention_map(pil_image_original_size) 

        # final important regions
        important_region_mask = (attention_map > self.attention_threshold) 
        
        # skip if no important regions are found
        if not np.any(important_region_mask):
            return pil_image_original_size

       # Convert to target size so that it can be campared with the attention map
        pil_image_resized_for_noise = pil_image_original_size.resize(
            (self.image_target_size, self.image_target_size), 
            Image.Resampling.LANCZOS 
        )
        img_array = np.array(pil_image_resized_for_noise).astype(np.float32) 
        
        noise_strength = np.random.uniform(20, 60) 
        
        # Generate noise matching the shape of the resized image array
        # Create an array of noise with 0 and variance of noise_strength
        noise = np.random.normal(0, noise_strength, img_array.shape) 
        
        # Look at rbg channels and apply noise to the important regions
        for c in range(img_array.shape[2]): 
            img_array[:, :, c][important_region_mask] += noise[:, :, c][important_region_mask]
        

        img_array_noised = np.clip(img_array, 0, 255).astype(np.uint8)
        noised_pil_image_at_target_size = Image.fromarray(img_array_noised) 
        
        # return the noised image
        return noised_pil_image_at_target_size 

def pregenerate_attention_noisy_data():
    print("Starting pre-generation of attention-guided noisy images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get current attention model
    attention_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(LABELS_MAP),
        ignore_mismatched_sizes=True,
        output_attentions=True
    )
    attention_model.load_state_dict(torch.load(FIRST_MODEL_PATH, map_location=device))
    attention_model.to(device)
    attention_model.eval()

    # Initialize the noise generator
    noise_gen = AttentionNoiseGenerator(attention_model, device, image_target_size=IMAGE_SIZE, patch_size=PATCH_SIZE)

    meta_df_full = pd.read_csv(METADATA_CSV)
    train_df_orig = meta_df_full[meta_df_full["set"] == "train"].copy()

    os.makedirs(ATTN_NOISE_DATA_DIR, exist_ok=True)

    new_metadata_rows = []
    processed_counter = 0

    # Loop through each image in the training set
    for idx, row in tqdm(train_df_orig.iterrows(), total=len(train_df_orig), desc="Generating Attention Noisy Images"):
        original_img_rel_path = row['image_path']
        original_img_full_path = os.path.join(ROOT_DIR, original_img_rel_path)
        pil_image = Image.open(original_img_full_path).convert('RGB')

        # Apply the noise to the image
        noised_pil_image = noise_gen.apply_noise_to_image(pil_image)

        # Save the noised image
        base_name = os.path.splitext(os.path.basename(original_img_rel_path))[0]
        new_img_filename = f"{base_name}_attn_noise_{processed_counter}.jpg" 
        new_img_save_full_path = os.path.join(ATTN_NOISE_DATA_DIR, new_img_filename)
        noised_pil_image.save(new_img_save_full_path)
        new_img_rel_save_path = os.path.join(ATTN_NOISE_DATA_DIR_NAME, new_img_filename)
        new_metadata_rows.append({
            'isic_id': row['isic_id'] + f"_attn_noise_{processed_counter}", 
            'diagnosis_fine': row['diagnosis_fine'],
            'image_path': new_img_rel_save_path,
            'diagnosis': row['diagnosis'],
            'label': row['label'],
            'set': 'train_attn_noise' 
        })
        processed_counter += 1

    # Save the meta data for new noised images
    if new_metadata_rows:
        attn_noise_df = pd.DataFrame(new_metadata_rows)
        attn_noise_df.to_csv(ATTN_NOISE_METADATA_CSV, index=False)
        print(f"Generated {len(attn_noise_df)} attention-noised image records.")
        print(f"Metadata for attention-noised images saved to: {ATTN_NOISE_METADATA_CSV}")

if __name__ == '__main__':
    pregenerate_attention_noisy_data()