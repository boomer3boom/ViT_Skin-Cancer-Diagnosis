import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification 
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import shutil
import re 

# Roots
ROOT_DIR = ""
print(f"CUDA Available: {torch.cuda.is_available()}")
# Original data
metadata_csv = os.path.join(ROOT_DIR, "data/metadata/clean_metadata.csv")
MODEL_PATH = ""
print("Loading metadata_csv...")
meta_df = pd.read_csv(metadata_csv)
labels_map = {
    "melanoma": 0,
    "nevus": 1,
    "basal cell carcinoma": 2,
    "squamous cell carcinoma": 3,
    "dermatofibroma": 4,
    "benign keratosis": 5,
    "actinic keratosis": 6,
    "vascular lesion": 7
}
labels_map_inv = {v: k for k, v in labels_map.items()}

adversarial_data_dir = ROOT_DIR

image_size = 224

# Dataset class for BCN dataset
class BCNDataset(Dataset):
    def __init__(self, meta_df, transform=None, root_dir=ROOT_DIR):
        self.paths = meta_df["image_path"].values
        self.labels = meta_df["label"].values.astype(int)

        # Augment the data, we call this transform data
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_rel_path = self.paths[idx]
        img_full_path = os.path.join(self.root_dir, img_rel_path)
        image = Image.open(img_full_path).convert('RGB')
        label = self.labels[idx]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, label


# Load the train, validate, and test sets
print("Preparing original dataframes...")
train_df_orig = meta_df[meta_df["set"] == "train"].copy()
val_df = meta_df[meta_df["set"] == "validation"].copy()
test_df = meta_df[meta_df["set"] == "test"].copy()
meta_df_indexed = meta_df.set_index('isic_id')
train_class_counts = train_df_orig['label'].value_counts().sort_index()
max_count = train_class_counts.max()

# preprocessing
adv_preprocess_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

device_adv_gen = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model instance
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(labels_map),
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device_adv_gen))
model.to(device_adv_gen)
criterion = nn.CrossEntropyLoss()

def fgsm_attack(image, epsilon, data_grad):
    # Generate adversarial example using FGSM
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()
    return perturbed_image


def denormalize_image(tensor_image):
    # Ensure the tensor is on CPU and detached from the computation graph
    tensor_image = tensor_image.cpu().detach()    
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1) 
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1) 

    # Denormalize the tensor: output = input * std + mean
    denormalized_tensor = tensor_image * std + mean
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)

    to_pil = transforms.ToPILImage()
    pil_image = to_pil(denormalized_tensor)

    return pil_image


# Create adv examples for under-represented classes in the training set
for class_label, current_count in train_class_counts.items():
    # Calculate number of examples to generate
    num_to_generate = max_count - current_count
    class_name = labels_map_inv.get(class_label, f"Unknown Class {class_label}")
    if num_to_generate <= 0:
        print(f"Class {class_label} ({class_name}) is already balanced or over-represented.")
        continue

    print('got upto here')

    print(f"Generating {num_to_generate} for class {class_label} ({class_name})")
    class_specific_df = train_df_orig[train_df_orig['label'] == class_label]

    for i in tqdm(range(num_to_generate), desc=f"Class {class_label} ({class_name}) Gen", unit="img", leave=False):
        original_row = class_specific_df.iloc[i % len(class_specific_df)] 
        original_img_rel_path = original_row['image_path']

        current_epsilon = np.random.uniform(0.01, 0.1) 

        original_img_full_path = os.path.join(ROOT_DIR, original_img_rel_path)
        pil_image = Image.open(original_img_full_path).convert('RGB')

        input_tensor = adv_preprocess_transform(pil_image).unsqueeze(0).to(device_adv_gen)
        input_tensor.requires_grad = True

        # Forward pass to get the model's prediction
        model.zero_grad()
        output = model(input_tensor).logits

        # Compute loss
        target_label_tensor = torch.tensor([class_label], dtype=torch.long).to(device_adv_gen)
        loss = criterion(output, target_label_tensor)
        loss.backward()

        # Generate adversarial example using FGSM
        data_grad = input_tensor.grad.data.squeeze(0)
        adversarial_tensor = fgsm_attack(input_tensor.squeeze(0).detach(), current_epsilon, data_grad)
        adv_pil_image = denormalize_image(adversarial_tensor)

        # Save the file
        base_name = os.path.splitext(os.path.basename(original_img_rel_path))[0]
        current_epsilon_str = str(round(current_epsilon, 4)).replace('.', '_')
        adv_img_filename = f"{base_name}_adv_eps{current_epsilon_str}_{original_row['isic_id']}.jpg"

        adv_img_save_full_path = os.path.join(adversarial_data_dir, adv_img_filename)
        #adv_pil_image.save(adv_img_save_full_path)

        #adv_img_rel_save_path = os.path.join(adversarial_data_dir_name, adv_img_filename)

        # Append the new adversarial example to the list
        # new_adv_rows.append({
        #     'isic_id': original_row['isic_id'] + f"_adv_eps{current_epsilon_str}",
        #     'diagnosis_fine': original_row['diagnosis_fine'],
        #     'image_path': adv_img_save_full_path,
        #     'diagnosis': original_row['diagnosis'],
        #     'label': class_label,
        #     'set': 'train_adv'
        # })