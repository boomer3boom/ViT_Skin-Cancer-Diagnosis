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

# Attention noise data
ATTN_NOISE_METADATA_CSV = os.path.join(ROOT_DIR, "data/metadata/attention_noise_metadata.csv")

# Adversarial data, note that we don't have the metadata for these images
ADVERSARIAL_IMAGES_DIR_NAME = "data_adversarial_mixed_eps"
ADVERSARIAL_IMAGES_FULL_PATH = os.path.join(ROOT_DIR, ADVERSARIAL_IMAGES_DIR_NAME)

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

# Used to identify final weights for loss function
class_weights_tensor = None
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

# If the attention noise already exist then we load it.
attn_noise_df = pd.DataFrame()
if os.path.exists(ATTN_NOISE_METADATA_CSV):
    print(f"Loading attention-noisy data metadata from: {ATTN_NOISE_METADATA_CSV}")
    attn_noise_df = pd.read_csv(ATTN_NOISE_METADATA_CSV)
    print(f"Loaded {len(attn_noise_df)} attention-noisy image records.")
else:
    print(f"WARNING: Attention-noisy metadata file not found at {ATTN_NOISE_METADATA_CSV}. Attention-noise set will be empty.")

# Note that we do not have the metadata for adv so we do the following:
print(f"Scanning for pre-generated adversarial images in: {ADVERSARIAL_IMAGES_FULL_PATH}")
new_adv_rows_from_scan = []
if os.path.isdir(ADVERSARIAL_IMAGES_FULL_PATH):
    for filename in tqdm(os.listdir(ADVERSARIAL_IMAGES_FULL_PATH), desc="Scanning Adversarial Dir"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Parse original ISIC ID from filename like "ISIC_0053457_adv_eps0_0469_11717.jpg"
            match = re.match(r"(ISIC_\d+)_adv_eps.*", filename)
            if match:
                original_isic_id = match.group(1)
                # Look up original metadata
                original_meta_row = meta_df_indexed.loc[original_isic_id]
                adv_image_relative_path = os.path.join(ADVERSARIAL_IMAGES_DIR_NAME, filename)

                # add the original metadata as metadata for the adversarial image
                new_adv_rows_from_scan.append({
                    'isic_id': os.path.splitext(filename)[0], 
                    'diagnosis_fine': original_meta_row['diagnosis_fine'],
                    'image_path': adv_image_relative_path,
                    'diagnosis': original_meta_row['diagnosis'],
                    'label': original_meta_row['label'],
                    'set': 'train_adv_scanned'  
                })
else:
    print(f"Warning: Adversarial image directory not found")

adversarial_df = pd.DataFrame()
if new_adv_rows_from_scan:
    adversarial_df = pd.DataFrame(new_adv_rows_from_scan)
    print(f"Found and processed {len(adversarial_df)} pre-generated adversarial image records by scanning the directory.")

#Combine all training data: Original + Attention-Noise + Scanned Adversarial
train_df_parts = [train_df_orig]
if not attn_noise_df.empty:
    train_df_parts.append(attn_noise_df)
if not adversarial_df.empty:
    train_df_parts.append(adversarial_df)

train_df = pd.concat(train_df_parts, ignore_index=True)
#train_df = train_df.sample(frac=1).reset_index(drop=True)

print(f"\nTotal combined training samples: {len(train_df)}")
if not train_df.empty:
    print("Recalculating class weights for the combined training set...")
    final_class_counts = train_df['label'].value_counts().sort_index()
    num_classes_final = len(labels_map)

    # Count the number of data in each class.
    full_final_class_counts = pd.Series([0] * num_classes_final, index=range(num_classes_final))
    full_final_class_counts.update(final_class_counts)

    # Do a weight of 1/count, with a small epsilon to avoid division by zero
    weights_final = 1.0 / (full_final_class_counts + 1e-6)
    weights_final = weights_final / weights_final.sum()
    class_weights_tensor = torch.tensor(weights_final.values, dtype=torch.float)
    print("Final class weights for loss function:", class_weights_tensor)
else:
    print("Warning: Combined training dataframe is empty.")

# Train and Evaluation data agumentation
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=170),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# eval just normalises the data
eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Create the BCN datasets
if not train_df.empty:
    train_dataset = BCNDataset(train_df, transform=train_transform, root_dir=ROOT_DIR)
else:
    print("No training data avaiable")

val_dataset = BCNDataset(val_df, transform=eval_transform, root_dir=ROOT_DIR)
test_dataset = BCNDataset(test_df, transform=eval_transform, root_dir=ROOT_DIR)

# Define some constants
batch_size = 150
num_workers = 1

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True if len(train_dataset) > 0 else False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Prepareing for training
num_labels_for_model = len(labels_map)

# Set up the pre-trained model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_labels_for_model,
    ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=8e-6)

# Take the inverse of the class weights for the loss function
class_weights_tensor_on_device = class_weights_tensor.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor_on_device)
num_epochs = 80

# --- Training Loop ---
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []

# Directory to save the best validation model
best_val_accuracy = 0.0
output_dir_results_name = "training_results_combined_scanned_adv"
output_dir_results = os.path.join(ROOT_DIR, "pythoncode", output_dir_results_name)
os.makedirs(output_dir_results, exist_ok=True)
best_model_filename = "best_model_combined_scanned_adv.pth"
best_model_save_path = os.path.join(output_dir_results, best_model_filename) 

# Training loop
for epoch in range(num_epochs):
    current_epoch = epoch + 1
    model.train()
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0 
    train_loop_tqdm = tqdm(train_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Training]")

    for i, (inputs, batch_labels) in enumerate(train_loop_tqdm):
        # If for whatever reason the batch is empty or contains -1 labels, skip it
        if inputs.nelement() == 0 or (isinstance(batch_labels, torch.Tensor) and -1 in batch_labels):
            continue
        
        inputs, batch_labels = inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).logits

        # Calculate loss and backpropagate
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        train_losses.append({'epoch': current_epoch, 'train_batch_loss': loss.item()})
        
        # Calculate training accuracy for the batch
        _, predicted_train = torch.max(outputs.data, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted_train == batch_labels).sum().item()
        
        train_loop_tqdm.set_postfix(loss=loss.item())

    # Calculate training accuracy for the epoch
    epoch_train_accuracy = 100 * train_correct / train_total
    train_accuracies.append({'epoch': current_epoch, 'train_epoch_accuracy': epoch_train_accuracy})

    # Now for evaluation with validation set
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    val_loop_tqdm = tqdm(val_loader, desc=f"Epoch {current_epoch}/{num_epochs} [Validation]")

    with torch.no_grad():
        for inputs, batch_labels in val_loop_tqdm:
            # If for whatever reason the batch is empty or contains -1 labels, skip it
            if inputs.nelement() == 0 or (isinstance(batch_labels, torch.Tensor) and -1 in batch_labels):
                continue
            
            # forward pass
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            outputs = model(inputs).logits

            # Calculate validation loss and accuracy
            val_loss_item = criterion(outputs, batch_labels)
            running_val_loss += val_loss_item.item()
            val_losses.append({'epoch': current_epoch, 'val_batch_loss': val_loss_item.item()})

            # Update validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            val_loop_tqdm.set_postfix(loss=val_loss_item.item())
    
    # Calculate average losses and accuracies
    avg_train_loss = running_train_loss / len(train_loader)
    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_accuracies.append({'epoch': current_epoch, 'val_epoch_accuracy': val_accuracy})
    print(f"Epoch {current_epoch} - Train Loss: {avg_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Check if current validation accuracy is the best seen so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        # Save the model
        torch.save(model.state_dict(), best_model_save_path)
        print(f"New best validation accuracy: {best_val_accuracy:.2f}%. Model saved to {best_model_save_path}")

print("Finished Training")

# Save all the loss and accuracy
train_losses_df = pd.DataFrame(train_losses)
train_losses_path = os.path.join(output_dir_results, "train_losses_with_epoch.csv")
train_losses_df.to_csv(train_losses_path, index=False)
print(f"Training losses with epoch saved to {train_losses_path}")

train_accuracies_df = pd.DataFrame(train_accuracies)
train_accuracies_path = os.path.join(output_dir_results, "train_accuracies_with_epoch.csv")
train_accuracies_df.to_csv(train_accuracies_path, index=False)
print(f"Training accuracies with epoch saved to {train_accuracies_path}")

val_losses_df = pd.DataFrame(val_losses)
val_losses_path = os.path.join(output_dir_results, "val_losses_with_epoch.csv")
val_losses_df.to_csv(val_losses_path, index=False)
print(f"Validation losses with epoch saved to {val_losses_path}")

val_accuracies_df = pd.DataFrame(val_accuracies)
val_accuracies_path = os.path.join(output_dir_results, "val_accuracies_with_epoch.csv")
val_accuracies_df.to_csv(val_accuracies_path, index=False)
print(f"Validation accuracies with epoch saved to {val_accuracies_path}")

# Save the final model
final_model_name = "final_model_combined_scanned_adv.pth"
final_model_path = os.path.join(output_dir_results, final_model_name)
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")