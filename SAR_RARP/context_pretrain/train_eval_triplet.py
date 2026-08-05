import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, accuracy_score
from dataload_rarp_resnet import CustomVideoDatasetTriplet

# ----------------------------
# Model Definition
# ----------------------------
class ResNetMultiClass(nn.Module):
    def __init__(self, resnet_type="resnet50", pretrained=True, num_classes=16):
        super(ResNetMultiClass, self).__init__()
        
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Replace the final FC layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # Simple pass-through for classification
        return self.resnet(x)

# ----------------------------
# Transforms
# ----------------------------
def get_transforms():
    """
    Customize your transforms as needed. 
    Here is a simple example with resizing, center cropping, etc.
    """
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, test_transform

# ----------------------------
# Training & Testing Routines
# ----------------------------
def train(model, dataloader, criterion, optimizer, device, accumulation_steps=16):
    model.train()
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    running_loss = 0.0

    for step, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)   # [batch_size, num_classes]
        loss = criterion(outputs, labels)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps  # undo division for logging

        # Predictions & metrics accumulation
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
    
    # If we have leftover steps not divisible by accumulation_steps
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Compute metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader.dataset)

    return f1, accuracy, avg_loss

def test(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)  # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)

    return f1, accuracy, avg_loss

# ----------------------------
 #Main Script
# ----------------------------
import argparse

# ---------------------------- Argument Parser ----------------------------
parser = argparse.ArgumentParser(description="Train ResNet for Gesture Classification on SAR-RARP50")

parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8
num_epochs = 100
batch_size=4

# Prepare datasets and data loaders
train_transform, test_transform = get_transforms()

train_dataset = CustomVideoDatasetTriplet(mode="train", transform=train_transform)
test_dataset  = CustomVideoDatasetTriplet(mode="test",  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize model, loss, and optimizer
model = ResNetMultiClass(resnet_type="resnet50", pretrained=True, num_classes=num_classes).to(device)

# Example class weights – update these as appropriate for your dataset
class_weight = torch.tensor([4.721196454948301, 20.75487012987013, 2.421401515151515, 23.5451197053407, 21.237541528239202, 6.683220073183482, 38.978658536585364, 43.04713804713805, 31.72456575682382, 297.3255813953488, 245.8653846153846], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)



# Optionally track results (e.g., if you have multiple folds)
results = {
    'train_f1': [], 'train_acc': [], 'train_loss': [],
    'test_f1': [], 'test_acc': [], 'test_loss': []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # --- TRAIN ---
    train_f1, train_acc, train_loss = train(
        model, train_loader, criterion, optimizer, device, 
        accumulation_steps=16
    )
    print(f"[Train] F1: {train_f1:.4f} | Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")

    # --- TEST ---
    test_f1, test_acc, test_loss = test(model, test_loader, criterion, device)
    print(f"[Test]  F1: {test_f1:.4f}  | Acc: {test_acc:.4f}  | Loss: {test_loss:.4f}")

    # Record metrics
    results['train_f1'].append(train_f1)
    results['train_acc'].append(train_acc)
    results['train_loss'].append(train_loss)
    results['test_f1'].append(test_f1)
    results['test_acc'].append(test_acc)
    results['test_loss'].append(test_loss)



# Save the model
save_dir = "./error_detection/resnet"
save_filename = "RARP_resnet50_ges_classification.pth"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_filename)

if not os.path.exists(save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
    }, save_path)
    print(f"\nModel and weights saved to {save_path}")
else:
    print(f"\n{save_path} already exists. Skipping save.")


