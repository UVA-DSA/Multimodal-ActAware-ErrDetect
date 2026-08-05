import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
import os
import torch.nn.functional as F

print("util.py from SAR_RARP loaded")


def compute_pos_weight2(train_loader, device):
    num_positive_samples = 0
    num_negative_samples = 0

    # Loop through the train_loader to count positive and negative samples
    for images,_, labels,names in train_loader:
        # Flatten labels and masks
        flattened_labels = labels.view(-1)

        # Apply the mask to filter relevant parts of labels
        flattened_labels = flattened_labels.float()

        # Count positive and negative samples
        num_positive_samples += (flattened_labels == 1).sum().item()
        num_negative_samples += (flattened_labels == 0).sum().item()

    # Compute pos_weight
    pos_weight_value = num_negative_samples / num_positive_samples
    pos_weight = torch.tensor(pos_weight_value, device=device)
    
    # Initialize BCEWithLogitsLoss with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    return pos_weight, criterion

def compute_pos_weight(train_loader, device):
    num_positive_samples = 0
    num_negative_samples = 0

    # Loop through the train_loader to count positive and negative samples
    for images, labels,names in train_loader:
        # Flatten labels and masks
        flattened_labels = labels.view(-1)

        # Apply the mask to filter relevant parts of labels
        flattened_labels = flattened_labels.float()

        # Count positive and negative samples
        num_positive_samples += (flattened_labels == 1).sum().item()
        num_negative_samples += (flattened_labels == 0).sum().item()

    # Compute pos_weight
    pos_weight_value = num_negative_samples / num_positive_samples
    pos_weight = torch.tensor(pos_weight_value, device=device)
    
    # Initialize BCEWithLogitsLoss with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    return pos_weight, criterion


# Define the number of epochs for testing
def train2(model, dataloader, criterion, optimizer, device, accumulation_steps=16):
    model.train()
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()  # Reset gradients at the start
    
    for step, (images,ges_embed, labels, video_names) in enumerate(tqdm(dataloader, desc="Training")):
        # Move images and labels to the device
        images = images.to(device)
        ges_embed = ges_embed.to(device)
        # pdb.set_trace()
        labels = labels.to(device).float()  # Ensure labels are float for binary loss functions

        # Forward pass
        outputs = model(images,ges_embed)  # Expected shape: [batch_size] or [batch_size, ...]
        
        # Flatten if necessary (if outputs and labels are already [batch_size], this is a no-op)
        flattened_outputs = outputs.view(-1)
        flattened_labels = labels.view(-1)

        # Compute loss (no mask)
        loss = criterion(flattened_outputs, flattened_labels) / accumulation_steps
        loss.backward()

        # Update model weights after accumulating gradients for `accumulation_steps`
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients after each update
        
        # Convert outputs to probabilities, then to predictions
        probs = torch.sigmoid(flattened_outputs)
        preds = (probs > 0.5).cpu().numpy()  # Apply threshold for binary classification
        all_preds.extend(preds.tolist())
        all_labels.extend(flattened_labels.cpu().numpy().tolist())

    # Handle the case if the total number of steps is not divisible by accumulation_steps
    total_steps = len(dataloader)
    if total_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Convert lists to NumPy arrays for metric calculations
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    
    # Calculate evaluation metrics
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = float(balanced_accuracy_score(all_labels, all_preds))
    jaccard = jaccard_score(all_labels, all_preds, average='binary')
    
    return f1, accuracy, jaccard

def test2(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images,ges_embed, labels, video_names in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            ges_embed = ges_embed.to(device)
            labels = labels.to(device).float()  # Ensure labels are float if needed

            outputs = model(images,ges_embed)  # Expected shape: [batch_size] or [batch_size, ...]

            # Flatten if necessary
            flattened_outputs = outputs.view(-1)
            flattened_labels = labels.view(-1)

            # Convert outputs to probabilities and then to binary predictions
            probs = torch.sigmoid(flattened_outputs)
            preds = (probs > 0.5).cpu().numpy().astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(flattened_labels.cpu().numpy().astype(int).tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = float(balanced_accuracy_score(all_labels, all_preds))
    jaccard = jaccard_score(all_labels, all_preds, average='binary')

    return f1, accuracy, jaccard
def train(model, dataloader, criterion, optimizer, device, accumulation_steps=16):
    model.train()
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()  # Reset gradients at the start
    
    for step, (images, labels, video_names) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device).float()  # Ensure labels are float if required by the criterion

        # Forward pass
        outputs = model(images)  # Expected shape: [batch_size] or [batch_size, 1]
        outputs = outputs.repeat(1, labels.shape[1])
        # Flatten the outputs and labels if needed
        flattened_outputs = outputs.view(-1)
        flattened_labels = labels.view(-1)

        # Compute loss and backpropagate
        loss = criterion(flattened_outputs, flattened_labels) / accumulation_steps
        loss.backward()

        # Update model weights after accumulating gradients
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Make predictions
        probs = torch.sigmoid(flattened_outputs)
        preds = (probs > 0.5).cpu().numpy().astype(int)
        
        # Collect predictions and labels for metrics
        all_preds.extend(preds.tolist())
        all_labels.extend(flattened_labels.cpu().numpy().astype(int).tolist())
    
    # If total steps not divisible by accumulation_steps, update at the end
    total_steps = len(dataloader)
    if total_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Convert lists to numpy arrays for metric calculations
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    
    # Calculate evaluation metrics
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = float(balanced_accuracy_score(all_labels, all_preds))
    jaccard = jaccard_score(all_labels, all_preds, average='binary')
    
    return f1, accuracy, jaccard

def test(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels, video_names in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)  # Expected shape: [batch_size] or [batch_size, 1]
            outputs = outputs.repeat(1, labels.shape[1])
            # Flatten if necessary
            flattened_outputs = outputs.view(-1)
            flattened_labels = labels.view(-1)

            # Convert outputs to probabilities and then to binary predictions
            probs = torch.sigmoid(flattened_outputs)
            preds = (probs > 0.5).cpu().numpy().astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(flattened_labels.cpu().numpy().astype(int).tolist())
    
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = float(balanced_accuracy_score(all_labels, all_preds))
    jaccard = jaccard_score(all_labels, all_preds, average='binary')
    
    return f1, accuracy, jaccard
