#!/usr/bin/env python3
"""
Feature Extraction Script for JIGSAWS Dataset

This script pre-extracts visual features from video frames and saves them to disk.
Supports both ResNet and CLIP vision encoder variants.

This avoids redundant CNN/ViT computation during training/evaluation since the 
encoder weights are frozen and the extracted features are deterministic.

Output structure:
    ./data/vid_features/{model_name}/{task}/{video_name}/features.pt
    
Each features.pt file is a dict mapping frame_number (int) -> feature tensor

Feature dimensions by model:
    - resnet18: 512
    - resnet50: 2048
    - resnet101: 2048
    - clip-vit-base-patch32: 768
    - clip-vit-base-patch16: 768
    - clip-vit-large-patch14: 1024
    - clip-vit-large-patch14-336: 1024
"""

import os
import glob
import re
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Available models and their properties
AVAILABLE_MODELS = {
    # ResNet variants
    "resnet18": {
        "type": "resnet",
        "feature_dim": 512,
        "description": "ResNet-18 (ImageNet pretrained)"
    },
    "resnet50": {
        "type": "resnet",
        "feature_dim": 2048,
        "description": "ResNet-50 (ImageNet pretrained)"
    },
    "resnet101": {
        "type": "resnet",
        "feature_dim": 2048,
        "description": "ResNet-101 (ImageNet pretrained)"
    },
    # CLIP Vision Transformer variants
    "clip-vit-base-patch32": {
        "type": "clip",
        "hf_name": "openai/clip-vit-base-patch32",
        "feature_dim": 768,
        "image_size": 224,
        "description": "CLIP ViT-B/32 (224x224)"
    },
    "clip-vit-base-patch16": {
        "type": "clip",
        "hf_name": "openai/clip-vit-base-patch16",
        "feature_dim": 768,
        "image_size": 224,
        "description": "CLIP ViT-B/16 (224x224)"
    },
    "clip-vit-large-patch14": {
        "type": "clip",
        "hf_name": "openai/clip-vit-large-patch14",
        "feature_dim": 1024,
        "image_size": 224,
        "description": "CLIP ViT-L/14 (224x224)"
    },
    "clip-vit-large-patch14-336": {
        "type": "clip",
        "hf_name": "openai/clip-vit-large-patch14-336",
        "feature_dim": 1024,
        "image_size": 336,
        "description": "CLIP ViT-L/14 (336x336, higher resolution)"
    },
}


def get_resnet_extractor(model_name, device="cuda"):
    """
    Load a ResNet model for feature extraction with the final FC layer removed.
    """
    if model_name == "resnet18":
        resnet = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        resnet = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        resnet = models.resnet101(pretrained=True)
    else:
        raise ValueError(f"Unsupported ResNet type: {model_name}")
    
    # Remove the final FC layer (same as model_gvr.py)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    
    return resnet


def get_clip_extractor(model_name, device="cuda"):
    """
    Load a CLIP vision encoder for feature extraction.
    Returns the vision model and its processor.
    """
    from transformers import CLIPVisionModel, CLIPProcessor
    
    model_info = AVAILABLE_MODELS[model_name]
    hf_name = model_info["hf_name"]
    
    vision_model = CLIPVisionModel.from_pretrained(hf_name)
    processor = CLIPProcessor.from_pretrained(hf_name)
    
    vision_model = vision_model.to(device)
    vision_model.eval()
    
    return vision_model, processor


def get_resnet_transform(normalize=True, image_size=224):
    """
    Get the transform for ResNet models (same as dataloader for consistency).
    """
    transform_ops = [
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_ops)


def extract_features_resnet(model, video_dir, output_path, transform, device, batch_size=32):
    """
    Extract features for all frames using ResNet.
    
    Args:
        model: The ResNet model for feature extraction
        video_dir: Path to directory containing frame_XXXX.png files
        output_path: Path to save the features.pt file
        transform: Image transform to apply
        device: torch device
        batch_size: Number of frames to process at once
    
    Returns:
        Number of frames processed
    """
    frame_pattern = re.compile(r'frame_(\d+)\.png')
    frame_files = sorted(glob.glob(os.path.join(video_dir, "frame_*.png")))
    
    if len(frame_files) == 0:
        print(f"[WARN] No frames found in {video_dir}")
        return 0
    
    frame_info = []
    for fpath in frame_files:
        match = frame_pattern.search(os.path.basename(fpath))
        if match:
            frame_num = int(match.group(1))
            frame_info.append((frame_num, fpath))
    
    frame_info.sort(key=lambda x: x[0])
    
    features_dict = {}
    
    with torch.no_grad():
        for i in range(0, len(frame_info), batch_size):
            batch_info = frame_info[i:i + batch_size]
            
            images = []
            frame_nums = []
            for frame_num, fpath in batch_info:
                try:
                    img = Image.open(fpath).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    frame_nums.append(frame_num)
                except Exception as e:
                    print(f"[WARN] Failed to process {fpath}: {e}")
                    continue
            
            if len(images) == 0:
                continue
            
            batch_tensor = torch.stack(images).to(device)
            batch_features = model(batch_tensor)
            
            for j, frame_num in enumerate(frame_nums):
                features_dict[frame_num] = batch_features[j].cpu()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(features_dict, output_path)
    
    return len(features_dict)


def extract_features_clip(model, processor, video_dir, output_path, device, batch_size=32):
    """
    Extract features for all frames using CLIP vision encoder.
    
    Args:
        model: The CLIP vision model for feature extraction
        processor: The CLIP processor for image preprocessing
        video_dir: Path to directory containing frame_XXXX.png files
        output_path: Path to save the features.pt file
        device: torch device
        batch_size: Number of frames to process at once
    
    Returns:
        Number of frames processed
    """
    frame_pattern = re.compile(r'frame_(\d+)\.png')
    frame_files = sorted(glob.glob(os.path.join(video_dir, "frame_*.png")))
    
    if len(frame_files) == 0:
        print(f"[WARN] No frames found in {video_dir}")
        return 0
    
    frame_info = []
    for fpath in frame_files:
        match = frame_pattern.search(os.path.basename(fpath))
        if match:
            frame_num = int(match.group(1))
            frame_info.append((frame_num, fpath))
    
    frame_info.sort(key=lambda x: x[0])
    
    features_dict = {}
    
    with torch.no_grad():
        for i in range(0, len(frame_info), batch_size):
            batch_info = frame_info[i:i + batch_size]
            
            images = []
            frame_nums = []
            for frame_num, fpath in batch_info:
                try:
                    img = Image.open(fpath).convert('RGB')
                    images.append(img)
                    frame_nums.append(frame_num)
                except Exception as e:
                    print(f"[WARN] Failed to process {fpath}: {e}")
                    continue
            
            if len(images) == 0:
                continue
            
            # Use CLIP processor for preprocessing
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            # Get pooler output (CLS token representation)
            outputs = model(pixel_values=pixel_values)
            batch_features = outputs.pooler_output  # Shape: (batch_size, hidden_dim)
            
            for j, frame_num in enumerate(frame_nums):
                features_dict[frame_num] = batch_features[j].cpu()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(features_dict, output_path)
    
    return len(features_dict)


def print_available_models():
    """Print a formatted table of available models."""
    print("\nAvailable Models:")
    print("-" * 80)
    print(f"{'Model Name':<30} {'Feature Dim':<15} {'Description'}")
    print("-" * 80)
    for name, info in AVAILABLE_MODELS.items():
        print(f"{name:<30} {info['feature_dim']:<15} {info['description']}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract visual features from JIGSAWS video frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ResNet50 features (default)
  python extract_features.py --model resnet50
  
  # Extract CLIP ViT-B/32 features
  python extract_features.py --model clip-vit-base-patch32
  
  # Extract CLIP ViT-L/14 features with higher resolution
  python extract_features.py --model clip-vit-large-patch14-336
  
  # List available models
  python extract_features.py --list-models
"""
    )
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Visual encoder model to use for feature extraction")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")
    parser.add_argument("--data_root", type=str, default="./data", 
                        help="Root data directory")
    parser.add_argument("--tasks", type=str, nargs="+", default=["Suturing", "Needle_Passing"], 
                        help="Tasks to process")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for extraction")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Apply normalization (default: True, only affects ResNet)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-extract features even if they already exist")
    
    args = parser.parse_args()
    
    if args.list_models:
        print_available_models()
        return
    
    model_info = AVAILABLE_MODELS[args.model]
    model_type = model_info["type"]
    feature_dim = model_info["feature_dim"]
    
    print(f"[INFO] Using device: {args.device}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Model type: {model_type}")
    print(f"[INFO] Feature dimension: {feature_dim}")
    print(f"[INFO] Tasks: {args.tasks}")
    
    # Load model based on type
    if model_type == "resnet":
        print(f"[INFO] Loading {args.model}...")
        model = get_resnet_extractor(args.model, args.device)
        transform = get_resnet_transform(args.normalize)
        processor = None
    elif model_type == "clip":
        print(f"[INFO] Loading CLIP vision encoder: {model_info['hf_name']}...")
        model, processor = get_clip_extractor(args.model, args.device)
        transform = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    total_videos = 0
    total_frames = 0
    
    for task in args.tasks:
        vid_frames_dir = os.path.join(args.data_root, "vid_frames", task)
        # Output directory now includes model name for organization
        vid_features_dir = os.path.join(args.data_root, "vid_features", args.model, task)
        
        if not os.path.exists(vid_frames_dir):
            print(f"[WARN] Video frames directory not found: {vid_frames_dir}")
            continue
        
        video_dirs = sorted([d for d in os.listdir(vid_frames_dir) 
                           if os.path.isdir(os.path.join(vid_frames_dir, d))])
        
        print(f"\n[INFO] Processing task: {task} ({len(video_dirs)} videos)")
        
        for video_name in tqdm(video_dirs, desc=f"Extracting {task}"):
            video_path = os.path.join(vid_frames_dir, video_name)
            output_path = os.path.join(vid_features_dir, video_name, "features.pt")
            
            if os.path.exists(output_path) and not args.force:
                continue
            
            if model_type == "resnet":
                n_frames = extract_features_resnet(
                    model, video_path, output_path, transform, args.device, args.batch_size
                )
            elif model_type == "clip":
                n_frames = extract_features_clip(
                    model, processor, video_path, output_path, args.device, args.batch_size
                )
            
            total_videos += 1
            total_frames += n_frames
    
    print(f"\n[INFO] Feature extraction complete!")
    print(f"[INFO] Model: {args.model} (dim={feature_dim})")
    print(f"[INFO] Processed {total_videos} videos, {total_frames} frames total")
    print(f"[INFO] Features saved to: {os.path.join(args.data_root, 'vid_features', args.model)}")


if __name__ == "__main__":
    main()
