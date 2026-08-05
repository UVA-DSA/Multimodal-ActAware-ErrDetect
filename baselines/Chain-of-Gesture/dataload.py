import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pandas as pd
import pickle
import csv
import os
import numpy as np
from PIL import Image




class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, mstcn=False):
        self.root_dir = root_dir
        self.transform = transform
        if train == True:
            self.video_folders = pd.read_csv(os.path.join(root_dir, 'train.csv'), header=None)[0].tolist()
        else:
            self.video_folders = pd.read_csv(os.path.join(root_dir, 'test.csv'), header=None)[0].tolist()
        self.mstcn = mstcn

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name.replace('.csv', '.pkl'))
        with open(video_path, 'rb') as f:
            video_data = pickle.load(f)
        features = video_data['feature'].astype('float32')
        e_labels = video_data['error_GT'].astype('float32')
        g_labels = video_data['gesture_GT'].astype('float32')
        # g_labels = [replacement_values[replace_values.index(x)] if x in replace_values else x for x in labels[0]]
        video_length = len(e_labels)
        
        if self.transform:
            features = self.transform(features)

        # Return the frames of the video as a list and its corresponding label
        return features, video_length, e_labels, g_labels, video_name



# Example usage
if __name__ == "__main__":
    dataset_path = './dataset/setting_f1/LOSO'

    custom_dataset = CustomVideoDataset(root_dir=dataset_path)

    # Example: Access the first video and its corresponding label
    #frames, video_length, e_labels, g_labels, video_name = custom_dataset[0]
    #print(f"Video frames: {len(frames)}, Label: {e_labels}, Gesture_labels: {g_labels}, Video_name: {video_name}")
