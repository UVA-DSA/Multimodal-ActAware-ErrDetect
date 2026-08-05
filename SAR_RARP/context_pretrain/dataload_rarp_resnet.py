import os
import pickle
import numpy as np
import re
from typing import Optional, Tuple, List
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
# for train gestures classifier
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
from PIL import Image
def extract_number(file_name: str) -> int:
    match = re.search(r"(\d+)", file_name)
    return int(match.group()) if match else 0


class CustomVideoDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train = True if "train" in self.root_dir else False
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)
        self.image_path = (
            "./SAR_RARP50/training_set" if self.train else "./SAR_RARP50/testing_set"
        )

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[List[str], int, List[int], str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        image_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"images")

        # Load video data
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)

        # Extract features and labels
        frame_names = video_data["image_name"]
        e_labels = video_data["error_GT"]
        frame_paths = [os.path.join(image_path, name) for name in frame_names]

        return frame_paths, len(e_labels), e_labels, video_name



class CustomVideoDatasetTriplet(Dataset):
    def __init__(self, mode: str = "train", transform: Optional[Callable] = None):
        """
        Args:
            mode (str): 'train' or 'test'
            transform (callable, optional): A torchvision transform to apply to each image
        """
        # Base directories
        self.image_dir = "./segmentation/SPAR/images"
        self.label_base = "./SPAR_RARP_SEG/SPAR_RARP"
        self.test_list_path = "./SPAR_RARP_SEG/SPAR_RARP_test"

        self.mode = mode
        self.transform = transform

        # Build train/test split
        vid_list = sorted(os.listdir(self.image_dir))
        test_list = sorted(os.listdir(self.test_list_path))
        train_list = list(set(vid_list) - set(test_list))

        if self.mode == "train":
            self.video_folders = train_list
            self.label_path_base = self.label_base
        else:
            self.video_folders = test_list
            # Notice test label folder has "_test" appended
            self.label_path_base = self.label_base + "_test"

        # ----------------------------------------------------
        # Build a flat list of (image_path, label_vector) for *all* frames
        # ----------------------------------------------------
        self.samples = []  # will hold tuples (img_path, label_vector)

        for video_name in self.video_folders:
            video_image_dir = os.path.join(self.image_dir, video_name)
            video_label_dir = os.path.join(self.label_path_base, video_name)
            triplet_label_file = os.path.join(video_label_dir, 'triplet.txt')

            if not os.path.isfile(triplet_label_file):
                # Skip missing label files
                continue

            # Load triplet.txt: "frame_id, [17 label columns]"
            df = pd.read_csv(triplet_label_file, header=None, dtype={0: str})

            # All frames in this folder
            frame_files = sorted(os.listdir(video_image_dir))
            frame_ids = {os.path.splitext(f)[0] for f in frame_files}

            # Filter only frames that exist
            df_filtered = df[df[0].isin(frame_ids)]

            # Build label map: {frame_id -> np.array([17-dim vector])}
            label_map = {row[0]: row[1:].values.astype(np.float32) for _, row in df_filtered.iterrows()}

            for frame_file in frame_files:
                frame_id, ext = os.path.splitext(frame_file)
                full_img_path = os.path.join(video_image_dir, frame_file)

                if frame_id in label_map:
                    label_vector = label_map[frame_id]
                else:
                    # If missing, create an all-zero label
                    label_vector = np.zeros(17, dtype=np.float32)

                # Add to samples
                self.samples.append((full_img_path, label_vector))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image (Tensor): [3, H, W]
            label (Tensor): [17] (multi-label, one-hot/multi-hot vector)
        """
        img_path, label_vector = self.samples[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply transform if given
        if self.transform:
            image = self.transform(image)

        # Convert label to torch.Tensor
        label = torch.tensor(label_vector, dtype=torch.float32)

        return image, label

    
class CustomVideoDatasetGesture(Dataset):
    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train' or 'test'
            transform (callable): A torchvision transform to apply to each image
        """
        # Base directories
        self.image_dir = "./segmentation/SPAR/images"
        self.label_base = "./SPAR_RARP_SEG/SPAR_RARP"
        self.test_list_path = "./SPAR_RARP_SEG/SPAR_RARP_test"

        self.mode = mode
        self.transform = transform
        
        # Build train/test split
        vid_list = sorted(os.listdir(self.image_dir))
        test_list = sorted(os.listdir(self.test_list_path))
        train_list = list(set(vid_list) - set(test_list))

        if self.mode == "train":
            self.video_folders = train_list
            self.label_path_base = self.label_base
        else:
            self.video_folders = test_list
            # Notice the test label folder has "_test" appended
            self.label_path_base = self.label_base + "_test"

        # ----------------------------------------------------
        # Build a flat list of (image_path, label) for *all* frames
        # ----------------------------------------------------
        self.samples = []  # will hold tuples (img_path, label)

        for video_name in self.video_folders:
            video_image_dir = os.path.join(self.image_dir, video_name)
            video_label_dir = os.path.join(self.label_path_base, video_name)
            action_label_file = os.path.join(video_label_dir, 'action_discrete.txt')

            if not os.path.isfile(action_label_file):
                # If missing, skip or raise an error
                continue

            # Load the label CSV, e.g.: "frame_id,label"
            df = pd.read_csv(action_label_file, header=None, names=['frame_id', 'label'], dtype={'frame_id': str})

            # All frames in this folder
            frame_files = sorted(os.listdir(video_image_dir))
            frame_ids = {os.path.splitext(f)[0] for f in frame_files}

            # Filter the CSV to frames we actually have
            df_filtered = df[df['frame_id'].isin(frame_ids)]

            # Build a dictionary for quick lookup: {frame_id -> label}
            label_map = dict(zip(df_filtered['frame_id'], df_filtered['label']))

            # For each valid frame in the folder, record (img_path, label)
            for frame_file in frame_files:
                frame_id, ext = os.path.splitext(frame_file)
                if frame_id in label_map:
                    full_img_path = os.path.join(video_image_dir, frame_file)
                    label = label_map[frame_id]
                    # Add to our samples
                    self.samples.append((full_img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single frame [C, H, W] and its label (as int).
        """
        img_path, label = self.samples[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if given
        if self.transform:
            image = self.transform(image)

        # Convert label to int, if needed
        label = int(label)

        return image, label

    
    
class Gesture_SegmentWrapper(Dataset):
    def __init__(
        self, 
        original_dataset: Dataset, 
        segment_length: int = 20, 
        step_size: int = 10, 
        augmentation_list: List[str] = ['original'], 
        normalize: bool = True
    ):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.augmentation_list = augmentation_list
        self.normalize = normalize

        # Build train and test transformations
        trainform, testform = self.transform()
        self.transform = trainform if original_dataset.train else testform

        # Build indices
        self.indices = []
        for idx in range(len(original_dataset)):
            features, video_length, e_labels, video_name = self.original_dataset[idx]
            total_frames = len(features)
            if total_frames >= segment_length:
                for start_frame in range(0, total_frames - segment_length + 1, step_size):
                    self.indices.append((idx, start_frame))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray, str]:
        original_index, start_frame = self.indices[index]
        frame_paths, video_length, e_labels, video_name = self.original_dataset[original_index]

        end_frame = start_frame + self.segment_length
        segment_paths = frame_paths[start_frame:end_frame]

        segment_images = []
        for img_path in segment_paths:
            try:
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(img)
                segment_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        segment_labels = np.full(self.segment_length, np.max(e_labels[start_frame:end_frame]))
        return torch.stack(segment_images), segment_labels, video_name

    def transform(self):
        """
        Constructs train and test transformations based on augmentation_list and normalization settings.
        """
        base_transforms = [transforms.Resize((240, 240)), transforms.CenterCrop(224), transforms.ToTensor()]
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Define augmentation transforms (if any)
        augmentations = []
        if 'original' not in self.augmentation_list:
            # Add other augmentations here if needed, e.g., random flips, rotations
            augmentations.append(transforms.RandomHorizontalFlip())

        # Test transform
        op_test = base_transforms + ([normalize_transform] if self.normalize else [])

        # Train transform
        op_train = base_transforms + augmentations + ([normalize_transform] if self.normalize else [])

        return transforms.Compose(op_train), transforms.Compose(op_test)
