"""
Gesture Classification CNN Model.
A simple 2-layer CNN for classifying hand gestures into 4 directions:
up, down, left, right.

Also includes the custom Dataset class and image transforms.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Image transforms ──────────────────────────────────────────────
IMG_SIZE = 64
NUM_CLASSES = 4
CLASS_NAMES = ['up', 'down', 'left', 'right']

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────
class GestureDataset(Dataset):
    """Custom dataset that loads gesture images from class subdirectories.

    Expected structure:
        root/
            up/
                img1.jpg
                img2.jpg
            down/
                ...
            left/
                ...
            right/
                ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ── CNN Model ─────────────────────────────────────────────────────
class GestureCNN(nn.Module):
    """Simple CNN with 2 convolutional layers for gesture classification.

    Architecture:
        Conv2d(3, 32) -> ReLU -> MaxPool
        Conv2d(32, 64) -> ReLU -> MaxPool
        Flatten -> FC(64*16*16, 128) -> ReLU -> Dropout -> FC(128, 4)

    Why ReLU?
        Non-linear activation functions like ReLU allow the network to learn
        complex, non-linear decision boundaries. Without them, stacking
        linear layers would collapse into a single linear transformation,
        making the model unable to learn non-trivial patterns.
    """

    def __init__(self):
        super(GestureCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 → 32

            # Layer 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 → 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
