import json
import random
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from UnetModel import UNet

# CLASSICAL_METHOD = os.getenv("CLASSICAL_METHOD", "otsu")
# CLASSICAL_METHOD = 'otsu'
# CLASSICAL_METHOD = 'otsu_roi'
# CLASSICAL_METHOD = 'adaptive_gaussian'
# CLASSICAL_METHOD = 'adaptive_gaussian_roi'
CLASSICAL_METHOD = 'adaptive_mean'
# CLASSICAL_METHOD = 'adaptive_mean_roi'

SEED = 42
IMAGE_DIR = Path("Radiographs")
MASK_DIR = Path("Segmentation/teeth_mask")
SPLIT_JSON = Path("data_split.json")
CLASSICAL_SEG_DIR = Path(f"classical_seg_{CLASSICAL_METHOD}")

IMAGE_SIZE = (256, 512)
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = f"best_unet_teeth_{CLASSICAL_METHOD}.pt"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeethDataset(Dataset):
    def __init__(self, image_paths, mask_paths, classical_paths, image_size=IMAGE_SIZE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.classical_paths = classical_paths
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        classical_seg = Image.open(self.classical_paths[idx]).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        classical_seg = self.mask_transform(classical_seg)

        classical_seg = (classical_seg > 0.5).float() ####
        mask = (mask > 0.5).float()

        stacked_input = torch.cat([image, classical_seg], dim=0)

        return stacked_input, mask


def resolve_path(base_dir: Path, name: str):
    path = Path(name)
    if path.exists():
        return path
    candidate = base_dir / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find file: {name}")


def load_split_paths(split_json_path: Path, image_dir: Path, mask_dir: Path, classical_dir: Path):
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_json_path}")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    if "train" not in split_data or "test" not in split_data:
        raise KeyError("data_split.json must contain 'train' and 'test' keys")

    train_entries = split_data["train"]
    test_entries = split_data["test"]

    train_imgs, train_masks, train_classical = parse_entries(train_entries, image_dir, mask_dir, classical_dir)
    test_imgs, test_masks, test_classical = parse_entries(test_entries, image_dir, mask_dir, classical_dir)

    return train_imgs, train_masks, train_classical, test_imgs, test_masks, test_classical


def parse_entries(entries, image_dir: Path, mask_dir: Path, classical_dir: Path):
    image_paths = []
    classical_paths = []
    mask_paths = []

    for entry in entries:
        if isinstance(entry, dict):
            image_name = entry.get("radiograph") 
            mask_name = entry.get("segmentation")
            if image_name is None or mask_name is None:
                raise ValueError(
                    "Each dict entry in data_split.json must contain image/mask fields, "
                    "e.g. {'image': 'x.png', 'mask': 'x.png'}"
                )
        elif isinstance(entry, str):
            image_name = entry
            mask_name = entry
        else:
            raise ValueError("Unsupported entry format in data_split.json")

        image_paths.append(resolve_path(image_dir, image_name))
        mask_paths.append(resolve_path(mask_dir, mask_name))
        classical_paths.append(resolve_path(classical_dir, image_name))

    return image_paths, mask_paths, classical_paths

import torch.nn.functional as F
def bce_dice_loss(logits, targets, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d_loss = dice_score_from_logits(logits, targets)
    return bce_weight * bce + (1 - bce_weight) * d_loss

def dice_score_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = 0.0
    dice_meter = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            loss_meter += loss.item() * images.size(0)
            dice_meter += dice_score_from_logits(logits, masks) * images.size(0)

    n = len(loader.dataset)
    return loss_meter / n, dice_meter / n


def main():
    set_seed(SEED)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Missing mask directory: {MASK_DIR}")
    if not CLASSICAL_SEG_DIR.exists():
        raise FileNotFoundError(f"Missing classical segmentation directory: {CLASSICAL_SEG_DIR}")

    train_imgs, train_masks, train_classical, test_imgs, test_masks, test_classical = load_split_paths(
        SPLIT_JSON, IMAGE_DIR, MASK_DIR, CLASSICAL_SEG_DIR
    )

    train_dataset = TeethDataset(train_imgs, train_masks, train_classical)
    test_dataset = TeethDataset(test_imgs, test_masks, test_classical)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    criterion = bce_dice_loss ## change to DICE loss
    # criterion = nn.BCEWithLogitsLoss() ## change to DICE loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_dice = -1.0

    print(f"Using device: {DEVICE}")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_dice = evaluate(model, test_loader, criterion, DEVICE)

        if test_dice > best_dice:
            best_dice = test_dice
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_dice": test_dice,
                "seed": SEED,
            }, CHECKPOINT_PATH)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Dice: {test_dice:.4f}"
        )

    print(f"Best model saved to {CHECKPOINT_PATH} with Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()