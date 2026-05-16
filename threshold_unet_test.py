import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from UnetModel import UNet


# CLASSICAL_METHOD = 'otsu'
# CLASSICAL_METHOD = 'otsu_roi'
# CLASSICAL_METHOD = 'adaptive_gaussian'
# CLASSICAL_METHOD = 'adaptive_gaussian_roi'
CLASSICAL_METHOD = 'adaptive_mean'
# CLASSICAL_METHOD = 'adaptive_mean_roi'

IMAGE_DIR = Path("Radiographs")
MASK_DIR = Path("Segmentation/teeth_mask")
SPLIT_JSON = Path("data_split.json")
CHECKPOINT_PATH = Path(f"best_unet_teeth_{CLASSICAL_METHOD}.pt")
OUTPUT_DIR = Path("figure")
OUTPUT_PATH = OUTPUT_DIR / f"Unet_Results_{CLASSICAL_METHOD}.png"

CLASSICAL_SEG_DIR = Path(f"classical_seg_{CLASSICAL_METHOD}")

IMAGE_SIZE = (256, 512)
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        classical_seg = (classical_seg > 0.5).float()
        mask = (mask > 0.5).float()

        stacked_input = torch.cat([image, classical_seg], dim=0)

        return stacked_input, mask, self.image_paths[idx].name


def resolve_path(base_dir: Path, name: str):
    path = Path(name)
    if path.exists():
        return path
    candidate = base_dir / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find file: {name}")


def parse_entries(entries, image_dir: Path, mask_dir: Path, classical_dir: Path):
    image_paths = []
    mask_paths = []
    classical_paths = []

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


def load_test_split(split_json_path: Path, image_dir: Path, mask_dir: Path, classical_dir: Path):
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_json_path}")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    if "test" not in split_data:
        raise KeyError("data_split.json must contain a 'test' key")

    test_entries = split_data["test"]
    test_imgs, test_masks, test_classical = parse_entries(test_entries, image_dir, mask_dir, classical_dir)
    return test_imgs, test_masks, test_classical


def dice_score(preds, targets, eps=1e-7):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2 * intersection + eps) / (union + eps)).item()


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    test_imgs, test_masks, test_classical = load_test_split(SPLIT_JSON, IMAGE_DIR, MASK_DIR, CLASSICAL_SEG_DIR)[:3]
    test_dataset = TeethDataset(test_imgs, test_masks, test_classical)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dice_scores = []
    samples = []

    with torch.no_grad():
        for idx, (images, masks, names) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            score = dice_score(preds, masks)
            dice_scores.append(score)

            if len(samples) < 3:
                samples.append(
                    {
                        "image": images[0, 0].cpu().numpy(),
                        "classical": images[0, 1].cpu().numpy(),
                        "gt": masks[0, 0].cpu().numpy(),
                        "pred": preds[0, 0].cpu().numpy(),
                        "name": names[0],
                        "dice": score,
                    }
                )


    mean_dice = float(np.mean(dice_scores))
    std_dice = float(np.std(dice_scores))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(15, 5))

    # =================================================
    for row, sample in enumerate(samples):
        axes[row, 0].imshow(sample["gt"], cmap="gray")
        axes[row, 0].set_title(f"GT: {sample['name']}")
        axes[row, 0].axis("off")
    
        axes[row, 1].imshow(sample["pred"], cmap="gray")
        axes[row, 1].set_title(f"Segmentation (Dice={sample['dice']:.4f})")
        axes[row, 1].axis("off")
    
        axes[row, 2].imshow(sample["image"], cmap="gray")
        axes[row, 2].set_title("Radiograph")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(sample["classical"], cmap="gray")
        axes[row, 3].set_title(f"Classical ({CLASSICAL_METHOD})")
        axes[row, 3].axis("off")
    # ==================================================

    fig.suptitle(
        f"Unet_Results ({CLASSICAL_METHOD})\n"
        f"Average Test Dice Score: {mean_dice:.4f}\n"
        f"Std Test Dice Score: {std_dice:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {OUTPUT_PATH}")
    print(f"Average Test Dice Score: {mean_dice:.4f}")
    print(f"Std Test Dice Score: {std_dice:.4f}")


if __name__ == "__main__":
    main()