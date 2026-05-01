import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from UnetModel import UNet


IMAGE_DIR = Path("Radiographs")
MASK_DIR = Path("Segmentation/teeth_mask")
SPLIT_JSON = Path("data_split.json")
CHECKPOINT_PATH = Path("best_unet_teeth.pt")
OUTPUT_DIR = Path("figure")
OUTPUT_PATH = OUTPUT_DIR / "Unet_Results.png"

IMAGE_SIZE = (256, 512)
BATCH_SIZE = 1
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TeethDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=IMAGE_SIZE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask, self.image_paths[idx].name


def resolve_path(base_dir: Path, name: str):
    path = Path(name)
    if path.exists():
        return path
    candidate = base_dir / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find file: {name}")


def parse_entries(entries, image_dir: Path, mask_dir: Path):
    image_paths = []
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

    return image_paths, mask_paths


def load_test_split(split_json_path: Path, image_dir: Path, mask_dir: Path):
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_json_path}")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    if "test" not in split_data:
        raise KeyError("data_split.json must contain a 'test' key")

    test_entries = split_data["test"]
    test_imgs, test_masks = parse_entries(test_entries, image_dir, mask_dir)
    return test_imgs, test_masks


def dice_score(preds, targets, eps=1e-7):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2 * intersection + eps) / (union + eps)).item()


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    test_imgs, test_masks = load_test_split(SPLIT_JSON, IMAGE_DIR, MASK_DIR)
    test_dataset = TeethDataset(test_imgs, test_masks)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
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
                        "gt": masks[0, 0].cpu().numpy(),
                        "pred": preds[0, 0].cpu().numpy(),
                        "name": names[0],
                        "dice": score,
                    }
                )


    mean_dice = float(np.mean(dice_scores))
    std_dice = float(np.std(dice_scores))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 5))

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
    # ==================================================

    fig.suptitle(
        f"Unet_Results\n"
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