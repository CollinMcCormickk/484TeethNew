import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SEED = 42
IMAGE_DIR = Path("Radiographs")
MASK_DIR = Path("Segmentation/teeth_mask")
SPLIT_JSON = Path("data_split.json")

IMAGE_SIZE = (256, 512)   # (H, W)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # not required, but kept for consistency
THRESH = 0.5


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeethDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=IMAGE_SIZE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
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

        return image, mask


def resolve_path(base_dir: Path, name):
    path = Path(name)
    if path.exists():
        return path
    candidate = base_dir / str(name)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find file: {name}")


def load_split_paths(split_json_path: Path, image_dir: Path, mask_dir: Path):
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_json_path}")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    if "train" not in split_data or "test" not in split_data:
        raise KeyError("data_split.json must contain 'train' and 'test' keys")

    train_entries = split_data["train"]
    test_entries = split_data["test"]

    train_imgs, train_masks = parse_entries(train_entries, image_dir, mask_dir)
    test_imgs, test_masks = parse_entries(test_entries, image_dir, mask_dir)

    return train_imgs, train_masks, test_imgs, test_masks


def parse_entries(entries, image_dir: Path, mask_dir: Path):
    image_paths = []
    mask_paths = []

    for entry in entries:
        if isinstance(entry, dict):
            image_name = entry.get("radiograph")
            mask_name = entry.get("segmentation")
            if image_name is None or mask_name is None:
                raise ValueError(
                    "Each dict entry in data_split.json must contain "
                    "'radiograph' and 'segmentation' fields"
                )
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            image_name, mask_name = entry
        elif isinstance(entry, str):
            image_name = entry
            mask_name = entry
        else:
            raise ValueError(
                "Unsupported entry format in data_split.json. "
                "Use dicts, 2-tuples/lists, or strings."
            )

        image_paths.append(resolve_path(image_dir, image_name))
        mask_paths.append(resolve_path(mask_dir, mask_name))

    return image_paths, mask_paths


def dice_score(pred, target, eps=1e-7):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + eps) / (union + eps)


def build_atlas(dataset):
    imgs = []
    masks = []

    for i in range(len(dataset)):
        image_t, mask_t = dataset[i]  # [1,H,W], [1,H,W]
        imgs.append(image_t.squeeze(0).numpy())
        masks.append(mask_t.squeeze(0).numpy())

    imgs = np.stack(imgs, axis=0)      # [N,H,W]
    masks = np.stack(masks, axis=0)    # [N,H,W]

    mean_img = imgs.mean(axis=0).astype(np.float32)
    mean_mask_prob = masks.mean(axis=0).astype(np.float32)
    mean_mask_bin = (mean_mask_prob >= THRESH).astype(np.uint8)

    return mean_img, mean_mask_prob, mean_mask_bin


def display_atlas(mean_img, mean_mask_prob, mean_mask_bin):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(mean_img, cmap="gray")
    plt.title("Mean Radiograph Atlas")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mean_mask_prob, cmap="hot")
    plt.title("Mean Segmentation Prob.")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mean_img, cmap="gray")
    plt.imshow(mean_mask_bin, cmap="spring", alpha=0.35)
    plt.title("Mean Segmentation Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("figure/AtlasMean")
    # plt.show()


def segment_with_atlas(test_img_t, mean_img, mean_mask_prob, thresh=THRESH):
    """
    test_img_t: torch tensor [1,H,W], values in [0,1]
    mean_img: [H,W]
    mean_mask_prob: [H,W]
    """
    test_img = test_img_t.squeeze(0).numpy().astype(np.float32)

    ref = mean_img.astype(np.float32)
    mov = test_img.astype(np.float32)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        300,
        1e-6
    )

    try:
        _, warp = cv2.findTransformECC(
            ref,
            mov,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5
        )
    except cv2.error as e:
        print("ECC registration failed; using unregistered atlas.")
        print("Reason:", e)
        pred_prob = mean_mask_prob.copy()
        pred_bin = (pred_prob >= thresh).astype(np.uint8)
        return pred_bin, pred_prob, np.eye(2, 3, dtype=np.float32)

    h, w = mean_img.shape

    pred_prob = cv2.warpAffine(
        mean_mask_prob.astype(np.float32),
        warp,
        (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    pred_bin = (pred_prob >= thresh).astype(np.uint8)
    return pred_bin, pred_prob, warp


def evaluate_atlas(dataset, mean_img, mean_mask_prob):
    dice_scores = []

    for i in range(len(dataset)):
        image_t, mask_t = dataset[i]
        gt = mask_t.squeeze(0).numpy().astype(np.uint8)

        pred_bin, _, _ = segment_with_atlas(image_t, mean_img, mean_mask_prob)
        d = dice_score(pred_bin, gt)
        dice_scores.append(d)

    dice_scores = np.array(dice_scores, dtype=np.float32)
    return dice_scores


def show_prediction(image_t, gt_t, pred_bin, pred_prob, title_prefix="Sample"):
    image_np = image_t.squeeze(0).numpy()
    gt_np = gt_t.squeeze(0).numpy()

    d = dice_score(pred_bin, gt_np)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title(f"{title_prefix} Radiograph")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gt_np, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(pred_prob, cmap="hot")
    plt.title("Pred Prob.")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(pred_bin, cmap="gray")
    plt.title("Atlas Prediction")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(image_np, cmap="gray")
    plt.imshow(pred_bin, cmap="spring", alpha=0.35)
    plt.title(f"Overlay Dice={d:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("figure/AtlasPred")
    # plt.show()


def main():
    set_seed(SEED)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Missing mask directory: {MASK_DIR}")

    train_imgs, train_masks, test_imgs, test_masks = load_split_paths(
        SPLIT_JSON, IMAGE_DIR, MASK_DIR
    )

    train_dataset = TeethDataset(train_imgs, train_masks)
    test_dataset = TeethDataset(test_imgs, test_masks)

    print(f"Using device: {DEVICE}")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    mean_img, mean_mask_prob, mean_mask_bin = build_atlas(train_dataset)

    print("Displaying atlas and mean segmentation...")
    display_atlas(mean_img, mean_mask_prob, mean_mask_bin)

    print("Running atlas segmentation on one test sample...")
    sample_idx = 0
    image_t, gt_t = test_dataset[sample_idx]
    pred_bin, pred_prob, _ = segment_with_atlas(image_t, mean_img, mean_mask_prob)
    show_prediction(image_t, gt_t, pred_bin, pred_prob, title_prefix="Test")

    print("Evaluating atlas on full test set...")
    test_dice = evaluate_atlas(test_dataset, mean_img, mean_mask_prob)

    print(f"Test Dice mean: {test_dice.mean():.4f}")
    print(f"Test Dice std : {test_dice.std():.4f}")
    print(f"Test Dice min : {test_dice.min():.4f}")
    print(f"Test Dice max : {test_dice.max():.4f}")

    plt.figure(figsize=(6, 4))
    plt.hist(test_dice, bins=10, edgecolor="black")
    plt.title("Atlas Segmentation Dice on Test Set")
    plt.xlabel("Dice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figure/AtlasDice")
    # plt.show()


if __name__ == "__main__":
    main()