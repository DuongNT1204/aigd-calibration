import json
import torch
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset as TorchDataset
from .transforms import apply_random_augmentations
import numpy as np
import cv2

# CLIP normalization constants (from CLIP's preprocessor_config.json)
# Source: https://huggingface.co/openai/clip-vit-large-patch14/blob/main/preprocessor_config.json
# Pre-initialized as tensors to avoid recreating them on each call
MEAN_CLIP = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD_CLIP = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)



def train_transforms(image: np.ndarray, target_size=None, crop_prob=0.5) -> np.ndarray:
    """Apply training augmentations and JPEG compression"""
    image, _, _, _ = apply_random_augmentations(
        image,
        target_size=target_size,
        level_probs={0: 0.1, 1: 0.2, 2: 0.4, 3: 0.3},
        crop_prob=crop_prob,
    )
    # image_np_array = compress_image_jpeg_pil(image, quality=75)
    return image


def val_transforms(image: np.ndarray, target_size=None) -> np.ndarray:
    """Apply validation augmentations and JPEG compression"""
    image, _, _, _ = apply_random_augmentations(
        image,
        target_size=target_size,
        level_probs={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
        crop_prob=0.0,
    )
    # image_np_array = compress_image_jpeg_pil(image, quality=75)
    return image


def base_transforms(image: np.ndarray, target_size=None) -> torch.Tensor:
    """Convert numpy array (HWC, uint8) to torch tensor (CHW, float32)"""
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return image


def normalize_clip(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor using CLIP statistics.
    
    Args:
        image: torch.Tensor of shape (C, H, W) with values in [0, 1]
    
    Returns:
        torch.Tensor normalized with CLIP mean/std
    """
    # Move mean/std to same device as input image
    mean = MEAN_CLIP.to(image.device)
    std = STD_CLIP.to(image.device)
    image = (image - mean) / std
    return image


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor using ImageNet statistics for timm pretrained models.
    """
    mean = MEAN_IMAGENET.to(image.device)
    std = STD_IMAGENET.to(image.device)
    image = (image - mean) / std
    return image


def base_transforms_clip(image: np.ndarray, target_size=None) -> torch.Tensor:
    """
    Convert numpy array (HWC, uint8) to normalized CLIP tensor (CHW, float32).
    Same as base_transforms but with CLIP normalization applied.
    """
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    image = normalize_clip(image)
    return image


def base_transforms_imagenet(image: np.ndarray, target_size=None) -> torch.Tensor:
    """
    Convert numpy array (HWC, uint8) to ImageNet-normalized tensor.
    """
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    image = normalize_imagenet(image)
    return image

    
class AIGDImageDataset(TorchDataset):
    """JSONL image dataset used by Lightning training and evaluation.

    This class keeps the same image loading, retry, augmentation, and
    normalization logic from the original Dataset implementation, but its
    default output is the dictionary format required by training/evaluation.
    """

    def __init__(
        self,
        jsonl_path=None,
        image_size=224,
        is_training=False,
        is_inference=False,
        normalization="imagenet",
        target_size=None,
        data_path=None,
        return_dict=True,
    ):
        if jsonl_path is None:
            jsonl_path = data_path
        if jsonl_path is None:
            raise ValueError("AIGDImageDataset requires jsonl_path or data_path")
        if target_size is None:
            target_size = (image_size, image_size)

        use_clip_norm = normalization == "clip"
        self.data_path = data_path
        self.is_training = is_training
        self.target_size = self._normalize_target_size(target_size)
        self.is_inference = is_inference
        self.use_clip_norm = use_clip_norm  # Use CLIP normalization if True
        if normalization is None:
            normalization = "clip" if use_clip_norm else "none"
        self.normalization = str(normalization).lower()
        self.return_dict = return_dict

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

        self.label_map = {
            'real': 0,
            '0': 0,
            'synthetic': 1,
            'semisynthetic': 1,
            'fake': 1,
            'ai': 1,
            'generated': 1,
            '1': 1,
        }

    @staticmethod
    def _normalize_target_size(target_size):
        if isinstance(target_size, int):
            return (target_size, target_size)
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            return (int(target_size[0]), int(target_size[1]))
        raise ValueError(f"target_size must be int or (H, W), got {target_size}")

    def _label_to_int(self, raw_label):
        key = str(raw_label).strip().lower()
        if key not in self.label_map:
            raise ValueError(f"Unsupported label {raw_label!r}. Known labels: {sorted(self.label_map)}")
        return self.label_map[key]

    def _make_output(self, image_tensor, label, item):
        if not self.return_dict:
            return image_tensor, label

        generator = item.get('generator', item.get('model', item.get('dataset', item.get('source', 'unknown'))))
        source = item.get('source', item.get('dataset', generator))
        return {
            "image": image_tensor,
            "label": label,
            "image_path": item.get("image_path", item.get("path", item.get("file", ""))),
            "generator": str(generator),
            "source": str(source),
        }
    
    def __getitem__(self, idx):
        max_retries = 10
        retry_count = 0
        original_idx = idx
        
        while retry_count < max_retries:
            current_idx = (original_idx + retry_count) % len(self.data)
            item = self.data[current_idx]
            label = torch.tensor(self._label_to_int(item['label']), dtype=torch.long)
    
            try:
                # Load image
                try:
                    image_path = item.get('image_path', item.get('path', item.get('file')))
                    if not image_path:
                        raise ValueError("Missing image_path/path/file")
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        image = np.array(image)
                except Exception:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError("cv2.imread returned None")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
                # Validate image
                if image is None or image.size == 0:
                    raise ValueError("Image is None or empty")
                if len(image.shape) != 3 or image.shape[2] != 3:
                    raise ValueError(f"Invalid image shape: {image.shape}")
    
                # Apply transforms
                if self.is_training:
                    image_augmented = train_transforms(image, self.target_size, crop_prob=0.5)

                    # if item['label'] == "semisynthetic" :
                    #     image_augmented = train_transforms(image, self.target_size, crop_prob=0.1)
                    # else:
                    #     image_augmented = train_transforms(image, self.target_size, crop_prob=0.1)
                else:
                    image_augmented = val_transforms(image, self.target_size)
    
                # Validate augmented image
                if image_augmented is None or image_augmented.size == 0:
                    raise ValueError("Augmented image is None or empty")
                if len(image_augmented.shape) != 3 or image_augmented.shape[2] != 3:
                    raise ValueError(f"Invalid augmented shape: {image_augmented.shape}")
    
                # Convert to tensor with optional CLIP normalization
                if self.normalization == "clip":
                    image_tensor = base_transforms_clip(image_augmented, self.target_size)
                elif self.normalization == "imagenet":
                    image_tensor = base_transforms_imagenet(image_augmented, self.target_size)
                elif self.normalization in {"none", "raw"}:
                    image_tensor = base_transforms(image_augmented, self.target_size)
                else:
                    raise ValueError(f"Unsupported normalization: {self.normalization}")
                
                return self._make_output(image_tensor, label, item)
    
            except Exception as e:
                if retry_count == 0:
                    print(f"[Warning] Skipping {item.get('image_path', item.get('path', 'unknown'))}: {e}")
                retry_count += 1
        
        # Fallback dummy sample
        print(f"[ERROR] All retries failed for idx {original_idx}")
        dummy_image = torch.zeros(3, self.target_size[0], self.target_size[1], dtype=torch.float32)
        fallback_item = {
            "image_path": "error",
            "generator": "error",
            "source": "error",
        }
        return self._make_output(dummy_image, torch.tensor(0, dtype=torch.long), fallback_item)

    def __len__(self):
        return len(self.data)



    
if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    dataset = AIGDImageDataset(
        '/workspace/image_json/val_1k.jsonl',
        is_training=False,
        target_size=(336, 336),
        normalization="clip",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Quick test
    sample = dataset[0]
    print(f"Sample - shape: {sample['image'].shape}, label: {sample['label']}")

    loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)

    start = time.time()
    for i, batch in enumerate(loader):
        if i == 10:
            break
    print(f"Loaded 11 batches in {time.time() - start:.2f}s")
