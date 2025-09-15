"""Base Transform class for vision data augmentation pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np


class BaseTransform(ABC):
    """
    Abstract base class for custom vision transforms.

    This class provides a standard interface for creating
    custom augmentation and preprocessing transforms.
    """

    def __init__(self, p: float = 1.0, **kwargs):
        """
        Initialize the transform.

        Args:
            p: Probability of applying the transform
            **kwargs: Additional transform-specific parameters
        """
        self.p = p
        self.params = kwargs

    @abstractmethod
    def apply(self, img: Union[Image.Image, torch.Tensor], **params) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the transform to an image.

        Args:
            img: Input image (PIL or Tensor)
            **params: Transform-specific parameters

        Returns:
            Transformed image
        """
        raise NotImplementedError("Subclasses must implement apply method")

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """
        Apply the transform with probability p.

        Args:
            img: Input image

        Returns:
            Transformed or original image
        """
        if random.random() < self.p:
            return self.apply(img, **self.get_params(img))
        return img

    def get_params(self, img: Union[Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Get random parameters for the transform.

        Args:
            img: Input image

        Returns:
            Dictionary of parameters
        """
        return {}

    def __repr__(self) -> str:
        """String representation of the transform."""
        return f"{self.__class__.__name__}(p={self.p}, params={self.params})"


class ComposeTransforms:
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[BaseTransform]):
        """
        Initialize composed transforms.

        Args:
            transforms: List of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """Apply all transforms sequentially."""
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        """String representation."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class MixupTransform(nn.Module):
    """
    Mixup augmentation for vision models.

    Reference: "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Initialize Mixup transform.

        Args:
            alpha: Beta distribution parameter
            p: Probability of applying mixup
        """
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Batch of labels (B,) or (B, num_classes)

        Returns:
            Mixed images, targets_a, targets_b, lambda value
        """
        if random.random() > self.p:
            return images, targets, targets, 1.0

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Random shuffle for mixing
        index = torch.randperm(batch_size, device=images.device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]

        # Return both targets for loss calculation
        targets_a = targets
        targets_b = targets[index]

        return mixed_images, targets_a, targets_b, lam


class CutMixTransform(nn.Module):
    """
    CutMix augmentation for vision models.

    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers
               with Localizable Features"
    """

    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Initialize CutMix transform.

        Args:
            alpha: Beta distribution parameter
            p: Probability of applying cutmix
        """
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Batch of labels

        Returns:
            Mixed images, targets_a, targets_b, lambda value
        """
        if random.random() > self.p:
            return images, targets, targets, 1.0

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Random shuffle for mixing
        index = torch.randperm(batch_size, device=images.device)

        # Get random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

        targets_a = targets
        targets_b = targets[index]

        return mixed_images, targets_a, targets_b, lam

    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class RandAugmentTransform(BaseTransform):
    """
    RandAugment: Practical automated data augmentation.

    Reference: "RandAugment: Practical automated data augmentation with a reduced search space"
    """

    def __init__(self, n: int = 2, m: int = 9, p: float = 1.0):
        """
        Initialize RandAugment.

        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude for all transformations
            p: Probability of applying RandAugment
        """
        super().__init__(p=p)
        self.n = n
        self.m = m
        self.augmentations = self._get_augmentation_list()

    def _get_augmentation_list(self) -> List[callable]:
        """Get list of available augmentations."""
        # This would include various augmentations like
        # AutoContrast, Equalize, Rotate, Solarize, etc.
        # Implementation details omitted for brevity
        return []

    def apply(self, img: Image.Image, **params) -> Image.Image:
        """Apply RandAugment to image."""
        ops = random.choices(self.augmentations, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img


class TestTimeAugmentation:
    """
    Test Time Augmentation (TTA) for improved inference.
    """

    def __init__(
        self,
        transforms: List[callable],
        n_augmentations: int = 5,
        aggregation: str = "mean"
    ):
        """
        Initialize TTA.

        Args:
            transforms: List of augmentation transforms
            n_augmentations: Number of augmentations to apply
            aggregation: How to aggregate predictions ('mean', 'max', 'voting')
        """
        self.transforms = transforms
        self.n_augmentations = n_augmentations
        self.aggregation = aggregation

    def __call__(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA to get robust predictions.

        Args:
            model: The model to use for predictions
            image: Input image

        Returns:
            Aggregated predictions
        """
        predictions = []

        for i in range(self.n_augmentations):
            if i == 0:
                # Original image
                aug_image = image
            else:
                # Apply random augmentation
                transform = random.choice(self.transforms)
                aug_image = transform(image)

            with torch.no_grad():
                pred = model(aug_image.unsqueeze(0))
                predictions.append(pred)

        # Aggregate predictions
        predictions = torch.cat(predictions, dim=0)

        if self.aggregation == "mean":
            return predictions.mean(dim=0)
        elif self.aggregation == "max":
            return predictions.max(dim=0)[0]
        elif self.aggregation == "voting":
            # For classification tasks
            votes = predictions.argmax(dim=1)
            return torch.bincount(votes).argmax()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")