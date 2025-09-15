"""Vision classification models."""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig

from src.core.base import BaseVisionModel
from src.core.registry import register_model


@register_model(
    name="resnet",
    task_type="classification",
    aliases=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
class ResNetClassifier(BaseVisionModel):
    """
    ResNet-based image classifier with flexible architecture.

    Supports ResNet-18/34/50/101/152 with optional pretrained weights
    and custom classification heads.
    """

    def __init__(
        self,
        config: DictConfig,
        num_classes: int = 1000,
        resnet_version: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        """
        Initialize ResNet classifier.

        Args:
            config: Configuration object
            num_classes: Number of output classes
            resnet_version: ResNet variant to use
            pretrained: Use pretrained ImageNet weights
            freeze_backbone: Freeze backbone during training
            dropout_rate: Dropout rate before final layer
            **kwargs: Additional arguments for base class
        """
        self.resnet_version = resnet_version
        self.dropout_rate = dropout_rate

        super().__init__(
            config=config,
            num_classes=num_classes,
            task_type="classification",
            backbone_name=resnet_version,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    def _build_backbone(self) -> nn.Module:
        """Build ResNet backbone."""
        # Get the appropriate ResNet model
        resnet_models = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        if self.resnet_version not in resnet_models:
            raise ValueError(f"Unknown ResNet version: {self.resnet_version}")

        # Load ResNet model
        resnet = resnet_models[self.resnet_version](pretrained=self.pretrained)

        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)

        # Store feature dimension for head
        if self.resnet_version in ["resnet18", "resnet34"]:
            self.feature_dim = 512
        else:
            self.feature_dim = 2048

        return backbone

    def _build_head(self) -> nn.Module:
        """Build classification head."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.num_classes)
        )

    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        return nn.CrossEntropyLoss()

    def _get_parameter_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for differential learning rates."""
        # Different learning rates for backbone and head
        backbone_params = self.backbone.parameters()
        head_params = self.head.parameters()

        return [
            {"params": backbone_params, "lr": self.learning_rate * 0.1},
            {"params": head_params, "lr": self.learning_rate}
        ]


@register_model(
    name="vgg",
    task_type="classification",
    aliases=["vgg11", "vgg13", "vgg16", "vgg19"]
)
class VGGClassifier(BaseVisionModel):
    """
    VGG-based image classifier.

    Supports VGG-11/13/16/19 with batch normalization options.
    """

    def __init__(
        self,
        config: DictConfig,
        num_classes: int = 1000,
        vgg_version: str = "vgg16",
        batch_norm: bool = True,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        **kwargs
    ):
        """Initialize VGG classifier."""
        self.vgg_version = vgg_version
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

        super().__init__(
            config=config,
            num_classes=num_classes,
            task_type="classification",
            backbone_name=vgg_version,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    def _build_backbone(self) -> nn.Module:
        """Build VGG backbone."""
        vgg_models = {
            "vgg11": models.vgg11_bn if self.batch_norm else models.vgg11,
            "vgg13": models.vgg13_bn if self.batch_norm else models.vgg13,
            "vgg16": models.vgg16_bn if self.batch_norm else models.vgg16,
            "vgg19": models.vgg19_bn if self.batch_norm else models.vgg19,
        }

        if self.vgg_version not in vgg_models:
            raise ValueError(f"Unknown VGG version: {self.vgg_version}")

        vgg = vgg_models[self.vgg_version](pretrained=self.pretrained)

        # Extract features (convolutional layers)
        backbone = vgg.features

        # Add adaptive pooling to handle variable input sizes
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.feature_dim = 512 * 7 * 7

        return backbone

    def _build_head(self) -> nn.Module:
        """Build classification head."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, self.num_classes)
        )

    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        return nn.CrossEntropyLoss()


@register_model(
    name="efficientnet",
    task_type="classification",
    aliases=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
             "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]
)
class EfficientNetClassifier(BaseVisionModel):
    """
    EfficientNet-based image classifier.

    Supports EfficientNet-B0 through B7 with compound scaling.
    """

    def __init__(
        self,
        config: DictConfig,
        num_classes: int = 1000,
        efficientnet_version: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        **kwargs
    ):
        """Initialize EfficientNet classifier."""
        self.efficientnet_version = efficientnet_version
        self.dropout_rate = dropout_rate
        self.stochastic_depth_prob = stochastic_depth_prob

        super().__init__(
            config=config,
            num_classes=num_classes,
            task_type="classification",
            backbone_name=efficientnet_version,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )

    def _build_backbone(self) -> nn.Module:
        """Build EfficientNet backbone."""
        efficientnet_models = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
            "efficientnet_b5": models.efficientnet_b5,
            "efficientnet_b6": models.efficientnet_b6,
            "efficientnet_b7": models.efficientnet_b7,
        }

        if self.efficientnet_version not in efficientnet_models:
            raise ValueError(f"Unknown EfficientNet version: {self.efficientnet_version}")

        # Load EfficientNet model
        efficientnet = efficientnet_models[self.efficientnet_version](
            pretrained=self.pretrained
        )

        # Extract features (everything except classifier)
        backbone = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )

        # Get feature dimension based on version
        feature_dims = {
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
            "efficientnet_b4": 1792,
            "efficientnet_b5": 2048,
            "efficientnet_b6": 2304,
            "efficientnet_b7": 2560,
        }
        self.feature_dim = feature_dims[self.efficientnet_version]

        return backbone

    def _build_head(self) -> nn.Module:
        """Build classification head."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate, inplace=True),
            nn.Linear(self.feature_dim, self.num_classes)
        )

    def _setup_loss(self) -> nn.Module:
        """Setup loss function with label smoothing."""
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth."""
        # Apply stochastic depth during training
        if self.training and self.stochastic_depth_prob > 0:
            # Implementation would go here
            pass

        return super().forward(x)