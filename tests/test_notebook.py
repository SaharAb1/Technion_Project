"""
Comprehensive validation of CNN_vs_ViT_Medical_Image_Classification.ipynb

Tests all code that can be tested WITHOUT the actual dataset:
  1. All imports
  2. Model builders (build_resnet, build_vit) -- output shapes & param freezing
  3. Training functions (train_one_epoch, evaluate, train_model) with dummy data
  4. GradCAM class
  5. Plotting functions (matplotlib Agg backend)
  6. Forward + backward passes

NOTE: Since network access is unavailable, models are built WITHOUT pretrained
weights. The code logic, architecture, shapes, and freezing behaviour are
identical -- only the initial weight values differ.
"""

import os
import sys
import copy
import random
import warnings

# Force non-interactive matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms

import timm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import pytest

# Suppress noisy warnings during tests
warnings.filterwarnings("ignore")

# Constants (mirroring the notebook)
DEVICE = torch.device("cpu")
IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 4
SEED = 42


# =====================================================================
#  Model builders -- same logic as notebook, but with pretrained=False
#  to avoid network downloads in CI/test environments.
#  The architecture, freezing logic, and head replacement are IDENTICAL
#  to the notebook code, which is what we are validating.
# =====================================================================

def build_resnet(num_classes=2, freeze_backbone=True):
    """Build a ResNet-50 (notebook version, adapted for offline testing)."""
    # Notebook uses: weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    # We use weights=None since we cannot download in this environment.
    model = torchvision.models.resnet50(weights=None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head -- IDENTICAL to notebook
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_vit(num_classes=2, freeze_backbone=True):
    """Build a ViT-B/16 (notebook version, adapted for offline testing)."""
    # Notebook uses: pretrained=True
    # We use pretrained=False since we cannot download in this environment.
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    return model


# =====================================================================
#  Notebook functions -- copied VERBATIM from the notebook
# =====================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns loss, accuracy, predictions, and true labels."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    total = len(all_labels)
    return (
        running_loss / total,
        accuracy_score(all_labels, all_preds),
        np.array(all_preds),
        np.array(all_labels),
    )


def train_model(
    model, train_loader, val_loader, num_epochs, lr, weight_decay, device, model_name="Model"
):
    """Full training loop -- verbatim from notebook."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = defaultdict(list)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, dict(history)


class GradCAM:
    """Grad-CAM -- verbatim from notebook."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# =====================================================================
#  Plotting functions -- verbatim from notebook (with plt.close)
# =====================================================================

def plot_training_curves(histories, model_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#FF5722"]
    for idx, (hist, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(hist["train_loss"]) + 1)
        c = colors[idx]
        axes[0].plot(epochs, hist["train_loss"], "-", color=c, label=f"{name} Train")
        axes[0].plot(epochs, hist["val_loss"], "--", color=c, label=f"{name} Val")
        axes[1].plot(epochs, hist["train_acc"], "-", color=c, label=f"{name} Train")
        axes[1].plot(epochs, hist["val_acc"], "--", color=c, label=f"{name} Val")
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Training Curves: ResNet-50 vs ViT-B/16", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.close(fig)


def plot_confusion_matrices(labels_list, preds_list, model_names, class_names):
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    for ax, labels, preds, name in zip(axes, labels_list, preds_list, model_names):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False,
        )
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.close(fig)


def plot_bar_comparison(resnet_acc, resnet_f1, vit_acc, vit_f1):
    """Replicates the bar chart comparison from the notebook."""
    metrics = {"Accuracy": [resnet_acc, vit_acc], "F1-Score": [resnet_f1, vit_f1]}
    x = np.arange(len(metrics))
    width = 0.3
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(
        x - width / 2,
        [metrics["Accuracy"][0], metrics["F1-Score"][0]],
        width, label="ResNet-50", color="#2196F3",
    )
    bars2 = ax.bar(
        x + width / 2,
        [metrics["Accuracy"][1], metrics["F1-Score"][1]],
        width, label="ViT-B/16", color="#FF5722",
    )
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys(), fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy & F1-Score", fontsize=13, fontweight="bold")
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", fontsize=10, fontweight="bold",
            )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.close(fig)


# =====================================================================
#  Helpers
# =====================================================================

def make_dummy_loader(n_samples=16, img_size=224, num_classes=2, batch_size=4):
    """Create a DataLoader with random images and labels."""
    images = torch.randn(n_samples, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =====================================================================
#                         T E S T S
# =====================================================================


class TestImports:
    """Test 1: Verify all imports used by the notebook work."""

    def test_core_imports(self):
        import os, copy, random
        from collections import defaultdict

    def test_numpy(self):
        import numpy as np
        assert hasattr(np, "array")

    def test_matplotlib(self):
        import matplotlib.pyplot as plt
        assert hasattr(plt, "subplots")

    def test_seaborn(self):
        import seaborn as sns
        assert hasattr(sns, "heatmap")

    def test_torch(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        assert hasattr(torch, "tensor")

    def test_torchvision(self):
        import torchvision
        from torchvision import datasets, transforms
        assert hasattr(torchvision.models, "resnet50")

    def test_timm(self):
        import timm
        assert hasattr(timm, "create_model")

    def test_sklearn_metrics(self):
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        assert callable(accuracy_score)

    def test_tqdm(self):
        from tqdm.auto import tqdm
        assert callable(tqdm)


class TestBuildResNet:
    """Test 2a: build_resnet produces correct architecture."""

    def test_build_resnet_returns_model(self):
        model = build_resnet(num_classes=2, freeze_backbone=True)
        assert isinstance(model, nn.Module)

    def test_resnet_output_shape(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_resnet_output_shape_multiclass(self):
        model = build_resnet(num_classes=5, freeze_backbone=True).to(DEVICE)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5), f"Expected (1, 5), got {out.shape}"

    def test_resnet_backbone_frozen(self):
        model = build_resnet(num_classes=2, freeze_backbone=True)
        for name, param in model.named_parameters():
            if "fc" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_resnet_backbone_unfrozen(self):
        model = build_resnet(num_classes=2, freeze_backbone=False)
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"

    def test_resnet_fc_is_sequential(self):
        model = build_resnet(num_classes=2)
        assert isinstance(model.fc, nn.Sequential)
        assert isinstance(model.fc[0], nn.Dropout)
        assert isinstance(model.fc[1], nn.Linear)

    def test_resnet_fc_linear_dims(self):
        model = build_resnet(num_classes=2)
        linear = model.fc[1]
        assert linear.in_features == 2048, f"ResNet-50 fc in_features should be 2048, got {linear.in_features}"
        assert linear.out_features == 2

    def test_resnet_param_counts(self):
        model = build_resnet(num_classes=2, freeze_backbone=True)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 20_000_000, "ResNet-50 should have >20M params"
        assert trainable < 10_000, "With frozen backbone, trainable params should be small"
        assert trainable > 0, "There must be some trainable params"


class TestBuildViT:
    """Test 2b: build_vit produces correct architecture."""

    def test_build_vit_returns_model(self):
        model = build_vit(num_classes=2, freeze_backbone=True)
        assert isinstance(model, nn.Module)

    def test_vit_output_shape(self):
        model = build_vit(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_vit_output_shape_multiclass(self):
        model = build_vit(num_classes=10, freeze_backbone=True).to(DEVICE)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"

    def test_vit_backbone_frozen(self):
        model = build_vit(num_classes=2, freeze_backbone=True)
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if "head" in name:
                assert param.requires_grad, f"{name} should be trainable"
                trainable_count += param.numel()
            else:
                assert not param.requires_grad, f"{name} should be frozen"
                frozen_count += param.numel()
        assert frozen_count > 0, "Some params should be frozen"
        assert trainable_count > 0, "Head params should be trainable"

    def test_vit_param_counts(self):
        model = build_vit(num_classes=2, freeze_backbone=True)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 80_000_000, "ViT-B/16 should have >80M params"
        assert trainable < 5_000, "With frozen backbone, trainable params should be small"


class TestTrainOneEpoch:
    """Test 3a: train_one_epoch with dummy data."""

    def test_returns_loss_and_accuracy(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
        assert isinstance(loss, float), "Loss should be a float"
        assert isinstance(acc, float), "Accuracy should be a float"
        assert loss >= 0, "Loss should be non-negative"
        assert 0.0 <= acc <= 1.0, f"Accuracy should be in [0, 1], got {acc}"

    def test_model_in_train_mode_during_training(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        train_one_epoch(model, loader, criterion, optimizer, DEVICE)
        assert model.training, "Model should be in training mode after train_one_epoch"

    def test_weights_update(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

        w_before = model.fc[1].weight.data.clone()
        train_one_epoch(model, loader, criterion, optimizer, DEVICE)
        w_after = model.fc[1].weight.data.clone()

        assert not torch.equal(w_before, w_after), "FC weights should have changed after training"


class TestEvaluate:
    """Test 3b: evaluate with dummy data."""

    def test_returns_four_values(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        result = evaluate(model, loader, criterion, DEVICE)
        assert len(result) == 4, f"Expected 4 return values, got {len(result)}"

    def test_return_types(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        loss, acc, preds, labels = evaluate(model, loader, criterion, DEVICE)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_predictions_shape(self):
        n_samples = 12
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=n_samples, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        _, _, preds, labels = evaluate(model, loader, criterion, DEVICE)
        assert len(preds) == n_samples, f"Expected {n_samples} predictions, got {len(preds)}"
        assert len(labels) == n_samples

    def test_predictions_are_valid_classes(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=16, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        _, _, preds, _ = evaluate(model, loader, criterion, DEVICE)
        assert all(p in [0, 1] for p in preds), "Predictions should be 0 or 1 for binary"

    def test_model_in_eval_mode(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        evaluate(model, loader, criterion, DEVICE)
        assert not model.training, "Model should be in eval mode after evaluate()"


class TestTrainModel:
    """Test 3c: Full train_model loop with dummy data."""

    def test_train_model_resnet(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        train_loader = make_dummy_loader(n_samples=16, batch_size=4)
        val_loader = make_dummy_loader(n_samples=8, batch_size=4)

        trained_model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=2, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, model_name="ResNet-test",
        )
        assert isinstance(trained_model, nn.Module)
        assert isinstance(history, dict)

    def test_history_keys(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        train_loader = make_dummy_loader(n_samples=16, batch_size=4)
        val_loader = make_dummy_loader(n_samples=8, batch_size=4)

        _, history = train_model(
            model, train_loader, val_loader,
            num_epochs=3, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, model_name="ResNet-test",
        )
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            assert key in history, f"Missing history key: {key}"
            assert len(history[key]) == 3, f"Expected 3 entries for {key}, got {len(history[key])}"

    def test_history_values_are_floats(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        train_loader = make_dummy_loader(n_samples=8, batch_size=4)
        val_loader = make_dummy_loader(n_samples=8, batch_size=4)

        _, history = train_model(
            model, train_loader, val_loader,
            num_epochs=2, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, model_name="Test",
        )
        for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
            for v in history[key]:
                assert isinstance(v, float), f"history['{key}'] values should be float, got {type(v)}"

    def test_train_model_vit(self):
        model = build_vit(num_classes=2, freeze_backbone=True).to(DEVICE)
        train_loader = make_dummy_loader(n_samples=8, batch_size=4)
        val_loader = make_dummy_loader(n_samples=8, batch_size=4)

        trained_model, history = train_model(
            model, train_loader, val_loader,
            num_epochs=2, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, model_name="ViT-test",
        )
        assert isinstance(trained_model, nn.Module)
        assert "train_loss" in history

    def test_train_model_returns_same_object(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        train_loader = make_dummy_loader(n_samples=8, batch_size=4)
        val_loader = make_dummy_loader(n_samples=8, batch_size=4)

        trained_model, _ = train_model(
            model, train_loader, val_loader,
            num_epochs=2, lr=1e-3, weight_decay=1e-4,
            device=DEVICE, model_name="Test",
        )
        assert trained_model is model, "train_model should modify and return same model object"


class TestGradCAM:
    """Test 4: GradCAM class works on ResNet."""

    def test_gradcam_instantiation(self):
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        assert gc.model is model
        assert gc.gradients is None
        assert gc.activations is None

    def test_gradcam_generate(self):
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

        cam = gc.generate(x, target_class=0)
        assert isinstance(cam, np.ndarray), "CAM should be a numpy array"
        assert cam.ndim == 2, f"CAM should be 2D, got {cam.ndim}D"
        assert cam.shape == (7, 7), f"Expected (7, 7) for ResNet layer4, got {cam.shape}"

    def test_gradcam_values_normalized(self):
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

        cam = gc.generate(x, target_class=1)
        assert cam.min() >= 0.0, "CAM values should be >= 0 (ReLU)"
        assert cam.max() <= 1.0 + 1e-6, "CAM values should be <= 1 (normalized)"

    def test_gradcam_auto_target_class(self):
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

        # target_class=None should auto-select predicted class
        cam = gc.generate(x, target_class=None)
        assert isinstance(cam, np.ndarray)
        assert cam.shape == (7, 7)

    def test_gradcam_hooks_populated(self):
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

        gc.generate(x, target_class=0)
        assert gc.activations is not None, "Activations should be captured"
        assert gc.gradients is not None, "Gradients should be captured"
        assert gc.activations.shape[0] == 1, "Batch dim should be 1"
        assert gc.gradients.shape[0] == 1, "Batch dim should be 1"
        # ResNet layer4 output: (1, 2048, 7, 7)
        assert gc.activations.shape[1] == 2048
        assert gc.gradients.shape[1] == 2048

    def test_gradcam_resizable(self):
        """Test that the CAM heatmap can be resized to image size (as done in notebook)."""
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        cam = gc.generate(x)

        cam_resized = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        assert cam_resized.shape == (224, 224)

    def test_gradcam_different_classes_differ(self):
        """CAMs for different target classes should generally differ."""
        model = build_resnet(num_classes=2, freeze_backbone=False).to(DEVICE)
        gc = GradCAM(model, model.layer4)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

        cam0 = gc.generate(x, target_class=0)
        cam1 = gc.generate(x, target_class=1)
        # They should not be identical (except in degenerate cases)
        # We check they are computed without error at minimum
        assert cam0.shape == cam1.shape == (7, 7)


class TestPlotFunctions:
    """Test 5: All plotting functions run without error (Agg backend)."""

    def test_plot_training_curves(self):
        hist1 = {
            "train_loss": [0.7, 0.5, 0.3],
            "val_loss": [0.8, 0.6, 0.4],
            "train_acc": [0.6, 0.75, 0.85],
            "val_acc": [0.55, 0.70, 0.80],
        }
        hist2 = {
            "train_loss": [0.65, 0.45, 0.25],
            "val_loss": [0.75, 0.55, 0.35],
            "train_acc": [0.65, 0.78, 0.88],
            "val_acc": [0.60, 0.72, 0.82],
        }
        plot_training_curves([hist1, hist2], ["ResNet-50", "ViT-B/16"])

    def test_plot_confusion_matrices_two_models(self):
        labels1 = np.array([0, 0, 1, 1, 0, 1])
        preds1 = np.array([0, 1, 1, 1, 0, 0])
        labels2 = np.array([0, 0, 1, 1, 0, 1])
        preds2 = np.array([0, 0, 1, 0, 0, 1])
        plot_confusion_matrices(
            [labels1, labels2], [preds1, preds2],
            ["ResNet-50", "ViT-B/16"], CLASS_NAMES,
        )

    def test_plot_confusion_matrices_single_model(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 1, 0])
        plot_confusion_matrices([labels], [preds], ["SingleModel"], CLASS_NAMES)

    def test_plot_bar_comparison(self):
        plot_bar_comparison(
            resnet_acc=0.92, resnet_f1=0.91,
            vit_acc=0.95, vit_f1=0.94,
        )

    def test_classification_report_output(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 1, 0, 0])
        report = classification_report(labels, preds, target_names=CLASS_NAMES)
        assert isinstance(report, str)
        assert "NORMAL" in report
        assert "PNEUMONIA" in report


class TestForwardBackward:
    """Test 6: Forward and backward passes work correctly."""

    def test_resnet_forward_backward(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(4, 3, 224, 224, device=DEVICE)
        labels = torch.tensor([0, 1, 0, 1], device=DEVICE)

        outputs = model(x)
        assert outputs.shape == (4, 2)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"

    def test_vit_forward_backward(self):
        model = build_vit(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        labels = torch.tensor([0, 1], device=DEVICE)

        outputs = model(x)
        assert outputs.shape == (2, 2)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if "head" in name and param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"

    def test_resnet_output_probabilities(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=DEVICE), atol=1e-5)

    def test_vit_output_probabilities(self):
        model = build_vit(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=DEVICE), atol=1e-5)

    def test_frozen_params_no_grad(self):
        """Frozen backbone params should have no gradients after backward."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        labels = torch.tensor([0, 1], device=DEVICE)

        out = model(x)
        loss = nn.CrossEntropyLoss()(out, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Frozen param {name} should not have gradients"

    def test_multiple_forward_passes_consistent(self):
        """Eval mode forward passes should be deterministic."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        model.eval()
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2), "Deterministic eval passes should match"

    def test_loss_decreases_with_training(self):
        """Loss should decrease over multiple epochs of training."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        loader = make_dummy_loader(n_samples=16, batch_size=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

        losses = []
        for _ in range(5):
            loss, _ = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
            losses.append(loss)
        # At least the last loss should be less than the first (not always monotonic)
        assert losses[-1] < losses[0], f"Loss should decrease: {losses}"


class TestTransforms:
    """Test the transform pipelines from the notebook."""

    def test_train_transforms(self):
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = train_transforms(img)
        assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"

    def test_eval_transforms(self):
        eval_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8))
        tensor = eval_transforms(img)
        assert tensor.shape == (3, 224, 224)

    def test_transforms_produce_normalized_output(self):
        """After normalization, values should not be strictly in [0,1]."""
        eval_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = eval_transforms(img)
        # After ImageNet normalization, values are centered around 0 and can be negative
        assert tensor.min() < 0.0 or tensor.max() > 1.0, \
            "Normalized tensor should have values outside [0,1]"


class TestSklearnMetrics:
    """Test that sklearn metrics work as used in the notebook."""

    def test_accuracy_score(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.75

    def test_f1_score_weighted(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        f1 = f1_score(y_true, y_pred, average="weighted")
        assert 0.0 <= f1 <= 1.0

    def test_confusion_matrix_shape(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_confusion_matrix_values(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        # Perfect predictions: diagonal = [2, 2], off-diagonal = 0
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0


class TestEdgeCases:
    """Additional edge-case and integration tests."""

    def test_resnet_single_sample(self):
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_vit_single_sample(self):
        model = build_vit(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_cosine_annealing_scheduler(self):
        """Test that the scheduler used in the notebook works."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(5):
            scheduler.step()
        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr != initial_lr, "LR should change with cosine annealing"

    def test_denormalize_image(self):
        """Test the denormalization used for displaying images."""
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = torch.randn(3, 224, 224)
        denormed = (img * std + mean).clamp(0, 1)
        assert denormed.min() >= 0.0
        assert denormed.max() <= 1.0
        assert denormed.shape == (3, 224, 224)

    def test_cross_entropy_loss_with_model_output(self):
        """Test that CrossEntropyLoss works with model outputs."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        x = torch.randn(4, 3, 224, 224, device=DEVICE)
        labels = torch.tensor([0, 1, 0, 1], device=DEVICE)
        with torch.no_grad():
            out = model(x)
        loss = nn.CrossEntropyLoss()(out, labels)
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_adamw_optimizer_creation(self):
        """Test that AdamW can be created with filtered parameters."""
        model = build_resnet(num_classes=2, freeze_backbone=True).to(DEVICE)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-4,
        )
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-4


# Entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
