#!/usr/bin/env python3
"""
================================================================================
Improved Self-Supervised Learning for MCI Detection
================================================================================
Author: Improved Research Pipeline
Version: 6.0.0 (Production - ResNet50 - ALL CRITICAL FIXES APPLIED + DataLoader Stability)

Title: Improved Self-Supervised Learning for MCI Detection
Concept: Pre-train 3D CNNs on unlabeled MRI using improved SimCLR framework to learn 
         latent neuroanatomical features, then fine-tune for MCI detection with better techniques.

Key improvements (v6.0.0 production fixes):
- FIX: DataLoader deadlock issue - disabled persistent_workers to prevent worker hangs
- FIX: Reduced default num_workers to prevent multiprocessing issues with large medical images
- FIX: Disabled default data caching to prevent memory exhaustion
- FIX: Added DataLoader validation before training starts
- FIX: Added comprehensive error handling and timeouts
- FIX: Better logging for batch size and training parameters
- Previous v5.0.0 fixes retained: Backbone loading, UnfreezeCallback, SyncBatchNorm conversion,
  DDP key remapping, weight validation, normalization, augmentation, contrastive loss, class balancing.
"""

from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import json
import random
import logging
import argparse
import warnings
import re
import pickle
import gc
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Tuple, Optional, Union, Any, Callable,
    Literal, TypeVar
)
from enum import Enum, auto
from collections import defaultdict, Counter
from functools import lru_cache
from contextlib import contextmanager
from abc import ABC, abstractmethod
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# Scientific Computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import zoom, rotate, gaussian_filter, map_coordinates
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Deep Learning - PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    TQDMProgressBar, Callback
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_info

# Medical Imaging
import nibabel as nib

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# IMPROVED LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Production-ready Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N,) where each value is 0 <= targets[i] <= C-1
        """
        # Compute cross entropy with numerical stability
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Clamp probabilities for numerical stability
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedNTXentLoss(nn.Module):
    """
    Improved NT-Xent Loss with better numerical stability and scaling.
    
    Addresses issues with temperature scaling and normalization.
    """
    
    def __init__(self, temperature: float = 0.07, use_cosine_similarity: bool = True):
        """
        Initialize improved NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling similarities
            use_cosine_similarity: Whether to use cosine similarity (recommended)
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.eps = 1e-8
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute improved NT-Xent loss.
        
        Args:
            z_i: Projections from view 1 [B, projection_dim]
            z_j: Projections from view 2 [B, projection_dim]
        
        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize embeddings to unit sphere (important for contrastive learning)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings: [z_i; z_j] -> [2B, D]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix: [2B, 2B]
        if self.use_cosine_similarity:
            sim = torch.mm(z, z.t())  # Dot product on unit sphere is cosine similarity
        else:
            # Negative squared Euclidean distance
            sim = torch.cdist(z.unsqueeze(0), z.unsqueeze(0), p=2).squeeze(0)
            sim = -sim.pow(2)
        
        # Divide by temperature
        sim = sim / self.temperature
        
        # Create mask to exclude self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))
        
        # Create labels for positive pairs
        # For sample i (0 to B-1), positive is at i+B
        # For sample i (B to 2B-1), positive is at i-B
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(sim, labels, reduction='mean')
        
        return loss


# ============================================================================
# IMPROVED CONFIGURATION CLASSES
# ============================================================================

@dataclass
class PathConfig:
    """
    Path configuration for ADNI project.
    All paths are configurable for different systems.
    """
    # Base directory
    base_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env"))
    
    # Input data paths
    preprocessed_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/Processed_Final"))
    metadata_csv: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/ADNIMERGE.csv"))
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/SSL_MCI_Project/output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/SSL_MCI_Project/checkpoints"))
    figures_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/SSL_MCI_Project/figures"))
    logs_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/SSL_MCI_Project/logs"))
    cache_dir: Path = field(default_factory=lambda: Path("/home/24pcs001/ADNI_Work/new_env/SSL_MCI_Project/cache"))
    
    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        for attr_name in ['base_dir', 'preprocessed_dir', 'metadata_csv', 
                          'output_dir', 'checkpoint_dir', 'figures_dir', 
                          'logs_dir', 'cache_dir']:
            value = getattr(self, attr_name)
            if isinstance(value, str):
                setattr(self, attr_name, Path(value))
    
    def create_all(self) -> None:
        """Create all output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate that required input paths exist."""
        if not self.preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {self.preprocessed_dir}")
        if not self.metadata_csv.exists():
            print(f"Warning: Metadata CSV not found: {self.metadata_csv}")
        return True


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Target image dimensions (standard for 3D CNNs with MNI space)
    target_shape: Tuple[int, int, int] = (182, 218, 182)
    
    # Data splits (subject-level to prevent data leakage)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Classes for classification
    all_classes: Tuple[str, ...] = ("CN", "SMC", "EMCI", "MCI", "LMCI", "AD")
    binary_classes: Tuple[str, ...] = ("CN", "MCI")  # Focus on CN vs MCI detection
    
    # Cross-validation
    n_folds: int = 10
    
    # Data loading - Production settings (conservative for stability)
    num_workers: int = 2  # ✅ PRODUCTION FIX: Reduced from 8 to prevent deadlocks with 3D medical data
    pin_memory: bool = True
    prefetch_factor: int = 2  # ✅ PRODUCTION FIX: Reduced from 4 for stability
    persistent_workers: bool = False  # ✅ PRODUCTION FIX: Disabled to prevent worker deadlocks
    
    def __post_init__(self):
        """Validate configuration."""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"


@dataclass
class AugmentationConfig:
    """3D augmentation configuration for SimCLR."""
    # Spatial augmentations (more conservative for medical images)
    random_crop_scale: Tuple[float, float] = (0.85, 1.0)
    random_flip_prob: float = 0.5
    random_rotation_degrees: float = 5.0  # More conservative for medical images
    
    # Intensity augmentations (medical-specific)
    brightness_range: Tuple[float, float] = (0.95, 1.05)  # More conservative
    contrast_range: Tuple[float, float] = (0.95, 1.05)    # More conservative
    gaussian_noise_std: float = 0.005  # Reduced for medical images
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 0.5)  # Reduced for medical images
    elastic_deform_prob: float = 0.05
    elastic_alpha: float = 2.0
    elastic_sigma: float = 6.0
    
    # Probability of applying each augmentation type
    intensity_aug_prob: float = 0.8
    spatial_aug_prob: float = 0.8


@dataclass
class SimCLRConfig:
    """SimCLR pre-training configuration - DGX optimized."""
    # Model architecture
    backbone: str = "resnet50"  # ✅ Changed to resnet34  # Options: resnet18, resnet34, resnet50
    projection_dim: int = 128
    hidden_dim: int = 512
    
    # Training - optimized for multi-GPU (32GB per GPU)
    batch_size_per_gpu: int = 48
    epochs: int = 500
    base_lr: float = 0.3
    weight_decay: float = 1e-4
    temperature: float = 0.1  # Increased temperature for better separation
    
    # Optimizer
    optimizer: str = "lars"  # Options: lars, sgd, adamw
    momentum: float = 0.9
    warmup_epochs: int = 10
    
    # DGX-specific settings
    use_fp16: bool = True
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    sync_batchnorm: bool = True  # Important for multi-GPU


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration for MCI classification."""
    # Labeled data fractions to evaluate
    labeled_fractions: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
    
    # Training
    batch_size: int = 4  # ✅ Reduced for stability      # Reduced for stability
    epochs: int = 200  # ✅ Increased for better convergence        # Increased for better convergence
    lr: float = 0.0001  # ✅ Reduced for stable training       # Production ready learning rate
    weight_decay: float = 1e-4
    
    # Fine-tuning strategy
    freeze_backbone: bool = True
    freeze_epochs: int = 10
    
    # Regularization
    dropout: float = 0.2
    label_smoothing: float = 0.0  # Disabled for stable training
    
    # Early stopping
    patience: int = 25  # Increased patience
    min_delta: float = 0.001
    gradient_clip_val: float = 1.0


    
    # Class balancing - ✅ FIXED: Disabled weighted sampling to prevent double weighting
    focal_loss_alpha: float = 0.5  # ✅ Changed from 1.0 for better balance
    focal_loss_gamma: float = 2.0  # Focusing parameter
    cn_weight_boost: float = 1.0
    mci_weight_boost: float = 1.0  # Changed from 1.3
    use_weighted_sampling: bool = False  # ✅ DISABLED - CrossEntropyLoss already handles class weights
    class_weight_method: str = "balanced"  # Changed from "effective" for stability
    sampler_weight_method: str = "sqrt_inv"
    class_weight_clip_min: float = 0.5
    class_weight_clip_max: float = 2.0

@dataclass
class EvaluationConfig:
    """Evaluation and analysis configuration."""
    # Bootstrap confidence intervals
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    # Visualization
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    
    # Statistical tests
    alpha: float = 0.05


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    simclr: SimCLRConfig = field(default_factory=SimCLRConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment settings
    seed: int = 42
    experiment_name: str = "ssl_emci_v1"
    debug: bool = False
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        # Create directories
        self.paths.create_all()
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self._to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            data=DataConfig(**config_dict.get('data', {})),
            augmentation=AugmentationConfig(**config_dict.get('augmentation', {})),
            simclr=SimCLRConfig(**config_dict.get('simclr', {})),
            finetune=FineTuneConfig(**config_dict.get('finetune', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            seed=config_dict.get('seed', 42),
            experiment_name=config_dict.get('experiment_name', 'ssl_mci_experiment'),
            debug=config_dict.get('debug', False)
        )
    
    def _to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'paths': {k: str(v) for k, v in asdict(self.paths).items()},
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'simclr': asdict(self.simclr),
            'finetune': asdict(self.finetune),
            'evaluation': asdict(self.evaluation),
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'debug': self.debug
        }


# ============================================================================
# IMPROVED LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_dir: Path, experiment_name: str, debug: bool = False) -> logging.Logger:
    """
    Setup comprehensive logging system.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        debug: Enable debug level logging
    
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("SSL_MCI")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_format = ColoredFormatter(
        '%(asctime)s │ %(levelname)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (detailed logs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"{experiment_name}_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s │ %(levelname)s │ %(name)s │ %(funcName)s │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# IMPROVED UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CUDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Changed to False for reproducibility
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set numpy print options
    np.set_printoptions(precision=4, suppress=True)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary containing GPU details
    """
    info = {
        'available': torch.cuda.is_available(),
        'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }
    
    if info['available']:
        for i in range(info['count']):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            info['devices'].append({
                'id': i,
                'name': props.name,
                'memory_total_gb': round(mem_total, 2),
                'memory_allocated_gb': round(mem_allocated, 2),
                'memory_reserved_gb': round(mem_reserved, 2),
                'memory_free_gb': round(mem_total - mem_reserved, 2),
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            })
    
    return info


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_millions': round(total / 1e6, 2),
        'trainable_millions': round(trainable / 1e6, 2)
    }


def format_number(n: int) -> str:
    """Format number with commas for readability."""
    return f"{n:,}"


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the operation being timed
        logger: Optional logger instance
    """
    start = datetime.now()
    msg = f"⏳ Starting: {name}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    try:
        yield
    finally:
        elapsed = datetime.now() - start
        msg = f"✅ Completed: {name} in {elapsed}"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def convert_syncbn_to_bn(module: nn.Module) -> nn.Module:
    """
    Recursively convert all SyncBatchNorm layers to BatchNorm3d.

    CRITICAL when loading a checkpoint saved during multi-GPU DDP pre-training
    (which auto-converts BN -> SyncBN) and then fine-tuning on a single GPU.
    SyncBatchNorm requires a distributed process group and will crash / produce
    garbage on a single GPU.

    Copies weight, bias, running_mean, running_var so the learned statistics
    from pre-training are preserved.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm3d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum if child.momentum is not None else 0.1,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            )
            if child.affine:
                bn.weight = child.weight
                bn.bias = child.bias
            if child.track_running_stats:
                bn.running_mean = child.running_mean
                bn.running_var = child.running_var
                bn.num_batches_tracked = child.num_batches_tracked
            setattr(module, name, bn)
        else:
            convert_syncbn_to_bn(child)
    return module


def remap_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip DDP 'module.' prefix from every key in a state_dict.

    DDP wraps the model and prepends 'module.' to every parameter name.
    PyTorch Lightning may or may not strip this depending on version / strategy.
    This function strips it unconditionally so the dict loads into an un-wrapped model.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        while new_key.startswith("module."):
            new_key = new_key[len("module."):]
        new_sd[new_key] = v
    return new_sd


def validate_backbone_weights(backbone: nn.Module, logger: logging.Logger) -> bool:
    """
    Sanity-check that backbone weights are pre-trained, not freshly random.

    Kaiming-normal init for the first Conv3d (kernel=7, in_ch=1) gives
    std ≈ sqrt(2 / (1*7*7*7)) ≈ 0.076.  A trained network drifts away from
    that value.  We log the first 6 conv layers and warn if everything still
    looks like random init.

    Returns True if weights look pre-trained, False if they look random.
    """
    conv_stds = []
    for name, param in backbone.named_parameters():
        if 'conv' in name and 'weight' in name and param.dim() == 5:
            conv_stds.append(param.data.std().item())
            logger.info(f"   📏 {name}: mean={param.data.mean().item():.6f}, "
                        f"std={param.data.std().item():.6f}")
            if len(conv_stds) >= 6:
                break

    if not conv_stds:
        logger.warning("   ⚠️ No Conv3d layers found — cannot validate weights")
        return True

    # If the first layer std is still ≈ 0.076 (Kaiming for kernel=7, ch=1)
    # AND all subsequent layers are also at their Kaiming values, weights are random.
    # A trained network will have at least some layers that have drifted.
    first_std = conv_stds[0]
    if abs(first_std - 0.076) < 0.015 and len(conv_stds) > 2:
        # Double-check: are ALL stds suspiciously close to small Kaiming values?
        all_small = all(s < 0.15 for s in conv_stds)
        if all_small:
            logger.warning("   ⚠️ WARNING: Backbone weights look like random Kaiming init! "
                           "Pretrained weights may NOT have loaded correctly.")
            return False

    logger.info("   ✅ Backbone weights look pre-trained (not random init)")
    return True


# ============================================================================
# IMPROVED DATA STRUCTURES
# ============================================================================

@dataclass
class ScanInfo:
    """Information about a single MRI scan."""
    scan_id: str
    subject_id: str
    file_path: Path
    diagnosis: str
    age: Optional[float] = None
    sex: Optional[str] = None
    visit: Optional[str] = None
    phase: Optional[str] = None
    is_valid: bool = True
    file_size_mb: float = 0.0
    
    def __post_init__(self):
        """Compute file size if path exists."""
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        
        if self.file_path.exists():
            self.file_size_mb = self.file_path.stat().st_size / (1024 * 1024)


class DiagnosisGroup(Enum):
    """ADNI diagnosis groups."""
    CN = "CN"           # Cognitively Normal
    SMC = "SMC"         # Significant Memory Concern
    EMCI = "EMCI"       # Early Mild Cognitive Impairment
    MCI = "MCI"         # Mild Cognitive Impairment
    LMCI = "LMCI"       # Late Mild Cognitive Impairment
    AD = "AD"           # Alzheimer's Disease
    UNKNOWN = "UNKNOWN"


# ============================================================================
# IMPROVED DATA ORGANIZATION
# ============================================================================

class ADNIDataOrganizer:
    """
    Organize ADNI data with proper metadata linking.
    
    Features:
    - Discovers all preprocessed NIfTI files
    - Links with ADNIMERGE.csv metadata
    - Validates file integrity
    - Creates subject-level train/val/test splits
    - Handles class imbalance information
    """
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        """
        Initialize data organizer.
        
        Args:
            config: Experiment configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.scans: List[ScanInfo] = []
        self.metadata: Optional[pd.DataFrame] = None
    
    def run(self) -> Dict[str, List[ScanInfo]]:
        """
        Run complete data organization pipeline.
        
        Returns:
            Dictionary with train/val/test splits
        """
        self.logger.info("=" * 70)
        self.logger.info("📂 STEP 2: DATA ORGANIZATION & VALIDATION")
        self.logger.info("=" * 70)
        
        # Step 1: Discover preprocessed files
        nii_files = self._discover_files()
        
        # Step 2: Load ADNI metadata
        self.metadata = self._load_metadata()
        
        # Step 3: Link files with metadata
        self.scans = self._link_with_metadata(nii_files)
        
        # Step 4: Validate files
        valid_scans = self._validate_files(self.scans)
        
        # Step 5: Filter for target classes
        filtered_scans = self._filter_classes(valid_scans)
        
        # Step 6: Create subject-level splits
        splits = self._create_subject_level_splits(filtered_scans)
        
        # Step 7: Save split information
        self._save_splits(splits)
        
        # Step 8: Print summary
        self._print_summary(splits)
        
        return splits
    
    def _discover_files(self) -> List[Path]:
        """
        Find all preprocessed NIfTI files.
        
        Returns:
            List of paths to NIfTI files
        """
        preprocessed_dir = self.config.paths.preprocessed_dir
        
        if not preprocessed_dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
        
        # Find all .nii and .nii.gz files
        nii_files = []
        
        # Search patterns
        patterns = [
            "**/*.nii.gz",
            "**/*.nii",
            "*_preprocessed.nii.gz",
            "*_preprocessed.nii"
        ]
        
        for pattern in patterns:
            found = list(preprocessed_dir.glob(pattern))
            nii_files.extend(found)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in nii_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        
        nii_files = sorted(unique_files)
        
        self.logger.info(f"📁 Found {len(nii_files)} NIfTI files in {preprocessed_dir}")
        
        if len(nii_files) == 0:
            raise ValueError(f"No NIfTI files found in {preprocessed_dir}")
        
        return nii_files
    
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load ADNIMERGE.csv metadata.
        
        Returns:
            DataFrame with metadata
        """
        csv_path = self.config.paths.metadata_csv
        
        if not csv_path.exists():
            self.logger.warning(f"⚠️ Metadata file not found: {csv_path}")
            self.logger.info("Will extract information from filenames...")
            return pd.DataFrame()
        
        # Load CSV with low_memory=False to avoid dtype warnings
        df = pd.read_csv(csv_path, low_memory=False)
        
        self.logger.info(f"📋 Loaded ADNIMERGE with {len(df)} rows, {len(df.columns)} columns")
        
        # Log available columns (first 15)
        self.logger.info(f"   Key columns: {list(df.columns[:15])}")
        
        # Log diagnosis distribution if available
        dx_columns = ['DX', 'DX_bl', 'DIAGNOSIS', 'DX_GROUP']
        for col in dx_columns:
            if col in df.columns:
                self.logger.info(f"   {col} distribution:")
                for dx, count in df[col].value_counts().head(10).items():
                    self.logger.info(f"      {dx}: {count}")
                break
        
        return df
    
    def _link_with_metadata(self, nii_files: List[Path]) -> List[ScanInfo]:
        """
        Link preprocessed files with ADNI metadata.
        
        Args:
            nii_files: List of NIfTI file paths
        
        Returns:
            List of ScanInfo objects
        """
        self.logger.info("🔗 Linking files with metadata...")
        
        scans = []
        matched = 0
        unmatched = 0
        
        for nii_file in nii_files:
            # Extract information from filename
            filename = nii_file.stem
            # Remove common suffixes
            for suffix in ['_preprocessed', '_brain', '_stripped', '_norm']:
                filename = filename.replace(suffix, '')
            
            # Extract subject ID using ADNI pattern: XXX_S_XXXX
            subject_id = self._extract_subject_id(filename)
            
            # Default values
            diagnosis = "UNKNOWN"
            age = None
            sex = None
            visit = None
            
            # Try to match with metadata
            if not self.metadata.empty and subject_id:
                match_result = self._match_metadata(subject_id, filename)
                if match_result:
                    diagnosis = match_result.get('diagnosis', 'UNKNOWN')
                    age = match_result.get('age')
                    sex = match_result.get('sex')
                    visit = match_result.get('visit')
                    matched += 1
                else:
                    unmatched += 1
            else:
                unmatched += 1
            
            # Create ScanInfo object
            scan = ScanInfo(
                scan_id=filename,
                subject_id=subject_id or filename,
                file_path=nii_file,
                diagnosis=diagnosis,
                age=age,
                sex=sex,
                visit=visit
            )
            scans.append(scan)
        
        self.logger.info(f"   ✅ Matched with metadata: {matched}")
        self.logger.info(f"   ⚠️ Unmatched (using filename): {unmatched}")
        
        # Log diagnosis distribution after linking
        diagnosis_counts = Counter(s.diagnosis for s in scans)
        self.logger.info("📊 Diagnosis distribution after linking:")
        for diag, count in sorted(diagnosis_counts.items()):
            self.logger.info(f"   {diag}: {count}")
        
        return scans
    
    def _extract_subject_id(self, filename: str) -> Optional[str]:
        """
        Extract subject ID from filename.
        
        ADNI subject ID format: XXX_S_XXXX (e.g., 002_S_0295)
        
        Args:
            filename: Filename to parse
        
        Returns:
            Subject ID or None
        """
        # Pattern: 3 digits, underscore, S, underscore, 4 digits
        pattern = r'(\d{3}_S_\d{4})'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1)
        
        # Alternative patterns
        # Try to find any XXX_S_XXXX pattern with variable digits
        pattern2 = r'(\d+_S_\d+)'
        match2 = re.search(pattern2, filename)
        if match2:
            return match2.group(1)
        
        # Try splitting by underscore
        parts = filename.split('_')
        if len(parts) >= 3:
            for i in range(len(parts) - 2):
                if parts[i+1].upper() == 'S':
                    return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
        
        return None
    
    def _match_metadata(self, subject_id: str, filename: str) -> Optional[Dict]:
        """
        Match subject with ADNIMERGE metadata.
        
        Args:
            subject_id: Subject ID
            filename: Original filename
        
        Returns:
            Dictionary with matched metadata or None
        """
        if self.metadata.empty:
            return None
        
        # Try different matching strategies
        match = None
        
        # Strategy 1: Match by PTID (Patient ID)
        if 'PTID' in self.metadata.columns:
            match = self.metadata[self.metadata['PTID'] == subject_id]
        
        # Strategy 2: Match by RID (Roster ID - numeric part)
        if (match is None or len(match) == 0) and 'RID' in self.metadata.columns:
            # Extract numeric part from subject_id
            rid_parts = subject_id.split('_')
            if len(rid_parts) >= 3:
                rid = rid_parts[-1].lstrip('0') or '0'  # Remove leading zeros
                match = self.metadata[self.metadata['RID'].astype(str) == rid]
        
        # Strategy 3: Match by Subject column
        if (match is None or len(match) == 0) and 'Subject' in self.metadata.columns:
            match = self.metadata[self.metadata['Subject'] == subject_id]
        
        if match is None or len(match) == 0:
            return None
        
        # Get the first matching row (or most recent visit)
        if 'VISCODE' in match.columns:
            # Sort by visit code to get baseline or most recent
            match = match.sort_values('VISCODE')
        
        row = match.iloc[0]
        
        # Extract diagnosis
        diagnosis = "UNKNOWN"
        for dx_col in ['DX', 'DX_bl', 'DIAGNOSIS', 'DX_GROUP']:
            if dx_col in row and pd.notna(row[dx_col]):
                diagnosis = str(row[dx_col]).strip()
                diagnosis = self._standardize_diagnosis(diagnosis)
                break
        
        # Extract other fields
        result = {'diagnosis': diagnosis}
        
        if 'AGE' in row and pd.notna(row['AGE']):
            result['age'] = float(row['AGE'])
        
        if 'PTGENDER' in row and pd.notna(row['PTGENDER']):
            result['sex'] = str(row['PTGENDER'])
        elif 'Gender' in row and pd.notna(row['Gender']):
            result['sex'] = str(row['Gender'])
        
        if 'VISCODE' in row and pd.notna(row['VISCODE']):
            result['visit'] = str(row['VISCODE'])
        
        return result
    
    def _standardize_diagnosis(self, diagnosis: str) -> str:
        """
        Standardize ADNI diagnosis labels to binary CN/MCI.

        Maps various ADNI diagnosis codes:
        - CN, NL, NORMAL -> CN
        - MCI, EMCI, LMCI, SMC -> MCI
        - AD, DEMENTIA -> AD

        Args:
            diagnosis: Raw diagnosis string from ADNIMERGE

        Returns:
            Standardized label (CN/MCI/AD)

        Example:
            >>> _standardize_diagnosis("Early Mild Cognitive Impairment")
            'MCI'
        """
        diagnosis = diagnosis.upper().strip()
        
        # Cognitively Normal
        if diagnosis in ['CN', 'COGNITIVELY NORMAL', 'NL', 'NORMAL', 'NC', 'CONTROL']:
            return 'CN'
        
        # All MCI types -> MCI (including EMCI, LMCI, SMC as per preprocessing)
        if any(x in diagnosis for x in ['MCI', 'EMCI', 'LMCI', 'SMC', 'EARLY MCI', 'EARLY MILD', 'LATE MCI', 'MILD COGNITIVE', 'MEMORY CONCERN']):
            return 'MCI'
        
        # Alzheimer's Disease
        if any(x in diagnosis for x in ['AD', 'ALZHEIMER', 'DEMENTIA']):
            return 'AD'
        
        return diagnosis
    
    def _validate_files(self, scans: List[ScanInfo]) -> List[ScanInfo]:
        """
        Validate NIfTI files for integrity.
        
        Args:
            scans: List of ScanInfo objects
        
        Returns:
            List of valid ScanInfo objects
        """
        self.logger.info("🔍 Validating NIfTI files...")
        
        valid_scans = []
        invalid_count = 0
        invalid_reasons = Counter()
        
        for i, scan in enumerate(scans):
            if (i + 1) % 500 == 0:
                self.logger.info(f"   Validated {i + 1}/{len(scans)} files...")
            
            is_valid, reason = self._check_nifti_file(scan.file_path)
            
            if is_valid:
                scan.is_valid = True
                valid_scans.append(scan)
            else:
                scan.is_valid = False
                invalid_count += 1
                invalid_reasons[reason] += 1
                
                # Log first few invalid files
                if invalid_count <= 5:
                    self.logger.warning(f"   Invalid: {scan.scan_id} - {reason}")
        
        self.logger.info(f"✅ Valid files: {len(valid_scans)}")
        self.logger.info(f"❌ Invalid files: {invalid_count}")
        
        if invalid_reasons:
            self.logger.info("   Invalid file reasons:")
            for reason, count in invalid_reasons.most_common():
                self.logger.info(f"      {reason}: {count}")
        
        return valid_scans
    
    def _check_nifti_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Check if a NIfTI file is valid.
        
        Args:
            file_path: Path to NIfTI file
        
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check file exists
            if not file_path.exists():
                return False, "File not found"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size < 1000:  # Less than 1KB
                return False, "File too small"
            
            # Try to load the file
            img = nib.load(str(file_path))
            data = img.get_fdata()
            
            # Check for NaN values
            if np.isnan(data).any():
                return False, "Contains NaN values"
            
            # Check for Inf values
            if np.isinf(data).any():
                return False, "Contains Inf values"
            
            # Check if not all zeros
            if data.max() == 0:
                return False, "All zero values"
            
            # Check dimensions (should be 3D)
            if len(data.shape) != 3:
                return False, f"Not 3D: shape={data.shape}"
            
            # Check reasonable dimensions
            min_dim = min(data.shape)
            if min_dim < 10:
                return False, f"Dimension too small: {data.shape}"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Load error: {str(e)[:50]}"
    
    def _filter_classes(self, scans: List[ScanInfo]) -> List[ScanInfo]:
        """
        Filter scans for target classes (CN vs MCI for detection).
        
        Args:
            scans: List of ScanInfo objects
        
        Returns:
            Filtered list of ScanInfo objects
        """
        target_classes = set(self.config.data.binary_classes)
        
        # Count before filtering
        before_counts = Counter(s.diagnosis for s in scans)
        
        # Filter
        filtered = [s for s in scans if s.diagnosis in target_classes]
        
        # Count after filtering
        after_counts = Counter(s.diagnosis for s in filtered)
        
        self.logger.info(f"🎯 Filtering for classes: {target_classes}")
        self.logger.info(f"   Before: {len(scans)} scans")
        self.logger.info(f"   After: {len(filtered)} scans")
        
        for cls in target_classes:
            before = before_counts.get(cls, 0)
            after = after_counts.get(cls, 0)
            self.logger.info(f"   {cls}: {before} → {after}")
        
        if len(filtered) == 0:
            # If no matching classes, show what's available
            self.logger.warning("⚠️ No scans match target classes!")
            self.logger.warning(f"   Available diagnoses: {dict(before_counts)}")
            
            # Fall back to using all non-UNKNOWN scans
            filtered = [s for s in scans if s.diagnosis != "UNKNOWN"]
            self.logger.warning(f"   Falling back to all known diagnoses: {len(filtered)} scans")
        
        return filtered
    
    def _create_subject_level_splits(self, scans: List[ScanInfo]) -> Dict[str, List[ScanInfo]]:
        """
        Create train/val/test splits at SUBJECT level.
        
        This prevents data leakage from same subject appearing in different splits.
        
        Args:
            scans: List of ScanInfo objects
        
        Returns:
            Dictionary with train/val/test splits
        """
        self.logger.info("✂️ Creating subject-level data splits...")
        
        # Group scans by subject
        subjects: Dict[str, List[ScanInfo]] = defaultdict(list)
        for scan in scans:
            subjects[scan.subject_id].append(scan)
        
        # Get unique subjects and their diagnoses
        subject_ids = list(subjects.keys())
        
        # Assign diagnosis to each subject (most common across their scans)
        subject_diagnoses = []
        for subj_id in subject_ids:
            diagnoses = [s.diagnosis for s in subjects[subj_id]]
            most_common = Counter(diagnoses).most_common(1)[0][0]
            subject_diagnoses.append(most_common)
        
        self.logger.info(f"👥 Total subjects: {len(subject_ids)}")
        
        # Check we have enough subjects for stratified split
        diagnosis_counts = Counter(subject_diagnoses)
        min_count = min(diagnosis_counts.values())
        
        if min_count < 3:
            self.logger.warning(f"⚠️ Very few subjects per class (min={min_count}). Using simple random split.")
            stratify = None
        else:
            stratify = subject_diagnoses
        
        # Split: train vs (val + test)
        train_ratio = self.config.data.train_ratio
        val_ratio = self.config.data.val_ratio
        test_ratio = self.config.data.test_ratio
        
        try:
            train_subj, temp_subj, train_diag, temp_diag = train_test_split(
                subject_ids,
                subject_diagnoses,
                train_size=train_ratio,
                stratify=stratify,
                random_state=self.config.seed
            )
            
            # Split temp into val and test
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            
            val_subj, test_subj = train_test_split(
                temp_subj,
                train_size=val_ratio_adjusted,
                stratify=temp_diag if stratify else None,
                random_state=self.config.seed
            )
        except ValueError as e:
            self.logger.warning(f"⚠️ Stratified split failed: {e}")
            self.logger.warning("   Using random split instead...")
            
            # Random split
            random.seed(self.config.seed)
            random.shuffle(subject_ids)
            
            n_train = int(len(subject_ids) * train_ratio)
            n_val = int(len(subject_ids) * val_ratio)
            
            train_subj = subject_ids[:n_train]
            val_subj = subject_ids[n_train:n_train + n_val]
            test_subj = subject_ids[n_train + n_val:]
        
        # Collect scans for each split
        splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for subj_id in train_subj:
            splits['train'].extend(subjects[subj_id])
        for subj_id in val_subj:
            splits['val'].extend(subjects[subj_id])
        for subj_id in test_subj:
            splits['test'].extend(subjects[subj_id])
        
        return splits
    
    def _save_splits(self, splits: Dict[str, List[ScanInfo]]) -> None:
        """
        Save split information to files.
        
        Args:
            splits: Dictionary with data splits
        """
        output_dir = self.config.paths.output_dir
        
        # Save as CSV files
        for split_name, scans in splits.items():
            records = [{
                'scan_id': s.scan_id,
                'subject_id': s.subject_id,
                'file_path': str(s.file_path),
                'diagnosis': s.diagnosis,
                'age': s.age,
                'sex': s.sex,
                'visit': s.visit,
                'file_size_mb': s.file_size_mb
            } for s in scans]
            
            df = pd.DataFrame(records)
            csv_path = output_dir / f"{split_name}_split.csv"
            df.to_csv(csv_path, index=False)
        
        # Save as pickle for faster loading
        pickle_path = output_dir / "splits.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(splits, f)
        
        self.logger.info(f"💾 Saved splits to {output_dir}")
        self.logger.info(f"   CSV files: train_split.csv, val_split.csv, test_split.csv")
        self.logger.info(f"   Pickle: splits.pkl")
    
    def _print_summary(self, splits: Dict[str, List[ScanInfo]]) -> None:
        """
        Print detailed summary of data splits.
        
        Args:
            splits: Dictionary with data splits
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📊 DATA SPLIT SUMMARY")
        self.logger.info("=" * 60)
        
        total_scans = sum(len(s) for s in splits.values())
        total_subjects = len(set(
            scan.subject_id 
            for scans in splits.values() 
            for scan in scans
        ))
        
        self.logger.info(f"Total scans: {total_scans}")
        self.logger.info(f"Total subjects: {total_subjects}")
        self.logger.info("")
        
        for split_name in ['train', 'val', 'test']:
            scans = splits[split_name]
            n_scans = len(scans)
            n_subjects = len(set(s.subject_id for s in scans))
            pct = n_scans / total_scans * 100 if total_scans > 0 else 0
            
            self.logger.info(f"📁 {split_name.upper()}:")
            self.logger.info(f"   Scans: {n_scans} ({pct:.1f}%)")
            self.logger.info(f"   Subjects: {n_subjects}")
            
            # Per-class distribution
            diagnosis_counts = Counter(s.diagnosis for s in scans)
            for diag, count in sorted(diagnosis_counts.items()):
                diag_pct = count / n_scans * 100 if n_scans > 0 else 0
                self.logger.info(f"   {diag}: {count} ({diag_pct:.1f}%)")
            
            self.logger.info("")
        
        self.logger.info("=" * 60)


# ============================================================================
# IMPROVED 3D AUGMENTATIONS
# ============================================================================

class Augmentation3D:
    """
    Improved 3D augmentation pipeline for medical imaging.
    
    Uses more conservative and medical-appropriate augmentations.
    """
    
    def __init__(self, config: AugmentationConfig, training: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration
            training: Whether in training mode (applies augmentations)
        """
        self.config = config
        self.training = training
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to 3D volume.
        
        Args:
            volume: 3D numpy array (normalized to [0, 1])
        
        Returns:
            Augmented volume
        """
        if not self.training:
            return volume
        
        # Make a copy to avoid modifying original
        volume = volume.copy()
        
        # Intensity augmentations first (less disruptive)
        if random.random() < self.config.intensity_aug_prob:
            volume = self._apply_intensity_augmentations(volume)
        
        # Spatial augmentations second (preserves anatomical structure better)
        if random.random() < self.config.spatial_aug_prob:
            volume = self._apply_spatial_augmentations(volume)
        
        return volume
    
    def _apply_spatial_augmentations(self, volume: np.ndarray) -> np.ndarray:
        """Apply spatial augmentations."""
        # Random flip
        volume = self._random_flip(volume)
        
        # Random rotation (very conservative for medical images)
        volume = self._random_rotation(volume)
        
        # Random crop and resize (conservative)
        volume = self._random_crop_resize(volume)
        
        if random.random() < self.config.elastic_deform_prob:
            volume = self._elastic_deform(volume)

        return volume
    
    def _apply_intensity_augmentations(self, volume: np.ndarray) -> np.ndarray:
        """Apply intensity augmentations."""
        # Random brightness (very conservative)
        volume = self._random_brightness(volume)
        
        # Random contrast (very conservative)
        volume = self._random_contrast(volume)
        
        # Gaussian noise (very small)
        if random.random() < 0.3:  # Lower probability
            volume = self._add_gaussian_noise(volume)
        
        # Gaussian blur (very small)
        if random.random() < 0.1:  # Much lower probability
            volume = self._gaussian_blur(volume)
        
        return volume
    
    def _random_crop_resize(self, volume: np.ndarray) -> np.ndarray:
        """Random crop and resize back to original size."""
        scale = random.uniform(*self.config.random_crop_scale)
        
        if scale >= 0.99:
            return volume
        
        original_shape = volume.shape
        crop_shape = tuple(max(2, int(s * scale)) for s in original_shape)
        
        # Random crop position
        starts = [
            random.randint(0, max(0, o - c))
            for o, c in zip(original_shape, crop_shape)
        ]
        
        # Crop
        cropped = volume[
            starts[0]:starts[0] + crop_shape[0],
            starts[1]:starts[1] + crop_shape[1],
            starts[2]:starts[2] + crop_shape[2]
        ]
        
        if any(c <= 0 for c in cropped.shape):
            return volume

        # Resize back to original shape
        zoom_factors = [o / max(c, 1) for o, c in zip(original_shape, cropped.shape)]
        resized = zoom(cropped, zoom_factors, order=1, mode='nearest')
        
        return resized
    
    def _random_flip(self, volume: np.ndarray) -> np.ndarray:
        """Random flip along each axis."""
        for axis in range(3):
            if random.random() < self.config.random_flip_prob:
                volume = np.flip(volume, axis=axis)
        
        return np.ascontiguousarray(volume)
    
    def _random_rotation(self, volume: np.ndarray) -> np.ndarray:
        """Random rotation in a random plane."""
        max_angle = self.config.random_rotation_degrees
        
        if max_angle <= 0:
            return volume
        
        # Choose a random rotation plane
        axes_pairs = [(0, 1), (0, 2), (1, 2)]
        axes = random.choice(axes_pairs)
        
        # Random angle
        angle = random.uniform(-max_angle, max_angle)
        
        # Rotate
        rotated = rotate(
            volume, 
            angle, 
            axes=axes, 
            reshape=False, 
            order=1, 
            mode='nearest'
        )
        
        return rotated
    
    def _random_brightness(self, volume: np.ndarray) -> np.ndarray:
        """Random brightness adjustment."""
        factor = random.uniform(*self.config.brightness_range)
        adjusted = volume * factor
        return np.clip(adjusted, 0, 1)
    
    def _random_contrast(self, volume: np.ndarray) -> np.ndarray:
        """Random contrast adjustment."""
        factor = random.uniform(*self.config.contrast_range)
        mean = volume.mean()
        adjusted = (volume - mean) * factor + mean
        return np.clip(adjusted, 0, 1)
    
    def _add_gaussian_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(
            0, 
            self.config.gaussian_noise_std, 
            volume.shape
        ).astype(np.float32)
        noisy = volume + noise
        return np.clip(noisy, 0, 1)
    
    def _gaussian_blur(self, volume: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        sigma = random.uniform(*self.config.gaussian_blur_sigma)
        blurred = gaussian_filter(volume, sigma=sigma)
        return blurred

    def _elastic_deform(self, volume: np.ndarray) -> np.ndarray:
        alpha = self.config.elastic_alpha
        sigma = self.config.elastic_sigma
        if alpha <= 0 or sigma <= 0:
            return volume
        shape = volume.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='nearest') * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='nearest') * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='nearest') * alpha
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        indices = (
            (x + dx).reshape(-1),
            (y + dy).reshape(-1),
            (z + dz).reshape(-1)
        )
        deformed = map_coordinates(volume, indices, order=1, mode='nearest').reshape(shape)
        return deformed


class SimCLRAugmentation:
    """
    Improved SimCLR-specific augmentation that generates two different views.
    
    Uses more conservative augmentations for medical images.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize SimCLR augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.augment = Augmentation3D(config, training=True)
    
    def __call__(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two augmented views of the volume.
        
        Args:
            volume: Input 3D volume
        
        Returns:
            Tuple of two augmented views
        """
        view1 = self.augment(volume.copy())
        view2 = self.augment(volume.copy())
        return view1, view2


# ============================================================================
# IMPROVED DATASET CLASSES
# ============================================================================

class MRIClassificationDataset(Dataset):
    """
    Improved MRI Classification Dataset for fine-tuning.
    
    Features:
    - Better preprocessing pipeline
    - Proper normalization
    - Robust file loading
    """
    
    def __init__(self,
                 scans: List[ScanInfo],
                 target_shape: Tuple[int, int, int] = (182, 218, 182),
                 augmentation: Optional[Callable] = None,
                 label_encoder: Optional[LabelEncoder] = None,
                 cache_data: bool = False,
                 validate_on_load: bool = False):
        """
        Initialize dataset.
        
        Args:
            scans: List of ScanInfo objects
            target_shape: Target volume dimensions
            augmentation: Augmentation function
            label_encoder: Label encoder for classes
            cache_data: Cache loaded volumes in memory
            validate_on_load: Validate volumes during loading
        """
        self.scans = scans
        self.target_shape = target_shape
        self.augmentation = augmentation
        self.validate_on_load = validate_on_load
        
        # Setup label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            labels = [s.diagnosis for s in scans]
            self.label_encoder.fit(sorted(set(labels)))
        else:
            self.label_encoder = label_encoder
        
        # Class names
        self.class_names = list(self.label_encoder.classes_)
        self.num_classes = len(self.class_names)
        
        # Cache setup
        self.cache_data = cache_data
        self.cache: Dict[int, np.ndarray] = {}
    
    def __len__(self) -> int:
        return len(self.scans)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scan = self.scans[idx]
        
        try:
            # Load from cache or disk
            if self.cache_data and idx in self.cache:
                volume = self.cache[idx].copy()
            else:
                volume = self._load_and_preprocess(scan.file_path)
                if self.cache_data:
                    self.cache[idx] = volume.copy()
            
            # Apply augmentation
            if self.augmentation is not None:
                volume = self.augmentation(volume)
            
            # Convert to tensor [C, D, H, W]
            volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)
            
            # Get label
            label = self.label_encoder.transform([scan.diagnosis])[0]
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return {
                'volume': volume_tensor,
                'label': label_tensor,
                'scan_id': scan.scan_id,
                'subject_id': scan.subject_id,
                'diagnosis': scan.diagnosis
            }
            
        except Exception as e:
            # Return dummy sample on error
            logging.warning(f"Error loading {scan.scan_id}: {e}")
            return self._get_dummy_sample(idx)
    
    def _load_and_preprocess(self, file_path: Path) -> np.ndarray:
        """Load and preprocess NIfTI volume."""
        # Load NIfTI
        img = nib.load(str(file_path))
        volume = img.get_fdata().astype(np.float32)
        
        # Validate if required
        if self.validate_on_load:
            if np.isnan(volume).any():
                volume = np.nan_to_num(volume, nan=0.0)
            if np.isinf(volume).any():
                volume = np.nan_to_num(volume, posinf=1.0, neginf=0.0)
        
        # Resize to target shape if needed
        if volume.shape != self.target_shape:
            zoom_factors = [
                t / s for t, s in zip(self.target_shape, volume.shape)
            ]
            volume = zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Normalize to [0, 1] using robust statistics
        # Use percentile-based normalization to handle outliers
        p5 = np.percentile(volume, 5)
        p95 = np.percentile(volume, 95)
        if p95 - p5 > 1e-8:
            volume = np.clip((volume - p5) / (p95 - p5), 0, 1)
        else:
            volume = np.zeros(self.target_shape, dtype=np.float32)
        
        return volume.astype(np.float32)
    
    def _get_dummy_sample(self, idx: int) -> Dict[str, Any]:
        """Return dummy sample for failed loads."""
        scan = self.scans[idx]
        return {
            'volume': torch.zeros(1, *self.target_shape, dtype=torch.float32),
            'label': torch.tensor(0, dtype=torch.long),
            'scan_id': scan.scan_id,
            'subject_id': scan.subject_id,
            'diagnosis': scan.diagnosis
        }
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        labels = [s.diagnosis for s in self.scans]
        return self.label_encoder.transform(labels)
    
    def get_sample_weights(self, class_weights: Dict[str, float]) -> torch.Tensor:
        """
        Get per-sample weights for weighted sampling.
        
        Args:
            class_weights: Dictionary mapping class names to weights
        
        Returns:
            Tensor of sample weights
        """
        weights = [class_weights.get(s.diagnosis, 1.0) for s in self.scans]
        return torch.FloatTensor(weights)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        return dict(Counter(s.diagnosis for s in self.scans))


class SimCLRDataset(Dataset):
    """
    Improved SimCLR Dataset for self-supervised pre-training.
    
    Generates two augmented views of each sample for contrastive learning.
    """
    
    def __init__(self,
                 scans: List[ScanInfo],
                 target_shape: Tuple[int, int, int] = (182, 218, 182),
                 aug_config: Optional[AugmentationConfig] = None,
                 cache_data: bool = False):
        """
        Initialize SimCLR dataset.
        
        Args:
            scans: List of ScanInfo objects
            target_shape: Target volume dimensions
            aug_config: Augmentation configuration
            cache_data: Cache loaded volumes in memory
        """
        self.scans = scans
        self.target_shape = target_shape
        self.cache_data = cache_data
        self.cache: Dict[int, np.ndarray] = {}
        
        # Setup augmentation
        if aug_config is None:
            aug_config = AugmentationConfig()
        self.simclr_augment = SimCLRAugmentation(aug_config)
    
    def __len__(self) -> int:
        return len(self.scans)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scan = self.scans[idx]
        
        try:
            # Load from cache or disk
            if self.cache_data and idx in self.cache:
                volume = self.cache[idx].copy()
            else:
                volume = self._load_volume(scan.file_path)
                if self.cache_data:
                    self.cache[idx] = volume.copy()
            
            # Generate two augmented views
            view1, view2 = self.simclr_augment(volume)
            
            return {
                'view1': torch.from_numpy(view1).float().unsqueeze(0),
                'view2': torch.from_numpy(view2).float().unsqueeze(0),
                'scan_id': scan.scan_id,
                'idx': idx
            }
            
        except Exception as e:
            logging.warning(f"Error loading {scan.scan_id}: {e}")
            return self._get_dummy_sample(idx, scan.scan_id)
    
    def _load_volume(self, file_path: Path) -> np.ndarray:
        """Load and preprocess NIfTI volume."""
        img = nib.load(str(file_path))
        volume = img.get_fdata().astype(np.float32)
        
        # Handle NaN/Inf
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Resize if needed
        if volume.shape != self.target_shape:
            zoom_factors = [
                t / s for t, s in zip(self.target_shape, volume.shape)
            ]
            volume = zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Normalize to [0, 1] using robust statistics
        p5 = np.percentile(volume, 5)
        p95 = np.percentile(volume, 95)
        if p95 - p5 > 1e-8:
            volume = np.clip((volume - p5) / (p95 - p5), 0, 1)
        else:
            volume = np.zeros(self.target_shape, dtype=np.float32)
        
        return volume.astype(np.float32)
    
    def _get_dummy_sample(self, idx: int, scan_id: str) -> Dict[str, torch.Tensor]:
        """Return dummy sample for failed loads."""
        dummy = torch.zeros(1, *self.target_shape, dtype=torch.float32)
        return {
            'view1': dummy,
            'view2': dummy.clone(),
            'scan_id': scan_id,
            'idx': idx
        }


# ============================================================================
# IMPROVED CLASS WEIGHT CALCULATOR
# ============================================================================

class ClassWeightCalculator:
    """Calculate class weights for handling imbalanced datasets."""
    
    @staticmethod
    def compute_weights(labels: List[str], 
                       method: str = 'balanced') -> Dict[str, float]:
        """
        Compute class weights.
        
        Args:
            labels: List of string labels
            method: Weight computation method
                   - 'balanced': sklearn balanced formula
                   - 'sqrt_inv': Square root inverse frequency
                   - 'effective': Effective number of samples
        
        Returns:
            Dictionary mapping class names to weights
        """
        counts = Counter(labels)
        n_samples = len(labels)
        n_classes = len(counts)
        
        if method == 'balanced':
            # sklearn formula: n_samples / (n_classes * n_samples_per_class)
            weights = {
                cls: n_samples / (n_classes * count)
                for cls, count in counts.items()
            }
        
        elif method == 'sqrt_inv':
            # Square root inverse frequency
            weights = {
                cls: np.sqrt(n_samples / count)
                for cls, count in counts.items()
            }
        
        elif method == 'effective':
            # Effective number of samples (for long-tailed distributions)
            beta = 0.999
            weights = {
                cls: (1 - beta) / (1 - beta ** count)
                for cls, count in counts.items()
            }
        
        else:
            # Equal weights
            weights = {cls: 1.0 for cls in counts}
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v * n_classes / total for k, v in weights.items()}
        
        return weights
    
    @staticmethod
    def get_sample_weights(labels: List[str],
                          class_weights: Dict[str, float]) -> np.ndarray:
        """Get per-sample weights."""
        return np.array([class_weights[label] for label in labels])


# ============================================================================
# IMPROVED 3D CNN ARCHITECTURES
# ============================================================================

class ConvBlock3D(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 bias: bool = False):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """
    Improved 3D Residual Block for ResNet-18/34.
    
    Added improvements:
    - Better initialization
    - Potential for stochastic depth
    """
    
    expansion = 1
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock3D(nn.Module):
    """
    3D Bottleneck Block for ResNet-50/101/152.
    
    Architecture:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> + -> ReLU
        |___________________________________________________________________________|
    """
    
    expansion = 4
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        # 1x1 reduce
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 1x1 expand
        self.conv3 = nn.Conv3d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
# IMPROVED 3D RESNET MAIN ARCHITECTURE
# ============================================================================

class ResNet3D(nn.Module):
    """
    Improved 3D ResNet for volumetric medical imaging.
    
    Optimized for brain MRI analysis with:
    - Better initialization
    - More stable architecture
    - Proper feature dimension handling
    """
    
    def __init__(self,
                 block: type,
                 layers: List[int],
                 in_channels: int = 1,
                 num_classes: int = 2,
                 zero_init_residual: bool = True,
                 groups: int = 1,
                 width_per_group: int = 64):
        """
        Initialize ResNet3D.
        
        Args:
            block: Block type (ResidualBlock3D or BottleneckBlock3D)
            layers: Number of blocks in each layer
            in_channels: Number of input channels (1 for grayscale MRI)
            num_classes: Number of output classes
            zero_init_residual: Zero-initialize last BN in each residual branch
            groups: Number of groups for grouped convolution
            width_per_group: Width of each group
        """
        super().__init__()
        
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution layer
        self.conv1 = nn.Conv3d(
            in_channels, self.in_planes, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Feature dimension (for SSL)
        self.feature_dim = 512 * block.expansion
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(self,
                    block: type,
                    planes: int,
                    blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """
        Create a residual layer.
        
        Args:
            block: Block type
            planes: Number of planes (channels)
            blocks: Number of blocks
            stride: Stride for first block
        
        Returns:
            Sequential layer
        """
        downsample = None
        
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock3D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (without classifier).
        
        Args:
            x: Input tensor [B, C, D, H, W]
        
        Returns:
            Feature tensor [B, feature_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, D, H, W]
        
        Returns:
            Output logits [B, num_classes]
        """
        features = self.forward_features(x)
        return self.fc(features)


# ============================================================================
# RESNET FACTORY FUNCTIONS
# ============================================================================

def resnet3d_18(in_channels: int = 1, num_classes: int = 2, **kwargs) -> ResNet3D:
    """
    ResNet-18 3D.
    
    Total params: ~33M (with 1 input channel)
    """
    return ResNet3D(
        ResidualBlock3D, [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs
    )


def resnet3d_34(in_channels: int = 1, num_classes: int = 2, **kwargs) -> ResNet3D:
    """
    ResNet-34 3D.
    
    Total params: ~63M (with 1 input channel)
    """
    return ResNet3D(
        ResidualBlock3D, [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs
    )


def resnet3d_50(in_channels: int = 1, num_classes: int = 2, **kwargs) -> ResNet3D:
    """
    ResNet-50 3D.
    
    Total params: ~46M (with 1 input channel)
    """
    return ResNet3D(
        BottleneckBlock3D, [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs
    )


def get_backbone(name: str, in_channels: int = 1) -> ResNet3D:
    """
    Get backbone by name.
    
    Args:
        name: Backbone name (resnet18, resnet34, resnet50)
        in_channels: Number of input channels
    
    Returns:
        ResNet3D model
    """
    backbones = {
        'resnet18': resnet3d_18,
        'resnet34': resnet3d_34,
        'resnet50': resnet3d_50,
    }
    
    name = name.lower()
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(backbones.keys())}")
    
    return backbones[name](in_channels=in_channels)


# ============================================================================
# IMPROVED SIMCLR COMPONENTS
# ============================================================================

class ProjectionHead(nn.Module):
    """
    MLP Projection Head for SimCLR.
    
    Projects features to a lower-dimensional space for contrastive learning.
    Architecture: Linear -> BN -> ReLU -> Linear
    """
    
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 512,
                 out_dim: int = 128):
        """
        Initialize projection head.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (projection space)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize projection head weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, in_dim]
        
        Returns:
            Projected features [B, out_dim]
        """
        return self.net(x)


# ============================================================================
# IMPROVED LARS OPTIMIZER
# ============================================================================

class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    LARS is designed for large batch training and is commonly used
    in self-supervised learning.
    
    Reference:
        You et al., "Large Batch Training of Convolutional Networks"
        https://arxiv.org/abs/1708.03888
    
    Usage:
        optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
    """
    
    def __init__(self,
                 params,
                 lr: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 trust_coefficient: float = 0.001,
                 eps: float = 1e-8,
                 exclude_bias_and_bn: bool = True):
        """
        Initialize LARS optimizer.
        
        Args:
            params: Model parameters
            lr: Base learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 regularization)
            trust_coefficient: Trust coefficient for layer adaptation
            eps: Small constant for numerical stability
            exclude_bias_and_bn: Exclude bias and BatchNorm from LARS
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps
        )
        
        super().__init__(params, defaults)
        self.exclude_bias_and_bn = exclude_bias_and_bn
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Compute LARS trust ratio
                param_norm = torch.norm(p)
                grad_norm = torch.norm(grad)
                
                # LARS scaling factor
                trust_ratio = 1.0
                if param_norm > 0 and grad_norm > 0:
                    trust_ratio = (
                        group['trust_coefficient'] * param_norm /
                        (grad_norm + group['eps'])
                    )
                
                # Scale gradient by trust ratio
                scaled_grad = grad * trust_ratio
                
                # Apply momentum
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(scaled_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(scaled_grad)
                    
                    scaled_grad = buf
                
                # Update parameters
                p.add_(scaled_grad, alpha=-group['lr'])
        
        return loss


class LARSWrapper:
    """
    Wrapper for creating LARS optimizer with proper parameter groups.
    
    Handles exclusion of bias and BatchNorm parameters from LARS.
    """
    
    @staticmethod
    def get_parameter_groups(model: nn.Module,
                             weight_decay: float = 1e-4,
                             lr: float = 0.1) -> List[Dict]:
        """
        Get parameter groups with proper LARS handling.
        
        Bias and BatchNorm parameters are excluded from weight decay
        as per common practice.
        
        Args:
            model: PyTorch model
            weight_decay: Weight decay value
            lr: Learning rate
        
        Returns:
            List of parameter group dictionaries
        """
        # Separate parameters
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Exclude bias and BatchNorm from weight decay
            if name.endswith('.bias'):
                no_decay_params.append(param)
            elif 'bn' in name.lower() or 'batchnorm' in name.lower():
                no_decay_params.append(param)
            elif param.ndim == 1:  # Also catches some norm layers
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr}
        ]


# ============================================================================
# IMPROVED SIMCLR LIGHTNING MODULE
# ============================================================================

class SimCLRModule(LightningModule):
    """
    Improved SimCLR Self-Supervised Learning Module.
    
    Features:
    - 3D ResNet backbone
    - Projection head
    - Improved NT-Xent contrastive loss
    - LARS optimizer with warmup
    - Multi-GPU support with DDP
    - Mixed precision training
    - Better numerical stability
    """
    
    def __init__(self,
                 config: SimCLRConfig,
                 num_gpus: int = 1,
                 use_gradient_checkpointing: bool = False):
        """
        Initialize SimCLR module.
        
        Args:
            config: SimCLR configuration
            num_gpus: Number of GPUs for training
            use_gradient_checkpointing: Use gradient checkpointing for memory
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['use_gradient_checkpointing'])
        
        self.config = config
        self.num_gpus = max(1, num_gpus)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Build backbone
        self.backbone = self._build_backbone()
        
        # Projection head
        self.projection = ProjectionHead(
            in_dim=self.backbone.feature_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.projection_dim
        )
        
        # Loss function
        self.criterion = ImprovedNTXentLoss(temperature=config.temperature)
        
        # Training tracking
        self.train_losses: List[torch.Tensor] = []
        
        # Effective batch size for LR scaling
        self.effective_batch_size = config.batch_size_per_gpu * self.num_gpus
    
    def _build_backbone(self) -> ResNet3D:
        """Build and configure backbone network."""
        backbone_name = self.config.backbone.lower()
        
        if backbone_name == 'resnet18':
            backbone = resnet3d_18(in_channels=1)
        elif backbone_name == 'resnet34':
            backbone = resnet3d_34(in_channels=1)
        elif backbone_name == 'resnet50':
            backbone = resnet3d_50(in_channels=1)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Remove classifier (we use projection head instead)
        backbone.fc = nn.Identity()
        
        return backbone
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (for inference/fine-tuning).
        
        Args:
            x: Input tensor [B, 1, D, H, W]
        
        Returns:
            Features [B, feature_dim]
        """
        return self.backbone.forward_features(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for SimCLR.
        
        Args:
            batch: Dictionary containing 'view1' and 'view2'
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        view1 = batch['view1']
        view2 = batch['view2']
        
        # Extract features
        h1 = self.backbone.forward_features(view1)
        h2 = self.backbone.forward_features(view2)
        
        # Project to contrastive space
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        # Compute NT-Xent loss
        loss = self.criterion(z1, z2)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True, batch_size=view1.size(0))
        
        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, prog_bar=True)
        
        # Track for epoch-level logging
        self.train_losses.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        if self.train_losses:
            avg_loss = torch.stack(self.train_losses).mean()
            self.log('train_loss_epoch', avg_loss, sync_dist=True)
            self.train_losses.clear()
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            self.log('gpu_memory_peak_gb', memory_gb)
            torch.cuda.reset_peak_memory_stats()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Linear scaling rule for learning rate
        base_lr = self.config.base_lr
        lr = base_lr * self.effective_batch_size / 256
        
        # Get parameter groups
        param_groups = LARSWrapper.get_parameter_groups(
            self, weight_decay=self.config.weight_decay, lr=lr
        )
        
        # Create optimizer
        optimizer_name = self.config.optimizer.lower()
        
        if optimizer_name == 'lars':
            optimizer = LARS(
                param_groups,
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                trust_coefficient=0.001
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=True
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr * 0.1,  # AdamW typically needs lower LR
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler with warmup
        total_epochs = self.config.epochs
        warmup_epochs = self.config.warmup_epochs
        
        def lr_lambda(epoch: int) -> float:
            """Learning rate schedule: warmup + cosine decay."""
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def get_backbone(self) -> nn.Module:
        """Get the backbone for fine-tuning."""
        return self.backbone


# ============================================================================
# IMPROVED MCI CLASSIFIER MODULE
# ============================================================================

class MCIClassifier(LightningModule):
    """
    Improved MCI Classification Module for Fine-tuning.
    
    Features:
    - Uses pre-trained backbone from SimCLR
    - Supports frozen/unfrozen backbone training
    - Class-weighted loss for imbalanced data
    - Comprehensive metrics logging
    - Better regularization and training procedures
    """
    
    def __init__(self,
                 backbone: nn.Module,
                 num_classes: int = 2,
                 config: Optional[FineTuneConfig] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 class_names: Optional[List[str]] = None,
                 freeze_backbone: bool = False):  # Changed to False for better adaptation
        """
        Initialize classifier.
        
        Args:
            backbone: Pre-trained backbone network
            num_classes: Number of output classes
            config: Fine-tuning configuration
            class_weights: Weights for each class (for imbalanced data)
            class_names: Names of classes for logging
            freeze_backbone: Whether to freeze backbone initially
        """
        super().__init__()
        
        # Save hyperparameters (exclude non-serializable)
        self.save_hyperparameters(ignore=['backbone', 'class_weights'])
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.config = config or FineTuneConfig()
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self._logger = logging.getLogger("SSL_MCI")
        self.best_threshold = 0.5
        
        # ✅ FIX 3: Register class weights WITHOUT boosting
        # Boosting causes loss explosion when combined with WeightedRandomSampler
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Get feature dimension
        if hasattr(backbone, 'feature_dim'):
            feature_dim = backbone.feature_dim
        else:
            # ✅ FIX 4: Auto-detect feature dim by test forward pass
            try:
                dummy = torch.randn(1, 1, 96, 112, 96)
                with torch.no_grad():
                    dummy_out = backbone.forward_features(dummy)
                feature_dim = dummy_out.shape[1]
                print(f"✅ Auto-detected feature_dim = {feature_dim}")
            except:
                feature_dim = 2048  # ResNet50 default (safe fallback)
                print(f"⚠️ Using default feature_dim = {feature_dim}")
        
        # Classification head with improved architecture
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.config.dropout),
            nn.Linear(feature_dim, 512),  # Increased size
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # Added BN
            nn.Dropout(p=self.config.dropout),
            nn.Linear(512, 256),  # Additional layer
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=self.config.dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
        # Loss function - Using standard CrossEntropyLoss with class weights and label smoothing
        # Focal Loss was causing training instability (loss ~4.0+)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.config.label_smoothing if hasattr(self.config, 'label_smoothing') else 0.0,
            reduction='mean'
        )
        
        # Freeze backbone if requested
        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()
        
        # Metrics storage
        self.train_outputs: List[Dict] = []
        self.val_outputs: List[Dict] = []
        self.test_outputs: List[Dict] = []
        
        # Store test results for visualization
        self.test_results: Optional[Dict] = None
    
    def _initialize_classifier(self) -> None:
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
    
    def is_backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._backbone_frozen
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 1, D, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        features = self.backbone.forward_features(x)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch['volume']
        y = batch['label']
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end
        self.train_outputs.append({
            'loss': loss.detach().cpu(),
            'preds': preds.detach().cpu(),
            'labels': y.detach().cpu()
        })
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Compute epoch-level training metrics."""
        if not self.train_outputs:
            return
        
        all_preds = torch.cat([o['preds'] for o in self.train_outputs])
        all_labels = torch.cat([o['labels'] for o in self.train_outputs])
        all_losses = torch.stack([o['loss'] for o in self.train_outputs])
        avg_loss = all_losses.mean()
        self.log('train_loss_epoch', avg_loss, prog_bar=True, sync_dist=True)
        
        # Per-class accuracy
        for i, cls_name in enumerate(self.class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == all_labels[mask]).float().mean()
                self.log(f'train_acc_{cls_name}', cls_acc)
        
        self.train_outputs.clear()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        x = batch['volume']
        y = batch['label']
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.val_outputs.append({
            'loss': loss.detach(),
            'probs': probs.detach(),
            'preds': preds.detach(),
            'labels': y.detach()
        })
    
    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics."""
        if not self.val_outputs:
            return
        
        # Check if both classes are present in validation set
        all_labels = torch.cat([o['labels'] for o in self.val_outputs])
        unique_labels = torch.unique(all_labels)
        if len(unique_labels) < 2:
            # Log warning and return neutral metrics
            self._logger.warning(f"⚠️ Validation set contains only class {unique_labels.tolist()} - skipping validation metrics")
            self.log('val_loss', 1.0, prog_bar=True, sync_dist=True)
            self.log('val_acc', 0.5, prog_bar=True, sync_dist=True)
            self.log('val_auc', 0.5, prog_bar=True, sync_dist=True)
            self.log('val_f1', 0.0, prog_bar=True, sync_dist=True)
            self.log('val_sensitivity', 0.0, sync_dist=True)
            self.log('val_specificity', 0.0, sync_dist=True)
            self.log('val_ppv', 0.0, sync_dist=True)
            self.log('val_npv', 0.0, sync_dist=True)
            self.log('val_balanced_accuracy', 0.0, prog_bar=True, sync_dist=True)
            self.log('val_min_sens_spec', 0.0, prog_bar=True, sync_dist=True)
            self.log('val_best_threshold', 0.5, sync_dist=True)
            self.best_threshold = 0.5
            self.val_outputs.clear()
            return
        
        # Aggregate
        all_probs = torch.cat([o['probs'] for o in self.val_outputs])
        all_preds = torch.cat([o['preds'] for o in self.val_outputs])
        all_labels = torch.cat([o['labels'] for o in self.val_outputs])
        all_losses = torch.stack([o['loss'] for o in self.val_outputs])
        
        # Basic metrics
        avg_loss = all_losses.mean()
        acc = (all_preds == all_labels).float().mean()
        
        # Convert to numpy for sklearn metrics
        probs_np = all_probs.cpu().numpy()
        preds_np = all_preds.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        
        # AUC
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(labels_np, probs_np[:, 1])
            else:
                auc = roc_auc_score(labels_np, probs_np, multi_class='ovr')
        except ValueError:
            auc = 0.5
        
        # F1 Score
        f1 = f1_score(labels_np, preds_np, average='weighted')
        
        # Logging
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_auc', auc, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, prog_bar=True, sync_dist=True)
        
        # Binary classification specific metrics
        if self.num_classes == 2:
            try:
                fpr, tpr, thresholds = roc_curve(labels_np, probs_np[:, 1])
                tnr = 1 - fpr
                min_scores = np.minimum(tpr, tnr)
                best_idx = int(np.argmax(min_scores)) if min_scores.size > 0 else 0
                self.best_threshold = float(thresholds[best_idx]) if thresholds.size > 0 else 0.5
                self.log('val_min_sens_spec', float(min_scores[best_idx]), prog_bar=True, sync_dist=True)
                self.log('val_best_threshold', self.best_threshold, sync_dist=True)
            except Exception:
                self.best_threshold = 0.5
                self.log('val_min_sens_spec', 0.0, prog_bar=True, sync_dist=True)
                self.log('val_best_threshold', 0.5, sync_dist=True)
            
            preds_threshold = (probs_np[:, 1] >= self.best_threshold).astype(np.int64)
            cm = confusion_matrix(labels_np, preds_threshold)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                self.log('val_sensitivity', sensitivity, sync_dist=True)
                self.log('val_specificity', specificity, sync_dist=True)
                self.log('val_ppv', ppv, sync_dist=True)
                self.log('val_npv', npv, sync_dist=True)
                self.log('val_balanced_accuracy', (sensitivity + specificity) / 2, prog_bar=True, sync_dist=True)
        
        self.val_outputs.clear()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        x = batch['volume']
        y = batch['label']
        
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.test_outputs.append({
            'probs': probs.detach(),
            'preds': preds.detach(),
            'labels': y.detach(),
            'scan_ids': batch['scan_id']
        })
    
    def on_test_epoch_end(self) -> Dict[str, float]:
        """Compute comprehensive test metrics."""
        if not self.test_outputs:
            return {}
        
        # Check if both classes are present in test set
        all_labels = torch.cat([o['labels'] for o in self.test_outputs])
        unique_labels = torch.unique(all_labels)
        if len(unique_labels) < 2:
            self._logger.warning(f"⚠️ Test set contains only class {unique_labels.tolist()} - returning neutral metrics")
            return {
                'accuracy': 0.5,
                'auc': 0.5,
                'f1': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'balanced_accuracy': 0.25,
                'ppv': 0.0,
                'npv': 0.0
            }
        
        # Aggregate
        all_probs = torch.cat([o['probs'] for o in self.test_outputs]).cpu().numpy()
        all_preds = torch.cat([o['preds'] for o in self.test_outputs]).cpu().numpy()
        all_labels = torch.cat([o['labels'] for o in self.test_outputs]).cpu().numpy()
        
        # Collect scan IDs
        all_scan_ids = []
        for o in self.test_outputs:
            all_scan_ids.extend(o['scan_ids'])
        
        if self.num_classes == 2:
            all_preds = (all_probs[:, 1] >= self.best_threshold).astype(np.int64)
        
        metrics = self._compute_all_metrics(all_labels, all_preds, all_probs)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f'test_{name}', value, sync_dist=True)
        
        # Store for visualization
        self.test_results = {
            'probs': all_probs,
            'preds': all_preds,
            'labels': all_labels,
            'scan_ids': all_scan_ids,
            'metrics': metrics
        }
        
        self.test_outputs.clear()
        
        return metrics
    
    def _compute_all_metrics(self,
                              labels: np.ndarray,
                              preds: np.ndarray,
                              probs: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(labels, preds, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # AUC
        try:
            if self.num_classes == 2:
                metrics['auc'] = roc_auc_score(labels, probs[:, 1])
            else:
                metrics['auc'] = roc_auc_score(labels, probs, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.5
        
        # Binary-specific metrics
        if self.num_classes == 2:
            cm = confusion_matrix(labels, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Use different learning rates for backbone and classifier
        if self._backbone_frozen:
            # Only train classifier
            params = self.classifier.parameters()
            lr = self.config.lr
        else:
            # Different LR for backbone and classifier
            params = [
                {'params': self.backbone.parameters(), 'lr': self.config.lr * 0.1},
                {'params': self.classifier.parameters(), 'lr': self.config.lr}
            ]
            lr = self.config.lr
        
        optimizer = torch.optim.AdamW(
            params if isinstance(params, list) else [{'params': params, 'lr': lr}],
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine Annealing scheduler for better convergence
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=1,  # Don't multiply period
            eta_min=1e-7
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',  # Monitor F1 score instead of loss
                'interval': 'epoch',
                'frequency': 1
            }
        }


# ============================================================================
# IMPROVED TRAINING CALLBACKS
# ============================================================================

class UnfreezeBackboneCallback(Callback):
    """
    Callback to unfreeze backbone after N epochs AND re-configure the optimizer.

    WHY re-configure the optimizer?
    PyTorch Lightning calls configure_optimizers() exactly once at the start of
    training.  When freeze_backbone=True that call adds ONLY the classifier params
    to the optimizer.  Simply setting requires_grad=True on the backbone later does
    NOT retroactively add those params to the existing optimizer -- they stay frozen
    from the optimiser's perspective even though gradients now flow through them.

    WHY reset the GradScaler?
    When precision='16-mixed' Lightning manages a GradScaler internally.
    That scaler is calibrated for the OLD optimizer (head-only).  If we swap in a
    new optimizer without resetting the scaler, the old scale factor (potentially
    2^16 = 65536) is applied to the backbone gradients on the very first step.
    This causes immediate loss explosion (loss jumps to 10+) and the model never
    recovers.  We reset the scaler to scale=1 so it ramps up gently from scratch.

    This callback therefore:
      1. Calls pl_module.unfreeze_backbone()           (sets requires_grad=True)
      2. Creates a NEW AdamW optimizer with two param-groups:
           - backbone params  @ lr * 0.01  (very conservative to preserve features)
           - classifier params @ lr * 0.1  (also reduced to keep stable)
      3. Replaces trainer.optimizers with the new optimizer
      4. Resets the AMP GradScaler so it does not blow up the new optimizer
      5. Replaces the LR scheduler with a fresh CosineAnnealingWarmRestarts
    """

    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.unfrozen:
            return
        if trainer.current_epoch < self.unfreeze_epoch:
            return

        # --- Step 1: unfreeze backbone weights ---
        if hasattr(pl_module, 'unfreeze_backbone'):
            pl_module.unfreeze_backbone()
        self.unfrozen = True

        # --- Step 2: build new optimizer with VERY conservative LRs ---
        # After 10 epochs of head-only training the head has converged to a
        # certain feature expectation.  Hitting the backbone with a large LR
        # destroys those features instantly.  Use lr*0.01 for backbone.
        config = pl_module.config  # FineTuneConfig
        backbone_lr = config.lr * 0.01   # e.g. 1e-4 * 0.01 = 1e-6
        head_lr     = config.lr * 0.1    # e.g. 1e-4 * 0.1  = 1e-5
        new_optimizer = torch.optim.AdamW(
            [
                {'params': pl_module.backbone.parameters(), 'lr': backbone_lr},
                {'params': pl_module.classifier.parameters(), 'lr': head_lr}
            ],
            weight_decay=config.weight_decay
        )

        # --- Step 3: RESET the AMP GradScaler (critical for 16-mixed) ---
        # Lightning stores the scaler at trainer.scaler (PL 1.x / 2.x) or
        # inside the strategy.  We handle both paths.
        try:
            if hasattr(trainer, 'scaler') and trainer.scaler is not None:
                trainer.scaler._scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
                trainer.scaler._growth_tracker = torch.tensor(0.0, dtype=torch.float32, device='cuda')
                print(f"   🔄 GradScaler reset (scale=1.0)")
            elif hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'scaler'):
                scaler = trainer.strategy.scaler
                if scaler is not None:
                    scaler._scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
                    scaler._growth_tracker = torch.tensor(0.0, dtype=torch.float32, device='cuda')
                    print(f"   🔄 GradScaler reset via strategy (scale=1.0)")
        except Exception as e:
            print(f"   ⚠️  Could not reset GradScaler: {e}")
            print(f"        (training will still work if precision != 16-mixed)")

        # --- Step 4: fresh scheduler covering the remaining epochs ---
        remaining_epochs = trainer.max_epochs - trainer.current_epoch
        T_0 = min(50, max(10, remaining_epochs // 2))
        new_scheduler = CosineAnnealingWarmRestarts(
            new_optimizer,
            T_0=T_0,
            T_mult=1,
            eta_min=1e-8
        )

        # --- Step 5: inject into trainer ---
        trainer.optimizers = [new_optimizer]
        trainer.lr_schedulers = [
            {
                'scheduler': new_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_f1',
                'name': None,
                'reduce_on_plateau': False
            }
        ]

        print(f"\n🔓 Epoch {trainer.current_epoch}: backbone UNFROZEN + optimizer re-configured")
        print(f"   backbone lr = {backbone_lr:.2e},  head lr = {head_lr:.2e}")
        pl_module.log('backbone_unfrozen', 1.0)


class GPUMemoryMonitorCallback(Callback):
    """
    Callback to monitor and log GPU memory usage.
    
    Useful for debugging memory issues and optimizing batch sizes.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        """
        Initialize callback.
        
        Args:
            log_every_n_steps: Logging frequency
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch, batch_idx) -> None:
        """Log GPU memory after batch."""
        if batch_idx % self.log_every_n_steps != 0:
            return
        
        if not torch.cuda.is_available():
            return
        
        # Log memory for each GPU
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            pl_module.log(f'gpu{i}_memory_allocated_gb', allocated, prog_bar=False)
            pl_module.log(f'gpu{i}_memory_reserved_gb', reserved, prog_bar=False)


class GradientMonitorCallback(Callback):
    """
    Callback to monitor gradient statistics.
    
    Helps detect vanishing or exploding gradients.
    """
    
    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule,
                                 optimizer) -> None:
        """Log gradient norm before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        total_norm = 0.0
        num_params = 0
        
        for param in pl_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                num_params += 1
        
        total_norm = total_norm ** 0.5
        
        pl_module.log('gradient_norm', total_norm, prog_bar=False)
        
        # Warn if gradients are problematic
        if total_norm > 100:
            print(f"\n⚠️ Large gradient norm: {total_norm:.2f}")
        elif total_norm < 1e-7 and num_params > 0:
            print(f"\n⚠️ Vanishing gradients: {total_norm:.2e}")


# ============================================================================
# IMPROVED DATA MODULE
# ============================================================================

class ADNIDataModule(LightningDataModule):
    """
    Improved PyTorch Lightning DataModule for ADNI dataset.
    
    Features:
    - Automatic data loading from splits
    - Better weighted sampling for imbalanced classes
    - Configurable for pretraining and finetuning
    - Memory-efficient data pipeline
    - Better preprocessing pipeline
    """
    
    def __init__(self,
                 config: ExperimentConfig,
                 splits: Optional[Dict[str, List[ScanInfo]]] = None,
                 mode: Literal['pretrain', 'finetune', 'evaluate'] = 'finetune',
                 labeled_fraction: float = 1.0,
                 use_weighted_sampling: bool = True,
                 cache_data: bool = False):
        """
        Initialize data module.
        
        Args:
            config: Experiment configuration
            splits: Pre-computed data splits (if None, loads from disk)
            mode: 'pretrain' for SimCLR, 'finetune' for classification
            labeled_fraction: Fraction of labeled data to use (for SSL experiments)
            use_weighted_sampling: Use weighted sampling for class imbalance
            cache_data: Cache loaded volumes in memory
        """
        super().__init__()
        
        self.config = config
        self.splits = splits
        self.mode = mode
        self.labeled_fraction = labeled_fraction
        self.use_weighted_sampling = use_weighted_sampling
        self.cache_data = cache_data
        
        # Datasets (initialized in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_scans: List[ScanInfo] = []
        self.val_scans: List[ScanInfo] = []
        self.test_scans: List[ScanInfo] = []
        
        # Class information
        self.class_names: List[str] = []
        self.class_weights: Dict[str, float] = {}
        self.num_classes: int = 2
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Statistics
        self.stats = {
            'n_train': 0,
            'n_val': 0,
            'n_test': 0,
            'class_distribution': {}
        }
    
    def prepare_data(self) -> None:
        """
        Prepare data (called once on main process).
        
        Validates that required files exist.
        """
        # Check preprocessed directory
        if not self.config.paths.preprocessed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed directory not found: {self.config.paths.preprocessed_dir}"
            )
        
        # Check splits file if not provided
        if self.splits is None:
            splits_path = self.config.paths.output_dir / "splits.pkl"
            if not splits_path.exists():
                raise FileNotFoundError(
                    f"Splits file not found: {splits_path}\n"
                    "Run with --mode organize first to create data splits!"
                )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load splits if not provided
        if self.splits is None:
            self.splits = self._load_splits()
        
        # Get scans for each split
        train_scans = list(self.splits.get('train', []))
        val_scans = list(self.splits.get('val', []))
        test_scans = list(self.splits.get('test', []))
        
        # Setup label encoder from all data
        all_labels = [s.diagnosis for s in train_scans + val_scans + test_scans]
        self.class_names = sorted(set(all_labels))
        self.num_classes = len(self.class_names)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        # ✅ FIX 1: Compute class weights from FULL training data FIRST (before sampling)
        # This prevents extreme weights when using small labeled fractions
        full_train_labels = [s.diagnosis for s in train_scans]
        class_weight_method = 'balanced'
        if self.mode == 'finetune' and hasattr(self.config, 'finetune'):
            class_weight_method = getattr(self.config.finetune, 'class_weight_method', 'balanced')
        self.class_weights = ClassWeightCalculator.compute_weights(full_train_labels, method=class_weight_method)
        
        # THEN apply labeled fraction (for SSL experiments)
        if self.labeled_fraction < 1.0 and self.mode == 'finetune':
            train_scans = self._sample_labeled_data(train_scans, self.labeled_fraction)
        if self.mode == 'finetune' and hasattr(self.config, 'finetune'):
            clip_min = getattr(self.config.finetune, 'class_weight_clip_min', None)
            clip_max = getattr(self.config.finetune, 'class_weight_clip_max', None)
            if clip_min is not None or clip_max is not None:
                clipped_weights = {}
                for cls_name, weight in self.class_weights.items():
                    if clip_min is not None:
                        weight = max(weight, clip_min)
                    if clip_max is not None:
                        weight = min(weight, clip_max)
                    clipped_weights[cls_name] = weight
                self.class_weights = clipped_weights
        
        # Create datasets based on mode
        if self.mode == 'pretrain':
            # SimCLR pretraining: use train + val data (no labels needed)
            all_pretrain_scans = train_scans + val_scans
            
            self.train_dataset = SimCLRDataset(
                scans=all_pretrain_scans,
                target_shape=self.config.data.target_shape,
                aug_config=self.config.augmentation,
                cache_data=self.cache_data
            )
            
            # No validation dataset for pretraining
            self.val_dataset = None
            self.test_dataset = None
            
        else:
            # Finetuning mode
            # Training with augmentation
            train_augmentation = Augmentation3D(
                self.config.augmentation, training=True
            )
            
            self.train_dataset = MRIClassificationDataset(
                scans=train_scans,
                target_shape=self.config.data.target_shape,
                augmentation=train_augmentation,
                label_encoder=self.label_encoder,
                cache_data=self.cache_data
            )
            
            # Validation without augmentation
            self.val_dataset = MRIClassificationDataset(
                scans=val_scans,
                target_shape=self.config.data.target_shape,
                augmentation=None,  # No augmentation for validation!
                label_encoder=self.label_encoder,
                cache_data=self.cache_data
            )
            
            # Test without augmentation
            self.test_dataset = MRIClassificationDataset(
                scans=test_scans,
                target_shape=self.config.data.target_shape,
                augmentation=None,  # No augmentation for test!
                label_encoder=self.label_encoder,
                cache_data=self.cache_data
            )
        
        self.train_scans = train_scans
        self.val_scans = val_scans
        self.test_scans = test_scans

        # Update statistics
        self.stats['n_train'] = len(train_scans)
        self.stats['n_val'] = len(val_scans)
        self.stats['n_test'] = len(test_scans)
        self.stats['class_distribution'] = dict(Counter(
            s.diagnosis for s in train_scans
        ))
    
    def _load_splits(self) -> Dict[str, List[ScanInfo]]:
        """Load splits from pickle file."""
        splits_path = self.config.paths.output_dir / "splits.pkl"
        
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_path}")
        
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        return splits
    
    def _sample_labeled_data(self,
                              scans: List[ScanInfo],
                              fraction: float) -> List[ScanInfo]:
        """
        Sample a fraction of labeled data while maintaining class balance.
        
        Args:
            scans: List of scans
            fraction: Fraction to sample
        
        Returns:
            Sampled list of scans
        """
        if fraction >= 1.0:
            return scans
        
        # Group by class
        class_scans: Dict[str, List[ScanInfo]] = defaultdict(list)
        for scan in scans:
            class_scans[scan.diagnosis].append(scan)
        
        # Sample from each class proportionally
        sampled = []
        for cls, cls_scans in class_scans.items():
            n_sample = max(1, int(len(cls_scans) * fraction))
            # Use random.sample for reproducibility with global seed
            sampled.extend(random.sample(cls_scans, min(n_sample, len(cls_scans))))
        
        return sampled
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        
        # Determine batch size
        if self.mode == 'pretrain':
            batch_size = self.config.simclr.batch_size_per_gpu
        else:
            batch_size = self.config.finetune.batch_size
        
        sampler = None
        shuffle = True
        if self.use_weighted_sampling and self.mode == 'finetune':
            sampler_weight_method = getattr(self.config.finetune, 'sampler_weight_method', 'balanced')
            train_labels = [s.diagnosis for s in self.train_scans]
            sampler_class_weights = ClassWeightCalculator.compute_weights(train_labels, method=sampler_weight_method)
            clip_min = getattr(self.config.finetune, 'class_weight_clip_min', None)
            clip_max = getattr(self.config.finetune, 'class_weight_clip_max', None)
            if clip_min is not None or clip_max is not None:
                clipped_sampler_weights = {}
                for cls_name, weight in sampler_class_weights.items():
                    if clip_min is not None:
                        weight = max(weight, clip_min)
                    if clip_max is not None:
                        weight = min(weight, clip_max)
                    clipped_sampler_weights[cls_name] = weight
                sampler_class_weights = clipped_sampler_weights
            sample_weights = self.train_dataset.get_sample_weights(sampler_class_weights)
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        
        # Determine number of workers
        num_workers = self.config.data.num_workers
        
        # ✅ PRODUCTION FIX: Use config-based persistent_workers setting (not auto-calculated)
        # This prevents deadlocks with large 3D medical data
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True,
            persistent_workers=self.config.data.persistent_workers,  # ✅ Use config value
            prefetch_factor=self.config.data.prefetch_factor if num_workers > 0 else None
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        
        # ✅ PRODUCTION FIX: Use config persistent_workers setting
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.finetune.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None:
            return None
        
        # ✅ PRODUCTION FIX: Use config persistent_workers setting
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.finetune.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.persistent_workers
        )
    
    def get_class_weights_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Get class weights as tensor for loss function.
        
        Args:
            device: Device to place tensor on
        
        Returns:
            Tensor of class weights
        """
        weights = [self.class_weights.get(cls, 1.0) for cls in self.class_names]
        return torch.tensor(weights, dtype=torch.float32, device=device)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'n_train': self.stats['n_train'],
            'n_val': self.stats['n_val'],
            'n_test': self.stats['n_test'],
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'class_distribution': self.stats['class_distribution'],
            'class_weights': self.class_weights
        }


# ============================================================================
# IMPROVED EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """
    Improved Experiment Orchestrator for SSL MCI Detection.
    
    This class coordinates the entire experimental pipeline:
    1. Data organization and validation
    2. SimCLR self-supervised pre-training
    3. Fine-tuning with various labeled fractions
    4. Evaluation and visualization
    
    Key improvements:
    - Better error handling
    - More robust training procedures
    - Improved metrics
    - Better logging
    - Proper class balancing
    """
    
    def __init__(self, config: ExperimentConfig, gpu_ids: List[int]):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            gpu_ids: List of GPU IDs to use
        """
        self.config = config
        self.use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
        self.gpu_ids = gpu_ids if self.use_gpu else []
        self.num_gpus = len(self.gpu_ids) if self.use_gpu else 1
        self.device_type = 'gpu' if self.use_gpu else 'cpu'
        
        # Create all directories
        config.paths.create_all()
        
        # Setup logging
        self.logger = setup_logging(
            config.paths.logs_dir,
            config.experiment_name,
            debug=config.debug
        )
        
        # Set random seeds
        set_seed(config.seed)
        
        # Print experiment header
        self._print_header()
        
        # Save configuration
        config_path = config.paths.output_dir / "config.json"
        config.save(config_path)
        self.logger.info(f"💾 Configuration saved: {config_path}")
        
        # Initialize components
        self.data_organizer = ADNIDataOrganizer(config, self.logger)
        
        # Track experiment state
        self.splits: Optional[Dict[str, List[ScanInfo]]] = None
        self.pretrain_checkpoint: Optional[str] = None
    
    def _print_header(self) -> None:
        """Print experiment header with system info."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🧠 IMPROVED SELF-SUPERVISED LEARNING FOR MCI DETECTION")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info(f"📋 Experiment: {self.config.experiment_name}")
        self.logger.info(f"🎲 Random Seed: {self.config.seed}")
        self.logger.info(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # GPU Information
        gpu_info = get_gpu_info()
        self.logger.info(f"🖥️  GPU Information:")
        self.logger.info(f"   Available GPUs: {gpu_info['count']}")
        self.logger.info(f"   Using GPUs: {self.gpu_ids}")
        if not self.use_gpu:
            self.logger.info("   Using CPU")
        
        if gpu_info['available']:
            for dev in gpu_info['devices']:
                if dev['id'] in self.gpu_ids:
                    self.logger.info(
                        f"   GPU {dev['id']}: {dev['name']} | "
                        f"Memory: {dev['memory_total_gb']:.1f} GB | "
                        f"Free: {dev['memory_free_gb']:.1f} GB"
                    )
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("")
    
    def organize_data(self) -> Dict[str, List[ScanInfo]]:
        """
        Step 1: Organize and validate data.
        
        Returns:
            Dictionary with train/val/test splits
        """
        self.logger.info("")
        self.logger.info("🚀 STARTING DATA ORGANIZATION")
        self.logger.info("")
        
        # Check if splits already exist
        splits_path = self.config.paths.output_dir / "splits.pkl"
        
        if splits_path.exists():
            self.logger.info(f"📂 Found existing splits: {splits_path}")
            self.logger.info("   Loading from cache...")
            
            with open(splits_path, 'rb') as f:
                self.splits = pickle.load(f)
            
            # Print summary
            for split_name, scans in self.splits.items():
                n_scans = len(scans)
                n_subjects = len(set(s.subject_id for s in scans))
                self.logger.info(f"   {split_name}: {n_scans} scans, {n_subjects} subjects")
            
            return self.splits
        
        # Run data organization
        self.splits = self.data_organizer.run()
        
        return self.splits
    
    def pretrain(self,
                 splits: Optional[Dict[str, List[ScanInfo]]] = None,
                 resume_checkpoint: Optional[str] = None) -> str:
        """
        Step 2: SimCLR self-supervised pre-training.
        
        Args:
            splits: Data splits (uses cached if None)
            resume_checkpoint: Path to checkpoint for resuming
        
        Returns:
            Path to saved checkpoint
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🔄 STEP 2: SIMCLR SELF-SUPERVISED PRE-TRAINING")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        # Use provided splits or cached
        if splits is None:
            if self.splits is None:
                self.splits = self.organize_data()
            splits = self.splits
        
        # Check for existing checkpoint
        final_checkpoint = self.config.paths.checkpoint_dir / "simclr_final.ckpt"
        if final_checkpoint.exists() and resume_checkpoint is None:
            self.logger.info(f"📦 Found existing checkpoint: {final_checkpoint}")
            self.logger.info("   Skipping pre-training (use --resume-checkpoint to continue)")
            self.pretrain_checkpoint = str(final_checkpoint)
            return self.pretrain_checkpoint
        
        # Create data module
        datamodule = ADNIDataModule(
            config=self.config,
            splits=splits,
            mode='pretrain',
            cache_data=False  # Don't cache for large datasets
        )
        
        # Create model
        model = SimCLRModule(
            config=self.config.simclr,
            num_gpus=self.num_gpus,
            use_gradient_checkpointing=self.config.simclr.batch_size_per_gpu >= 16
        )
        
        # Log model info
        params = count_parameters(model)
        self.logger.info(f"📊 Model Architecture: {self.config.simclr.backbone}")
        self.logger.info(f"   Total Parameters: {format_number(params['total'])}")
        self.logger.info(f"   Trainable Parameters: {format_number(params['trainable'])}")
        self.logger.info(f"   Projection Dimension: {self.config.simclr.projection_dim}")
        self.logger.info("")
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"   Epochs: {self.config.simclr.epochs}")
        self.logger.info(f"   Batch Size per GPU: {self.config.simclr.batch_size_per_gpu}")
        self.logger.info(f"   Effective Batch Size: {self.config.simclr.batch_size_per_gpu * self.num_gpus}")
        self.logger.info(f"   Base Learning Rate: {self.config.simclr.base_lr}")
        self.logger.info(f"   Optimizer: {self.config.simclr.optimizer.upper()}")
        self.logger.info(f"   Temperature: {self.config.simclr.temperature}")
        self.logger.info(f"   Mixed Precision: {self.config.simclr.use_fp16}")
        self.logger.info("")
        
        # Setup callbacks
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                dirpath=str(self.config.paths.checkpoint_dir),
                filename='simclr-{epoch:03d}-{train_loss:.4f}',
                save_top_k=3,
                mode='min',
                monitor='train_loss',
                save_last=True,
                verbose=True
            ),
            # Learning rate monitoring
            LearningRateMonitor(logging_interval='epoch'),
            # GPU memory monitoring
            GPUMemoryMonitorCallback(log_every_n_steps=100),
            # Gradient monitoring
            GradientMonitorCallback(log_every_n_steps=50),
            # Progress bar
            TQDMProgressBar(refresh_rate=10)
        ]
        
        # Setup loggers
        loggers = [
            TensorBoardLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='simclr_pretrain',
                version=self.config.experiment_name
            ),
            CSVLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='simclr_csv',
                version=self.config.experiment_name
            )
        ]
        
        # Trainer configuration
        devices = self.gpu_ids if self.use_gpu and self.num_gpus > 1 else (self.gpu_ids[0] if self.use_gpu else 1)
        precision = '16-mixed' if self.use_gpu and self.config.simclr.use_fp16 else 32
        trainer_kwargs = {
            'max_epochs': self.config.simclr.epochs,
            'accelerator': self.device_type,
            'devices': devices,
            'precision': precision,
            'callbacks': callbacks,
            'logger': loggers,
            'gradient_clip_val': self.config.simclr.gradient_clip_val,
            'accumulate_grad_batches': self.config.simclr.accumulate_grad_batches,
            'log_every_n_steps': 10,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'deterministic': False,  # For speed
            'benchmark': True,  # For speed with fixed input size
        }
        
        # Multi-GPU strategy
        if self.use_gpu and self.num_gpus > 1:
            trainer_kwargs['strategy'] = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
            
            if self.config.simclr.sync_batchnorm:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.logger.info("   Using Synchronized BatchNorm")
        
        # Create trainer
        trainer = Trainer(**trainer_kwargs)
        
        # Resume from checkpoint if provided
        ckpt_path = None
        if resume_checkpoint and Path(resume_checkpoint).exists():
            ckpt_path = resume_checkpoint
            self.logger.info(f"📂 Resuming from checkpoint: {ckpt_path}")
        
        # Start training
        self.logger.info("")
        self.logger.info("🚀 Starting SimCLR pre-training...")
        self.logger.info("")
        
        # ✅ PRODUCTION FIX: Validate DataLoader before training
        try:
            self.logger.info("🔍 Validating DataLoader configuration...")
            self.logger.info(f"   Batch size per GPU: {self.config.simclr.batch_size_per_gpu}")
            self.logger.info(f"   Num GPUs: {self.num_gpus}")
            self.logger.info(f"   Effective batch size: {self.config.simclr.batch_size_per_gpu * self.num_gpus}")
            self.logger.info(f"   Num workers: {self.config.data.num_workers}")
            self.logger.info(f"   Persistent workers: {self.config.data.persistent_workers}")
            
            # Test DataLoader
            self.logger.info("   Testing DataLoader by loading one batch...")
            datamodule.setup()
            train_loader = datamodule.train_dataloader()
            test_iter = iter(train_loader)
            test_batch = next(test_iter)
            
            if isinstance(test_batch, (list, tuple)) and len(test_batch) >= 2:
                view1, view2 = test_batch[0], test_batch[1]
                self.logger.info(f"   ✅ DataLoader OK!")
                self.logger.info(f"      View 1 shape: {view1.shape}")
                self.logger.info(f"      View 2 shape: {view2.shape}")
            else:
                self.logger.info(f"   ✅ DataLoader OK (batch type: {type(test_batch)})")
            
            del test_batch, test_iter
            import gc
            gc.collect()
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"   ❌ DataLoader validation FAILED: {e}")
            self.logger.error("   Try reducing num_workers or disabling persistent_workers")
            import traceback
            self.logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"DataLoader initialization failed: {e}")
        
        try:
            trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Training interrupted by user")
        
        # Save final checkpoint
        final_ckpt_path = self.config.paths.checkpoint_dir / "simclr_final.ckpt"
        trainer.save_checkpoint(str(final_ckpt_path))
        
        self.pretrain_checkpoint = str(final_ckpt_path)
        
        self.logger.info("")
        self.logger.info("=" * 50)
        self.logger.info("✅ PRE-TRAINING COMPLETE!")
        self.logger.info(f"📦 Checkpoint saved: {final_ckpt_path}")
        self.logger.info("=" * 50)
        self.logger.info("")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        return self.pretrain_checkpoint
    
    def finetune(self,
                 splits: Optional[Dict[str, List[ScanInfo]]] = None,
                 pretrain_checkpoint: Optional[str] = None,
                 labeled_fraction: float = 1.0,
                 fold_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Step 3: Fine-tune for MCI classification.
        
        Args:
            splits: Data splits (uses cached if None)
            pretrain_checkpoint: Path to pretrained checkpoint
            labeled_fraction: Fraction of labeled data to use
        
        Returns:
            Dictionary of test metrics
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        fold_label = f" | Fold {fold_idx}" if fold_idx is not None else ""
        self.logger.info(f"🔄 STEP 3: FINE-TUNING ({labeled_fraction*100:.0f}% LABELED DATA){fold_label}")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        # Use provided or cached data
        if splits is None:
            if self.splits is None:
                self.splits = self.organize_data()
            splits = self.splits
        
        if pretrain_checkpoint is None:
            pretrain_checkpoint = self.pretrain_checkpoint
        
        # Create data module
        datamodule = ADNIDataModule(
            config=self.config,
            splits=splits,
            mode='finetune',
            labeled_fraction=labeled_fraction,
            use_weighted_sampling=self.config.finetune.use_weighted_sampling,
            cache_data=False
        )
        datamodule.setup()
        
        # Log data info
        stats = datamodule.get_stats()
        self.logger.info(f"📊 Dataset Statistics:")
        self.logger.info(f"   Training samples: {stats['n_train']}")
        self.logger.info(f"   Validation samples: {stats['n_val']}")
        self.logger.info(f"   Test samples: {stats['n_test']}")
        self.logger.info(f"   Classes: {stats['class_names']}")
        self.logger.info(f"   Class distribution: {stats['class_distribution']}")
        self.logger.info(f"   Class weights: {stats['class_weights']}")
        self.logger.info("")
        
        # Load pretrained backbone
        # ──────────────────────────────────────────────────────────────────
        # v5.0.0 FIXED loading pipeline:
        #   1. Strip DDP "module." prefix from checkpoint keys
        #   2. Load into a fresh SimCLRModule (no SyncBN, no DDP wrapper)
        #   3. Convert any residual SyncBatchNorm -> BatchNorm3d
        #   4. Validate that weights actually loaded (not still random)
        # ──────────────────────────────────────────────────────────────────
        if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
            self.logger.info(f"📦 Loading pretrained backbone: {pretrain_checkpoint}")

            try:
                checkpoint = torch.load(pretrain_checkpoint, map_location='cpu', weights_only=False)

                # --- 1. Remap keys: strip "module." DDP prefix if present ---
                raw_state_dict = checkpoint['state_dict']
                state_dict = remap_state_dict_keys(raw_state_dict)
                n_remapped = sum(1 for k in raw_state_dict if k.startswith("module."))
                if n_remapped > 0:
                    self.logger.info(f"   ℹ️  Remapped {n_remapped} keys (stripped DDP 'module.' prefix)")

                # --- 2. Create a fresh SimCLRModule and load the cleaned state_dict ---
                simclr_model = SimCLRModule(
                    config=self.config.simclr,
                    num_gpus=1  # single-GPU; no SyncBN conversion here
                )

                missing_keys, unexpected_keys = simclr_model.load_state_dict(state_dict, strict=False)
                n_loaded = len(state_dict) - len(missing_keys)
                self.logger.info(f"   ✅ load_state_dict: loaded={n_loaded}, "
                                 f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
                if missing_keys:
                    self.logger.warning(f"   ⚠️  Missing keys (first 10): {missing_keys[:10]}")
                if unexpected_keys:
                    self.logger.warning(f"   ⚠️  Unexpected keys (first 10): {unexpected_keys[:10]}")

                # --- 3. Extract backbone and convert SyncBatchNorm -> BatchNorm3d ---
                backbone = simclr_model.backbone
                backbone = convert_syncbn_to_bn(backbone)
                self.logger.info("   ✅ SyncBatchNorm -> BatchNorm3d conversion done")

                # --- 4. Validate the weights are actually pre-trained ---
                validate_backbone_weights(backbone, self.logger)

                # Log checkpoint metadata
                if 'epoch' in checkpoint:
                    self.logger.info(f"   ✅ Pretrained from epoch {checkpoint['epoch']}")
                if 'global_step' in checkpoint:
                    self.logger.info(f"   ✅ Global step: {checkpoint['global_step']}")

            except Exception as e:
                self.logger.warning(f"   ⚠️  Failed to load checkpoint: {e}")
                self.logger.warning("   Falling back to randomly initialized backbone")
                import traceback
                self.logger.warning(f"   Error details: {traceback.format_exc()}")
                backbone = get_backbone(self.config.simclr.backbone, in_channels=1)
                backbone.fc = nn.Identity()
        else:
            self.logger.warning("⚠️  No pretrained checkpoint provided")
            self.logger.warning("   Training from scratch (supervised baseline)")
            backbone = get_backbone(self.config.simclr.backbone, in_channels=1)
            backbone.fc = nn.Identity()

        # Get device for class weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = datamodule.get_class_weights_tensor(device)
        
        # Log class weights so we can verify they are sane (should be 0.8–2.0 range)
        self.logger.info(f"   Class weights: {class_weights.tolist()}")
        self.logger.info(f"   Class names:   {datamodule.class_names}")

        # Create classifier
        model = MCIClassifier(
            backbone=backbone,
            num_classes=datamodule.num_classes,
            config=self.config.finetune,
            class_weights=class_weights,
            class_names=datamodule.class_names,
            freeze_backbone=self.config.finetune.freeze_backbone
        )
        
        # Log model info
        params = count_parameters(model)
        self.logger.info(f"📊 Classifier Configuration:")
        self.logger.info(f"   Total Parameters: {format_number(params['total'])}")
        self.logger.info(f"   Trainable Parameters: {format_number(params['trainable'])}")
        self.logger.info(f"   Frozen Parameters: {format_number(params['frozen'])}")
        self.logger.info(f"   Backbone Frozen: {model.is_backbone_frozen()}")
        self.logger.info(f"   Dropout: {self.config.finetune.dropout}")
        self.logger.info(f"   Label Smoothing: {self.config.finetune.label_smoothing}")
        self.logger.info("")
        
        # Setup callbacks
        fold_prefix = f"fold{fold_idx:02d}-" if fold_idx is not None else ""
        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.config.paths.checkpoint_dir),
                filename=f'emci-{fold_prefix}frac{labeled_fraction:.2f}-' + '{epoch:02d}-{val_min_sens_spec:.4f}',
                save_top_k=1,
                mode='max',
                monitor='val_min_sens_spec',
                save_last=True,
                verbose=True
            ),
            EarlyStopping(
                monitor='val_min_sens_spec',
                patience=self.config.finetune.patience,
                mode='max',
                min_delta=self.config.finetune.min_delta,
                verbose=True
            ),
            LearningRateMonitor(logging_interval='epoch'),
            GPUMemoryMonitorCallback(log_every_n_steps=50),
            TQDMProgressBar(refresh_rate=5)
        ]
        
        # Add unfreeze callback if backbone is frozen
        if self.config.finetune.freeze_backbone and self.config.finetune.freeze_epochs > 0:
            callbacks.append(UnfreezeBackboneCallback(
                unfreeze_epoch=self.config.finetune.freeze_epochs
            ))
            self.logger.info(f"   Backbone will unfreeze at epoch {self.config.finetune.freeze_epochs}")
        
        # Setup loggers
        version_name = f"{fold_prefix}frac{labeled_fraction:.2f}_{self.config.experiment_name}"
        loggers = [
            TensorBoardLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='finetune',
                version=version_name
            ),
            CSVLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='finetune_csv',
                version=version_name
            )
        ]
        
        # Create trainer
        use_ddp = self.use_gpu and len(self.gpu_ids) > 1
        devices = self.gpu_ids if use_ddp else ([self.gpu_ids[0]] if self.use_gpu else 1)
        precision = '16-mixed' if self.use_gpu else 32
        strategy = DDPStrategy(find_unused_parameters=False) if use_ddp else "auto"
        sync_batchnorm = self.config.simclr.sync_batchnorm if use_ddp else False
        trainer = Trainer(
            max_epochs=self.config.finetune.epochs,
            accelerator=self.device_type,
            devices=devices,
            precision=precision,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=5,
            enable_progress_bar=True,
            enable_model_summary=True,
            num_sanity_val_steps=0,
            deterministic=False,
            benchmark=True,
            gradient_clip_val=self.config.finetune.gradient_clip_val
        )
        
        # Train
        self.logger.info("")
        self.logger.info("🚀 Starting fine-tuning...")
        self.logger.info("")
        
        # ✅ PRODUCTION FIX: Validate DataLoader before training
        # This catches issues early with clear error messages instead of silent hangs
        try:
            self.logger.info("🔍 Validating DataLoader configuration...")
            
            # Get batch size for logging
            if datamodule.mode == 'pretrain':
                batch_size = self.config.simclr.batch_size_per_gpu
            else:
                batch_size = self.config.finetune.batch_size
            
            self.logger.info(f"   Batch size: {batch_size}")
            self.logger.info(f"   Num workers: {self.config.data.num_workers}")
            self.logger.info(f"   Persistent workers: {self.config.data.persistent_workers}")
            self.logger.info(f"   Pin memory: {self.config.data.pin_memory}")
            
            # Calculate expected batches
            n_train = stats['n_train']
            expected_batches = n_train // batch_size
            self.logger.info(f"   Training samples: {n_train}")
            self.logger.info(f"   Expected batches per epoch: {expected_batches}")
            
            if expected_batches == 0:
                raise ValueError(
                    f"Batch size ({batch_size}) is larger than training samples ({n_train}). "
                    f"Please reduce batch size or increase training data."
                )
            
            # Test DataLoader by loading one batch
            self.logger.info("   Testing DataLoader by loading one batch...")
            train_loader = datamodule.train_dataloader()
            
            # Set a timeout for the test (prevent infinite hang)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("DataLoader test timed out after 60 seconds")
            
            # Only set alarm on Unix systems
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 60 second timeout
            
            try:
                test_iter = iter(train_loader)
                test_batch = next(test_iter)
                
                # Cancel timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                
                # Validate batch structure
                if isinstance(test_batch, (list, tuple)):
                    if len(test_batch) >= 2:
                        batch_data, batch_labels = test_batch[0], test_batch[1]
                        self.logger.info(f"   ✅ DataLoader OK!")
                        self.logger.info(f"      Batch data shape: {batch_data.shape}")
                        self.logger.info(f"      Batch labels shape: {batch_labels.shape}")
                        self.logger.info(f"      Data dtype: {batch_data.dtype}")
                        self.logger.info(f"      Data device: {batch_data.device}")
                    else:
                        self.logger.warning(f"   ⚠️  Unexpected batch structure: {len(test_batch)} elements")
                else:
                    self.logger.info(f"   ✅ DataLoader OK (batch type: {type(test_batch)})")
                
                # Cleanup
                del test_batch, test_iter
                import gc
                gc.collect()
                
            except TimeoutError as e:
                self.logger.error("   ❌ DATALOADER TIMEOUT!")
                self.logger.error("   " + str(e))
                self.logger.error("   SOLUTION: Reduce num_workers or disable persistent_workers")
                raise RuntimeError("DataLoader test timed out. Check num_workers and persistent_workers settings.")
            
            self.logger.info("")
            
        except StopIteration:
            self.logger.error("   ❌ DataLoader returned no batches!")
            self.logger.error("   This usually means:")
            self.logger.error("     1. Training dataset is empty")
            self.logger.error("     2. Batch size is larger than dataset")
            self.logger.error(f"     3. drop_last=True with only {n_train} samples and batch_size={batch_size}")
            raise RuntimeError(f"DataLoader validation failed: no batches available")
            
        except Exception as e:
            self.logger.error(f"   ❌ DataLoader validation FAILED: {e}")
            self.logger.error("   Common causes:")
            self.logger.error("     1. Corrupted NIfTI files")
            self.logger.error("     2. num_workers too high (try 0 or 2)")
            self.logger.error("     3. persistent_workers=True causing deadlock")
            self.logger.error("     4. Insufficient shared memory (/dev/shm)")
            self.logger.error("     5. Memory exhaustion from caching")
            import traceback
            self.logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"DataLoader initialization failed: {e}")
        
        # Proceed with training
        self.logger.info("🎯 DataLoader validated successfully, starting training...")
        self.logger.info("")
        
        try:
            trainer.fit(model, datamodule)
        except KeyboardInterrupt:
            self.logger.warning("⚠️ Training interrupted by user")
        
        # Test — MUST load the BEST checkpoint, not the last (possibly collapsed) epoch.
        # ModelCheckpoint saved the best weights to disk; ckpt_path="best" loads them.
        self.logger.info("")
        self.logger.info("\U0001f9ea Evaluating on test set (loading BEST checkpoint)...")

        # Find the best checkpoint path from the ModelCheckpoint callback
        best_ckpt_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_ckpt_path = cb.best_model_path
                self.logger.info(f"   Best checkpoint: {best_ckpt_path}")
                break

        if best_ckpt_path and Path(best_ckpt_path).exists():
            test_results = trainer.test(ckpt_path=best_ckpt_path, datamodule=datamodule)
        else:
            # Fallback: evaluate current model (not ideal but won't crash)
            self.logger.warning("   Best checkpoint not found — evaluating last epoch model")
            test_results = trainer.test(model, datamodule)
        
        # Get metrics
        if test_results:
            metrics = test_results[0]
        else:
            metrics = {}
        
        # Log results
        self.logger.info("")
        self.logger.info("=" * 50)
        self.logger.info(f"✅ FINE-TUNING COMPLETE ({labeled_fraction*100:.0f}% labeled)")
        self.logger.info("=" * 50)
        # Format final test results in a clear table
        self.logger.info("📊 FINAL TEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"  Accuracy:          {metrics.get('test_accuracy', 0):.4f}")
        self.logger.info(f"  AUC:               {metrics.get('test_auc', 0):.4f}")
        self.logger.info(f"  F1 Score:          {metrics.get('test_f1', 0):.4f}")
        self.logger.info(f"  Sensitivity (MCI): {metrics.get('test_sensitivity', 0):.4f}")
        self.logger.info(f"  Specificity (CN):  {metrics.get('test_specificity', 0):.4f}")
        self.logger.info(f"  Balanced Accuracy: {metrics.get('test_balanced_accuracy', 0):.4f}")
        self.logger.info(f"  PPV (MCI):         {metrics.get('test_ppv', 0):.4f}")
        self.logger.info(f"  NPV (CN):          {metrics.get('test_npv', 0):.4f}")
        self.logger.info("=" * 60)
        self.logger.info("")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        return metrics

    def _build_subject_groups(self, scans: List[ScanInfo]) -> Tuple[Dict[str, List[ScanInfo]], List[str], List[str]]:
        subjects: Dict[str, List[ScanInfo]] = defaultdict(list)
        for scan in scans:
            subjects[scan.subject_id].append(scan)
        subject_ids = list(subjects.keys())
        subject_labels = []
        for subj_id in subject_ids:
            diagnoses = [s.diagnosis for s in subjects[subj_id]]
            most_common = Counter(diagnoses).most_common(1)[0][0]
            subject_labels.append(most_common)
        return subjects, subject_ids, subject_labels

    def _split_train_val_subjects(self,
                                  subject_ids: List[str],
                                  subject_labels: List[str],
                                  val_ratio: float) -> Tuple[List[str], List[str]]:
        if len(subject_ids) < 2:
            return subject_ids, []
        diagnosis_counts = Counter(subject_labels)
        min_count = min(diagnosis_counts.values()) if diagnosis_counts else 0
        stratify = subject_labels if min_count >= 2 else None
        try:
            train_subj, val_subj = train_test_split(
                subject_ids,
                train_size=1.0 - val_ratio,
                stratify=stratify,
                random_state=self.config.seed
            )
        except ValueError:
            random.seed(self.config.seed)
            random.shuffle(subject_ids)
            n_val = max(1, int(len(subject_ids) * val_ratio))
            val_subj = subject_ids[:n_val]
            train_subj = subject_ids[n_val:]
        return train_subj, val_subj

    def cross_validate(self,
                       splits: Optional[Dict[str, List[ScanInfo]]] = None,
                       pretrain_checkpoint: Optional[str] = None,
                       labeled_fraction: float = 1.0,
                       n_folds: Optional[int] = None) -> Dict[str, Any]:
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🔁 CROSS-VALIDATION")
        self.logger.info("=" * 70)
        self.logger.info("")

        if splits is None:
            splits = self.organize_data()

        all_scans = []
        for split_list in splits.values():
            all_scans.extend(split_list)

        subjects, subject_ids, subject_labels = self._build_subject_groups(all_scans)
        n_folds = n_folds or self.config.data.n_folds

        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")

        diagnosis_counts = Counter(subject_labels)
        min_count = min(diagnosis_counts.values()) if diagnosis_counts else 0

        if min_count >= n_folds:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)
            split_iter = splitter.split(subject_ids, subject_labels)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)
            split_iter = splitter.split(subject_ids)

        fold_results: List[Dict[str, float]] = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(split_iter, start=1):
            set_seed(self.config.seed + fold_idx)
            test_subjects = [subject_ids[i] for i in test_idx]
            train_val_subjects = [subject_ids[i] for i in train_val_idx]
            train_val_labels = [subject_labels[i] for i in train_val_idx]

            val_ratio = self.config.data.val_ratio / max(self.config.data.train_ratio + self.config.data.val_ratio, 1e-6)
            train_subj, val_subj = self._split_train_val_subjects(train_val_subjects, train_val_labels, val_ratio)

            fold_splits = {
                'train': [],
                'val': [],
                'test': []
            }
            for subj_id in train_subj:
                fold_splits['train'].extend(subjects[subj_id])
            for subj_id in val_subj:
                fold_splits['val'].extend(subjects[subj_id])
            for subj_id in test_subjects:
                fold_splits['test'].extend(subjects[subj_id])

            self.logger.info(f"Fold {fold_idx}/{n_folds}")
            self.logger.info(f"  Train subjects: {len(train_subj)}")
            self.logger.info(f"  Val subjects:   {len(val_subj)}")
            self.logger.info(f"  Test subjects:  {len(test_subjects)}")

            metrics = self.finetune(
                splits=fold_splits,
                pretrain_checkpoint=pretrain_checkpoint,
                labeled_fraction=labeled_fraction,
                fold_idx=fold_idx
            )
            fold_results.append(metrics)

        summary = {}
        if fold_results:
            metric_keys = sorted({k for r in fold_results for k in r.keys()})
            for key in metric_keys:
                values = [r.get(key, 0.0) for r in fold_results]
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("📊 CROSS-VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        for key, stats in summary.items():
            self.logger.info(f"{key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        self.logger.info("=" * 60)

        return {
            'folds': fold_results,
            'summary': summary
        }
    
    def run_labeled_fraction_experiments(self,
                                          splits: Optional[Dict[str, List[ScanInfo]]] = None,
                                          pretrain_checkpoint: Optional[str] = None) -> Dict[float, Dict[str, float]]:
        """
        Run experiments with different labeled data fractions.
        
        Args:
            splits: Data splits
            pretrain_checkpoint: Pretrained checkpoint path
        
        Returns:
            Dictionary mapping fraction to metrics
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📊 LABELED FRACTION EXPERIMENTS")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        fractions = self.config.finetune.labeled_fractions
        self.logger.info(f"Testing fractions: {[f'{f*100:.0f}%' for f in fractions]}")
        self.logger.info("")
        
        all_results = {}
        
        for i, fraction in enumerate(fractions):
            self.logger.info("")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"📊 Experiment {i+1}/{len(fractions)}: {fraction*100:.0f}% labeled data")
            self.logger.info(f"{'='*50}")
            
            try:
                metrics = self.finetune(
                    splits=splits,
                    pretrain_checkpoint=pretrain_checkpoint,
                    labeled_fraction=fraction
                )
                
                # Store results
                all_results[fraction] = {
                    k.replace('test_', ''): v 
                    for k, v in metrics.items() 
                    if isinstance(v, (int, float))
                }
                
            except Exception as e:
                self.logger.error(f"❌ Error with fraction {fraction}: {e}")
                all_results[fraction] = {'error': str(e)}
            
            # Clear memory between experiments
            clear_gpu_memory()
        
        # Save results to JSON
        results_path = self.config.paths.output_dir / "fraction_experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(
                {str(k): v for k, v in all_results.items()},
                f, indent=2
            )
        self.logger.info(f"💾 Results saved: {results_path}")
        
        return all_results

    def evaluate(self,
                 splits: Optional[Dict[str, List[ScanInfo]]] = None,
                 checkpoint_path: Optional[str] = None,
                 labeled_fraction: float = 1.0) -> Dict[str, float]:
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📊 EVALUATION MODE")
        self.logger.info("=" * 70)
        self.logger.info("")

        if splits is None:
            if self.splits is None:
                self.splits = self.organize_data()
            splits = self.splits

        if checkpoint_path is None:
            ckpts = list(self.config.paths.checkpoint_dir.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {self.config.paths.checkpoint_dir}")
            ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_path = str(ckpts[0])

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        datamodule = ADNIDataModule(
            config=self.config,
            splits=splits,
            mode='finetune',
            labeled_fraction=labeled_fraction,
            use_weighted_sampling=False,
            cache_data=False
        )
        datamodule.setup()

        backbone = get_backbone(self.config.simclr.backbone, in_channels=1)
        backbone.fc = nn.Identity()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = datamodule.get_class_weights_tensor(device)

        model = MCIClassifier(
            backbone=backbone,
            num_classes=datamodule.num_classes,
            config=self.config.finetune,
            class_weights=class_weights,
            class_names=datamodule.class_names,
            freeze_backbone=False
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            self.logger.warning(f"Missing keys (first 10): {missing_keys[:10]}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys (first 10): {unexpected_keys[:10]}")

        version_name = f"eval_{self.config.experiment_name}"
        loggers = [
            TensorBoardLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='evaluate',
                version=version_name
            ),
            CSVLogger(
                save_dir=str(self.config.paths.logs_dir),
                name='evaluate_csv',
                version=version_name
            )
        ]

        devices = [self.gpu_ids[0]] if self.use_gpu else 1
        precision = '16-mixed' if self.use_gpu else 32
        trainer = Trainer(
            accelerator=self.device_type,
            devices=devices,
            precision=precision,
            logger=loggers,
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=False,
            benchmark=True
        )

        self.logger.info(f"📦 Evaluating checkpoint: {checkpoint_path}")
        test_results = trainer.test(model, datamodule=datamodule)
        metrics = test_results[0] if test_results else {}

        if model.test_results:
            labels = model.test_results.get('labels', [])
            preds = model.test_results.get('preds', [])
            if len(labels) and len(preds):
                cm = confusion_matrix(labels, preds)
                self.logger.info("Confusion Matrix:")
                self.logger.info(f"{cm}")

        self.logger.info("")
        self.logger.info("=" * 50)
        self.logger.info("✅ EVALUATION COMPLETE")
        self.logger.info("=" * 50)
        self.logger.info(f"  Accuracy:          {metrics.get('test_accuracy', 0):.4f}")
        self.logger.info(f"  AUC:               {metrics.get('test_auc', 0):.4f}")
        self.logger.info(f"  F1 Score:          {metrics.get('test_f1', 0):.4f}")
        self.logger.info(f"  Sensitivity (MCI): {metrics.get('test_sensitivity', 0):.4f}")
        self.logger.info(f"  Specificity (CN):  {metrics.get('test_specificity', 0):.4f}")
        self.logger.info(f"  Balanced Accuracy: {metrics.get('test_balanced_accuracy', 0):.4f}")
        self.logger.info(f"  PPV (MCI):         {metrics.get('test_ppv', 0):.4f}")
        self.logger.info(f"  NPV (CN):          {metrics.get('test_npv', 0):.4f}")
        self.logger.info("=" * 50)

        return metrics
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline.
        
        Executes:
        1. Data organization
        2. SimCLR pre-training
        3. Fine-tuning with multiple labeled fractions
        4. Evaluation and visualization
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🚀 RUNNING FULL EXPERIMENTAL PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        results = {
            'experiment_name': self.config.experiment_name,
            'timestamp_start': datetime.now().isoformat(),
            'seed': self.config.seed,
            'gpu_ids': self.gpu_ids
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Organization
            self.logger.info("")
            self.logger.info("📂 Step 1/4: Data Organization")
            with timer("Data Organization", self.logger):
                splits = self.organize_data()
                results['data'] = {
                    'n_train': len(splits['train']),
                    'n_val': len(splits['val']),
                    'n_test': len(splits['test'])
                }
            
            # Step 2: Pre-training
            self.logger.info("")
            self.logger.info("🔄 Step 2/4: SimCLR Pre-training")
            with timer("SimCLR Pre-training", self.logger):
                pretrain_ckpt = self.pretrain(splits)
                results['pretrain_checkpoint'] = pretrain_ckpt
            
            # Step 3: Labeled Fraction Experiments
            self.logger.info("")
            self.logger.info("🧪 Step 3/4: Labeled Fraction Experiments")
            with timer("Labeled Fraction Experiments", self.logger):
                fraction_results = self.run_labeled_fraction_experiments(
                    splits, pretrain_ckpt
                )
                results['fraction_experiments'] = {
                    str(k): v for k, v in fraction_results.items()
                }
            
            results['status'] = 'completed'
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        finally:
            # Calculate total time
            total_time = datetime.now() - start_time
            results['timestamp_end'] = datetime.now().isoformat()
            results['total_time'] = str(total_time)
            
            # Save final results
            final_results_path = self.config.paths.output_dir / "final_results.json"
            with open(final_results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print final summary
            self.logger.info("")
            self.logger.info("=" * 70)
            self.logger.info("🎉 PIPELINE COMPLETE!")
            self.logger.info("=" * 70)
            self.logger.info("")
            self.logger.info(f"📁 Output Directory: {self.config.paths.output_dir}")
            self.logger.info(f"📊 Figures: {self.config.paths.figures_dir}")
            self.logger.info(f"📝 Logs: {self.config.paths.logs_dir}")
            self.logger.info(f"💾 Checkpoints: {self.config.paths.checkpoint_dir}")
            self.logger.info("")
            self.logger.info(f"⏱️ Total Time: {total_time}")
            self.logger.info("")
            self.logger.info("=" * 70)
        
        return results


# ============================================================================
# UNIT TESTS
# ============================================================================

def run_unit_tests() -> bool:
    failures: List[str] = []

    organizer = ADNIDataOrganizer.__new__(ADNIDataOrganizer)
    diagnosis_cases = [
        ("EMCI", "MCI"),
        ("CN", "CN"),
        ("Early Mild Cognitive Impairment", "MCI"),
        ("NL", "CN"),
        ("AD", "AD")
    ]
    for raw, expected in diagnosis_cases:
        result = organizer._standardize_diagnosis(raw)
        if result != expected:
            failures.append(f"standardize_diagnosis({raw}) -> {result} (expected {expected})")

    random.seed(0)
    np.random.seed(0)
    aug_config = AugmentationConfig()
    augmentation = Augmentation3D(aug_config, training=True)
    volume = np.random.rand(32, 32, 32).astype(np.float32)
    augmented = augmentation(volume)
    if augmented.shape != volume.shape:
        failures.append(f"augmentation shape {augmented.shape} != {volume.shape}")
    if augmented.min() < 0 or augmented.max() > 1:
        failures.append("augmentation values out of range [0, 1]")

    simclr_aug = SimCLRAugmentation(aug_config)
    view1, view2 = simclr_aug(volume)
    if view1.shape != volume.shape or view2.shape != volume.shape:
        failures.append("simclr augmentation views have incorrect shapes")

    if failures:
        for failure in failures:
            print(f"❌ {failure}")
        return False

    print("✅ Unit tests passed")
    return True


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Improved Self-Supervised Learning for MCI Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Organize data and create splits
  python ssl_emci_detection_improved.py --mode organize

  # Step 2: Pre-train with SimCLR on multiple GPUs
  python ssl_emci_detection_improved.py --mode pretrain --gpus 1,2,3,4

  # Step 3: Fine-tune with 10% labeled data
  python ssl_emci_detection_improved.py --mode finetune --labeled-fraction 0.1 \\
      --pretrain-checkpoint checkpoints/simclr_final.ckpt

  # Run complete pipeline
  python ssl_emci_detection_improved.py --mode full-pipeline --gpus 1,2,3,4,5,6,7

  # Run with custom experiment name
  python ssl_emci_detection_improved.py --mode full-pipeline --gpus 1,2 \\
      --experiment-name my_experiment

Author: Improved Research Pipeline
Version: 5.0.0 (Production - ResNet50 - Backbone Loading + Unfreeze + SyncBN fixes)
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['organize', 'pretrain', 'finetune', 'evaluate', 'crossval', 'full-pipeline', 'test'],
        default='full-pipeline',
        help='Pipeline mode to run (default: full-pipeline)'
    )
    
    # GPU configuration
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='GPU IDs to use, comma-separated (default: 0). Example: 1,2,3,4'
    )
    
    # Data configuration
    parser.add_argument(
        '--labeled-fraction',
        type=float,
        default=0.1,
        help='Fraction of labeled data for fine-tuning (default: 0.1)'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=None,
        help='Number of folds for cross-validation (default: config setting)'
    )
    
    # Checkpoint paths
    parser.add_argument(
        '--pretrain-checkpoint',
        type=str,
        default=None,
        help='Path to pre-trained SimCLR checkpoint for fine-tuning'
    )

    parser.add_argument(
        '--eval-checkpoint',
        type=str,
        default=None,
        help='Path to fine-tuned checkpoint for evaluation'
    )
    
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for resuming training'
    )
    
    # Training parameters (overrides)
    parser.add_argument(
        '--epochs-pretrain',
        type=int,
        default=None,
        help='Override number of pre-training epochs'
    )
    
    parser.add_argument(
        '--epochs-finetune',
        type=int,
        default=None,
        help='Override number of fine-tuning epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size per GPU'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet18', 'resnet34', 'resnet50'],
        default=None,
        help='Override backbone architecture'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for the experiment (default: auto-generated with timestamp)'
    )
    
    # Path overrides
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override preprocessed data directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run without actual training'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the Improved SSL MCI Detection pipeline.
    """
    # Parse arguments
    args = parse_args()

    if args.mode == 'test':
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
    
    if not gpu_ids:
        gpu_ids = [0]
    
    # Set CUDA visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Create configuration
    config = ExperimentConfig(seed=args.seed)
    
    # Override experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"ssl_mci_improved_{timestamp}"
    
    # Override paths if provided
    if args.data_dir:
        config.paths.preprocessed_dir = Path(args.data_dir)
    
    if args.output_dir:
        config.paths.output_dir = Path(args.output_dir)
        config.paths.checkpoint_dir = Path(args.output_dir) / "checkpoints"
        config.paths.figures_dir = Path(args.output_dir) / "figures"
        config.paths.logs_dir = Path(args.output_dir) / "logs"
    
    if args.n_folds:
        config.data.n_folds = args.n_folds

    if args.epochs_pretrain:
        config.simclr.epochs = args.epochs_pretrain
    
    if args.epochs_finetune:
        config.finetune.epochs = args.epochs_finetune
    
    if args.batch_size:
        config.simclr.batch_size_per_gpu = args.batch_size
        config.finetune.batch_size = args.batch_size
    
    if args.backbone:
        config.simclr.backbone = args.backbone
    
    # Set debug mode
    config.debug = args.debug
    
    # Map GPU IDs (CUDA_VISIBLE_DEVICES makes them 0,1,2,...)
    mapped_gpu_ids = list(range(len(gpu_ids)))
    
    # Create experiment runner
    runner = ExperimentRunner(config, mapped_gpu_ids)
    
    # Execute based on mode
    try:
        if args.mode == 'organize':
            runner.organize_data()
        
        elif args.mode == 'pretrain':
            splits = runner.organize_data()
            if not args.dry_run:
                runner.pretrain(splits, resume_checkpoint=args.resume_checkpoint)
            else:
                print("🔍 Dry run - would execute pre-training")
        
        elif args.mode == 'finetune':
            splits = runner.organize_data()
            if not args.dry_run:
                runner.finetune(
                    splits=splits,
                    pretrain_checkpoint=args.pretrain_checkpoint,
                    labeled_fraction=args.labeled_fraction
                )
            else:
                print("🔍 Dry run - would execute fine-tuning")
        
        elif args.mode == 'evaluate':
            splits = runner.organize_data()
            if not args.dry_run:
                runner.evaluate(
                    splits=splits,
                    checkpoint_path=args.eval_checkpoint,
                    labeled_fraction=args.labeled_fraction
                )
            else:
                print("🔍 Dry run - would execute evaluation")

        elif args.mode == 'crossval':
            splits = runner.organize_data()
            if not args.dry_run:
                runner.cross_validate(
                    splits=splits,
                    pretrain_checkpoint=args.pretrain_checkpoint,
                    labeled_fraction=args.labeled_fraction,
                    n_folds=args.n_folds
                )
            else:
                print("🔍 Dry run - would execute cross-validation")
        
        elif args.mode == 'full-pipeline':
            if not args.dry_run:
                runner.run_full_pipeline()
            else:
                print("🔍 Dry run - would execute full pipeline:")
                print("   1. Data organization")
                print("   2. SimCLR pre-training")
                print("   3. Fine-tuning experiments")
                print("   4. Evaluation and visualization")
                runner.organize_data()
    
    except KeyboardInterrupt:
        print("\n")
        print("⚠️ Interrupted by user")
        print("   Partial results may be saved in output directory")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\n✨ Done!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
