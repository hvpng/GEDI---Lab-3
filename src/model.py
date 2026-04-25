"""Model definitions and experiment runners for the GEDI lab reproduction.

This module contains an independent re-implementation of the GEDI clustering
algorithm.  It is written from scratch and does not copy the authors' public
code.

Key components
--------------
GEDIConfig         — dataclass of all hyperparameters
GEDIModel          — nn.Module with encoder, projector and cluster centres
energy / logits    — free-energy and soft-assignment heads
loss_inv           — augmentation-invariance loss  (L_INV)
loss_prior         — cluster-uniformity prior      (L_PRIOR)
loss_gen           — contrastive-divergence loss   (L_GEN)  via SGLD
train_gedi         — full training loop
gedi_predict       — hard cluster assignment from a trained model
run_clustering_suite  — compare GEDI with sklearn baselines
run_ablation_study    — ablate each of the three loss terms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

from src.metrics import evaluate_clustering


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Container for a single experiment outcome."""

    dataset: str
    method: str
    scores: Dict[str, float]


@dataclass
class GEDIConfig:
    """Hyperparameters for the GEDI model and training loop.

    Attributes:
        in_features:    Dimensionality of input features.
        hidden_dim:     Width of the encoder's output (and projector).
        n_clusters:     Number of clusters c.
        tau:            Temperature for energy / softmax.
        train_iterations: Number of optimization steps.
        batch_size:     Mini-batch size.
        lr:             Adam learning rate.
        lambda_inv:     Weight for L_INV.
        lambda_prior:   Weight for L_PRIOR.
        lambda_gen:     Weight for L_GEN.
        sgld_steps:     Langevin steps per SGLD call.
        sgld_step_size: Step size η in the Langevin update.
        sgld_noise_std: Noise std σ in the Langevin update.
        buffer_size:    Replay-buffer capacity for SGLD.
        use_loss_inv:   Toggle L_INV (ablation flag).
        use_loss_prior: Toggle L_PRIOR (ablation flag).
        encoder_hidden_dims: Hidden layer widths for the encoder MLP.
            Defaults to [100, 100] (paper-like). Override to change
            architecture, e.g. [128] for a shallower network.
        use_loss_gen:   Toggle L_GEN (ablation flag).
        random_state:   Global seed.
    """

    in_features: int = 2
    hidden_dim: int = 2
    n_clusters: int = 2
    tau: float = 0.5
    train_iterations: int = 20000
    batch_size: int = 400
    lr: float = 1e-3
    lambda_inv: float = 50.0
    lambda_prior: float = 10.0
    lambda_gen: float = 1.0
    l2_reg: float = 0.0 
    sgld_steps: int = 1
    sgld_step_size: float = 0.01**2/2
    sgld_noise_std: float = 0.01
    buffer_size: int = 10000
    use_loss_inv: bool = True
    use_loss_prior: bool = True
    use_loss_gen: bool = True
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [100, 100])
    projector_hidden: int | None = None  # None = auto: 4 for toy (h<=2), 2*h otherwise
    encoder_type: str = 'mlp'  # 'mlp' or 'resnet8' (Appendix M, Table 8)
    aug_noise_std: float = 0.03  # std of Gaussian augmentation for L_INV (Table 7/9: 0.03 toy/image; Section 4.8: 0.05 text)
    random_state: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden_dims: List[int], out_dim: int) -> nn.Sequential:
    """Build a fully-connected ReLU MLP.

    Args:
        in_dim:      Input dimensionality.
        hidden_dims: List of hidden layer widths.
        out_dim:     Output dimensionality.

    Returns:
        nn.Sequential containing Linear -> ReLU blocks followed by a
        final Linear layer (no activation).
    """
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# ResNet-8 encoder (image datasets: SVHN, Fashion-MNIST)
# ──────────────────────────────────────────────────────────────────────────────

class _ResBlockDown(nn.Module):
    """Pre-activation residual block with AvgPool2D(2) downsampling.

    Block 1 (``first=True``): no leading activation, Conv→LReLU→Conv→AvgPool.
    Blocks 2+ (``first=False``): LReLU→Conv→LReLU→Conv→AvgPool (pre-activation).
    Shortcut: Conv1×1(in→out) if channels differ, then AvgPool2D(2).
    """

    def __init__(self, in_ch: int, out_ch: int, first: bool = False) -> None:
        super().__init__()
        self.first  = first
        self.lrelu  = nn.LeakyReLU(0.2, inplace=True)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=True)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.pool   = nn.AvgPool2d(2)
        sc: List[nn.Module] = []
        if in_ch != out_ch:
            sc.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        sc.append(nn.AvgPool2d(2))
        self.shortcut: nn.Module = nn.Sequential(*sc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.first:
            out = self.lrelu(self.conv1(x))
            out = self.conv2(out)
        else:
            out = self.conv1(self.lrelu(x))
            out = self.conv2(self.lrelu(out))
        return self.pool(out) + self.shortcut(x)


class _ResBlock(nn.Module):
    """Pre-activation residual block without downsampling (identity shortcut).

    LReLU(0.2) → Conv(F→F) → LReLU(0.2) → Conv(F→F) + skip.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.lrelu(x))
        out = self.conv2(self.lrelu(out))
        return out + x


class ResNet8Encoder(nn.Module):
    """ResNet-8 encoder for 32×32 RGB images (Appendix M, Table 8).

    F=128 channels throughout (constant width, no channel expansion).
    Activation: LeakyReLU(0.2), no BatchNorm (typical for EBMs).

    Architecture::

        Block 1: Conv(3→F) → LReLU → Conv(F→F) → AvgPool(2)   32→16  [shortcut: Conv1×1 + AvgPool]
        Block 2: LReLU → Conv(F→F) → LReLU → Conv(F→F) → AvgPool(2)  [shortcut: AvgPool]  16→8
        Block 3: LReLU → Conv(F→F) → LReLU → Conv(F→F)          8×8  [shortcut: identity]
        Block 4: LReLU → Conv(F→F) → LReLU → Conv(F→F)          8×8  [shortcut: identity]
        LReLU → AdaptiveAvgPool2d(1) → Linear(F, hidden_dim)

    Accepts either (B, 3, 32, 32) float tensors or (B, 3072) flat CHW vectors.
    Output shape: (B, hidden_dim).
    """

    def __init__(self, hidden_dim: int = 128, F: int = 128) -> None:
        super().__init__()
        self.block1 = _ResBlockDown(3, F, first=True)   # 32×32 → 16×16
        self.block2 = _ResBlockDown(F, F, first=False)  # 16×16 →  8×8
        self.block3 = _ResBlock(F)                       #  8×8  →  8×8
        self.block4 = _ResBlock(F)                       #  8×8  →  8×8
        self.lrelu  = nn.LeakyReLU(0.2, inplace=True)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(F, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:                       # (B, 3072) → (B, 3, 32, 32)
            x = x.view(-1, 3, 32, 32)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.lrelu(out)
        return self.fc(self.pool(out).flatten(1))



# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class GEDIModel(nn.Module):
    """GEDI energy-based generative clustering model.

    Architecture (synthetic / low-dimensional setting, following the paper):
        encoder   f : R^d  → R^h   (MLP: d → 100 → 100 → h)
        projector g : R^h  → R^h   (MLP: h → h → h)
        cluster centres  U ∈ R^{h × c}   (learnable parameter)

    Energy function (free energy / negative log-partition):
        E(x) = −logsumexp( Uᵀ g(f(x)) / τ )          scalar per sample

    Cluster assignment probability:
        p(y | x) = softmax( Uᵀ g(f(x)) / τ )          shape (B, c)
    """

    def __init__(self, cfg: GEDIConfig) -> None:
        """Initialise encoder, projector and cluster centres from config.

        Args:
            cfg: GEDIConfig instance specifying architecture and hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        # Encoder: MLP or ResNet-8 (Appendix M, Table 8)
        if cfg.encoder_type == 'resnet8':
            self.encoder: nn.Module = ResNet8Encoder(h)
        else:
            self.encoder = _mlp(cfg.in_features, cfg.encoder_hidden_dims, h)

        self.projector = nn.Sequential(
            nn.Linear(h, 2 * h),
            nn.ReLU(),
            nn.Linear(2 * h, cfg.n_clusters, bias=False) 
        )

    # ------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output f(x), shape (B, h)."""
        return self.encoder(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Đầu ra phân phối cụm chưa chuẩn hóa g(f(x)) / τ, shape (B, c).
        
        Projector ánh xạ trực tiếp đặc trưng h-dim sang không gian c-dim của các cụm.
        """
        return self.projector(self._embed(x)) / self.cfg.tau

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Free-energy E(x) = −logsumexp(logits), shape (B,)."""
        return -torch.logsumexp(self.logits(x), dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Cluster probabilities p(y|x) = softmax(logits), shape (B, c)."""
        return torch.softmax(self.logits(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute free-energy for each input sample.

        Args:
            x: Input tensor, shape (B, d).

        Returns:
            Energy values E(x), shape (B,).
        """
        return self.energy(x)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def loss_inv(
    model: GEDIModel,
    x: torch.Tensor,
    x_aug: torch.Tensor,
) -> torch.Tensor:
    """Augmentation-invariance loss L_INV.

    Tính toán hàm mất mát Cross-Entropy chéo, trong đó:
    - Target: Dự đoán xác suất qua softmax của Augmented view (x_aug).
    - Prediction: Logits chưa qua softmax của Clean view (x).
    Giúp gán cùng một cụm cho các biến thể khác nhau của cùng một điểm dữ liệu.
    """
    
    # Sửa lỗi 1: Augmented view (z2) làm Target, Clean view (z1) làm Prediction
    z1 = model.logits(x)      
    z2 = model.logits(x_aug)  
    
    target = torch.softmax(z2, dim=-1).detach()
    return -(target * z1).sum(dim=-1).mean() + torch.logsumexp(z1, dim=-1).mean()


def loss_prior(model: GEDIModel, x: torch.Tensor) -> torch.Tensor:
    """Cluster-uniformity prior L_PRIOR.

    Uniform-cluster prior from the paper.

    For a uniform prior p(y) = 1/c, the loss becomes:
        L_PRIOR = sum_y p(y) log(mean_j p(y | x_j))
                = mean_y log(mean_j p(y | x_j))

    Minimising the negative of this term encourages the batch-average cluster
    distribution to stay close to uniform and prevents cluster collapse.

    Args:
        model: The GEDI model.
        x:     Batch of samples, shape (B, d).

    Returns:
        Scalar loss to minimise.
    """
    p_mean = model.predict_proba(x).mean(dim=0)
    return -torch.log(p_mean + 1e-8).mean()


def _sgld_sample(
    model: GEDIModel,
    x_init: torch.Tensor,
    cfg: GEDIConfig,
    minim: float = -15.0,
    maxim: float = 15.0,
) -> torch.Tensor:
    """Sinh mẫu từ p(x) ∝ exp(−E(x)) thông qua Stochastic Gradient Langevin Dynamics (SGLD).

    Update rule có kẹp gradient và kẹp không gian dữ liệu để tránh exploding:
        grad_t  = clamp(∇_x E(x_t), -1, 1)
        x_{t+1} = clamp(x_t − η · grad_t + √η · ε, minim, maxim), với ε ∼ N(0, σ²I)
    """
    was_training = model.training
    original_requires_grad = [param.requires_grad for param in model.parameters()]

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    x = x_init.clone().detach().requires_grad_(True)
    try:
        for _ in range(cfg.sgld_steps):
            energy_sum = model.energy(x).sum()
            grad = torch.autograd.grad(energy_sum, x)[0]
            
            # Sửa lỗi 2a: Kẹp (Clamp) gradient
            grad = torch.clamp(grad, -100.0, 100.0)
            
            noise = torch.randn_like(x) * cfg.sgld_noise_std
            x_next = x - cfg.sgld_step_size * grad + noise
            
            # Sửa lỗi 2b: Kẹp (Clamp) giá trị vector
            x = torch.clamp(x_next, minim, maxim).detach().requires_grad_(True)
            
        return x.detach()
    finally:
        for param, requires_grad in zip(model.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)
        if was_training:
            model.train()
        else:
            model.eval()


def loss_gen(
    model: GEDIModel,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
) -> torch.Tensor:
    """Contrastive-divergence generative loss L_GEN.

    Minimises E(x_real) and maximises E(x_fake), so the energy landscape
    assigns low energy to real data and high energy to generated samples.

    Args:
        model:  The GEDI model.
        x_real: Real data samples,        shape (B,  d).
        x_fake: SGLD-generated samples,   shape (B', d).

    Returns:
        Scalar loss.
    """
    return model.energy(x_real).mean() - model.energy(x_fake).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_gedi(
    model: GEDIModel,
    X: np.ndarray,
    cfg: GEDIConfig,
) -> GEDIModel:
    """Huấn luyện mô hình GEDI trên ma trận đặc trưng NumPy.

    Tổng hợp các thành phần Loss:
        L_total = λ_inv · L_INV + λ_prior · L_PRIOR + λ_gen · L_GEN + L2_Priors

    Lưu ý cơ chế SGLD Replay Buffer:
        - 95% mẫu lấy từ Buffer cũ.
        - 5% mẫu được thay thế bằng nhiễu mới (Fresh noise) sinh từ phân phối Đều (Uniform)
          dựa trên biên min/max của tập dữ liệu thực.
    """
    if not (cfg.use_loss_inv or cfg.use_loss_prior or cfg.use_loss_gen):
        raise ValueError(
            "At least one loss term must be enabled: "
            "use_loss_inv, use_loss_prior, or use_loss_gen."
        )

    torch.manual_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # In ra thiết bị sử dụng để huấn luyện (CPU hoặc GPU)
    print(f"Training on device: {device}")
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Tính min, max để phục vụ cho Uniform Noise và Clamping
    v_min = X_t.min().item()
    v_max = X_t.max().item()
    
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=min(cfg.batch_size, len(X_t)),
        shuffle=True,
    )
    loader_iter = cycle(loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    buf_idx = rng.integers(0, len(X_t), size=cfg.buffer_size)
    buffer = X_t[buf_idx].clone()

    model.train()
    for step in range(cfg.train_iterations):
        (x_batch,) = next(loader_iter)

        # Nếu là dữ liệu SVHN (kích thước 3x32x32 = 3072 pixel)
        if x_batch.shape[1] == 3072: 
            # Reshape tensor về định dạng ảnh (B, C, H, W)
            x_img = x_batch.view(-1, 3, 32, 32)
            
            # Định nghĩa các phép Augmentation theo Table 9 của bài báo
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.1),
                T.RandomGrayscale(p=0.1)
            ])
            
            x_aug_img = transform(x_img)
            # Duỗi phẳng lại và cộng nhiễu Gaussian 0.03
            x_aug = x_aug_img.reshape(-1, 3072) + torch.randn_like(x_batch) * 0.03
        else:
            # Cho Moons, Circles và Text (chỉ cộng nhiễu)
            x_aug = x_batch + torch.randn_like(x_batch) * cfg.aug_noise_std

        loss_terms: List[torch.Tensor] = []

        if cfg.use_loss_inv:
            loss_terms.append(cfg.lambda_inv * loss_inv(model, x_batch, x_aug))

        if cfg.use_loss_prior:
            loss_terms.append(cfg.lambda_prior * loss_prior(model, x_batch))

        if cfg.use_loss_gen:
            b_idx = rng.integers(0, len(buffer), size=len(x_batch))
            x_init = buffer[b_idx].clone()
            fresh_mask = torch.rand(len(x_init), device=device) < 0.05
            
            # Sửa lỗi 3: Sinh nhiễu khởi tạo theo phân phối Đều (Uniform) thay vì Gaussian
            n_fresh = fresh_mask.sum().item()
            if n_fresh > 0:
                x_init[fresh_mask] = torch.empty(n_fresh, x_init.shape[1], dtype=x_init.dtype, device=x_init.device).uniform_(v_min, v_max)

            # Truyền giới hạn để kẹp biên khi sinh mẫu
            x_fake = _sgld_sample(model, x_init, cfg, minim=v_min, maxim=v_max)
            buffer[b_idx] = x_fake.detach()

            loss_terms.append(cfg.lambda_gen * loss_gen(model, x_batch, x_fake))

        if not loss_terms:
            continue

        total_loss = torch.stack(loss_terms).sum()
        if cfg.l2_reg > 0.0:
            z2 = model.encoder(x_aug)
            prior_z = 0.5 * (z2 ** 2).mean()
            weight1 = model.projector[0].weight
            weight2 = model.projector[2].weight
            prior_w = (
                0.5 * (weight1 ** 2).sum(1).mean()
                + 0.5 * (weight2 ** 2).sum(1).mean()
            )
            total_loss = total_loss + cfg.l2_reg * (prior_z + prior_w)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    return model


def gedi_predict(model: GEDIModel, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Return hard cluster assignments using batching to avoid OOM."""
    model.eval()
    
    device = next(model.parameters()).device
    
    all_preds = []
    for i in range(0, len(X), batch_size):
        batch_x = X[i : i + batch_size]
        with torch.no_grad():
            x_t = torch.tensor(batch_x, dtype=torch.float32).to(device)
            probs = model.predict_proba(x_t)
            preds = probs.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            
    return np.concatenate(all_preds)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ──────────────────────────────────────────────────────────────────────────────

def run_clustering_suite(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    random_state: int = 42,
    return_model: bool = False,
    gedi_cfg_overrides: Dict | None = None,
) -> pd.DataFrame | Tuple[pd.DataFrame, GEDIModel]:
    """Benchmark GEDI against sklearn baselines on the same dataset.

    The sklearn baselines (KMeans, Spectral, GMM, Agglomerative) serve as
    simple reference points.  The paper's deep baselines (JEM, Barlow Twins,
    SwAV) are not re-implemented here; their reported scores are available via
    ``src.utils.get_paper_reference_scores`` for notebook-level comparison.

    Args:
        X:                  Input features, shape (N, d).
        y:                  Ground-truth labels, shape (N,).
        dataset_name:       Name attached to the result rows.
        random_state:       Seed for stochastic models.
        return_model:       If True, also returns the trained GEDI model.
        gedi_cfg_overrides: Optional dict of GEDIConfig kwargs that override
                            the defaults (e.g. ``{'hidden_dim': 64,
                            'encoder_hidden_dims': [256]}`` for Fashion-MNIST).

    Returns:
        If return_model is False:
            DataFrame with columns: Dataset, Method, ACC, NMI, ARI,
            Silhouette, DBI, CHI. Sorted by NMI descending.
        If return_model is True:
            Tuple of (DataFrame, trained GEDIModel).
    """
    n_clusters = int(len(np.unique(y)))
    in_features = X.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sklearn_models = {
        "KMeans": KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "Spectral": SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=random_state,
        ),
        "GaussianMixture": GaussianMixture(
            n_components=n_clusters, random_state=random_state
        ),
    }

    rows: List[Dict] = []
    for method_name, clf in sklearn_models.items():
        if hasattr(clf, "fit_predict"):
            y_pred = clf.fit_predict(X_scaled)
        else:
            y_pred = clf.fit(X_scaled).predict(X_scaled)
        scores = evaluate_clustering(X_scaled, y, y_pred)
        rows.append({"Dataset": dataset_name, "Method": method_name, **scores})

    # GEDI
    cfg_kwargs: Dict = dict(
        in_features=in_features,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    if gedi_cfg_overrides:
        cfg_kwargs.update(gedi_cfg_overrides)
    cfg = GEDIConfig(**cfg_kwargs)
    gedi = GEDIModel(cfg)
    train_gedi(gedi, X_scaled, cfg)
    y_pred_gedi = gedi_predict(gedi, X_scaled)
    scores_gedi = evaluate_clustering(X_scaled, y, y_pred_gedi)
    rows.append({"Dataset": dataset_name, "Method": "GEDI", **scores_gedi})

    results = pd.DataFrame(rows)
    results = results.sort_values(["NMI", "ACC"], ascending=False).reset_index(drop=True)
    if return_model:
        return results, gedi
    return results


def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ablate each GEDI loss term to measure its individual contribution.

    Variants (mirrors the paper's ablation in Table 3):
        - Full GEDI  : L_INV + L_PRIOR + L_GEN
        - No L_GEN   : remove contrastive-divergence generative loss
        - No L_INV   : remove augmentation-invariance loss
        - No L_PRIOR : remove cluster-uniformity prior

    Args:
        X:            Input features, shape (N, d).
        y:            Ground-truth labels, shape (N,).
        dataset_name: Dataset name for the result rows.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with one row per variant and all six evaluation metrics.
    """
    n_clusters = int(len(np.unique(y)))
    in_features = X.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # (label, use_gen, use_inv, use_prior)
    settings = [
        ("Full GEDI",  True,  True,  True),
        ("No L_GEN",   False, True,  True),
        ("No L_INV",   True,  False, True),
        ("No L_PRIOR", True,  True,  False),
    ]

    rows: List[Dict] = []
    for variant, use_gen, use_inv, use_prior in settings:
        cfg = GEDIConfig(
            in_features=in_features,
            n_clusters=n_clusters,
            use_loss_gen=use_gen,
            use_loss_inv=use_inv,
            use_loss_prior=use_prior,
            random_state=random_state,
        )
        model = GEDIModel(cfg)
        train_gedi(model, X_scaled, cfg)
        y_pred = gedi_predict(model, X_scaled)
        scores = evaluate_clustering(X_scaled, y, y_pred)
        rows.append({"Dataset": dataset_name, "Variant": variant, **scores})

    return pd.DataFrame(rows).sort_values("NMI", ascending=False).reset_index(drop=True)

