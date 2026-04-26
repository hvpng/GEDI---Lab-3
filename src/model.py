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
    tau: float = 0.1
    train_iterations: int = 20000
    batch_size: int = 400
    lr: float = 1e-3
    lambda_inv: float = 50.0
    lambda_prior: float = 10.0
    lambda_gen: float = 1.0
    sgld_steps: int = 1
    sgld_step_size: float = 0.000072
    sgld_noise_std: float = 0.01
    sgld_grad_clip: float = 1.0   # ε: gradient clamped to [−ε, ε] at each SGLD step
    buffer_size: int = 10000
    use_loss_inv: bool = True
    use_loss_prior: bool = True
    use_loss_gen: bool = True
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [100, 100])
    projector_hidden: int | None = None  # None = auto: 2*h (matches paper: h → 2h → c)
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
        projector g : R^h  → R^c   (MLP: h → 2h → c)

    Energy function (free energy / negative log-partition):
        E(x) = −logsumexp( g(f(x)) / τ )          scalar per sample

    Cluster assignment probability:
        p(y | x) = softmax( g(f(x)) / τ )          shape (B, c)
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

        # Projector: h → proj_hidden → c  (original paper: h → 2h → c)
        proj_hidden = cfg.projector_hidden if cfg.projector_hidden is not None else 2 * h
        self.projector = _mlp(h, [proj_hidden], cfg.n_clusters)

    # ------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Projector output g(f(x)), shape (B, c)."""
        return self.projector(self.encoder(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits g(f(x)) / τ, shape (B, c)."""
        return self._embed(x) / self.cfg.tau   # (B, c)

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

    Cross-entropy with target = softmax(z2) [augmented view] and
    prediction = log_softmax(z1) [clean view], matching the paper formula:
        L_INV = -(z2.softmax(1) * z1).sum(1).mean() + z1.logsumexp(1).mean()
              = -mean[ sum_c softmax(z2)_c * log_softmax(z1)_c ]
    Gradients are allowed to flow through both branches, which matches the
    paper's definition q(y|x) ≡ p(y|x; Θ).

    Args:
        model:  The GEDI model.
        x:      Original samples,   shape (B, d).
        x_aug:  Augmented views,    shape (B, d).

    Returns:
        Scalar loss.
    """
    target = model.predict_proba(x_aug)
    log_pred = torch.log_softmax(model.logits(x), dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


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
    x_min: float | None = None,
    x_max: float | None = None,
) -> torch.Tensor:
    """Sample from p(x) ∝ exp(−E(x)) via Stochastic Gradient Langevin Dynamics.

    Update rule:
        x_{t+1} = x_t − (η/2) · ∇_x E(x_t) + √η · ε,   ε ∼ N(0, σ²I)

    Two clamping operations are applied at each step (matching the original):
      1. Gradient clamping: ∇E is clamped to [−ε, ε] (``cfg.sgld_grad_clip``).
      2. Sample clamping:   x is clamped to [x_min, x_max] when provided,
         keeping samples inside the valid data manifold.

    Args:
        model:  Energy model; only input gradients are used (no param grad).
        x_init: Starting points, shape (B, d).
        cfg:    Config with sgld_steps, sgld_step_size, sgld_noise_std,
                sgld_grad_clip.
        x_min:  Lower bound for sample clamping (data-range minimum).
        x_max:  Upper bound for sample clamping (data-range maximum).

    Returns:
        Samples after ``cfg.sgld_steps`` Langevin steps, shape (B, d).
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
            # 1) Gradient clamping: prevents exploding updates (orig: clamp(f', -eps, eps))
            grad = torch.clamp(grad, -cfg.sgld_grad_clip, cfg.sgld_grad_clip)
            noise = torch.randn_like(x) * cfg.sgld_noise_std
            x = (x - cfg.sgld_step_size * grad + noise).detach()
            # 2) Sample clamping: keeps x inside the valid data manifold
            if x_min is not None and x_max is not None:
                x = torch.clamp(x, x_min, x_max)
            x = x.requires_grad_(True)
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
    """Train a GEDIModel on a numpy feature matrix.

    Combines the three loss terms according to the flags in ``cfg``, plus
    unconditional L2 regularization priors::

        L = λ_inv · L_INV  +  λ_prior · L_PRIOR  +  λ_gen · L_GEN
            + 0.5·‖z₂‖² + 0.5·∑_c ‖U_c‖²

    SGLD samples are maintained in a replay buffer (80 % from buffer,
    20 % fresh noise) following the standard contrastive-divergence setup.

    Args:
        model: Initialised GEDIModel (modified in-place).
        X:     Feature matrix, shape (N, d).
        cfg:   Training hyperparameters.

    Returns:
        The trained model.

    Raises:
        ValueError: If all loss terms are disabled via ablation flags,
            no gradient update can be performed.
    """
    if not (cfg.use_loss_inv or cfg.use_loss_prior or cfg.use_loss_gen):
        raise ValueError(
            "At least one loss term must be enabled: "
            "use_loss_inv, use_loss_prior, or use_loss_gen."
        )

    torch.manual_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)

    X_t = torch.tensor(X, dtype=torch.float32)
    # Data-range bounds used for SGLD sample clamping
    x_min = float(X_t.min().item())
    x_max = float(X_t.max().item())

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=min(cfg.batch_size, len(X_t)),
        shuffle=True,
    )
    loader_iter = cycle(loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Replay buffer for SGLD (start from random real samples)
    buf_idx = rng.integers(0, len(X_t), size=cfg.buffer_size)
    buffer = X_t[buf_idx].clone()

    model.train()
    for step in range(cfg.train_iterations):
        (x_batch,) = next(loader_iter)

        # Gaussian augmentation used for the invariance term.
        x_aug = x_batch + torch.randn_like(x_batch) * cfg.aug_noise_std

        loss_terms: List[torch.Tensor] = []

        if cfg.use_loss_inv:
            loss_terms.append(cfg.lambda_inv * loss_inv(model, x_batch, x_aug))

        if cfg.use_loss_prior:
            loss_terms.append(cfg.lambda_prior * loss_prior(model, x_batch))

        if cfg.use_loss_gen:
            # 80 % from replay buffer, 20 % fresh noise
            b_idx = rng.integers(0, len(buffer), size=len(x_batch))
            x_init = buffer[b_idx].clone()
            fresh_mask = torch.rand(len(x_init)) < 0.05
            x_init[fresh_mask] = torch.empty_like(x_init[fresh_mask]).uniform_(x_min, x_max)

            x_fake = _sgld_sample(model, x_init, cfg, x_min=x_min, x_max=x_max)
            buffer[b_idx] = x_fake.detach()

            # Warm-up: linearly ramp lambda_gen from 0.1 → 1.0 over first 1000 steps
            warmup_steps = 1000
            if step < warmup_steps:
                lambda_gen_eff = 0.1 + (cfg.lambda_gen - 0.1) * step / warmup_steps
            else:
                lambda_gen_eff = cfg.lambda_gen
            loss_terms.append(lambda_gen_eff * loss_gen(model, x_batch, x_fake))

        if not loss_terms:
            continue

        # L2 regularization (Prior) — stabilise EBM training.
        # prior : L2 penalty on augmented-branch projector outputs z2
        #         (prevents exploding activations in the augmented view).
        z2 = model._embed(x_aug)
        prior = 0.5 * (z2 ** 2).mean()
        loss_terms.append(prior)

        total_loss = torch.stack(loss_terms).sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    return model


def gedi_predict(model: GEDIModel, X: np.ndarray) -> np.ndarray:
    """Return hard cluster assignments from a trained GEDIModel.

    Args:
        model: Trained GEDIModel in eval mode.
        X:     Feature matrix, shape (N, d).

    Returns:
        Integer cluster ids, shape (N,).
    """
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        return model.predict_proba(x_t).argmax(dim=-1).numpy()


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

