from collections import OrderedDict
import torch
import torch.nn as nn

def load_model(model, model_path):
    # Load the original state_dict
    state_dict = torch.load(model_path)

    # Adjust the state_dict to match the model keys with 'module.' prefix
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # add module. prefix
        new_state_dict[name] = v

    # Load the adjusted state_dict
    model.load_state_dict(new_state_dict)


class GaussianIO(nn.Module):
    def __init__(self, z_dim, beta=0.995, eps=1e-6):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.eps = eps

        # Register buffers so they are saved with state_dict but not trained via backprop
        # This solves the "Save/Load" problem automatically
        self.register_buffer('mu0_ema', torch.zeros(z_dim))
        self.register_buffer('mu1_ema', torch.zeros(z_dim))
        self.register_buffer('cov_ema', torch.eye(z_dim) * eps)
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))

    def forward(self, mu, label):
        """
        mu: [B, D]
        label: [B] (0 or 1)
        Returns: mu0_ema, s, Kinv, cov_ema
        """
        if self.training:
            self._update_stats(mu, label)

        # Calculate s and Kinv using the current EMA state
        # We use valid (non-gradient) buffers to create the detector parameters
        s = self.mu1_ema - self.mu0_ema
        Kinv = torch.inverse(self.cov_ema)

        return self.mu0_ema, s, Kinv, self.cov_ema

    def _update_stats(self, mu, label):
        N, D = mu.shape
        label = label.view(-1).long()

        # Handle Empty Batches to prevent NaN
        # If a class is missing in this batch, we skip the update for that class
        mask0 = (label == 0)
        mask1 = (label == 1)

        # Update is only performed if we have at least 1 sample of that class
        if mask0.sum() > 0:
            mu0_b = mu[mask0].mean(0)
            # Use beta only if initialized, else 0.0 (fast start)
            beta0 = self.beta if self.initialized else 0.0
            self.mu0_ema.data.mul_(beta0).add_(mu0_b.detach(), alpha=1 - beta0)

            # Covariance update (simplified to use global centering for stability)
            # Or keep your current logic:
            F0 = mu[mask0] - mu0_b
            if F0.size(0) > 1:
                cov_b = (F0.t() @ F0) / (F0.size(0) - 1)
                self.cov_ema.data.mul_(beta0).add_(cov_b.detach() + self.eps * torch.eye(D, device=mu.device),
                                                   alpha=1 - beta0)

        if mask1.sum() > 0:
            mu1_b = mu[mask1].mean(0)
            beta1 = self.beta if self.initialized else 0.0
            self.mu1_ema.data.mul_(beta1).add_(mu1_b.detach(), alpha=1 - beta1)

        self.initialized.fill_(True)


def normal_IO_train_torch(mu: torch.Tensor,
                          label: torch.Tensor,
                          beta: float = 0.995,
                          eps: float = 1e-6):
    """
    EMA (Exponential Moving Average) update built-in, but updates are done in no_grad() context, 
    entering the computation graph only after detach():
    Returns the exponentially moving averaged mu0_ema, signal vector s, and covariance inverse Kinv.
    """
    label = label.view(-1).long()
    N, D = mu.shape

    # Initialize on first call
    if not hasattr(normal_IO_train_torch, 'initialized'):
        normal_IO_train_torch.mu0_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch.mu1_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch.cov_ema = torch.eye(D,   device=mu.device, dtype=mu.dtype) * eps
        normal_IO_train_torch.initialized = True

    # Batch statistics
    mu0_b = mu[label==0].mean(0)
    mu1_b = mu[label==1].mean(0)
    F0    = mu[label==0] - mu0_b
    cov_b = (F0.t() @ F0) / max(F0.size(0)-1, 1)

    # EMA update without gradient tracking
    with torch.no_grad():
        b = beta
        # print("b =", b)
        m0 = normal_IO_train_torch.mu0_ema
        m1 = normal_IO_train_torch.mu1_ema
        c0 = normal_IO_train_torch.cov_ema

        m0_new = b*m0 + (1-b)*mu0_b
        m1_new = b*m1 + (1-b)*mu1_b
        c_new  = b*c0 + (1-b)*(cov_b + eps*torch.eye(D, device=mu.device, dtype=mu.dtype))

        # Detach and store back to function attributes
        normal_IO_train_torch.mu0_ema = m0_new.detach()
        normal_IO_train_torch.mu1_ema = m1_new.detach()
        normal_IO_train_torch.cov_ema = c_new.detach()

    # Pull the latest EMA state into the computation graph
    mu0_ema = normal_IO_train_torch.mu0_ema.requires_grad_(False)
    mu1_ema = normal_IO_train_torch.mu1_ema.requires_grad_(False)
    cov_ema = normal_IO_train_torch.cov_ema.requires_grad_(False)

    # Construct s, Kinv and let them participate in subsequent computation graph
    s    = mu1_ema - mu0_ema                # (D,)
    Kinv = torch.inverse(cov_ema)           # (D,D)

    return mu0_ema, s, Kinv, cov_ema


def normal_IO_train_torch1(mu: torch.Tensor,
                          label: torch.Tensor,
                          beta: float = 0.995,
                          eps: float = 1e-6):
    """
    EMA (Exponential Moving Average) update built-in, but updates are done in no_grad() context,
    entering the computation graph only after detach():
    Returns the exponentially moving averaged mu0_ema, signal vector s, and covariance inverse Kinv.
    """
    label = label.view(-1).long()
    N, D = mu.shape

    # Initialize on first call
    if not hasattr(normal_IO_train_torch1, 'initialized'):
        normal_IO_train_torch1.mu0_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch1.mu1_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch1.cov_ema = torch.eye(D,   device=mu.device, dtype=mu.dtype) * eps
        normal_IO_train_torch1.initialized = True

    # Batch statistics
    mu0_b = mu[label==0].mean(0)
    mu1_b = mu[label==1].mean(0)

    F0    = mu[label==0] - mu0_b
    cov0 = (F0.t() @ F0) / max(F0.size(0) - 1, 1)

    # Calculate Class 1 Covariance (Crucial for SKS?)
    F1 = mu[label == 1] - mu1_b
    cov1 = (F1.t() @ F1) / max(F1.size(0) - 1, 1)

    # Pooled Covariance
    cov_pooled = 0.5 * (cov0 + cov1)

    # cov_b = (F0.t() @ F0) / max(F0.size(0)-1, 1)

    # EMA update without gradient tracking
    with torch.no_grad():
        b = beta
        # print("b =", b)
        m0 = normal_IO_train_torch1.mu0_ema
        m1 = normal_IO_train_torch1.mu1_ema
        c0 = normal_IO_train_torch1.cov_ema

        m0_new = b*m0 + (1-b)*mu0_b
        m1_new = b*m1 + (1-b)*mu1_b
        c_new  = b*c0 + (1-b)*(cov_pooled + eps*torch.eye(D, device=mu.device, dtype=mu.dtype))

        # Detach and store back to function attributes
        normal_IO_train_torch1.mu0_ema = m0_new.detach()
        normal_IO_train_torch1.mu1_ema = m1_new.detach()
        normal_IO_train_torch1.cov_ema = c_new.detach()

    # Pull the latest EMA state into the computation graph
    mu0_ema = normal_IO_train_torch1.mu0_ema.requires_grad_(False)
    mu1_ema = normal_IO_train_torch1.mu1_ema.requires_grad_(False)
    cov_ema = normal_IO_train_torch1.cov_ema.requires_grad_(False)

    # Construct s, Kinv and let them participate in subsequent computation graph
    s    = mu1_ema - mu0_ema                # (D,)
    Kinv = torch.inverse(cov_ema)           # (D,D)

    return mu0_ema, s, Kinv, cov_ema


# def normal_IO_train_torch1(mu: torch.Tensor,
#                            logvar: torch.Tensor,
#                            label: torch.Tensor,
#                            beta: float = 0.99,
#                            eps: float = 1e-6):
#     """
#     Weighted EMA update of means using batch-level average log-variance,
#     covariance updated with regular EMA, returns mu0_ema, s, Kinv.
#
#     Parameters
#     ----------
#     mu      : Tensor, shape (N, D)
#               Feature means of this batch
#     logvar  : Tensor, shape (N, D)
#               Log-variance corresponding to each sample in this batch
#     label   : Tensor, shape (N,)
#               Binary classification labels 0/1
#     beta    : float
#               EMA decay factor
#     eps     : float
#               Covariance regularization constant
#
#     Returns
#     -------
#     mu0_ema : Tensor, shape (D,)
#               EMA-updated "no-signal" global mean
#     s       : Tensor, shape (D,)
#               Signal vector = mu1_ema − mu0_ema
#     Kinv    : Tensor, shape (D, D)
#               EMA-updated covariance inverse
#     """
#     label = label.view(-1).long()
#     N, D = mu.shape
#
#     # Initialize global EMA state on first call
#     if not hasattr(normal_IO_train_torch1, 'initialized'):
#         normal_IO_train_torch1.mu0_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
#         normal_IO_train_torch1.mu1_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
#         normal_IO_train_torch1.cov_ema = torch.eye(D, device=mu.device, dtype=mu.dtype) * eps
#         normal_IO_train_torch1.initialized = True
#
#     # -- 1. Class means and covariance of this batch -- #
#     mu0_b = mu[label == 0].mean(dim=0)  # (D,)
#     mu1_b = mu[label == 1].mean(dim=0)  # (D,)
#     F0    = mu[label == 0] - mu0_b      # (N0, D)
#     cov_b = (F0.t() @ F0) / max(F0.size(0) - 1, 1)  # (D, D)
#
#     # -- 2. Extract average log-variance for each class from logvar -- #
#     avg_logvar0 = logvar[label == 0].mean(dim=0)  # (D,)
#     avg_logvar1 = logvar[label == 1].mean(dim=0)  # (D,)
#
#     # Convert logvar of each dimension to scalar confidence, then compute mean
#     w0_dim = torch.sigmoid(-avg_logvar0)  # (D,)
#     w1_dim = torch.sigmoid(-avg_logvar1)  # (D,)
#     w0 = w0_dim.mean()                    # scalar ∈ (0,1)
#     w1 = w1_dim.mean()
#
#     # -- 3. EMA update (no_grad) -- #
#     with torch.no_grad():
#         b  = beta
#         m0 = normal_IO_train_torch1.mu0_ema  # (D,)
#         m1 = normal_IO_train_torch1.mu1_ema  # (D,)
#         C0 = normal_IO_train_torch1.cov_ema  # (D, D)
#
#         # Update means: weighted EMA
#         # When w0→1, this batch mean mu0_b has maximum influence; when w0→0, almost preserve m0
#         m0_new = b*m0 + (1 - b)*(w0 * mu0_b + (1 - w0) * m0)
#         m1_new = b*m1 + (1 - b)*(w1 * mu1_b + (1 - w1) * m1)
#
#         # Update covariance: standard EMA
#         C_new = b*C0 + (1 - b)*(cov_b + eps * torch.eye(D, device=mu.device, dtype=mu.dtype))
#
#         # Detach and store back to attributes
#         normal_IO_train_torch1.mu0_ema = m0_new.detach()
#         normal_IO_train_torch1.mu1_ema = m1_new.detach()
#         normal_IO_train_torch1.cov_ema = C_new.detach()
#
#     # -- 4. Calculate s and Kinv, return -- #
#     mu0_ema = normal_IO_train_torch1.mu0_ema.requires_grad_(False)  # (D,)
#     mu1_ema = normal_IO_train_torch1.mu1_ema.requires_grad_(False)  # (D,)
#     cov_ema = normal_IO_train_torch1.cov_ema.requires_grad_(False)  # (D, D)
#
#     s    = mu1_ema - mu0_ema             # (D,)
#     Kinv = torch.inverse(cov_ema)        # (D, D)
#
#     return mu0_ema, s, Kinv


def normal_IO_train_torch2(mu: torch.Tensor,
                           logvar: torch.Tensor,
                           label: torch.Tensor,
                           beta: float = 0.99,
                           eps: float = 1e-6):
    """
    EMA covariance estimation based on "dimension-level shrinkage" and returns mu0_ema, s, Kinv.

    Parameters
    ----------
    mu      : Tensor, shape (N, D)
              Feature means of this batch
    logvar  : Tensor, shape (N, D)
              Log-variance corresponding to each sample in this batch
    label   : Tensor, shape (N,)
              Binary classification labels 0/1
    beta    : float
              EMA decay factor, range (0,1)
    eps     : float
              Covariance diagonal regularization constant, prevents singularity

    Returns
    -------
    mu0_ema : Tensor, shape (D,)
              EMA-updated "no-signal" global mean
    s       : Tensor, shape (D,)
              Signal vector = mu1_ema − mu0_ema
    Kinv    : Tensor, shape (D, D)
              EMA-updated covariance inverse
    """
    label = label.view(-1).long()
    N, D = mu.shape

    # Initialize global EMA state on first call
    if not hasattr(normal_IO_train_torch2, 'initialized'):
        normal_IO_train_torch2.mu0_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch2.mu1_ema = torch.zeros(D, device=mu.device, dtype=mu.dtype)
        normal_IO_train_torch2.cov_ema = torch.eye(D, device=mu.device, dtype=mu.dtype) * eps
        normal_IO_train_torch2.initialized = True

    # -- 1. Class means and covariance of this batch -- #
    mu0_b = mu[label == 0].mean(dim=0)  # (D,)
    mu1_b = mu[label == 1].mean(dim=0)  # (D,)
    F0    = mu[label == 0] - mu0_b      # (N0, D)
    cov_b = (F0.t() @ F0) / max(F0.size(0) - 1, 1)  # (D, D)

    # -- 2. Get average logvar of "no-signal" class from this batch's logvar -- #
    avg_logvar0 = logvar[label == 0].mean(dim=0)  # (D,)

    # Use sigmoid(-avg_logvar0) to convert logvar to shrinkage coefficients alpha_i between (0,1)
    alpha = torch.sigmoid(-avg_logvar0)  # (D,)

    # -- 3. EMA update: update global means and covariance under no_grad() -- #
    with torch.no_grad():
        b  = beta
        m0 = normal_IO_train_torch2.mu0_ema  # (D,)
        m1 = normal_IO_train_torch2.mu1_ema  # (D,)
        C0 = normal_IO_train_torch2.cov_ema  # (D, D)

        # -- Update global means (standard EMA) -- #
        m0_new = b*m0 + (1 - b)*mu0_b
        m1_new = b*m1 + (1 - b)*mu1_b

        # -- Dimension-level shrinkage update covariance -- #
        diag_cov_b = cov_b.diagonal()  # (D,)
        diag_C0    = C0.diagonal()     # (D,)

        # Use alpha_i to shrink each dimension: new_var_i = α_i * cov_b_ii + (1 - α_i) * diag_C0_i
        new_diag = alpha * diag_cov_b + (1 - alpha) * diag_C0 + eps  # (D,)

        # Construct new C_new: preserve old off-diagonal, update diagonal to new_diag
        C_new = C0.clone()
        idx = torch.arange(D, device=mu.device)
        C_new[idx, idx] = new_diag

        # Detach and store back to function attributes
        normal_IO_train_torch2.mu0_ema = m0_new.detach()
        normal_IO_train_torch2.mu1_ema = m1_new.detach()
        normal_IO_train_torch2.cov_ema = C_new.detach()

    # -- 4. Pull into computation graph, construct s and Kinv -- #
    mu0_ema = normal_IO_train_torch2.mu0_ema.requires_grad_(False)  # (D,)
    mu1_ema = normal_IO_train_torch2.mu1_ema.requires_grad_(False)  # (D,)
    cov_ema = normal_IO_train_torch2.cov_ema.requires_grad_(False)  # (D, D)

    s    = mu1_ema - mu0_ema             # (D,)
    Kinv = torch.inverse(cov_ema)        # (D, D)

    return mu0_ema, s, Kinv


def normal_IO_test_torch(mu: torch.Tensor,
                         mu0_mean: torch.Tensor,
                         s: torch.Tensor,
                         Kinv: torch.Tensor):
    """
    Testing phase: compute complete test statistic λ = 2 s^T K^{-1}(f - mu0_mean) - s^T K^{-1}s using PyTorch tensors.

    Parameters
    ----------
    mu        : Tensor, shape (M, D)
        Feature vectors of test samples
    mu0_mean  : Tensor, shape (D,)
        No-signal mean estimated during training phase
    s         : Tensor, shape (D,)
        Signal vector estimated during training phase
    Kinv      : Tensor, shape (D, D)
        Covariance inverse estimated during training phase

    Returns
    -------
    lambda_full : Tensor, shape (M,)
        Test statistic for each test sample
    """
    # Centering
    F = mu - mu0_mean.unsqueeze(0)         # (M, D)

    # Linear term: 2 * s^T Kinv F^T  -> (M,)
    lin = 2 * torch.einsum('i,ij,nj->n', s, Kinv, F)

    # Constant term: s^T Kinv s
    const = s @ (Kinv @ s)                 # scalar

    lambda_full = lin - const
    return lambda_full
