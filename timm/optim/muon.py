""" Muon Optimizer

Improved Muon optimizer implementation with flexible handling of high-dimensional tensors.

Combines PyTorch-style structure with options for:
- Batched spatial processing for convolutions in addition to flatten
- Optional spatial normalization
- Selectable coefficient presets
- Automatic fallback to AdamW for 1D / scalar parameters (biases, norms, etc.) and optional fallback via param groups
- AdaMuon (https://arxiv.org/abs/2507.11005)
- mUP eps damping factor (https://arxiv.org/abs/2512.05620v1)

TODO look into mUP LR scaling and independent weight-decay scale

Based on implementation by Keller Jordan, see
- https://github.com/KellerJordan/Muon/blob/master/muon.py
- https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
- https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py
- https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py

Hacked together by Ross Wightman
"""
import logging
import numbers
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ._types import ParamsT
from .adamw import adamw
from .nadamw import nadamw

_logger = logging.getLogger(__name__)

# Constants from Keller Jordan's Muon
MUON_EPS = 1e-7
DEFAULT_NS_STEPS = 5

_COEFFICIENTS = {
    "original": [
        # Keller Jordan's Muon https://kellerjordan.github.io/posts/muon/
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # From https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express https://arxiv.org/abs/2505.16932
        # From https://github.com/NoahAmsel/PolarExpress/tree/main with safety 1e-2
        (8.237312490495555, -23.157747414558198, 16.680568411445915),
        (4.082441999064835, -2.893047735332586, 0.5252849256975648),
        (3.9263479922546582, -2.8547468034765298, 0.5318022422894988),
        (3.2982187133085143, -2.424541981026706, 0.48632008358844075),
        (2.2970369434552573, -1.63662558125903, 0.4002628455953627),
        (1.8763805351440397, -1.2347896577722228, 0.35891887501668385),
        (1.8564423485617974, -1.2132449880935525, 0.3568003487825883),
        (1.8749994008682747, -1.2499988017229169, 0.3749994008546422),
    ],
    "polar_express_safer": [
        # from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
        # w/ safety 2e-2
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.0429299351667245, -2.808917465908704, 0.5000178451051299),
        (3.8916678022926563, -2.7724841532176825, 0.5060648178503389),
        (3.285753657755658, -2.3681294933425394, 0.46449024233003117),
        (2.3005307116270983, -1.6111665557258408, 0.3833374427545273),
        (1.8631210546382593, -1.2042160621002727, 0.3421879560523383),
        (1.8382572152247512, -1.1779263289537742, 0.3396513038637379),
        (1.8749999923301852, -1.2499999836060613, 0.374999991275876),
    ],
}


NSCoeff = Union[str, Tuple[float, float, float], List[Tuple[float, float, float]]]


def scale_eps_for_ns(
        eps: float,
        shape: Tuple[int, ...],
) -> float:
    """Scale epsilon for Newton-Schulz based on matrix dimensions (μP-style).

    For μP compatibility, epsilon should scale as eps * sqrt(din/dout) to maintain
    consistent damping behavior across different model widths.

    Reference: https://arxiv.org/abs/2512.05620

    Args:
        eps: Base epsilon value
        shape: Shape of the matrix (out, in) or (batch, out, in)

    Returns:
        Scaled epsilon value
    """
    # Get din, dout from shape (handle both 2D and 3D batched)
    # FIXME TBD paper includes depth in the damping scale, e.g: eps * (din / dout) ** 0.5 / N
    dout, din = (shape[-2], shape[-1])
    return eps * (din / dout) ** 0.5


def zeropower_via_newtonschulz(
        G: torch.Tensor,
        steps: int,
        coefficients: List[Tuple[float, float, float]],
        eps: float = MUON_EPS,
        safety_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        scale_eps: bool = False,
) -> torch.Tensor:
    """Newton-Schulz quintic iteration to compute the zeroth power / orthogonalization of gradient.

    Supports batched operation over leading dimensions.

    See
    - https://github.com/KellerJordan/Muon/blob/master/muon.py
    - https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py
    - https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py

    Args:
        G: Input gradient tensor of shape (m, n) or (batch, m, n)
        steps: Number of Newton-Schulz iterations
        coefficients: Coefficients (a, b, c) for the iteration
        eps: Numerical stability epsilon for norm
        safety_factor: Multiplicative safety factor for norm (1.01 is common safety value in 'polar express' variants)
        dtype: Computation dtype
        scale_eps: If True, scale epsilon by sqrt(din/dout) for μP compatibility

    Returns:
        Orthogonalized tensor of same shape as G
    """
    assert G.ndim in (2, 3), f"Input must be 2D or 3D, got {G.ndim}D. Flatten batch dims first."
    num_cs = len(coefficients)
    assert num_cs >= 1 and len(coefficients[0]) == 3
    # match coefficients with # of steps, truncate or repeat last
    coeff_sequence = coefficients[:steps] if steps <= num_cs else \
        coefficients + [coefficients[-1]] * (steps - num_cs)

    # Scale epsilon by sqrt(din/dout) for μP compatibility if requested
    if scale_eps:
        eps = scale_eps_for_ns(eps, G.shape)

    X = G.to(dtype=dtype, copy=True)

    # Transpose if needed (operate on dimension with fewer elements)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    if scale_eps:
        # more of a damping factor in this case, use add instead of clamp
        X.div_(X.norm(2, dim=(-2, -1), keepdim=True).mul(safety_factor).add_(eps))
    else:
        X.div_(X.norm(2, dim=(-2, -1), keepdim=True).mul(safety_factor).clamp_min_(eps))

    # Batched vs unbatched fused MM
    mm_fn = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Pre-allocate
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Perform Newton-Schulz iterations
    for a, b, c in coeff_sequence:
        mm_fn(A, X, X.mT, beta=0.0, alpha=1.0, out=A)  # A = X @ X.mT
        mm_fn(A, A, A, beta=b, alpha=c, out=B)  # B = b * A + c * A @ A
        mm_fn(X, B, X, beta=a, alpha=1.0, out=C)  # C = a * X + B @ X
        X, C = C, X  # swap refs to avoid copy

    if transposed:
        X = X.mT

    return X


def get_lr_scale(
        param_shape: torch.Size,
        adjust_lr_fn: str = "match_rms_adamw",
) -> float:
    """Adjust learning rate based on parameter shape for Muon.

    Args:
        param_shape: Shape of the parameter tensor
        adjust_lr_fn: Scaling function name
            - "original": sqrt(max(1, out/in)) - Original Muon impl
            - "match_rms_adamw": 0.2 * sqrt(max(out, in)) - Kimi scaling
            - "rms_to_rms": sqrt(out/in) - Scion/Bernstein scaling
    """
    out_chs, in_chs = (param_shape[-2], param_shape[-1]) if len(param_shape) > 1 else (1., 1.)

    if adjust_lr_fn == "original":
        # Original Muon impl (https://kellerjordan.github.io/posts/muon/)
        return max(1, out_chs / in_chs) ** 0.5
    elif adjust_lr_fn == "match_rms_adamw":
        # Kimi (https://arxiv.org/abs/2502.16982)
        return 0.2 * max(out_chs, in_chs) ** 0.5
    elif adjust_lr_fn == "rms_to_rms":
        # Scion (https://arxiv.org/abs/2502.07529, https://github.com/LIONS-EPFL/scion)
        # Bernstein et al. (https://jeremybernste.in/writing/deriving-muon)
        return (out_chs / in_chs) ** 0.5
    else:
        assert False, f'Invalid scaling function "{adjust_lr_fn}" for Muon'


def get_adamuon_lr_scale(
        param_shape: torch.Size,
        adjust_lr_fn: str = "match_rms_adamw",
) -> Tuple[float, bool]:
    """Adjust learning rate based on parameter shape for AdaMuon.

    Args:
        param_shape: Shape of the parameter tensor
        adjust_lr_fn: Scaling function name

    Returns:
        Tuple of (scale_factor, use_rms_norm)
    """
    out_chs, in_chs = (param_shape[-2], param_shape[-1]) if len(param_shape) > 1 else (1., 1.)

    if adjust_lr_fn == "match_rms_adamw":
        # AdaMuon paper: normalize by RMS, then scale by 0.2 * sqrt(numel)
        # https://arxiv.org/abs/2507.11005
        return 0.2 * (out_chs * in_chs) ** 0.5, True
    elif adjust_lr_fn == "rms_to_rms":
        return (out_chs / in_chs) ** 0.5, False
    elif adjust_lr_fn == "rsqrt_in":
        return in_chs ** -0.5, False
    else:
        assert False, f'Invalid scaling function "{adjust_lr_fn}" for AdaMuon'


def _is_suitable_for_muon(
        param: torch.Tensor,
        min_dim_size: int = 4,
        max_aspect_ratio: float = 128.,
        return_reason: bool = False,
) -> Union[bool, Tuple[bool, str]]:
    """Check if a parameter is suitable for Muon optimization.

    Args:
        param: Parameter tensor
        min_dim_size: Minimum size for non-unit dimensions
        max_aspect_ratio: Maximum allowed aspect ratio
        return_reason: If True, return (bool, reason_string), else just bool (faster)

    Returns:
        If return_reason=False: bool indicating suitability
        If return_reason=True: Tuple of (is_suitable, reason_string)

    Examples:
        (64, 128) -> True (or (True, "ok") if return_reason=True)
        (96, 3, 4, 4) -> True - will be flattened to (96, 48)
        (4, 2048) -> False - extreme aspect ratio
        (64,) -> False - insufficient dims
        (1, 196, 768) -> False - leading unit dims

    NOTE: these rules were created to balance complexity with covering common timm model cases
    Please let me know if there are non-optimal cases that you run into.
    """

    s = param.shape
    # Must have at least 2 non-unit dimensions
    if param.ndim < 2 or sum(1 for dim_size in s if dim_size > 1) < 2:
        return (False, "insufficient_dims") if return_reason else False

    # Unit dimension in first two positions indicates:
    # - Position embeddings (1, seq, dim)
    # - Depthwise convs (out, 1, h, w)
    # - Other degenerate cases possibly not caught by first rule
    if s[0] == 1 or s[1] == 1:
        return (False, "leading_unit_dims") if return_reason else False

    if param.ndim >= 3:
        # For 3D+ tensors, check what dimensions will be AFTER flattening
        # since that's what gets passed to Newton-Schulz iteration
        # Flatten mode: (out, in, *spatial) -> (out, in * spatial_prod)
        out_ch = s[0]
        in_ch_with_spatial = 1
        for d in s[1:]:
            in_ch_with_spatial *= d
        check_dims = (out_ch, in_ch_with_spatial)
    else:
        # For 2D tensors, check as-is
        check_dims = s

    # Both dims should be >= minimum size
    min_size = min(check_dims)
    if min_size < min_dim_size:
        if return_reason:
            return False, f"min_dim_too_small:{min_size}"
        return False

    # Aspect ratio shouldn't be too extreme
    max_size = max(check_dims)
    aspect_ratio = max_size / min_size
    if aspect_ratio > max_aspect_ratio:
        if return_reason:
            return False, f"extreme_aspect_ratio:{aspect_ratio:.1f}"
        return False

    return (True, "ok") if return_reason else True


def reshape_for_muon(
        tensor: torch.Tensor,
        mode: str = "flatten",
) -> Tuple[torch.Tensor, torch.Size]:
    """Reshape high-dimensional tensor for Muon processing.

    Args:
        tensor: Input tensor of shape (out, in, *spatial)
        mode: How to handle spatial dimensions
            - "flatten": Flatten spatial into output dimension (out, in*H*W)
            - "batched": Batch over spatial positions (spatial_prod, out, in) for per-position orthogonalization

    Returns:
        Reshaped tensor and original shape for restoration
    """
    original_shape = tensor.shape
    if tensor.ndim == 2:
        return tensor, original_shape
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got {tensor.ndim}")

    out_ch, in_ch = tensor.shape[:2]
    if mode == "flatten":
        # Flatten: (out, in, *spatial) -> (out, in * spatial_prod)
        return tensor.reshape(out_ch, -1), original_shape
    elif mode == "batched":
        # Batched: (out, in, *spatial) -> (spatial_prod, out, in)
        # Move spatial dimension to front so zeropower_via_newtonschulz batches over it
        reshaped = tensor.reshape(out_ch, in_ch, -1)  # (out, in, spatial_prod)
        reshaped = reshaped.permute(2, 0, 1)  # (spatial_prod, out, in)
        return reshaped, original_shape
    else:
        raise ValueError(f"Unknown mode: {mode}")


def muon(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_bufs: List[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        ns_coefficients: NSCoeff,
        eps: float,
        safety_factor: float,
        adjust_lr_fn: Optional[str],
        conv_mode: str,
        normalize_spatial: bool,
        scale_eps: bool,
) -> None:
    """Functional API that performs Muon algorithm computation."""
    _single_tensor_muon(
        params,
        grads,
        momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        ns_coefficients=ns_coefficients,
        eps=eps,
        safety_factor=safety_factor,
        adjust_lr_fn=adjust_lr_fn,
        conv_mode=conv_mode,
        normalize_spatial=normalize_spatial,
        scale_eps=scale_eps,
    )


def adamuon(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_bufs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        beta2: float,
        ns_steps: int,
        ns_coefficients: NSCoeff,
        eps: float,
        safety_factor: float,
        adjust_lr_fn: Optional[str],
        conv_mode: str,
        normalize_spatial: bool,
        scale_eps: bool,
) -> None:
    """Functional API that performs AdaMuon algorithm computation.

    AdaMuon extends Muon with element-wise second moment estimation applied
    to orthogonalized update directions, providing Adam-like adaptive scaling
    while preserving Muon's geometric benefits.

    Reference: https://arxiv.org/abs/2507.11005
    """
    _single_tensor_adamuon(
        params,
        grads,
        momentum_bufs,
        exp_avg_sqs,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        beta2=beta2,
        ns_steps=ns_steps,
        ns_coefficients=ns_coefficients,
        eps=eps,
        safety_factor=safety_factor,
        adjust_lr_fn=adjust_lr_fn,
        conv_mode=conv_mode,
        normalize_spatial=normalize_spatial,
        scale_eps=scale_eps,
    )


def _single_tensor_muon(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_bufs: List[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        ns_coefficients: NSCoeff,
        eps: float,
        safety_factor: float,
        adjust_lr_fn: Optional[str],
        conv_mode: str,
        normalize_spatial: bool,
        scale_eps: bool,
) -> None:
    """Single tensor Muon update."""
    ns_coefficients = resolve_ns_coefficients(ns_coefficients, _COEFFICIENTS)

    for i, param in enumerate(params):
        grad = grads[i]
        momentum_buf = momentum_bufs[i]

        # Apply weight decay
        param.mul_(1 - lr * weight_decay)

        # Update momentum buffer
        momentum_buf.lerp_(grad, 1. - momentum)
        update = grad.lerp_(momentum_buf, momentum) if nesterov else momentum_buf.clone()

        # Reshape for processing (handle 3D+ tensors like conv weights)
        if update.ndim >= 3:
            update_reshaped, original_shape = reshape_for_muon(update, mode=conv_mode)
        else:
            update_reshaped = update
            original_shape = update.shape

        # Apply Newton-Schulz orthogonalization
        update_ortho = zeropower_via_newtonschulz(
            update_reshaped,
            ns_steps,
            ns_coefficients,
            eps=eps,
            safety_factor=safety_factor,
            scale_eps=scale_eps,
        )

        # Adjust learning rate based on parameter shape
        if adjust_lr_fn:
            scale = get_lr_scale(update_ortho.shape, adjust_lr_fn)
        else:
            scale = 1.0

        # Apply spatial normalization and permute back if in batched mode
        if conv_mode == "batched" and update_ortho.ndim >= 3:
            if normalize_spatial:
                scale *= update_ortho.shape[0] ** -0.5
            # Permute back: (spatial_prod, out, in) -> (out, in, spatial_prod)
            update_ortho = update_ortho.permute(1, 2, 0)

        # Reshape back to original shape
        update_ortho = update_ortho.reshape(original_shape)

        # Apply update
        param.add_(update_ortho, alpha=-lr * scale)


def _single_tensor_adamuon(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_bufs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        beta2: float,
        ns_steps: int,
        ns_coefficients: NSCoeff,
        eps: float,
        safety_factor: float,
        adjust_lr_fn: Optional[str],
        conv_mode: str,
        normalize_spatial: bool,
        scale_eps: bool,
) -> None:
    """Single tensor AdaMuon update.

    AdaMuon applies second-moment estimation to the orthogonalized directions,
    then rescales using RMS-alignment to maintain stable step sizes.

    Algorithm:
        1. Update momentum buffer: M = β₁·M + (1-β₁)·G
        2. Orthogonalize: O = Newton-Schulz(M) or Newton-Schulz(nesterov_update)
        3. Update second moment: v = β₂·v + (1-β₂)·O²
        4. Bias correct: v̂ = v/(1-β₂^t)
        5. Adaptive scaling: Ô = O / (√v̂ + ε)
        6. RMS-aligned rescaling and apply update
    """
    ns_coefficients = resolve_ns_coefficients(ns_coefficients, _COEFFICIENTS)

    for i, param in enumerate(params):
        grad = grads[i]
        momentum_buf = momentum_bufs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Increment step
        step_t += 1
        step = step_t.item()

        # Apply weight decay (decoupled)
        param.mul_(1 - lr * weight_decay)

        # Update momentum buffer
        momentum_buf.lerp_(grad, 1. - momentum)
        update = grad.lerp_(momentum_buf, momentum) if nesterov else momentum_buf.clone()

        # Reshape for processing (handle 3D+ tensors like conv weights)
        if update.ndim >= 3:
            update_reshaped, original_shape = reshape_for_muon(update, mode=conv_mode)
        else:
            update_reshaped = update
            original_shape = update.shape

        # Apply Newton-Schulz orthogonalization
        update_ortho = zeropower_via_newtonschulz(
            update_reshaped,
            ns_steps,
            ns_coefficients,
            eps=eps,
            safety_factor=safety_factor,
            scale_eps=scale_eps,
        )

        # Reshape back to original shape for second moment tracking
        if conv_mode == "batched" and update_ortho.ndim >= 3:
            # Permute back: (spatial_prod, out, in) -> (out, in, spatial_prod)
            update_ortho = update_ortho.permute(1, 2, 0)
        update_ortho = update_ortho.reshape(original_shape)

        # Update second moment on orthogonalized directions (element-wise)
        exp_avg_sq.mul_(beta2).addcmul_(update_ortho, update_ortho, value=1.0 - beta2)

        # Get shape-based LR scaling and whether to apply RMS normalization
        if adjust_lr_fn:
            scale, use_rms_norm = get_adamuon_lr_scale(update_ortho.shape, adjust_lr_fn)
        else:
            scale, use_rms_norm = 1.0, False

        if use_rms_norm:
            # Bias correction not needed if scaling by norm
            denom = exp_avg_sq.sqrt().add_(eps)
        else:
            # Bias correction for second moment
            bias_correction2 = 1.0 - beta2 ** step
            denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

        # Adaptive scaling: divide by sqrt of bias-corrected second moment
        # This is the key AdaMuon modification
        update_adaptive = update_ortho / denom

        # RMS-aligned rescaling: normalize by update norm, then scale by shape factor
        # Used by AdaMuon paper approach (match_rms_adamw), not by μP approach (rms_to_rms)
        if use_rms_norm:
            # eq(8) in AdaMuon paper, 0.2 / RMS(update) = 0.2 * sqrt(ndim) / frob(update)
            update_norm = update_adaptive.norm().add_(eps)
            update_adaptive = update_adaptive / update_norm

        # Apply spatial normalization if in batched mode
        if conv_mode == "batched" and len(original_shape) >= 3:
            if normalize_spatial:
                spatial_prod = 1
                for d in original_shape[2:]:
                    spatial_prod *= d
                scale *= spatial_prod ** -0.5

        # Apply update
        param.add_(update_adaptive, alpha=-lr * scale)


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz

    Combines Muon for 2D+ parameters (weight matrices) with AdamW for 1D parameters (biases, norms) and
    parameter groups with 'use_fallback=True' set (or 'use_muon=False' for compatibility).

    Supports two algorithms:
    - "muon": Standard Muon algorithm with momentum + orthogonalization
    - "adamuon": AdaMuon algorithm that adds element-wise second moment estimation
                 to orthogonalized directions for Adam-like adaptive scaling
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 0.02,
            weight_decay: float = 0,
            momentum: float = 0.95,
            nesterov: bool = False,
            ns_steps: int = DEFAULT_NS_STEPS,
            ns_coefficients: NSCoeff = "quintic",
            eps: float = MUON_EPS,
            safety_factor: float = 1.0,
            adjust_lr_fn: Optional[str] = "match_rms_adamw",
            conv_mode: str = "flatten",
            normalize_spatial: bool = True,
            adamw_lr: Optional[float] = None,
            betas: Tuple[float, float] = (0.9, 0.95),
            algo: str = "muon",
            scale_eps: bool = False,
            verbose: bool = False,
    ):
        """ Create Muon optimizer.
        Args:
            params: Iterable of parameters or dicts defining parameter groups
            lr: Learning rate (default: 0.02 for Muon parameters)
            weight_decay: Weight decay coefficient
            momentum: Momentum factor for Muon
            nesterov: Whether to use Nesterov momentum
            ns_steps: Number of Newton-Schulz iterations
            ns_coefficients: Coefficients for NS iteration
            eps: Numerical stability epsilon
            safety_factor: Multiplicative safety factor for NS norm
            adjust_lr_fn: LR adjustment function - "original", "match_rms_adamw", or "rms_to_rms".
                For adamuon mode, can set to None to disable (RMS rescaling handles scaling).
            conv_mode: How to handle convolutions - "flatten" or "batched"
            normalize_spatial: Whether to normalize by sqrt(spatial_size) in batched mode
            adamw_lr: Learning rate for AdamW (1D params), defaults to lr if not specified
            betas: Beta coefficients - (beta1, beta2) where beta1 is used for AdamW fallback
                and beta2 is used for both AdamW fallback and AdaMuon second moment
            algo: Algorithm - "muon" for standard Muon, "adamuon" for AdaMuon with
                adaptive second moment estimation (https://arxiv.org/abs/2507.11005)
            scale_eps: If True, scale epsilon by sqrt(din/dout) in Newton-Schulz for μP
                compatibility (https://arxiv.org/abs/2512.05620)
            verbose: Log parameter routing decisions (Muon vs AdamW)

        Example:
            ```python
            # Simple usage - automatically uses Muon for 2D+ params, AdamW for 1D
            optimizer = Muon(model.parameters(), lr=0.02)

            # Use AdaMuon algorithm for adaptive scaling
            optimizer = Muon(model.parameters(), lr=6e-4, algo="adamuon")

            # Manual control over parameter groups
            optimizer = Muon([
                {'params': weight_matrices, 'lr': 0.02},
                {'params': biases, 'use_fallback': True, 'lr': 3e-4}, # use AdamW if use_fallback=True
            ])
            ```
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if conv_mode not in ["flatten", "batched"]:
            raise ValueError(f"Invalid conv_mode: {conv_mode}")
        if algo not in ["muon", "adamuon"]:
            raise ValueError(f"Invalid algo: {algo}. Must be 'muon' or 'adamuon'")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            safety_factor=safety_factor,
            adjust_lr_fn=adjust_lr_fn,
            conv_mode=conv_mode,
            normalize_spatial=normalize_spatial,
            adamw_lr=adamw_lr if adamw_lr is not None else lr,
            betas=betas,
            algo=algo,
            scale_eps=scale_eps,
            verbose=verbose,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('algo', 'muon')
            group.setdefault('scale_eps', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        verbose = self.defaults.get("verbose", False)

        # Tracking for logging (populated on first encounter of each param)
        muon_count = 0
        adamw_count = 0
        routing_reasons = {} if verbose else None

        for group in self.param_groups:
            algo = group.get("algo", "muon")

            # Separate params into Muon and AdamW groups
            muon_params = []
            muon_grads = []
            muon_momentum_bufs = []
            # Additional state for adamuon mode
            muon_exp_avg_sqs = []
            muon_state_steps = []

            adamw_params = []
            adamw_grads = []
            adamw_exp_avgs = []
            adamw_exp_avg_sqs = []
            adamw_state_steps = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                # Determine routing on first encounter (cache in state)
                if "use_muon" not in state:
                    # Check explicit flags first (support both 'use_fallback' and 'use_muon' for compatibility)
                    reason = None
                    if group.get("use_fallback", False):
                        # use_fallback=True means use AdamW (use_muon=False)
                        state["use_muon"] = False
                        if verbose:
                            reason = "use_fallback_flag"
                    elif "use_muon" in group:
                        # Explicit use_muon flag for compatibility with other Muon implementations
                        state["use_muon"] = group["use_muon"]
                        if verbose:
                            reason = "use_muon_flag"
                    else:
                        # Check shape suitability
                        if verbose:
                            suitable, reason = _is_suitable_for_muon(p, return_reason=True)
                        else:
                            suitable = _is_suitable_for_muon(p, return_reason=False)
                        state["use_muon"] = suitable

                    # Track routing decision for logging
                    if routing_reasons is not None and reason is not None:
                        shape_str = "x".join(str(s) for s in p.shape)
                        if shape_str not in routing_reasons:
                            routing_reasons[shape_str] = []
                        routing_reasons[shape_str].append(reason)

                # Use cached routing decision
                use_muon = state["use_muon"]
                if use_muon:
                    # Collect Muon params
                    muon_params.append(p)
                    muon_grads.append(p.grad)
                    muon_count += 1

                    # State initialization for Muon/AdaMuon
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    muon_momentum_bufs.append(state["momentum_buffer"])

                    # Additional state for adamuon mode
                    if algo == "adamuon":
                        if "step" not in state:
                            state["step"] = torch.tensor(0.)
                            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        muon_exp_avg_sqs.append(state["exp_avg_sq"])
                        muon_state_steps.append(state["step"])
                else:
                    # Collect AdamW/NAdamW params
                    adamw_params.append(p)
                    adamw_grads.append(p.grad)
                    adamw_count += 1

                    # State initialization for AdamW
                    if "step" not in state:
                        state["step"] = torch.tensor(0.)
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    adamw_exp_avgs.append(state["exp_avg"])
                    adamw_exp_avg_sqs.append(state["exp_avg_sq"])
                    adamw_state_steps.append(state["step"])

            # Apply Muon/AdaMuon updates
            if muon_params:
                if algo == "adamuon":
                    _, beta2 = group["betas"]
                    adamuon(
                        muon_params,
                        muon_grads,
                        muon_momentum_bufs,
                        muon_exp_avg_sqs,
                        muon_state_steps,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        beta2=beta2,
                        ns_steps=group["ns_steps"],
                        ns_coefficients=group["ns_coefficients"],
                        eps=group["eps"],
                        safety_factor=group["safety_factor"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        conv_mode=group["conv_mode"],
                        normalize_spatial=group["normalize_spatial"],
                        scale_eps=group["scale_eps"],
                    )
                else:
                    muon(
                        muon_params,
                        muon_grads,
                        muon_momentum_bufs,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                        ns_coefficients=group["ns_coefficients"],
                        eps=group["eps"],
                        safety_factor=group["safety_factor"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        conv_mode=group["conv_mode"],
                        normalize_spatial=group["normalize_spatial"],
                        scale_eps=group["scale_eps"],
                    )

            # Apply AdamW updates
            if adamw_params:
                beta1, beta2 = group["betas"]
                if group["nesterov"]:
                    # use nadamw for fallback optimizer if nesterov is enabled
                    nadamw(
                        adamw_params,
                        adamw_grads,
                        adamw_exp_avgs,
                        adamw_exp_avg_sqs,
                        adamw_state_steps,
                        foreach=None,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["adamw_lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        caution=False,
                        maximize=False,
                        capturable=False,
                        max_lr=None,
                    )
                else:
                    adamw(
                        adamw_params,
                        adamw_grads,
                        adamw_exp_avgs,
                        adamw_exp_avg_sqs,
                        [],  # max_exp_avg_sqs (not using amsgrad)
                        adamw_state_steps,
                        foreach=None,
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["adamw_lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        caution=False,
                        maximize=False,
                        capturable=False,
                        max_lr=None,
                )

        # Log routing summary when we have new routing decisions
        if routing_reasons and len(routing_reasons) > 0:
            # Concise summary
            _logger.info(f"Muon parameter routing: {muon_count} Muon, {adamw_count} AdamW")

            # Group by reason for detailed breakdown
            reason_groups = {}
            for shape_str, reasons in sorted(routing_reasons.items()):
                for reason in reasons:
                    if reason not in reason_groups:
                        reason_groups[reason] = []
                    reason_groups[reason].append(shape_str)

            # Log summary counts per reason
            reason_summary = []
            for reason, shapes in sorted(reason_groups.items()):
                reason_summary.append(f"{reason}={len(shapes)}")
            _logger.info(f"  Breakdown: {', '.join(reason_summary)}")

            # Detailed breakdown at INFO level
            if _logger.isEnabledFor(logging.INFO):
                for reason, shapes in sorted(reason_groups.items()):
                    optimizer_name = "Muon" if reason == "ok" else "AdamW"
                    _logger.info(f"    {reason} -> {optimizer_name}:")
                    for shape in shapes[:10]:
                        _logger.info(f"      {shape}")
                    if len(shapes) > 10:
                        _logger.info(f"      ... and {len(shapes) - 10} more")

        return loss


def resolve_ns_coefficients(
        value: Union[str, Sequence[float], Sequence[Sequence[float]]],
        presets: Mapping[str, Sequence[Sequence[float]]]
) -> List[Tuple[float, float, float]]:
    # tiny helpers (kept inline for succinctness)
    is_seq = lambda x: isinstance(x, Sequence) and not isinstance(x, (str, bytes))
    is_real = lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool)

    def as_coeff(x: Sequence[float]) -> Tuple[float, float, float]:
        if not is_seq(x) or len(x) != 3 or not all(is_real(v) for v in x):
            raise ValueError(f"Coefficient must be length-3 of real numbers, got: {x!r}")
        a, b, c = x  # type: ignore[misc]
        return float(a), float(b), float(c)

    if isinstance(value, str):
        if value not in presets:
            valid = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown coefficients preset '{value}'. Valid options: {valid}")
        seq = presets[value]
        if not is_seq(seq) or len(seq) == 0:
            raise ValueError(f"Preset '{value}' is empty or invalid")
        return [as_coeff(item) for item in seq]  # validate & cast

    if not is_seq(value):
        raise TypeError(
            "Coefficients must be a preset name (str), a 3-sequence (a,b,c), "
            "or a sequence of 3-sequences."
        )

    # Decide single triple vs list-of-triples by structure
    if len(value) == 3 and all(is_real(v) for v in value):  # type: ignore[index]
        return [as_coeff(value)]  # single triple -> wrap

    # Otherwise treat as list/tuple of triples
    out = []
    for i, item in enumerate(value):  # type: ignore[assignment]
        if not is_seq(item):
            raise TypeError(f"Item {i} is not a sequence: {item!r}")
        out.append(as_coeff(item))
    if not out:
        raise ValueError("Coefficient list cannot be empty")
    return out