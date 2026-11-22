# Copyright (c) 2026 Prior Labs GmbH (tabpfn) and the WeightedPFN authors (modifications).
# Adapted from the tabpfn package, version 9.2.0 (https://github.com/priorlabs/tabpfn), licensed under the Apache
# License, Version 2.0 (see LICENSE-Apache-2.0). Modified by the WeightedPFN authors: ICLAttention.forward,
# _chunked_class_attention and TabPFNV3p5._preprocess_raw (with the empirical-CDF context of _build_ecdf_context) are
# reimplemented with sample weights. See NOTICE.
"""Weighted attention and weighted column preprocessing for TabPFN-3.5 (installed as a runtime patch; tabpfn is not modified).

TabPFN-3.5 attends from rows to the training (context) rows in two places: every ICL block (``ICLAttention``) and the
classification head's retrieval decoder (``_chunked_class_attention``).  WeightedPFN adds log w_i of each training row to
the attention scores of that key, in both places and for all heads.  With weighted preprocessing, the per-column statistics that
TabPFN-3.5 fits on the context inside the network (NaN imputation mean, standardisation mean / std, ECDF context) are computed
from the weighted context as well.

Weights are processed per task: cap at the 0.8 quantile of the context weights (zeros included), normalise to mean one in log
space, take logs; a zero weight masks its row.  With all weights equal the model is the unweighted network.

Patches are inactive (original TabPFN code paths) unless weights were set with ``active_weights``.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

_STATE = {"log_w": None, "prep": False}
_LOCK = threading.RLock()  # one weighted prediction at a time per process (the weights are process-global)
_ORIGINAL = {}


def process_weights(weights, quantile=0.8):
    """Context weights -> float32 log attention bias (quantile cap, mean one in log space, zeros -> -inf)."""
    w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    if w.ndim != 1 or not len(w):
        raise ValueError("sample_weight must be a non-empty vector.")
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite and non-negative.")
    if not (w > 0).any():
        raise ValueError("sample_weight needs at least one positive weight.")
    if quantile is not None:
        if not 0 < quantile <= 1:
            raise ValueError("weight_quantile must be in (0, 1] or None.")
        cap = torch.quantile(w, quantile, interpolation="linear")
        positive = w > 0
        cap = torch.where((cap == 0) & positive.any(), w.masked_fill(~positive, torch.inf).min(), cap)
        w = torch.minimum(w, cap)
    positive = w > 0
    log_w = torch.where(positive, w, torch.ones_like(w)).log().masked_fill(~positive, -torch.inf)
    log_w = log_w - (torch.logsumexp(log_w, 0) - math.log(len(log_w)))
    return log_w.float()


@contextmanager
def active_weights(log_w, prep=True):
    """Use ``log_w`` (or no weighting when None) for the TabPFN forward passes inside the block."""
    with _LOCK:
        _STATE["log_w"], _STATE["prep"] = log_w, bool(prep) and log_w is not None
        try:
            yield
        finally:
            _STATE["log_w"], _STATE["prep"] = None, False


def _weighted_sdpa(q_BSHD, k_BSJD, v_BSJD, softmax_scaling_layer, log_w):
    if softmax_scaling_layer is not None:
        q_BSHD = softmax_scaling_layer(q_BSHD, k_BSJD.shape[1])
    q, k, v = (t.transpose(1, 2) for t in (q_BSHD, k_BSJD.to(q_BSHD.dtype), v_BSJD.to(q_BSHD.dtype)))
    bias = log_w.to(device=q.device, dtype=q.dtype).clamp_min(-1e4)[None, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
                                         enable_gqa=k.shape[1] != q.shape[1])
    return out.transpose(1, 2)


def _icl_attention_forward(self, x_BRE, single_eval_pos, *, cached_kv=None, return_kv=False):
    log_w = _STATE["log_w"]
    N = x_BRE.shape[1] if single_eval_pos is None else single_eval_pos
    if log_w is None or cached_kv is not None or return_kv:
        if log_w is not None:
            raise RuntimeError("WeightedPFN needs a forward without KV cache (fit_mode='fit_preprocessors').")
        return _ORIGINAL["icl_forward"](self, x_BRE, single_eval_pos, cached_kv=cached_kv, return_kv=return_kv)
    if len(log_w) != N:
        raise RuntimeError(f"WeightedPFN: {len(log_w)} context weights for {N} training rows.")
    B, R, _ = x_BRE.shape
    q = self.q_norm(self.q_projection(x_BRE).view(B, R, self.num_heads, self.head_dim))
    x_train = x_BRE[:, :N]
    k = self.k_norm(self.k_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim))
    v = self.v_projection(x_train).view(B, N, self.num_kv_heads, self.head_dim)
    if self.num_kv_heads_test is not None and single_eval_pos is not None and N < R:
        out_train = _weighted_sdpa(q[:, :N], k, v, self.softmax_scaling_layer, log_w)
        h = self.num_kv_heads_test
        out_test = _weighted_sdpa(q[:, N:], k[:, :, :h], v[:, :, :h], self.softmax_scaling_layer, log_w)
        out = torch.cat([out_train, out_test], dim=1)
    else:
        out = _weighted_sdpa(q, k, v, self.softmax_scaling_layer, log_w)
    return self.out_projection(out.reshape(B, R, self.head_dim * self.num_heads)), None


def _class_attention(q_BSHD, k_BJHD, v_BJHT, softmax_scaling_layer=None):
    """tabpfn's _chunked_class_attention with the key bias (same folding of the class axis into the batch)."""
    log_w = _STATE["log_w"]
    if log_w is None:
        return _ORIGINAL["class_attention"](q_BSHD, k_BJHD, v_BJHT, softmax_scaling_layer=softmax_scaling_layer)
    B, S, H, D = q_BSHD.shape
    T, J = v_BJHT.shape[-1], v_BJHT.shape[1]
    if J != len(log_w):
        raise RuntimeError(f"WeightedPFN decoder: {len(log_w)} context weights for {J} training rows.")
    num_chunks = math.ceil(T / D)
    if num_chunks * D - T > 0:
        v_BJHT = F.pad(v_BJHT, (0, num_chunks * D - T))
    v_f = v_BJHT.reshape(B, J, H, num_chunks, D).permute(0, 3, 1, 2, 4).reshape(B * num_chunks, J, H, D).contiguous()
    q_f = q_BSHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, S, H, D).contiguous()
    k_f = k_BJHD.unsqueeze(1).expand(-1, num_chunks, -1, -1, -1).reshape(B * num_chunks, J, H, D).contiguous()
    out = _weighted_sdpa(q_f, k_f, v_f, softmax_scaling_layer, log_w)
    return out.reshape(B, num_chunks, S, H, D).permute(0, 2, 3, 1, 4).reshape(B, S, H, num_chunks * D)[..., :T]


def _weighted_impute_means(x_train_NBC, w_N):
    """Weighted mean of the finite context values per (estimator, column); 0 where a column has none (as TabPFN)."""
    finite = torch.isfinite(x_train_NBC)
    ww = w_N.view(-1, 1, 1) * finite
    total = (torch.where(finite, x_train_NBC.double(), 0.0) * ww).sum(0)
    return torch.nan_to_num(total / ww.sum(0), 0).to(x_train_NBC.dtype)


def _weighted_scaler_fit(x_PBC, w_P):
    """Weighted standardisation statistics of the context rows with w > 0 (already imputed, finite)."""
    total, rows = w_P.sum(), len(w_P)
    x = x_PBC.double()
    mean = (w_P.view(-1, 1, 1) * x).sum(0) / total
    var = (w_P.view(-1, 1, 1) * (x - mean).square()).sum(0) / (total - total / rows if rows > 1 else total)
    std = var.sqrt()
    std = torch.where(std == 0, torch.ones_like(std), std)
    if rows == 1:
        std = torch.ones_like(std)
    return {"mean": mean.to(x_PBC.dtype), "std": std.to(x_PBC.dtype)}


def _weighted_ecdf_context(x_BRiC, num_train, num_buckets, w_N):
    """TabPFN's ``_build_ecdf_context`` with weights: (3, B, C, K) = edges, weight strictly below, weight at most each edge."""
    from tabpfn.architectures import tabpfn_v3_5 as arch
    positive = w_N > 0
    rows_BPC = x_BRiC[:, :num_train][:, positive]
    w_P = w_N[positive].double()
    sorted_BCP, order = rows_BPC.transpose(1, 2).to(arch.ECDF_CONTEXT_DTYPE).contiguous().sort(dim=-1)
    cum = torch.cat([torch.zeros(*order.shape[:2], 1, dtype=torch.float64, device=order.device), w_P[order].cumsum(-1)], -1)
    n = sorted_BCP.shape[-1]
    k = min(num_buckets, n)
    is_new = torch.ones_like(sorted_BCP, dtype=torch.int32)
    is_new[..., 1:] = (sorted_BCP[..., 1:] != sorted_BCP[..., :-1]).to(torch.int32)
    distinct_idx = is_new.cumsum(-1).sub(1)
    num_distinct = distinct_idx[..., -1:] + 1
    steps = torch.arange(k, device=order.device, dtype=arch.ECDF_CONTEXT_DTYPE)
    if k > 1:
        distinct_BCK = steps * ((num_distinct - 1).to(arch.ECDF_CONTEXT_DTYPE) / (k - 1))
        # equal-weight-mass positions; for equal weights exactly TabPFN's row positions round(j (n - 1) / (k - 1))
        rows_K = (steps.double() * ((n - 1) / (k - 1))).round()
        target = (rows_K + 1) * (cum[..., -1:] / n) - 1e-9 * cum[..., -1:]
        position = torch.searchsorted(cum[..., 1:].contiguous(), target.contiguous(), side="left").clamp(max=n - 1)
        targets = torch.where(num_distinct <= k, distinct_BCK.round().to(torch.int32), distinct_idx.gather(-1, position)).contiguous()
    else:
        targets = steps.round().to(torch.int32).expand(*sorted_BCP.shape[:2], k).contiguous()
    below = torch.searchsorted(distinct_idx.contiguous(), targets, side="left")
    at_most = torch.searchsorted(distinct_idx.contiguous(), targets, side="right")
    edges = sorted_BCP.gather(-1, below)
    return torch.stack([edges, cum.gather(-1, below).to(arch.ECDF_CONTEXT_DTYPE), cum.gather(-1, at_most).to(arch.ECDF_CONTEXT_DTYPE)])


def _preprocess_raw(self, x_RiBC, num_train, scaler_cache=None):
    """TabPFNV3p5._preprocess_raw with weighted column statistics while weighted preprocessing is active."""
    from tabpfn.architectures import tabpfn_v3_5 as arch
    log_w = _STATE["log_w"]
    original = _ORIGINAL["preprocess_raw"]
    if log_w is None or not _STATE["prep"] or scaler_cache is not None or num_train <= 0:
        return original(self, x_RiBC, num_train, scaler_cache)
    if len(log_w) != num_train:
        raise RuntimeError(f"WeightedPFN preprocessing: {len(log_w)} context weights for {num_train} training rows.")
    w = log_w.to(device=x_RiBC.device, dtype=torch.float64).exp()
    positive = w > 0
    if bool(positive.all()) and bool((w == w[0]).all()):
        return original(self, x_RiBC, num_train, scaler_cache)  # equal weights: TabPFN's own path
    nan_ind_BRiC = arch._generate_nan_and_inf_indicator(x_RiBC).transpose(0, 1)
    is_finite = torch.isfinite(x_RiBC)
    means = _weighted_impute_means(x_RiBC[:num_train], w)
    x_RiBC = torch.where(is_finite, x_RiBC, means.unsqueeze(0).expand_as(x_RiBC))
    scaler_cache = _weighted_scaler_fit(x_RiBC[:num_train][positive], w[positive])
    x_RiBC = torch.where(is_finite, x_RiBC, scaler_cache["mean"].unsqueeze(0).expand_as(x_RiBC))
    x_imputed_BRiC = x_RiBC.transpose(0, 1)
    scaler_cache[arch._ECDF_CONTEXT_KEY] = _weighted_ecdf_context(x_imputed_BRiC, num_train, self.ecdf_num_buckets, w)
    ecdf_BRiC = arch._in_context_ecdf(x_imputed_BRiC, scaler_cache[arch._ECDF_CONTEXT_KEY])
    x_RiBC = self.standard_scaler.transform(x_RiBC, fitted_cache=scaler_cache)
    return x_RiBC.transpose(0, 1), nan_ind_BRiC, ecdf_BRiC, scaler_cache


def install():
    """Patch tabpfn's TabPFN-3.5 architecture once per process (no effect while no weights are active)."""
    with _LOCK:
        if _ORIGINAL:
            return
        from tabpfn.architectures import tabpfn_v3_5 as arch
        _ORIGINAL.update(icl_forward=arch.ICLAttention.forward, class_attention=arch._chunked_class_attention,
                         preprocess_raw=arch.TabPFNV3p5._preprocess_raw)
        arch.ICLAttention.forward = _icl_attention_forward
        arch._chunked_class_attention = _class_attention
        arch.TabPFNV3p5._preprocess_raw = _preprocess_raw
