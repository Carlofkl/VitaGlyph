"""Inference-only attention operations for VitaGlyph's final generation stage."""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F


def attention_saliency(query, key, scale, chunk_size=128):
    """Mean attention received by each spatial key; never allocate the full NxN map.

    Inputs are [batch, heads, tokens, head_dim]. Accumulate in float32 so
    small attention probabilities remain usable with half-precision models.
    """
    result = torch.zeros(key.shape[0], key.shape[2], device=key.device)
    keys = key.float().transpose(-1, -2)
    for start in range(0, query.shape[2], chunk_size):
        scores = query[:, :, start : start + chunk_size].float() @ keys
        probs = (scores * scale).softmax(dim=-1)
        result += probs.sum(dim=2).mean(dim=1)
    return result / query.shape[2]


def normalize_map(value):
    lower = value.amin(dim=(-2, -1), keepdim=True)
    span = value.amax(dim=(-2, -1), keepdim=True) - lower
    # Uniform attention contains no spatial evidence; do not turn it into a box.
    return torch.where(span > 1e-6, (value - lower) / span.clamp_min(1e-6), 0.0)


def neural_sketch(maps, size, sigma=1.0, threshold=0.5):
    """Aggregate saliency, Gaussian-smooth, binarize and extract its inner contour.

    The paper does not specify map aggregation or binarization; these explicit
    choices are documented in docs/generation.md and exposed by the CLI.
    """
    if not maps:
        raise RuntimeError("No spatial attention maps captured; raise attention_max_tokens.")
    value = torch.stack([
        F.interpolate(normalize_map(m.float()), size=size, mode="bilinear", align_corners=False)
        for m in maps
    ]).mean(dim=0)
    if sigma > 0:
        radius = max(1, math.ceil(3 * sigma))
        coords = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
        kernel = torch.exp(-0.5 * (coords / sigma).square())
        kernel /= kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        value = F.conv2d(F.pad(value, (radius,) * 4, mode="replicate"), kernel)
    binary = (normalize_map(value) > threshold).to(value.dtype)
    eroded = -F.max_pool2d(-binary, kernel_size=3, stride=1, padding=1)
    return binary - eroded


class AttentionState:
    def __init__(self, latent_size, alpha=0.5, cross_branch=True, max_tokens=1024):
        self.latent_size = latent_size
        self.alpha = alpha
        self.cross_branch = cross_branch
        self.max_tokens = max_tokens
        self.branch = "unconditional"
        self.collect_maps = False
        self.subject_keys = {}
        self.maps = []

    def begin_step(self):
        self.subject_keys.clear()
        self.maps.clear()

    def select(self, branch, collect_maps=False):
        self.branch = branch
        self.collect_maps = collect_maps
        self.maps.clear()

    def spatial_size(self, tokens):
        height, width = self.latent_size
        while height * width > tokens:
            height, width = (height + 1) // 2, (width + 1) // 2
        if height * width != tokens:
            raise ValueError(f"Cannot map {tokens} attention tokens to latent size {self.latent_size}.")
        return height, width


class CrossBranchAttnProcessor:
    """SD 1.x self-attention: surrounding queries attend to mixed branch keys.

    Values stay in their own branch, as the paper only specifies key mixing.
    Text cross-attention processors are never replaced.
    """

    def __init__(self, name, state):
        self.name = name
        self.state = state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("CrossBranchAttnProcessor expects unmasked spatial self-attention.")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        shape = hidden_states.shape
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        batch, tokens, _ = hidden_states.shape
        query, key, value = (layer(hidden_states) for layer in (attn.to_q, attn.to_k, attn.to_v))
        query, key, value = (
            x.view(batch, tokens, attn.heads, -1).transpose(1, 2) for x in (query, key, value)
        )
        if self.state.cross_branch:
            if self.state.branch == "subject":
                self.state.subject_keys[self.name] = key.detach()
            elif self.state.branch == "surrounding":
                subject_key = self.state.subject_keys.get(self.name)
                if subject_key is None or subject_key.shape != key.shape:
                    raise RuntimeError(f"Missing compatible subject keys for {self.name}.")
                key = self.state.alpha * key + (1 - self.state.alpha) * subject_key
        if self.state.collect_maps and tokens <= self.state.max_tokens:
            saliency = attention_saliency(query, key, attn.scale)
            self.state.maps.append(saliency.view(batch, 1, *self.state.spatial_size(tokens)))
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=attn.scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, tokens, -1).to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if len(shape) == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(shape)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


@contextmanager
def install_attention(unet, state):
    """Restore original processors and release cached features even on failure."""
    originals = dict(unet.attn_processors)
    processors = {
        name: CrossBranchAttnProcessor(name, state) if name.endswith("attn1.processor") else processor
        for name, processor in originals.items()
    }
    if not any(isinstance(p, CrossBranchAttnProcessor) for p in processors.values()):
        raise ValueError("The supplied UNet has no supported SD spatial self-attention layers.")
    unet.set_attn_processor(processors)
    try:
        yield
    finally:
        unet.set_attn_processor(originals)
        state.begin_step()
