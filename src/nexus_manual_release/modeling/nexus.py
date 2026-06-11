import inspect
import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast



# ============================================================
# CONFIG
# ============================================================


class NexusConfig(PretrainedConfig):
    model_type = "nexus"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = 704,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 16000,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 100000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn


# ============================================================
# BUILDING BLOCKS
# ============================================================


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e5,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            inv_dim = lambda b: (
                (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            )
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, num_kv_heads * n_rep, head_dim)
    )


# ============================================================
# ATTENTION
# ============================================================


class Attention(nn.Module):
    def __init__(self, args: NexusConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # Deploy wrappers pass cos/sin as [1, seq_len, head_dim]; squeeze to
        # the [seq_len, head_dim] that apply_rotary_pos_emb expects.
        # No-op during training where cos/sin are already 2D.
        if cos.dim() == 3:
            cos = cos.squeeze(0)
        if sin.dim() == 3:
            sin = sin.squeeze(0)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(attention_mask is None),
        )

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ============================================================
# FFN
# ============================================================


class FeedForward(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# ============================================================
# TRANSFORMER BLOCK
# ============================================================


class NexusBlock(nn.Module):
    def __init__(self, layer_id: int, config: NexusConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_id = layer_id
        # QAT hooks — None in normal operation, set by inject_residual_fake_quant()
        self._attn_res_fq = None
        self._mlp_res_fq = None

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):

        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        if self._attn_res_fq is not None:
            hidden_states = self._attn_res_fq(hidden_states)  # ← exact attn residual output

        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        if self._mlp_res_fq is not None:
            hidden_states = self._mlp_res_fq(hidden_states)  # ← exact mlp residual output

        return hidden_states, present_key_value


# ============================================================
# CORE MODEL
# ============================================================


class NexusModel(nn.Module):
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [NexusBlock(l, config) for l in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False  # controlled by NexusForCausalLM

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_embeddings_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """Forward pass for full transformer.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs
            attention_mask (Optional[torch.Tensor]): Attention mask
            past_key_values (Optional[List]): Cached attention keys/values from previous steps
            use_cache (bool): Whether to cache attention outputs

        Returns:
            Tuple[torch.Tensor, List]:
                - Last hidden states
                - Present key/value states
        """
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Compute start_pos early — needed for both the boundary check and
        # the default position-embedding slice.
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        max_pos = self.freqs_cos.shape[0]
        if start_pos + seq_length > max_pos:
            # Extend buffers on-the-fly if sequence exceeds precomputed length
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.hidden_size // self.config.num_attention_heads,
                end=start_pos + seq_length,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            freqs_cos = freqs_cos.to(device=self.freqs_cos.device, dtype=self.freqs_cos.dtype)
            freqs_sin = freqs_sin.to(device=self.freqs_sin.device, dtype=self.freqs_sin.dtype)
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin

        # Get position embeddings for current sequence
        if position_embeddings_override is not None:
            position_embeddings = position_embeddings_override
        else:
            position_embeddings = (
                freqs_cos[start_pos : start_pos + seq_length],
                freqs_sin[start_pos : start_pos + seq_length],
            )

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                # gradient checkpointing: recompute activations on backward pass
                # use_cache must be False during checkpointing — KV cache incompatible
                def create_custom_forward(module):
                    def custom_forward(hidden_states, cos, sin, attention_mask):
                        pos_emb = (cos, sin)
                        out, _ = module(
                            hidden_states,
                            pos_emb,
                            past_key_value=None,
                            use_cache=False,
                            attention_mask=attention_mask,
                        )
                        return out

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    position_embeddings[0],
                    position_embeddings[1],
                    attention_mask,
                    use_reentrant=False,
                )
                presents.append(None)
            else:
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = hidden_states.new_zeros(())
        return hidden_states, presents, aux_loss


# ============================================================
# CAUSAL LM HEAD  (+ optimizers + MFU helpers)
# ============================================================


class NexusForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NexusConfig
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]
    supports_gradient_checkpointing = True

    def __init__(self, config: NexusConfig = None):
        self.config = config or NexusConfig()
        super().__init__(self.config)
        self.model = NexusModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # initialise weights (mirrors the llama2 training script convention)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_hidden_layers)
                )

        self.model.embed_tokens.weight = self.lm_head.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # ------------------------------------------------------------------
    # gradient checkpointing — required by Unsloth / HF Trainer
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def is_gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    # ------------------------------------------------------------------
    # weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a normalised embedding vector for each sequence in the batch.

        Runs a full decoder forward pass and takes the last-token hidden state.
        Under causal attention the last token has attended to the entire sequence,
        making it the richest single-vector summary available.

        Used by:
        - build_index.py        : encode every manual chunk offline → int8 index
        - contrastive_train.py  : encode query / positive / negative triplets
        - On-device query encoding before cosine scan (embedding mode)

        Args:
            input_ids: [B, T] token IDs

        Returns:
            [B, hidden_size] float32 L2-normalised embedding vectors.
            For create_6m_config_2_1 this is [B, 288].
        """
        hidden_states, _, _ = self.model(input_ids=input_ids)
        emb = hidden_states[:, -1, :]            # last token: [B, hidden_size]
        return F.normalize(emb, p=2, dim=-1)     # L2 normalise → unit sphere

    # ------------------------------------------------------------------
    # optimizer  (ported from llama2 training script)
    # ------------------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decayed tensors   : {len(decay_params):4d}  →  {num_decay:,} params")
        print(f"Non-decayed tensors: {len(nodecay_params):4d}  →  {num_nodecay:,} params")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_kwargs = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_kwargs)
        print(f"Using fused AdamW : {use_fused}")
        return optimizer

    # ------------------------------------------------------------------
    # MFU estimator
    # ------------------------------------------------------------------

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, flops_promised: float) -> float:
        cfg = self.config
        N = sum(p.numel() for p in self.parameters())
        L = cfg.num_hidden_layers
        H = cfg.num_attention_heads
        Q = cfg.hidden_size // cfg.num_attention_heads
        T = cfg.max_position_embeddings

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        return flops_achieved / flops_promised


# ============================================================
# STANDARD CONFIGS
# ============================================================


def create_8m_config_16k() -> NexusConfig:
    """8 M parameter Nexus model — fits in 256-token MCU context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=192,
        intermediate_size=448,
        max_position_embeddings=256,
        num_attention_heads=4,
        num_hidden_layers=14,
        num_key_value_heads=2,
        vocab_size=16000,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_8m_config() -> NexusConfig:
    """8 M parameter Nexus model — fits in 256-token MCU context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=256,
        intermediate_size=384,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_hidden_layers=14,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_6m_config_2_1() -> NexusConfig:
    """8 M parameter Nexus model — fits in 256-token MCU context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=288,
        intermediate_size=768,
        max_position_embeddings=256,
        num_attention_heads=6,
        num_hidden_layers=6,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_6m_config_2_3_1() -> NexusConfig:
    """6.5M · Nexus 2.3.1 — balanced, low KV cache.

    h=288, 8L, FFN=512 → head_dim=72, KV~1.18MB
    Wider FFN (1.78×) recovers quality lost from shallow depth.
    KV: 2 × 2 × 72 × 8 × 256 × 4 = ~1.18 MB FP32
    """
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=288,
        intermediate_size=512,  # 1.78× hidden
        max_position_embeddings=256,
        num_attention_heads=4,  # head_dim=72
        num_hidden_layers=8,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_7m_config_2_3_2() -> NexusConfig:
    """7.0M · Nexus 2.3.2 — deeper mid, moderate KV cache.

    h=256, 10L, FFN=512 → head_dim=64 (ideal), KV~1.31MB
    Best head_dim in the trio. Extra 2 layers vs 2.3.1 for
    more abstraction, still well under V2's 3.5MB KV cost.
    KV: 2 × 2 × 64 × 10 × 256 × 4 = ~1.31 MB FP32
    """
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=256,
        intermediate_size=512,  # 2.0× hidden — strongest FFN of the trio
        max_position_embeddings=256,
        num_attention_heads=4,  # head_dim=64 ✓
        num_hidden_layers=10,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_7m_config_2_3_3() -> NexusConfig:
    """7.5M · Nexus 2.3.3 — wide+shallow, ultra-low KV cache.

    h=320, 6L, FFN=640 → head_dim=80, KV~0.59MB
    Matches V1's layer count, significantly wider.
    Lowest KV cache of all configs. Trade: shallowest depth.
    KV: 2 × 2 × 80 × 6 × 256 × 4 = ~0.59 MB FP32
    """
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=320,
        intermediate_size=640,  # 2.0× hidden
        max_position_embeddings=256,
        num_attention_heads=4,  # head_dim=80
        num_hidden_layers=6,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_5m_config_2_3_4() -> NexusConfig:
    """5 M parameter Nexus model — fits in 256-token MCU context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=256,
        intermediate_size=384,
        max_position_embeddings=256,
        num_attention_heads=4,
        num_hidden_layers=8,
        num_key_value_heads=2,
        vocab_size=4096,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_13m_config() -> NexusConfig:
    """13 M parameter Nexus model — 1024-token context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=320,
        intermediate_size=512,
        max_position_embeddings=1024,
        num_attention_heads=8,
        num_hidden_layers=16,
        num_key_value_heads=2,
        vocab_size=16000,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


def create_25m_config() -> NexusConfig:
    """25 M parameter Nexus model — 2048-token context."""
    return NexusConfig(
        dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=512,
        intermediate_size=1024,
        max_position_embeddings=2048,
        num_attention_heads=8,
        num_hidden_layers=16,
        num_key_value_heads=2,
        vocab_size=16000,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        flash_attn=True,
        tie_word_embeddings=True,
    )


# ============================================================
# HF HUB UTILITIES
# ============================================================


def push_to_huggingface_hub(model, config, output_dir, args):
    try:
        import tempfile
        from datetime import datetime
        from pathlib import Path

        from huggingface_hub import create_repo, upload_file, upload_folder
    except ImportError:
        print("❌ pip install huggingface_hub")
        raise

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set --hf_token or HF_TOKEN env variable")
    if not args.hf_repo_id:
        raise ValueError("Repository ID required. Set --hf_repo_id (e.g., 'username/model-name')")

    pytorch_path = os.path.join(output_dir, args.pytorch_filename)
    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pytorch_path}")

    tmp = Path(tempfile.mkdtemp(prefix="nexus_hf_"))

    # ── Config ────────────────────────────────────────────────────────
    skip = {
        "transformers_version",
        "torch_dtype",
        "model_type",
        "rope_scaling",
        "_attn_implementation_autoset",
        "auto_map",
        "id2label",
        "label2id",
    }
    arch_dict = {k: v for k, v in config.to_dict().items() if k not in skip}

    config.save_pretrained(str(tmp))
    print(f"[config] Saved → {tmp}/config.json")

    # ── README ────────────────────────────────────────────────────────
    def md_table(data: dict, title: str) -> str:
        rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in data.items())
        return f"### {title}\n\n| Parameter | Value |\n|-----------|-------|\n{rows}\n"

    total_params = sum(p.numel() for p in model.parameters())
    readme = f"""---
license: apache-2.0
tags:
  - edge-ai
  - language-model
  - pytorch
  - microcontroller
  - nexus
  - infineon
library_name: custom
---

# Nexus Edge LM — `{args.model.upper()}`

> **Raw PyTorch checkpoint** — designed for PSoC Edge microcontrollers with **≤ 32 MB SRAM**.

---

## 🔖 Architecture

{md_table(arch_dict, "Model Configuration")}

---

## 📦 Repository Contents

| File | Description |
|------|-------------|
| `{args.pytorch_filename}` | PyTorch checkpoint `{{model_state_dict, config, total_params}}` |
| `config.json` | `NexusConfig` serialised via `PretrainedConfig.save_pretrained()` |

---

*Total parameters: {total_params:,}  ({total_params / 1e6:.2f} M)*
*Auto-generated: {datetime.utcnow().strftime("%Y-%m-%d")}*
"""
    (tmp / "README.md").write_text(readme, encoding="utf-8")

    # ── Create repo (idempotent) ───────────────────────────────────────
    print(f"[hub] Creating repo: {args.hf_repo_id}")
    create_repo(
        repo_id=args.hf_repo_id,
        repo_type="model",
        private=args.hf_private,
        exist_ok=True,
        token=token,
    )

    # ── Upload README ─────────────────────────────────────────────────
    print("[hub] Uploading README.md …")
    upload_file(
        path_or_fileobj=str(tmp / "README.md"),
        path_in_repo="README.md",
        repo_id=args.hf_repo_id,
        repo_type="model",
        token=token,
        commit_message="Add model card",
    )

    # ── Upload config.json ────────────────────────────────────────────
    print("[hub] Uploading config.json …")
    upload_file(
        path_or_fileobj=str(tmp / "config.json"),
        path_in_repo="config.json",
        repo_id=args.hf_repo_id,
        repo_type="model",
        token=token,
        commit_message="Add NexusConfig",
    )

    # ── Upload .pt checkpoint ─────────────────────────────────────────
    mb = Path(pytorch_path).stat().st_size / 1e6
    print(f"[hub] Uploading checkpoint ({mb:.1f} MB) …")
    upload_file(
        path_or_fileobj=pytorch_path,
        path_in_repo=args.pytorch_filename,
        repo_id=args.hf_repo_id,
        repo_type="model",
        token=token,
        commit_message=args.hf_commit_message,
    )

    # ── Upload tokenizer (optional) ───────────────────────────────────
    if args.tokenizer_dir and os.path.isdir(args.tokenizer_dir):
        print(f"[hub] Uploading tokenizer from {args.tokenizer_dir} …")
        upload_folder(
            folder_path=args.tokenizer_dir,
            path_in_repo="tokenizer",
            repo_id=args.hf_repo_id,
            repo_type="model",
            token=token,
            commit_message="Add tokenizer",
            ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
        )
        print("✓ Tokenizer pushed!")

    print(f"\n🎉 Model live at: https://huggingface.co/{args.hf_repo_id}")


# ============================================================
# MAIN
# ============================================================


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="8m_2_1",
        choices=["8m", "13m", "25m", "5m_2_3_4", "8m_2_1", "6m_2_3_1", "7m_2_3_2", "7m_2_3_3"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/teamspace/studios/this_studio/edge-gpt/training/local_test_models",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_init_loss", action="store_true")
    parser.add_argument("--pytorch_filename", type=str, default="nexus.pt")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--hf_private", action="store_true")
    parser.add_argument("--hf_commit_message", type=str, default="Upload Nexus checkpoint")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Optional path to tokenizer directory to upload alongside the checkpoint",
    )
    args = parser.parse_args()

    args = parser.parse_args()

    config_fn = {
        "5m_2_3_4": create_5m_config_2_3_4,
        "6m_2_3_1": create_6m_config_2_3_1,
        "7m_2_3_2": create_7m_config_2_3_2,
        "7m_2_3_3": create_7m_config_2_3_3,
        "8m": create_8m_config,
        "6m_2_1": create_6m_config_2_1,
        "13m": create_13m_config,
        "25m": create_25m_config,
    }
    config = config_fn[args.model]()

    print(f"Creating Nexus {args.model.upper()} …")
    model = NexusForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}  ({total / 1e6:.2f} M)")
    print(f"Trainable params: {trainable:,}  ({trainable / 1e6:.2f} M)")

    if args.device != "cpu":
        model = model.to(args.device)

    if args.eval_init_loss:
        model.eval()
        x = torch.randint(0, 4096, (4, 256))
        y = torch.randint(0, 4096, (4, 256))
        with torch.no_grad():
            out = model(input_ids=x, labels=y)
        print(f"loss  : {out.loss.item():.4f}")
        print(f"logits: {out.logits.shape}")
        print(f"init loss target: {torch.log(torch.tensor(4096.0)):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save PyTorch checkpoint
    pt_path = os.path.join(args.output_dir, args.pytorch_filename)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "total_params": total},
        pt_path,
    )
    print(f"✓ Saved PyTorch checkpoint → {pt_path}")

    if args.push_to_hub:
        push_to_huggingface_hub(model, config, args.output_dir, args)


if __name__ == "__main__":
    main()
