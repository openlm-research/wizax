from functools import partial
from typing import Callable
from dataclasses import dataclass

import numpy as np
import mlxu
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
import einops
from ringattention import ringattention
from scalax.sharding import (
    MeshShardingHelper, TreePathShardingRule, with_sharding_annotation
)
from scalax.utils import JaxRNG

import wizax.nn as nn


class LLaMAShardingConfig(object):
    """Sharding config for llama model."""

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.mesh_dim = '1,-1,1,1'
        config.shard_model_along_sequence = False
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._ring_attention_function = None

    def get_mesh(self):
        axis_dims = self.config.mesh_dim
        if axis_dims.startswith('!'):
            # Allow splitting a physical mesh axis if needed
            mesh_axis_splitting = True
            axis_dims = axis_dims[1:]
        else:
            mesh_axis_splitting = False

        names = ('replica', 'fsdp', 'sequence', 'tensor')
        dims = [int(x) for x in axis_dims.split(',')]
        assert len(dims) == len(names)
        return MeshShardingHelper(dims, names, mesh_axis_splitting)

    def get_model_sharding_rule(self):
        """ Get the tree path based partition rule for LLaMA model. """
        if self.config.shard_model_along_sequence:
            model_all_gather_axis = ('fsdp', 'sequence')
        else:
            model_all_gather_axis = 'fsdp'
        return TreePathShardingRule(
            # embeddings
            ('transformer/embedding/embedding', PS('tensor', model_all_gather_axis)),
            # atention
            ('self_attention/(k_proj|q_proj|v_proj)', PS(model_all_gather_axis, 'tensor')),
            ('self_attention/o_proj', PS('tensor', model_all_gather_axis)),
            # mlp
            ('feedforward/up_proj', PS(model_all_gather_axis, 'tensor')),
            ('feedforward/down_proj', PS('tensor', model_all_gather_axis)),
            ('feedforward/gate_proj', PS(model_all_gather_axis, 'tensor')),
            # layer norms
            ('input_layernorm/scale', PS(None)),
            ('post_attention_layernorm/scale', PS(None)),
            # output head
            ('lm_head_norm/scale', PS(None)),
            ('lm_head/unembedding', PS(model_all_gather_axis, 'tensor')),
            ('.*', PS(None)),
        )

    def get_intermediate_sharding_rules(self):
        return {
            'data': PS(('replica', 'fsdp'), 'sequence'),
            'ffw_intermediate': PS(('replica', 'fsdp'), 'sequence', 'tensor'),
            'attention_kqv': PS(('replica', 'fsdp'), 'sequence', 'tensor'),
            'mask': PS(('replica', 'fsdp'), 'sequence'),
        }

    def get_batch_sharding(self):
        return PS(('replica', 'fsdp'), 'sequence')


def get_ring_attention_function(
    chunk_size,
    deterministic=True,
    attention_dropout=0.0,
    dropout_rng=None,
):
    return shard_map(
        partial(
            ringattention,
            axis_name='sequence',
            float32_logits=True,
            cache_idx=None,
            blockwise_kwargs=dict(
                causal_block_size=1,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
                attn_pdrop=attention_dropout,
                query_chunk_size=chunk_size,
                key_chunk_size=chunk_size,
                policy=jax.checkpoint_policies.nothing_saveable,
                dtype=jnp.float32,
                precision=None,
                prevent_cse=True,
            )
        ),
        mesh=MeshShardingHelper.get_global_mesh(),
        in_specs=(
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), None, None, None),
            PS(('replica', 'fsdp'), None),
        ),
        out_specs=PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
        check_rep=False
    )


def apply_rotary_emb(xq, xk, position_ids, max_pos, theta=10000.0):
    input_dtype = xq.dtype
    with jax.ensure_compile_time_eval():
        dim = xq.shape[-1]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(max_pos)
        freqs = jnp.outer(t, freqs).astype(jnp.float32)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)
    freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(input_dtype), xk_out.astype(input_dtype)


@dataclass(frozen=True)
class RMSNorm:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self):
        return {
            'scale': jnp.ones(
                (self.config.hidden_size,),
                dtype=self.param_dtype
            )
        }

    def __call__(self, params, x):
        x = x.astype(self.dtype)
        scale = param['scale'].astype(self.dtype)
        rms = jnp.sqrt(
            self.config.rms_norm_eps + jnp.mean(x ** 2, axis=-1, keepdims=True)
        )
        return x / rms * scale


@dataclass(frozen=True)
class Embedding:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        rng = JaxRNG(rng)
        return {
            'embedding': nn.init_normal(
                rng(),
                (self.config.vocab_size, self.config.hidden_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
                scaling_mode='constant',
            ),
        }

    def __call__(self, params, input_ids):
        embedding = params['embedding'].astype(self.dtype)
        x = jnp.take(embedding, input_ids, axis=0)
        x = with_sharding_annotation(x, 'embedding')
        return x


@dataclass(frozen=True)
class LMHead:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        rng = JaxRNG(rng)
        return {
            'unembedding': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.vocab_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
        }

    def __call__(self, params, hidden_states):
        unembedding = params['unembedding'].astype(self.dtype)
        logits = jnp.matmul(hidden_states, kernel)
        logits = with_sharding_annotation(logits, 'logits')
        return logits


@dataclass(frozen=True)
class FeedForward:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        rng = JaxRNG(rng)
        return {
            'gate_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.intermediate_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
            'down_proj': nn.init_normal(
                rng(),
                (self.config.intermediate_size, self.config.hidden_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
            'up_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.intermediate_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),

        }

    def __call__(self, params, x):
        gate_proj = params['gate_proj'].astype(self.dtype)
        down_proj = params['down_proj'].astype(self.dtype)
        up_proj = params['up_proj'].astype(self.dtype)
        x = jnp.matmul(
            jax.nn.silu(
                with_sharding_annotation(jnp.matmul(x, gate_proj), 'ffw_intermediate')
            ) * with_sharding_annotation(jnp.matmul(x, up_proj), 'ffw_intermediate'),
            down_proj
        )
        x = with_sharding_annotation(x, 'ffw_output')
        return x


@dataclass(frozen=True)
class Attention:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        assert self.config.hidden_size % self.config.num_key_value_heads == 0
        assert self.config.hidden_size % self.config.num_attention_heads == 0
        num_query_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        rng = JaxRNG(rng)

        return {
            'q_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.hidden_size),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
            'k_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.hidden_size // num_query_groups),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
            'v_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.hidden_size // num_query_groups),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
            'o_proj': nn.init_normal(
                rng(),
                (self.config.hidden_size, self.config.hidden_size // num_query_groups),
                scale=self.config.initializer_scale,
                dtype=self.param_dtype,
            ),
        }

    def __call__(self, params, hidden_states, attention_mask, position_ids, segment_ids):
        sequence_length = hidden_states.shape[1]
        num_query_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        hidden_states = hidden_states.astype(self.dtype)
        xq = jnp.matmul(hidden_states, params['q'].astype(self.dtype))
        xk = jnp.matmul(hidden_states, params['k'].astype(self.dtype))
        xv = jnp.matmul(hidden_states, params['v'].astype(self.dtype))
        xq = einops.rearrange(
            xq, 'b s (h d) -> b s h d',
            h=self.config.num_attention_heads,
        )
        xk = einops.repeat(
            xk, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )
        xv = einops.repeat(
            xv, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )
        xq = with_sharding_annotation(xq, 'attention_kqv')
        xk = with_sharding_annotation(xk, 'attention_kqv')
        xv = with_sharding_annotation(xv, 'attention_kqv')

        xq, xk = apply_rotary_emb(
            xq, xk, position_ids,
            max_pos=self.config.max_position_embeddings,
            theta=self.config.rope_theta,
        )
        attention_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        attention_bias = einops.rearrange(attention_bias, 'b s -> b 1 1 s')
        attention_output = get_ring_attention_function(
            chunk_size=self.config.attention_chunk_size,
            deterministic=deterministic,
            attention_dropout=self.config.attention_dropout,
            dropout_rng=dropout_rng,
        )(xq, xk, xv, attention_bias, segment_ids).astype(self.dtype)

        attention_output = einops.rearrange(attention_output, 'b s h d -> b s (h d)')
        x_out = jnp.matmul(
            attention_output,
            params['o_proj'].astype(self.dtype)
        )
        return x_out


@dataclass(frozen=True)
class TransformerBlock():
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        rng = JaxRNG(rng)
        layernorm = RMSNorm(
            config=self.config,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
        )
        return {
            'input_layernorm': layernorm.init(),
            'post_attention_layernorm': layernorm.init(),
            'self_attention': Attention(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ).init(rng()),
            'feedforward': FeedForward(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ).init(rng()),
        }

    def __call__(self, params, hidden_states, attention_mask, position_ids, segment_ids):
        layernorm = RMSNorm(
            config=self.config,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            param_dtype=self.param_dtype,
        )
        attention = Attention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        feedforward = FeedForward(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        x_out = layernorm(params['input_layernorm'], hidden_states)
        x_out = attention(
            params['self_attention'],
            x_out,
            attention_mask,
            position_ids,
            segment_ids,
        )
        mlp_inputs = x_out + hidden_states
        x_out = layernorm(params['post_attention_layernorm'], mlp_inputs)
        x_out = feedforward(params['feedforward'], x_out)
        return x_out + mlp_inputs


@dataclass(frozen=True)
class LLaMAModel:
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def init(self, rng):
        rng = JaxRNG(rng)
        params = {
            'embedding': Embedding(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ).init(rng()),
            'lm_head': LMHead(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ).init(rng()),
            'lm_head_norm': RMSNorm(
                config=self.config,
                dtype=jnp.float32,
                param_dtype=self.param_dtype,
            ).init(),
        }
        for i in range(self.config.num_hidden_layers):
            params[f'transformer_block_{i}'] = TransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ).init(rng())
        return params

    def __call__(self, input_ids, attention_mask, position_ids, segment_ids):
        remat_policy = {
            'block': jax.checkpoint_policies.nothing_saveable,
            'dots': jax.checkpoint_policies.checkpoint_dots,
            'dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            'none': jax.checkpoint_policies.everything_saveable,
        }[self.config.remat]
        remat_fn = lambda fn: jax.checkpoint(fn, policy=remat_policy)
        embedding = remat(Embedding(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        ))
        lm_head = remat(LMHead(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        ))
        lm_head_norm = remat(RMSNorm(
            config=self.config,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
        ))
        transformer_block = remat(TransformerBlock(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        ))

        hidden_states = embedding(self.params['embedding'], input_ids)
        for i in range(self.config.num_hidden_layers):
            hidden_states = transformer_block(
                self.params[f'transformer_block_{i}'],
                hidden_states,
                attention_mask,
                position_ids,
                segment_ids,
            )

        hidden_states = lm_head_norm(hidden_states)
        logits = lm_head(self.params['lm_head'], hidden_states)
        return logits
