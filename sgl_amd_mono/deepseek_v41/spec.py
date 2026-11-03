# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on gfx950: the monokernel spec and its forward (RFC #43949).

A routed decode or DSpark verify step runs ``_Built.forward``, which takes and returns
what ``DeepseekV4ForCausalLM.forward`` does: SGLang's input preparation, the embedding,
every layer in two launches, the head's combine and norm, and SGLang's logits
processor, with SGLang's own modules where SGLang owns the state.

- Every layer is K1, then K2. K1 runs the attention seam and front and writes the
  step's SWA KV records and K2's key table; K2 runs the attention core, wo_a, wo_b,
  the FFN seam, the MoE and both all-reduces. The core reads KV records from every K1
  workgroup and each gfx950 XCD has its own L2, so a launch boundary is the
  device-wide release. K1 folds the previous layer's pending post into the residual;
  where none is owed (the first layer's broadcast embedding; an Engram layer, whose
  write runs on the settled residual) it takes the residual as is, collapsed with the
  previous pre (copy 0 on the first).
- An index source (ratio 1 or 2): K1 also writes its normed input, its qr and, where
  its indexer's chain is reproduced at the step's width, the index query
  (``gfx950/attention/index_query.py``; else SGLang's ``_index_query``). Between the
  launches SGLang's ``forward_low_ratio_sources`` runs a KV source's compressor on
  them, as SGLang does before the attention; K2i (``gfx950/indexer``) scores the index
  keys and SGLang's top-k kernels write the top-512 into SGLang's sparse indices as
  its decode indexer decides (``_select``); K2 builds the key table from them.
- A window-only layer (ratio 0) reads no compressed cache.
- DSpark's target layers: K1 also writes the mean of the residual it rebuilds, the
  residual SGLang's loop captures.
- Token parallel (``gfx950/token_par.py``, from 12 rows and a multiple of the TP
  size): a rank's residual and mix rows past its own are stale, so after the last
  layer one all-gather brings every row of the residual, the last pre and the DSpark
  captures from its owner.

A model with a layer of another kind is refused. The FFN launch (``gfx950/layer.py``)
runs in no forward: the kernel checks use it as the reference for K2's FFN half. The
kernels read SGLang's weights in place except the copies logged at startup: this
rank's shard of each covered layer's FP8 shared expert
(``--enforce-shared-experts-fusion`` converts SGLang's to MXFP4), the E8M0 codes of
the f32 block scales, the f32 router bias, this rank's f32 attention sinks, one RoPE
table per distinct table (layers 0 and 1 have their own), and wo_a as MXFP8 where
SGLang dequantized it to bf16 (K2 widens it back to bf16 in registers and keeps the
attention output in bf16, as the normal path's wo_a does). Pad rows write KV slot 0,
the reserved dummy slot, as the normal path does.
"""

from __future__ import annotations

import importlib.util
import re
from typing import Any

import msgspec
import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
        StepMetadata,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
        StepMetadata,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.deepseek_v41 import CANDIDATES, HF_CONFIG

MAX_ROWS = 48  # the rows a launch takes at most (``layer.MAX_TOKENS``)

_SHARED = ("gate_proj", "up_proj", "down_proj")
# the checkpoint's shared expert tensors, as the model loader reads them
_CKPT_SHARED = re.compile(
    r"layers\.(?P<i>\d+)\.ffn\.shared_experts\.(?P<w>w[123])\.(?P<kind>weight|scale)"
)
_CKPT_PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
# (freqs_cis, table) pairs per process, so the target and the draft share equal tables
ROPE_TABLES: list = []

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "StepMetadata",
        "tokens_per_request",
        "unread_metadata",
    ),
    # an index source whose rows SGLang's decode indexer does not score
    "kernels/ops/attention/dsv4/__init__.py": ("topk_transform_paged",),
    "kernels/ops/attention/dsv4/topk.py": (
        "topk_transform_paged(scores)",
        "topk_transform_paged(seq_lens)",
        "topk_transform_paged(page_tables)",
        "topk_transform_paged(out_page_indices)",
        "topk_transform_paged(page_size)",
        "topk_transform_paged(out_raw_indices)",
    ),
    # SGLang's indexer query at decode row counts (``_index_query``)
    "kernels/ops/attention/dsv4/fp4_indexer_hip.py": (
        "index_q_rope_pack_weights_flydsl",
        "index_q_rope_pack_weights_flydsl(q)",
        "index_q_rope_pack_weights_flydsl(freqs_cis)",
        "index_q_rope_pack_weights_flydsl(positions)",
        "index_q_rope_pack_weights_flydsl(rope_dim)",
        "index_q_rope_pack_weights_flydsl(head_weight_partials)",
        "index_q_rope_pack_weights_flydsl(weight_scale)",
        "index_q_rope_pack_weights_flydsl(num_heads)",
    ),
    "kernels/ops/gemm/router_gemv_hip.py": (
        "rocm_router_gemv_split_k",
        "rocm_router_gemv_split_k(x)",
        "rocm_router_gemv_split_k(w)",
    ),
    "srt/layers/attention/dsv4/unified_kv_gate.py": ("is_unified_kv_triton",),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast", "mhc_combine"),
    "kernels/ops/layernorm/mhc_boundary_hip.py": ("tensors",),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "moe_runner_backend",
        "speculative_algorithm",
        "enable_encoder_swa_bounded_replay",
        "fp8_gemm_runner_backend",
        "context_length",
        "expert_distribution_recorder_mode",
        "disaggregation_mode",
        "disaggregation_transfer_backend",
    ),
    "srt/environ.py": (
        "SGLANG_DSV4_FP4_EXPERTS",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_USE_AITER_MOE_GU_ITLV",
        "SGLANG_DSV4_KV_LAYOUT",
        "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE",
        "SGLANG_HIP_DSV41_INDEX_K_PAGE8",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    # an index source's compressor, between K1 and K2
    "srt/layers/attention/deepseek_v4_backend.py": (
        "forward_low_ratio_sources(layer)",
        "forward_low_ratio_sources(x)",
        "forward_low_ratio_sources(q_lora)",
        "forward_low_ratio_sources(positions)",
        "forward_low_ratio_sources(forward_batch)",
        "forward_low_ratio_sources(run_indexer)",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "forward_low_ratio_sources",
        "forward_metadata",
        "core_attn_metadata",
        "get_swa_out_cache_loc",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
        # its indexer's inputs and outputs: K2i stands in for its scorer
        "low_ratio_indexer_metadata",
        "sparse_raw_indices",
        "drop_folded_sparse_indices",
    ),
    # SGLang's decode indexer: its per-step decisions
    "srt/layers/attention/dsv4/low_ratio_backend_hip.py": (
        "low_ratio_decode_rows_are_identity",
        "low_ratio_decode_rows_are_identity(backend)",
        "low_ratio_decode_rows_are_identity(forward_batch)",
        "low_ratio_decode_rows_are_identity(ratio)",
        "low_ratio_decode_rows_fit_candidate_span",
        "low_ratio_decode_rows_fit_candidate_span(backend)",
        "low_ratio_decode_rows_fit_candidate_span(forward_batch)",
    ),
    "srt/layers/attention/dsv4/metadata.py": (
        "page_table",
        "compressed_seq_lens",
        "compressed_page_size",
    ),
    "srt/layers/attention/dsv4/dsv41_sparse.py": (
        "is_candidate_source",
        "uses_candidates",
        "candidate_topk_blocks",
        "candidate_block_size",
        "n_heads",
        "wq_b",
        "weights_proj",
        "weights_proj_hip_max_tokens",
        "rope_head_dim",
        "head_weight_scale",
    ),
    "srt/layers/quantization/fp8.py": (
        "aiter_bpreshuffled",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
    ),
    "srt/layers/quantization/fp8_hip.py": (
        "mxfp8_native_ready",
        "mxfp8_aiter_preshuffled",
    ),
    "srt/layers/quantization/mxfp4.py": ("is_shuffled",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "srt/layers/logits_processor.py": ("forward(hidden_states_before_norm)",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_swa_raw_buffer",
        "get_swa_key_page_size",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        "get_index_k_fp4_payload_buffer",
        "get_index_k_fp4_scale_buffer",
        "low_ratio_index_k_is_split",
        "get_index_k_page_size",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/configs/deepseek_v4.py": ("vision_n_layers", "image_token_id"),
    "srt/layers/engram.py": (
        "layer_hash_index",
        "forward(cp_all_tokens)",
        "engram_gate",
        "embed",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
        "_shared",
        "_table_ptrs",
        "row_start",
        "rows",
        "tp_size",
        "dim",
    ),
    "kernels/ops/embeddings/engram_gather.py": ("engram_gather",),
    "kernels/ops/quantization/fp4_fake_quant.py": ("FP8_BLOCK_SIZE",),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": ("hc_boundary_fused",),
    "srt/models/deepseek_v2.py": (
        "experts",
        "gate",
        "e_score_correction_bias",
        "is_nextn",
        "is_hash",
        "forward_normal",
    ),
    # DeepseekV4ForCausalLM.forward, which the routed forward stands in for
    "srt/models/deepseek_v4.py": (
        "prepare_language_model_inputs",
        "logits_processor",
        "lm_head",
        "capture_aux_hidden_states",
        "model",
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "embed_tokens",
        "hc_mult",
        "norm",
        "start_layer",
        "end_layer",
        "engram_hasher",
        '"freqs_cis_c4"',
        '"freqs_cis_c128"',
        "layers",
        "config",
        "dspark_layers_to_capture",
        "self_attn",
        "mlp",
        "engram",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "input_layernorm",
        "post_attention_layernorm",
        "hc_post",
        "compressor",
        "indexer",
        "q_head_norm",
        "fuse_wqa_wkv",
        "n_local_heads",
        "wqkv_a",
        "wq_b",
        "wo_a",
        "wo_b",
        "q_norm",
        "kv_norm",
        "freqs_cis",
        "attn_sink",
        "compress_ratio",
    ),
    "srt/configs/model_config.py": ("context_len",),
    # a token-parallel step's rows from their owners (``_gather_rows``)
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "all_gather(dim)",
        "inplace_all_reduce",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/utils/common.py": ("get_bool_env_var",),
}

# each launch's output against the normal path on the same GPU (the kernel tests)
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # K1: the seam's mixes and the SWA KV records as the normal path writes them
        "k1.kv.max_abs": 0.0,
        "k1.mix.max_abs": 1e-6,
        # wq_b summed in another order; bf16 rounding flips only
        "k1.q.cosine": 0.99999,
        "k1.q.rel": 5e-3,
        # an index source's normed input as the normal path's norm writes it
        "k1.normed.max_abs": 0.0,
        # its qr: q_norm's sum of squares in another order, so bf16 flips
        "k1.qr.rel": 4e-3,
        # K2: the attention core output, wo_a's MXFP8 output and wo_b's partial
        "k2.attn_out.cosine": 0.9998,
        "k2.wo_a.rel": 5e-3,
        "k2.wo_b.rel": 1e-3,
        # the FFN launch (K2 runs the same stages)
        "ffn.mix.rel": 1e-4,
        "ffn.normed.cosine": 0.9999,
        "ffn.normed.rel": 0.012,
        "ffn.moe.cosine": 0.9999,
        "ffn.moe.rel": 0.015,
        # K2i: a row's max difference over its max logit vs aiter's FP4 (FMA order differs)
        "k2i.logits.rel": 1e-5,
        # a candidate consumer's top-512 as a set: SGLang's top-k in the candidate blocks
        "k2i.sel.max_abs": 0.0,
    },
)


def _covered_layers(config) -> list[int]:
    """The layers K1 and K2 can run: not hash-routed, compress ratio 0, 1 or 2."""
    first = getattr(config, "num_hash_layers", 0) or 0
    return [
        i
        for i in range(first, config.num_hidden_layers)
        if config.compress_ratios[i] in (0, 1, 2)
    ]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: PD roles other than a MoRI decode server, a
    value SGLang's attention gate must not take, aiter's own flag, and FlyDSL."""
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.utils import get_bool_env_var

    reasons = []
    mode = cfg.disaggregation_mode
    if mode not in (None, "null"):
        if mode != "decode":
            reasons.append("PD monokernels support the decode role only")
        elif cfg.disaggregation_transfer_backend != "mori":
            reasons.append("PD monokernels require the mori transfer backend")
    if is_unified_kv_triton():
        reasons.append("needs the paged SWA cache, not unified_kv")
    if get_bool_env_var("AITER_FORCE_A8W4"):
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout, not "
            "AITER_FORCE_A8W4"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )

    heads = Dims(get_parallel().tp_size).heads
    covered = _covered_layers(hf_config)
    other = sorted(set(range(hf_config.num_hidden_layers)) - set(covered))
    reasons = [f"layers {other}: no kernels (hash-routed, or another compress ratio)"]
    reasons, unseen = reasons if other else [], []
    for i in covered:
        layer = model.model.layers[i]
        a, moe = layer.self_attn, layer.mlp
        if not layer.hc_boundary_fused or moe.is_nextn or moe.is_hash:
            reasons.append(f"layer {i}: needs the fused mHC boundary and a routed MoE")
        if not getattr(moe.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: MoE weights not in aiter's shuffle")
        if any(
            ("shared expert fp8", i, p, k) not in copies
            for p in _SHARED
            for k in ("weight", "scale")
        ):
            unseen.append(i)
        if a.q_head_norm or not a.fuse_wqa_wkv or a.n_local_heads != heads:
            reasons.append(
                f"layer {i}: K1 needs the fused wqkv_a, {heads} local heads and no "
                "q head norm"
            )
        if (a.compressor is not None or a.indexer is not None) and (
            a.compress_ratio not in (1, 2)
        ):
            reasons.append(
                f"layer {i}: a compressor or indexer at compress ratio "
                f"{a.compress_ratio}; SGLang's low-ratio sources serve ratio 1 or 2"
            )
        reasons += _indexer_reasons(i=i, attn=a)
        reasons += attn_layout_reasons(f"layer {i}", a)
    if shuffled_layout(model.model.layers[i].self_attn for i in covered) is None:
        reasons.append(
            "wqkv_a / wq_b / wo_b are aiter-preshuffled on some layers and row-major "
            "on others"
        )
    if unseen:
        reasons.append(
            "the shared expert's FP8 checkpoint tensors were not seen at load "
            f"(layers {unseen})"
        )
    return reasons


def _qkv_linears(attn) -> list:
    return [attn.wqkv_a] if attn.fuse_wqa_wkv else [attn.wq_a, attn.wkv]


def attn_layout_reasons(where: str, attn) -> list[str]:
    """K1 and K2 read wo_a row-major, and wqkv / wq_b / wo_b row-major or in the
    (16, 16) preshuffle of SGLang's MXFP8 aiter route; any other in-place reorder
    of an attention weight is unsupported."""
    linears = _qkv_linears(attn) + [attn.wq_b, attn.wo_a, attn.wo_b]
    if any(
        getattr(lin, "aiter_bpreshuffled", False)
        or getattr(lin, "mxfp8_native_ready", False)
        for lin in linears
    ) or getattr(attn.wo_a, "mxfp8_aiter_preshuffled", False):
        return [
            f"{where}: an FP8 linear method reordered an attention weight in place "
            "into a layout K1 / K2 do not read"
        ]
    return []


def _iq_readable(attn) -> bool:
    """Whether K1 reads the indexer's wq_b for the index query: MXFP8 e4m3 in the
    layout of the layer's own wq_b (row-major or aiter's (16, 16) preshuffle);
    else SGLang's chain computes it (``_index_query``), which reads any layout."""
    w = attn.indexer.wq_b
    return (
        w.weight.dtype == torch.float8_e4m3fn
        and not getattr(w, "aiter_bpreshuffled", False)
        and not getattr(w, "mxfp8_native_ready", False)
        and getattr(w, "mxfp8_aiter_preshuffled", False)
        == getattr(attn.wq_b, "mxfp8_aiter_preshuffled", False)
    )


def shuffled_layout(attns) -> bool | None:
    """Whether wqkv / wq_b / wo_b of these attentions are in aiter's (16, 16)
    preshuffle; None when they mix layouts."""
    seen = {
        bool(getattr(lin, "mxfp8_aiter_preshuffled", False))
        for a in attns
        for lin in _qkv_linears(a) + [a.wq_b, a.wo_b]
    }
    return seen.pop() if len(seen) == 1 else (False if not seen else None)


def _indexer_reasons(*, i: int, attn) -> list[str]:
    """What K2i needs of layer i's indexer past ``HF_CONFIG``: a candidate source's
    blocks (optional HF-config keys, so read off the indexer), and SGLang's
    decode-row query path, which ``_index_query`` runs."""
    ix = attn.indexer
    if ix is None:
        return []
    reasons = []
    if (ix.is_candidate_source or ix.uses_candidates) and (
        ix.candidate_topk_blocks,
        ix.candidate_block_size,
    ) != (CANDIDATES["candidate_topk_blocks"], CANDIDATES["candidate_block_size"]):
        reasons.append(f"layer {i}: K2i scores candidate blocks of 2048 x 8 rows")
    if ix.weights_proj_hip_max_tokens < MAX_ROWS or attn.freqs_cis.dtype != (
        torch.complex64
    ):
        reasons.append(
            f"layer {i}: K2i's query needs SGLang's split-K head-weight GEMV for "
            f"{MAX_ROWS} rows and a complex64 RoPE table"
        )
    return reasons


FULL_EMBEDDING = "_mono_full_embedding"


def embed(module, input_ids: torch.Tensor) -> torch.Tensor:
    """The tokens' embeddings: from the whole table kept at load, else the module."""
    full = module.__dict__.get(FULL_EMBEDDING)
    if full is None:
        return module(input_ids)
    return torch.nn.functional.embedding(input_ids, full)


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep the whole embedding table, and this rank's shard of each covered
    layer's FP8 shared expert, by its checkpoint name
    (``layers.{i}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}``; the checkpoint's
    ``mtp.*`` layers do not match)."""
    from sglang.srt.runtime_context import get_parallel

    if name == "embed.weight":
        # a whole copy per rank skips the vocab-parallel all-reduce, which only adds zeros
        model.model.embed_tokens.__dict__[FULL_EMBEDDING] = tensor.to(
            torch.cuda.current_device(), copy=True
        )
        return
    m = _CKPT_SHARED.fullmatch(name)
    if m is None or int(m["i"]) not in _covered_layers(model.config):
        return
    i, proj = int(m["i"]), _CKPT_PROJ[m["w"]]
    p = get_parallel()
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert fp8", i, proj, m["kind"]] = shard.contiguous()


class _Layer(msgspec.Struct):
    """One covered layer: its module and weights, and its caches (bound at the
    layer's first launch)."""

    i: int
    module: Any
    ffn: Any  # FFNWeights
    k1: Any = None  # K1Weights
    k2: Any = None  # K2Weights
    ratio: int = 0  # the compress ratio: which sparse key table K1 reads (0: none)
    # an index source: SGLang's compressor and indexer run between K1 and K2
    index: bool = False
    swa: Any = None  # the SWA cache, uint8 [pages, page bytes]
    swa_block: int = 0
    comp: Any = None  # the compressed cache, the same layout
    comp_block: int = 0
    plane: Any = None  # an indexer's FP4 index plane: (payload, scales)


def _keep(copies: dict, t: torch.Tensor, *key) -> torch.Tensor:
    copies[key] = t
    return t


def _codes(copies: dict, i: int, name: str, lin) -> torch.Tensor:
    from sgl_amd_mono.deepseek_v41.gfx950.weights import e8m0

    return _keep(
        copies,
        e8m0(lin.weight_scale_inv, f"layer {i} {name}"),
        "e8m0 scale codes",
        i,
        name,
    )


def _weights(copies: dict, tables: list, layer, i: int) -> _Layer:
    """Layer i's tensors as the launches read them."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.runner import (
        FFNWeights,
        K1Weights,
        K2Weights,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.weights import (
        mxfp8_exact,
        rope_table,
    )

    p, a, moe = get_parallel(), layer.self_attn, layer.mlp
    d, e, u8, e4m3 = Dims(p.tp_size), moe.experts, torch.uint8, torch.float8_e4m3fn
    n = moe.gate.weight.shape[0]  # the routed experts, before a fused shared one

    def gate_up(kind):
        parts = [copies.pop(("shared expert fp8", i, q, kind)) for q in _SHARED[:2]]
        return _keep(copies, torch.cat(parts), "shared expert fp8", i, "gate_up", kind)

    def down(kind):
        return copies["shared expert fp8", i, "down_proj", kind]

    ffn = FFNWeights(
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.post_attention_layernorm.weight,
        gate_w=moe.gate.weight,
        bias=_keep(
            copies, moe.gate.e_score_correction_bias.float(), "f32 router bias", i
        ),
        w13=e.w13_weight[:n],
        w13_s=e.w13_weight_scale_inv.view(u8)[: n * e.w13_weight.shape[1]],
        w2=e.w2_weight[:n],
        w2_s=e.w2_weight_scale_inv.view(u8)[: n * e.w2_weight.shape[1]],
        sgu=gate_up("weight").view(e4m3),
        sgu_s=gate_up("scale"),
        sw2=down("weight").view(e4m3),
        sw2_s=down("scale"),
    )
    ffn.check(p.tp_size, n)
    L = _Layer(i, layer, ffn)
    if a.fuse_wqa_wkv:
        wqkv, wqkv_s = a.wqkv_a.weight, _codes(copies, i, "wqkv_a", a.wqkv_a)
    else:  # the DSpark draft's wq_a and wkv as K1's one weight (a copy)
        u8 = [lin.weight.view(torch.uint8) for lin in (a.wq_a, a.wkv)]
        wqkv = _keep(copies, torch.cat(u8).view(a.wq_a.weight.dtype), "wqkv fp8", i)
        wqkv_s = _keep(
            copies,
            torch.cat(
                [
                    _codes({}, i, n_, lin)
                    for n_, lin in (("wq_a", a.wq_a), ("wkv", a.wkv))
                ]
            ),
            "e8m0 scale codes",
            i,
            "wqkv",
        )
    cos_sin = rope_table(a.freqs_cis, tables)
    L.k1 = K1Weights(
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm=layer.input_layernorm.weight,
        wqkv=wqkv,
        wqkv_s=wqkv_s,
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wqb=a.wq_b.weight,
        wqb_s=_codes(copies, i, "wq_b", a.wq_b),
        cos_sin=cos_sin,
    )
    ix = getattr(a, "indexer", None)
    if ix is not None and _iq_readable(a):
        L.k1.iq_w = ix.wq_b.weight
        L.k1.iq_ws = _codes(copies, i, "index wq_b", ix.wq_b)
        L.k1.iw_w = ix.weights_proj.weight
        L.k1.iw_scale = float(ix.head_weight_scale)
    L.k1.check(d)
    if a.wo_a.weight.dtype == e4m3:
        woa, woa_s = a.wo_a.weight, _codes(copies, i, "wo_a", a.wo_a)
    else:  # SGLang dequantized the checkpoint's FP8 wo_a to bf16 at load
        woa, woa_s = mxfp8_exact(
            a.wo_a.weight.reshape(d.groups * 1024, d.group_k), f"layer {i} wo_a"
        )
        _keep(copies, woa, "wo_a mxfp8", i, "weight")
        _keep(copies, woa_s, "wo_a mxfp8", i, "scale")
    sink = a.attn_sink.detach()[p.tp_rank * d.heads : (p.tp_rank + 1) * d.heads]
    L.k2 = K2Weights(
        woa=woa,
        woa_s=woa_s,
        wob=a.wo_b.weight,
        wob_s=_codes(copies, i, "wo_b", a.wo_b),
        sink=_keep(copies, sink.float().contiguous(), "f32 attention sinks", i),
        cos_sin=cos_sin,
    )
    L.k2.check(d)
    L.ratio = a.compress_ratio
    # the DSpark draft's window-only attention has neither
    L.index = (
        getattr(a, "compressor", None) is not None
        or getattr(a, "indexer", None) is not None
    )
    return L


def _boundary_out(layer, defer_post, x, residual, post, comb, pre):
    """What the fused boundary returns: (hidden_states, pre, pending post); with
    ``defer_post`` this layer's post stays pending for the next layer."""
    post = post.view(post.shape[0], -1)
    if defer_post:
        return None, pre, (x, residual, post, comb)
    return layer.hc_post(x, residual, post, comb), pre, None


def _launch(
    runner,
    L,
    plan,
    positions,
    forward_batch,
    pending_post,
    prev_pre,
    capture=None,
    tok_par=False,
):
    """K1, then K2, of layer ``L``: the previous layer's pending post (x, residual,
    post, comb) and pre in; K2's (x, residual, post, comb, pre) out, this layer's
    post pending. With no post owed (x None) the residual arrives settled: the
    first layer's broadcast embedding (no pre: copy 0), an Engram layer's
    residual, or a draft stage's input."""
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    from sgl_amd_mono.deepseek_v41.gfx950.runner import SLOT_DTYPES

    x, residual, post, comb = pending_post
    if L.swa is None:  # the layer's caches, at its first launch
        pool = get_token_to_kv_pool()
        L.swa, L.swa_block = pool.get_swa_raw_buffer(L.i), pool.get_swa_key_page_size()
        if L.ratio:
            # SGLang types the compressed cache as e4m3; K2 addresses its records' bytes
            L.comp = pool.get_extra_key_buffer(L.i).view(torch.uint8)
            L.comp_block = pool.get_extra_key_page_size(L.i)
    backend, M = get_attn_backend(), residual.shape[0]
    if "k1" not in plan.state:
        # the forward's first K1 converts them where needed; a graph records the copies
        slot = backend.get_swa_out_cache_loc(forward_batch)[:M]
        if not (slot.dtype in SLOT_DTYPES and slot.is_contiguous()):
            slot = slot.long().contiguous()
        plan.state["k1"] = (positions[:M].long().contiguous(), slot)
    pos, slot = plan.state["k1"]
    md = backend.forward_metadata.core_attn_metadata
    if L.ratio:  # the top-k of the layer's index source (its own, after K1)
        topk = (md.sparse_page_indices(L.ratio), md.sparse_topk_lengths(L.ratio))
    else:
        topk = (runner.no_topk, runner.no_topk_lens)
    keys = (md.swa_page_indices, md.swa_topk_lengths, *topk)
    if x is None:
        residual = residual.contiguous()  # the broadcast embedding may be a view
        if prev_pre is None:
            prev_pre = runner.first_pre[:M]
    runner.k1(
        L.k1, x, residual, post, comb, prev_pre, pos, slot, L.swa, L.swa_block,
        keys, index=L.index, tok_par=tok_par,
        aux=None if capture is None else capture(L, M),
    )  # fmt: skip
    if L.index:
        _select(
            runner=runner, L=L, plan=plan, backend=backend, positions=positions,
            pos=pos, forward_batch=forward_batch, M=M,
        )  # fmt: skip
    # K2 reads K1's outputs from the runner's buffers
    return runner.k2(
        L.k2, L.ffn, M, pos, L.swa, L.swa_block, L.comp, L.comp_block,
        slot=slot, keys=keys, index=L.index, tok_par=tok_par,
    )  # fmt: skip


def _select(*, runner, L, plan, backend, positions, pos, forward_batch, M) -> None:
    """An index source between K1 and K2.

    A KV source's compressor runs on K1's normed input, as SGLang's low-ratio path
    runs it before the attention (the other index sources read its keys). The top-512
    then lands in SGLang's sparse indices of the ratio as SGLang's decode indexer
    (``low_ratio_index_topk_hip_decode``) decides, asking SGLang's own functions (a
    captured variant's answer under a graph): every row in order where SGLang scores
    nothing, else SGLang's top-k of K2i's logits (``K2iRunner.select``) -- every layer
    of a FULL step, or on a two-level step the candidate source (``producer``) and the
    layers after it, inside its blocks (``reindex``)."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
    from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
        low_ratio_decode_rows_are_identity,
        low_ratio_decode_rows_fit_candidate_span,
    )

    attn, r = L.module.self_attn, L.ratio
    if attn.compressor is not None:
        backend.forward_low_ratio_sources(
            layer=attn,
            x=runner.normed[:M],
            q_lora=None,
            positions=positions,
            forward_batch=forward_batch,
            run_indexer=False,
        )
    ix = attn.indexer
    if ix is None:
        return
    fm = backend.forward_metadata
    core, meta = fm.core_attn_metadata, fm.low_ratio_indexer_metadata(r)
    page_idx, raw_idx = core.sparse_page_indices(r), core.sparse_raw_indices(r)
    # as SGLang's indexer: it rewrites this ratio's indices in place
    core.drop_folded_sparse_indices(r)
    if low_ratio_decode_rows_are_identity(
        backend=backend, forward_batch=forward_batch, ratio=r
    ):
        # every row sees at most index_topk rows: its top-k is every row, in order
        topk_transform_paged(
            scores=torch.empty(
                (page_idx.shape[0], 1), dtype=torch.float32, device=page_idx.device
            ),
            seq_lens=meta.compressed_seq_lens,
            page_tables=meta.page_table,
            out_page_indices=page_idx,
            page_size=meta.compressed_page_size,
            out_raw_indices=raw_idx,
        )
        return
    two_level = (
        ix.is_candidate_source or ix.uses_candidates
    ) and not low_ratio_decode_rows_fit_candidate_span(
        backend=backend, forward_batch=forward_batch
    )
    kind = "full" if not two_level else "reindex" if ix.uses_candidates else "producer"
    if L.plane is None:  # the layer's index plane, at its first launch (eager)
        L.plane = _index_plane(L=L, M=M, meta=meta, raw_idx=raw_idx)
    # the step's int32 bounds of this ratio, shared by all its index sources
    bounds = plan.state.get(("k2i bounds", r))
    if bounds is None:
        bounds = meta.compressed_seq_lens
        if bounds.dtype != torch.int32 or not bounds.is_contiguous():
            bounds = bounds.to(torch.int32).contiguous()
        plan.state["k2i bounds", r] = bounds
    k2i, per_req = runner.k2i, plan.key.tokens_per_request
    if kind != "reindex" and ("k2i plan", r) not in plan.state:
        # the step's walk plan of this ratio: its FULL and producer layers share it
        k2i.walk_plan(ratio=r, bounds=bounds, rows=M, per_req=per_req)
        plan.state["k2i plan", r] = True
    if L.k1.iq_w is not None and runner.iq_fused(M, plan.state["tok_par"]):
        query = (runner.iq[:M], runner.iqs[:M], runner.iw[:M])  # K1 computed it
    else:
        query = _index_query(
            attn=attn, x=runner.normed[:M], q_lora=runner.qr[:M], positions=pos
        )
    published = k2i.select(
        kind=kind,
        rows=M,
        per_req=per_req,
        ratio=r,
        query=query,
        plane=L.plane,
        page_table=meta.page_table,
        bounds=bounds,
        raw_out=raw_idx,
        page_out=page_idx,
        # the candidate source's blocks of this step, for the layers after it
        candidates=plan.state["k2i candidates"] if kind == "reindex" else None,
        # the step's REINDEX layers share one table (same blocks, page table, bounds)
        reindex_table_built=kind == "reindex" and ("k2i reindex", r) in plan.state,
    )
    if kind == "reindex":
        plan.state["k2i reindex", r] = True
    if kind == "producer":
        plan.state["k2i candidates"] = published


def _index_query(*, attn, x, q_lora, positions) -> tuple:
    """SGLang's indexer query for decode rows (``low_ratio_backend_hip``'s split-K
    GEMV path, its path at K2i's row counts): the FP4 query, its scales and the
    scaled bf16 head weights."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        index_q_rope_pack_weights_flydsl,
    )
    from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k

    ix = attn.indexer
    q, _ = ix.wq_b(q_lora)
    return index_q_rope_pack_weights_flydsl(
        q=q,
        freqs_cis=attn.freqs_cis,
        positions=positions,
        rope_dim=ix.rope_head_dim,
        head_weight_partials=rocm_router_gemv_split_k(x=x, w=ix.weights_proj.weight),
        weight_scale=ix.head_weight_scale,
        num_heads=ix.n_heads,
    )


def _index_plane(*, L, M: int, meta, raw_idx) -> tuple:
    """The layer's FP4 index plane, after the checks K2i needs of SGLang's index
    cache (it exists only after the forward's KV sizing, so not at attach)."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    from sgl_amd_mono.deepseek_v41.gfx950.indexer.runner import (
        PLANE_ROWS,
        check_plane,
    )

    pool = get_token_to_kv_pool()
    reasons = []
    if not pool.low_ratio_index_k_is_split(L.i):
        reasons.append("the index keys are not in the split FP4 plane")
    if pool.get_index_k_page_size(L.ratio) != PLANE_ROWS:
        reasons.append(f"index pages of {pool.get_index_k_page_size(L.ratio)} rows")
    if meta.compressed_page_size != PLANE_ROWS or meta.page_table.shape[0] != M:
        reasons.append(
            f"an index page table of {tuple(meta.page_table.shape)} at "
            f"{meta.compressed_page_size} rows a page, for {M} rows"
        )
    if raw_idx is None:
        reasons.append("no sparse_raw_indices for the step")
    if reasons:
        raise RuntimeError(
            f"monokernel: layer {L.i}: K2i needs SGLang's FP4 index plane of "
            f"{PLANE_ROWS}-row pages: " + "; ".join(reasons)
        )
    payload = pool.get_index_k_fp4_payload_buffer(L.i)
    scale = pool.get_index_k_fp4_scale_buffer(L.i)
    check_plane(payload=payload, scale=scale)
    return payload.view(torch.uint8), scale


class _Built:
    """One rank's monokernels: the runner, each layer's weights, and the forward."""

    def __init__(self, lm, runner, layers: dict):
        self.lm, self.model, self.runner, self._layers = lm, lm.model, runner, layers
        self._normal_moe = lm.model.layers[min(layers)].mlp
        # DSpark's captured layers -> the buffer K1 writes their capture into
        self._captures: dict[int, torch.Tensor] = {}

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__
    ):
        """``DeepseekV4ForCausalLM.forward`` for a routed step: SGLang's input
        preparation and logits processor around the backbone."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            out = self._backbone(
                plan, input_ids, positions, forward_batch, input_embeds
            )
        aux = None
        if lm.capture_aux_hidden_states:
            out, aux = out
        hidden, pre_hc_head = out
        lp = lm.logits_processor
        sampling_info = getattr(forward_batch, "sampling_info", None)
        lp._amd_mono_top1_verify = bool(
            forward_batch.forward_mode.is_target_verify()
            and not getattr(forward_batch, "return_logprob", False)
            and (sampling_info is None or sampling_info.is_all_greedy)
            and not getattr(sampling_info, "has_custom_logit_processor", False)
            and getattr(sampling_info, "grammar_mask", None) is None
            and getattr(sampling_info, "logit_bias", None) is None
            and getattr(sampling_info, "acc_linear_penalties", None) is None
        )
        try:
            return lp(
                input_ids,
                hidden,
                lm.lm_head,
                forward_batch,
                aux,
                hidden_states_before_norm=None if aux is not None else pre_hc_head,
            )
        finally:
            lp._amd_mono_top1_verify = False

    def _backbone(self, plan, input_ids, positions, forward_batch, input_embeds):
        """``DeepseekV4Model.forward``: the embedding, each layer as K1 and K2,
        SGLang's Engram writes, then the head's combine and norm."""
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast, mhc_combine

        m = self.model
        self._warm_normal_moe()
        x = (
            input_embeds
            if input_embeds is not None
            else embed(m.embed_tokens, input_ids)
        )
        # the compressor caches each step's rotary tables on the batch
        for attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, attr):
                delattr(forward_batch, attr)
        hash_ids = None
        engram_rows = {}
        if m.engram_hasher is not None:
            hash_ids = m.engram_hasher(input_ids, forward_batch)
            engram_rows = self._engram_rows(hash_ids)
        aux = None if m.dspark_layers_to_capture is None else []
        plan.state["dspark_aux"] = aux
        # image rows bypass Engram on every step, as in SGLang: a graph cannot branch on mm inputs
        image = None
        if m.config.vision_n_layers > 0:
            image = (input_ids == m.config.image_token_id)[:, None, None]
        pending, pre = (None, hc_broadcast(x, m.hc_mult), None, None), None
        plan.state["tok_par"] = self.runner.tok_par(x.shape[0])
        for i in range(m.start_layer, m.end_layer):
            engram = m.layers[i].engram
            if engram is not None:  # on the residual the last layer settled
                before = pending[1]
                rows = engram_rows.get(engram.layer_hash_index)
                if rows is None:
                    residual = engram(
                        before, hash_ids[:, engram.layer_hash_index], forward_batch
                    )
                else:  # Engram.forward on the staged rows
                    from sglang.srt.layers.engram import engram_gate

                    kv, _ = engram.wkv(rows.flatten(-2))
                    residual = engram_gate(
                        before,
                        kv,
                        engram.q_weight,
                        engram.k_weight,
                        engram.eps,
                        engram.clamp_value,
                    )
                if image is not None:
                    residual = torch.where(image, before, residual)
                pending = (None, residual, None, None)
            # an Engram layer reads the settled residual: no post owed into it
            defer = i + 1 < m.end_layer and m.layers[i + 1].engram is None
            hidden, pre, post = self._layer(
                self._layers[i], plan, positions, forward_batch, pending, pre, defer
            )
            pending = post if defer else (None, hidden, None, None)
        residual = pending[1]
        tok_par = plan.state["tok_par"]
        if tok_par and aux is None:
            # the head's combine reads every row's residual and pre
            residual, pre = self._gather_rows([residual, pre])
        pre_hc_head = residual.flatten(1)
        # mhc_combine widens x on load: no f32 copy first
        hidden = m.norm(mhc_combine(pre_hc_head, pre, m.hc_mult, residual.dtype))
        if aux is None:
            return hidden, pre_hc_head
        if tok_par:
            # the drafter needs only the head's output and captures, gathered as ROCm/ATOM does
            hidden, *taps = self._gather_rows([hidden, *aux])
            aux[:] = taps
            pre_hc_head = None
        return (hidden, pre_hc_head), aux

    def _engram_rows(self, hash_ids) -> dict:
        """Every Engram layer's rows of the step by hash layer, staged ahead of the
        layers as ROCm/ATOM does.

        Each rank gathers its shards' rows (zeros for the rest) and one all-reduce
        reassembles them all, in place of SGLang's all-reduce per Engram layer. Empty
        where SGLang's lookup stays (one rank, a shared host table)."""
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather
        from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
        from sglang.srt.distributed.parallel_state import inplace_all_reduce
        from sglang.srt.runtime_context import get_parallel

        m = self.model
        engrams = [
            m.layers[i].engram
            for i in range(m.start_layer, m.end_layer)
            if m.layers[i].engram is not None
        ]
        embeds = [e.embed for e in engrams]
        if not embeds or hash_ids.shape[0] == 0:
            return {}
        if any(e.tp_size == 1 or e._shared for e in embeds):
            return {}
        dim = embeds[0].dim
        out = torch.empty(
            (len(engrams), *hash_ids.shape[:1], hash_ids.shape[-1], dim),
            dtype=torch.bfloat16,
            device=hash_ids.device,
        )
        for k, (engram, embed) in enumerate(zip(engrams, embeds)):
            rows = out[k].view(-1, dim)
            if embed.rows == 0:
                rows.zero_()
                continue
            ids = hash_ids[:, engram.layer_hash_index].reshape(-1)
            engram_gather(
                *embed._table_ptrs(),
                ids,
                rows,
                dim,
                FP8_BLOCK_SIZE,
                row_lo=embed.row_start,
                row_hi=embed.row_start + embed.rows,
            )
        # integer addition keeps every bf16 bit: one shard owns each row
        inplace_all_reduce(
            out.view(torch.int32), group_name=get_parallel().tp_group.unique_name
        )
        return {e.layer_hash_index: out[k] for k, e in enumerate(engrams)}

    def _layer(
        self, L, plan, positions, forward_batch, pending_post, prev_pre, defer_post
    ):
        """One layer: the previous layer's pending post (x, residual, post, comb)
        and pre in; the fused boundary's outputs out."""
        self._warm_normal_moe()
        out = _launch(
            self.runner, L, plan, positions, forward_batch, pending_post, prev_pre,
            capture=lambda L, M: self._capture(plan, L, M),
            tok_par=plan.state["tok_par"],
        )  # fmt: skip
        return _boundary_out(L.module, defer_post, *out)

    def _gather_rows(self, tensors: list) -> list:
        """A token-parallel step's ``tensors``, every row from its owner (each rank
        computed its own rows alone): their bytes as bf16 rows, one all-gather."""
        from sglang.srt.distributed import get_tp_group

        r = self.runner
        n = tensors[0].shape[0] // r.tp
        rows = slice(r.rank * n, (r.rank + 1) * n)
        mine = [t[rows].contiguous().view(n, -1).view(torch.bfloat16) for t in tensors]
        every = get_tp_group().all_gather(torch.cat(mine, 1), dim=0)
        out, at = [], 0
        for t, m in zip(tensors, mine):
            out.append(every[:, at : at + m.shape[1]].contiguous().view(t.dtype))
            at += m.shape[1]
        return [o.view(t.shape) for o, t in zip(out, tensors)]

    def _capture(self, plan, L, M):
        """Where K1 writes layer ``L``'s DSpark capture, the mean of its input
        residual's streams; None for a layer DSpark does not read."""
        from sgl_amd_mono.deepseek_v41.gfx950.runner import no_capture

        targets = self.model.dspark_layers_to_capture
        if targets is None or L.i not in targets:
            return None
        buf = self._captures.get(L.i)
        if buf is None:
            no_capture("DSpark's capture buffers")
            buf = self._captures[L.i] = self.runner.res_mid.new_empty(
                MAX_ROWS, self.runner.res_mid.shape[-1]
            )
        aux = buf[:M]
        plan.state["dspark_aux"].append(aux)
        return aux

    def _warm_normal_moe(self) -> None:
        """At the first eager forward, run the normal MoE at 1 to MAX_ROWS rows: the
        decode graphs run the monokernels instead, so no later extend step compiles
        its kernels (per rank, no collective)."""
        moe = self._normal_moe
        if moe is None or torch.cuda.is_current_stream_capturing():
            return
        self._normal_moe = None
        x = torch.randn(
            MAX_ROWS,
            HF_CONFIG["hidden_size"],
            dtype=torch.bfloat16,
            device=self.runner.device,
        )
        for m in range(1, MAX_ROWS + 1):
            moe.forward_normal(x[:m], return_moe_output=True)


def _longest_context(hf_config) -> int:
    """The longest context a request can have, SGLang's ``context_len``; a ratio-1
    index source sees that many compressed rows (plus a verify step's draft rows)."""
    return int(hf_config.context_len) + 2 * MAX_ROWS


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
    )

    first = len(ROPE_TABLES)
    layers = {
        i: _weights(ctx.copies, ROPE_TABLES, model.model.layers[i], i)
        for i in _covered_layers(hf_config)
    }
    for n, (_, t) in enumerate(ROPE_TABLES[first:], first):
        ctx.copies["rope table", n] = t
    # the peer handshake is collective: every rank gets here (attach agreed first)
    p = get_parallel()
    device = model.model.layers[0].input_layernorm.weight.device
    runner = DSV41MonoRunner(
        p.tp_size,
        p.tp_rank,
        p.tp_group.cpu_group,
        device,
        shuffled=bool(shuffled_layout(L.module.self_attn for L in layers.values())),
    )
    runner.set_deadline(ctx.deadline_s)
    k2i = any(L.module.self_attn.indexer is not None for L in layers.values())
    if k2i:
        # the indexer's buffers, sized here: before SGLang sizes the KV cache
        runner.enable_k2i(_longest_context(hf_config))
    # K1 with and without an owed post (none on the first layer and after an Engram write)
    k1 = {
        (fold, L.index, L.k1.iq_w is not None)
        for L in layers.values()
        for fold in (True, False)
    }
    k2 = {(L.ratio > 0, L.index) for L in layers.values()}
    # ranks share FlyDSL's disk cache (a lock per kernel), so each starts at its own slice
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // p.tp_size
    for m in widths[start:] + widths[:start]:
        runner.precompile(m, k1=k1, k2=k2, k2i=k2i, tok_par=runner.tok_par(m))
    from sgl_amd_mono.deepseek_v41.gfx950.lm_head_top1 import lm_head_top1, warmup

    lp = model.logits_processor
    lp._amd_mono_top1_fn = lambda hidden, weight: lm_head_top1(
        hidden, weight, p.tp_rank * weight.shape[0]
    )
    warmup(model.lm_head.weight)
    return _Built(model, runner, layers)


SPEC = MonoSpec(
    name="deepseek_v41_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(2, 4),
            resident_blocks=BLOCKS,
        )
    },
    # MoRI transfers into the same KV, compressor-state and index-key buffers the kernels use
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsv4"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # other backends reorder wqkv_a, wq_b, wo_b in place; K1 and K2 read them row-major
            "fp8_gemm_runner_backend": frozenset({"aiter"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
            # the paged SWA cache, not a request window
            "enable_encoder_swa_bounded_replay": frozenset({False}),
            # the launches record no expert distribution (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        hf_config={**HF_CONFIG, "model_type": "deepseek_v41"},
        env={
            # the MXFP4 experts in aiter's gate / up interleaved layout
            "SGLANG_DSV4_FP4_EXPERTS": True,
            "SGLANG_DSV4_FP4_DEQUANT": False,
            "SGLANG_USE_AITER_MOE_GU_ITLV": True,
            # fp8 nope and bf16 rope, 584 B a token
            "SGLANG_DSV4_KV_LAYOUT": "v4",
            # K1 quantizes the SWA nope payload from f32, as SGLang's fused writer does
            "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE": False,
            # the scorer reads the ratio-1 index-K pages in the page-64 byte layout
            "SGLANG_HIP_DSV41_INDEX_K_PAGE8": False,
            # benchmark routing overrides: K2 routes with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
    load_tensor=_load_tensor,
    # a routed step scores every index source with K2i (_select)
    unread_metadata=frozenset({StepMetadata.LOW_RATIO_DECODE_SCHEDULE}),
)
