# Long-context implementation

The training configuration targets 16,384 input bytes. With four-byte patches,
the expensive global body sees 4,096 positions. `max_seq_len` remains a byte
length everywhere outside the attention module; RoPE converts it to patch
positions internally.

## Implemented path

- Padding is inferred from `PAD=256` or accepted as an explicit byte mask.
- Padded bytes are zeroed before and after the local encoder/decoder paths.
- Byte masks are reduced to patch-validity masks for global attention.
- Q and K receive per-head RMSNorm before RoPE.
- RoPE is unchanged within `rope_original_max_seq_len` and uses configurable
  `linear` or `yarn` scaling beyond it.
- A configurable fraction of training forwards use block-causal attention:
  keys in the current block and all earlier blocks are visible; future blocks
  are not. The choice is deterministic for a given batch of diffusion times.
- RDT depth may increase by one visit per context-length doubling, capped by
  `max_loop_iters`.
- Dataset preparation emits a mixture from 128 through 16,384 bytes while
  retaining short examples.

## Default extension configuration

```text
max_seq_len                  = 16384 bytes
byte_patch_size              = 4
rope_original_max_seq_len    = 4096 bytes
rope_scaling_type            = yarn
rope_scaling_factor          = 4
qk_norm                      = true
block_causal_prob            = 0.5
block_causal_min_size        = 128 bytes
block_causal_max_size        = 512 bytes
rdt_context_loop_threshold   = 4096 bytes
```

## Training order

For an existing 4K checkpoint, do not immediately train only 16K examples.
Continue with 4K/8K batches, then introduce 16K while retaining short windows.
QK RMSNorm introduces new scale parameters and therefore requires a checkpoint
migration or non-strict load followed by adaptation. A legacy model reconstructed
from a legacy checkpoint keeps QK normalization and RoPE scaling disabled.

## Validation gates

Before advertising a larger effective window, measure:

1. short-context bits per byte;
2. single- and multi-needle retrieval by evidence position;
3. long code dependency and multilingual byte reconstruction;
4. full-bidirectional versus block-causal quality;
5. three-bit versus four-bit KV retrieval;
6. attention entropy and maximum logits by layer and length.

`tests/test_long_context.py` provides structural regression coverage. It does
not replace continued training or retrieval evaluation.
