# SPDX-License-Identifier: Apache-2.0
"""Compare native-runtime logits with the PyTorch model on one canvas.

One machine (has torch and the runtime):
    python scripts/check_runtime_parity.py --checkpoint ckpt.pt \
        --gguf export/model.gguf --runtime runtime/build/yeoubi

Split (torch on Colab, runtime on the laptop):
    colab:  python scripts/check_runtime_parity.py --checkpoint ckpt.pt --save-ref ref.npz
    laptop: python scripts/check_runtime_parity.py --ref ref.npz \
                --gguf model.gguf --runtime C:/.../yeoubi.exe

--act none switches activation quantization off on both sides, so both run
the same math up to ggml's 8-bit activation blocks: an exact graph check with
tight limits. a4 (sample.py's default) and a8 get looser limits: that block
noise moves values across INT4/INT8 rounding edges, and every flip is a whole
quantization step. --block N checks the cached path (BlockDiffusionSampler):
the prompt is committed to PyTorch's KVCache, then one block is scored.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# byte_codec.py constants, repeated so the compare side needs no torch.
MASK_TOKEN_ID, VOCAB_SIZE = 258, 258
LIMITS = {"none": (0.9999, 0.97), "a8": (0.999, 0.9), "a4": (0.999, 0.8)}
# The KV cache is 4-bit in every mode, so a cached check is never "exact".
BLOCK_LIMITS = (0.999, 0.8)


def reference(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Run the PyTorch model; returns (canvas ids, logits[T, VOCAB_SIZE])."""
    import torch

    from yeoubi.export import load_model_from_checkpoint
    from yeoubi.quantization import HybridQuantizer, KVCache
    from yeoubi.sample import _model_fwd

    model, _, config = load_model_from_checkpoint(argparse.Namespace(
        checkpoint=args.checkpoint, trust_checkpoint=args.trust_checkpoint,
        moe_layers_override=None,
    ))
    model.eval()
    if args.act == "none":
        for module in model.modules():
            if isinstance(module, HybridQuantizer):
                module.enabled = False
    elif args.act == "a8":
        model.set_activation_mode("A8")
    if getattr(config, "recurrent_state_init_std", 0.0) > 0:
        print("warning: recurrent_state_init_std > 0 adds random noise; expect drift")

    patch = config.byte_patch_size if config.use_byte_patches else 1
    prompt = args.prompt.encode("utf-8")
    loops = args.loops or None
    with torch.no_grad():
        if args.block:
            layers = (model.cache_layers_for_loops(loops)
                      if hasattr(model, "cache_layers_for_loops") else config.n_layers)
            kv = KVCache(n_layers=layers, default_bits=config.kv_cache_bits,
                         bos_bits=config.kv_cache_bos_bits)
            commit = len(prompt) // patch * patch
            for start in range(0, commit, args.block):
                chunk = torch.tensor([list(prompt[start:start + args.block])])
                _model_fwd(model, chunk, torch.zeros(1), kv_cache=kv, rope_offset=start, n_loops=loops)
            carry = list(prompt[commit:])
            ids = np.array(carry + [MASK_TOKEN_ID] * (args.block - len(carry)))
            kv.ephemeral = True
            logits, _ = _model_fwd(model, torch.tensor(ids[None]), torch.tensor([args.t]),
                                   kv_cache=kv, rope_offset=commit, n_loops=loops)
        else:
            T = max(args.length, len(prompt) + 1)
            T = -(-T // patch) * patch  # the runtime rounds the canvas up the same way
            ids = np.array(list(prompt) + [MASK_TOKEN_ID] * (T - len(prompt)))
            logits, _ = _model_fwd(model, torch.tensor(ids[None]), torch.tensor([args.t]), n_loops=loops)
    return ids, logits[0, :, :VOCAB_SIZE].float().numpy()


def runtime(args, saved: dict) -> np.ndarray:
    ids = saved["ids"]
    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = os.path.join(tmp, "prompt.bin")
        out_file = os.path.join(tmp, "logits.bin")
        with open(prompt_file, "wb") as f:
            f.write(bytes(saved["prompt"].tolist()))
        cmd = [args.runtime, "-m", args.gguf, "-f", prompt_file, "--t", str(saved["t"]),
               "--act", str(saved["act"]), "--dump-logits", out_file]
        cmd += ["--block", str(saved["block"])] if saved["block"] else ["-n", str(len(ids))]
        if saved["loops"]:
            cmd += ["--loops", str(saved["loops"])]
        if args.device:
            cmd += ["--device", args.device]
        subprocess.run(cmd, check=True)
        return np.fromfile(out_file, dtype=np.float32).reshape(len(ids), -1)[:, :VOCAB_SIZE]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", help="PyTorch checkpoint (needs torch)")
    p.add_argument("--save-ref", help="write PyTorch logits here and stop")
    p.add_argument("--ref", help="compare against logits saved by --save-ref (no torch)")
    p.add_argument("--gguf")
    p.add_argument("--runtime", help="path to the built yeoubi executable")
    p.add_argument("--device", default=None, help="runtime --device (default: best)")
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--block", type=int, default=0, help="check the KV-cached path with this block size")
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--loops", type=int, default=0)
    p.add_argument("--act", choices=("a4", "a8", "none"), default="a4")
    p.add_argument("--trust_checkpoint", action="store_true")
    args = p.parse_args()

    if args.ref:
        saved = dict(np.load(args.ref))
    elif args.checkpoint:
        ids, ref = reference(args)
        saved = dict(ids=ids, logits=ref, t=args.t, loops=args.loops, act=args.act,
                     block=args.block, prompt=np.frombuffer(args.prompt.encode("utf-8"), np.uint8))
        if args.save_ref:
            np.savez(args.save_ref, **saved)
            print(f"saved reference logits to {args.save_ref}")
            return
    else:
        p.error("need --checkpoint or --ref")
    if not (args.gguf and args.runtime):
        p.error("comparing needs --gguf and --runtime")

    ref, got = saved["logits"], runtime(args, saved)
    act, block = str(saved["act"]), int(saved["block"])
    masked = saved["ids"] == MASK_TOKEN_ID
    cos = (ref * got).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-12)
    agree = (ref.argmax(-1) == got.argmax(-1))[masked].mean()
    tag = f"{act}{f' block={block}' if block else ''}"
    print(f"[{tag}] max |diff| {np.abs(ref - got).max():.4g}   min cosine {cos.min():.5f}   "
          f"argmax agreement on masked bytes {agree:.3f}")
    min_cos, min_agree = BLOCK_LIMITS if block else LIMITS[act]
    if cos.min() < min_cos or agree < min_agree:
        sys.exit("FAIL: runtime and PyTorch disagree")
    print("OK")


if __name__ == "__main__":
    main()
