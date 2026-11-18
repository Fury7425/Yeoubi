# SPDX-License-Identifier: Apache-2.0
"""Compare native-runtime logits with the PyTorch model on one canvas.

    python scripts/check_runtime_parity.py --checkpoint checkpoints/final.pt \
        --gguf export/model.gguf --runtime runtime/build/yeoubi [--device Vulkan0]

Exits non-zero when the two disagree. Small differences are expected: the
runtime skips the INT8 fake-quant (see topk() in runtime/yeoubi.c) and ggml
backends quantize activations into their own blocks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yeoubi.byte_codec import MASK_TOKEN_ID, VOCAB_SIZE  # noqa: E402
from yeoubi.export import load_model_from_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--runtime", required=True, help="path to the built yeoubi executable")
    p.add_argument("--device", default=None, help="runtime --device (default: best)")
    p.add_argument("--prompt", default="The quick brown fox")
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--loops", type=int, default=0)
    p.add_argument("--trust_checkpoint", action="store_true")
    args = p.parse_args()

    model, _, config = load_model_from_checkpoint(argparse.Namespace(
        checkpoint=args.checkpoint, trust_checkpoint=args.trust_checkpoint,
        moe_layers_override=None,
    ))
    model.eval()
    if getattr(config, "recurrent_state_init_std", 0.0) > 0:
        print("warning: recurrent_state_init_std > 0 adds random noise; expect drift")

    patch = config.byte_patch_size if config.use_byte_patches else 1
    prompt = args.prompt.encode("utf-8")
    T = max(args.length, len(prompt) + 1)
    T = -(-T // patch) * patch  # the runtime rounds the canvas up the same way
    ids = list(prompt) + [MASK_TOKEN_ID] * (T - len(prompt))

    kwargs = {"n_loops": args.loops} if args.loops and hasattr(model, "rdt_config") else {}
    with torch.no_grad():
        ref, _ = model(torch.tensor([ids]), torch.tensor([args.t]), **kwargs)
    ref = ref[0, :, :VOCAB_SIZE].float().numpy()

    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = os.path.join(tmp, "prompt.bin")
        out_file = os.path.join(tmp, "logits.bin")
        with open(prompt_file, "wb") as f:
            f.write(prompt)
        cmd = [args.runtime, "-m", args.gguf, "-f", prompt_file, "-n", str(T),
               "--t", str(args.t), "--dump-logits", out_file]
        if args.loops:
            cmd += ["--loops", str(args.loops)]
        if args.device:
            cmd += ["--device", args.device]
        subprocess.run(cmd, check=True)
        got = np.fromfile(out_file, dtype=np.float32).reshape(T, -1)[:, :VOCAB_SIZE]

    masked = np.asarray(ids) == MASK_TOKEN_ID
    cos = (ref * got).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-12)
    agree = (ref.argmax(-1) == got.argmax(-1))[masked].mean()
    print(f"max |diff| {np.abs(ref - got).max():.4g}   min cosine {cos.min():.5f}   "
          f"argmax agreement on masked bytes {agree:.3f}")
    if cos.min() < 0.99 or agree < 0.9:
        sys.exit("FAIL: runtime and PyTorch disagree")
    print("OK")


if __name__ == "__main__":
    main()
