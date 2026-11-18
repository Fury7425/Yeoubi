# SPDX-License-Identifier: Apache-2.0
"""Compare native-runtime logits with the PyTorch model on one canvas.

One machine (has torch and the runtime):
    python scripts/check_runtime_parity.py --checkpoint ckpt.pt \
        --gguf export/model.gguf --runtime runtime/build/yeoubi

Split (torch on Colab, runtime on the laptop):
    colab:  python scripts/check_runtime_parity.py --checkpoint ckpt.pt --save-ref ref.npz
    laptop: python scripts/check_runtime_parity.py --ref ref.npz \
                --gguf model.gguf --runtime C:/.../yeoubi.exe

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# byte_codec.py constants, repeated so the compare side needs no torch.
MASK_TOKEN_ID, VOCAB_SIZE = 258, 258


def reference(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Run the PyTorch model; returns (ids, logits[T, VOCAB_SIZE])."""
    import torch

    from yeoubi.export import load_model_from_checkpoint

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
    ids = np.array(list(prompt) + [MASK_TOKEN_ID] * (T - len(prompt)))

    kwargs = {"n_loops": args.loops} if args.loops and hasattr(model, "rdt_config") else {}
    with torch.no_grad():
        logits, _ = model(torch.tensor(ids[None]), torch.tensor([args.t]), **kwargs)
    return ids, logits[0, :, :VOCAB_SIZE].float().numpy()


def runtime(args: argparse.Namespace, ids: np.ndarray, t: float, loops: int) -> np.ndarray:
    prompt = bytes(int(i) for i in ids if i != MASK_TOKEN_ID)
    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = os.path.join(tmp, "prompt.bin")
        out_file = os.path.join(tmp, "logits.bin")
        with open(prompt_file, "wb") as f:
            f.write(prompt)
        cmd = [args.runtime, "-m", args.gguf, "-f", prompt_file, "-n", str(len(ids)),
               "--t", str(t), "--dump-logits", out_file]
        if loops:
            cmd += ["--loops", str(loops)]
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
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--loops", type=int, default=0)
    p.add_argument("--trust_checkpoint", action="store_true")
    args = p.parse_args()

    if args.ref:
        saved = np.load(args.ref)
        ids, ref = saved["ids"], saved["logits"]
        t, loops = float(saved["t"]), int(saved["loops"])
    elif args.checkpoint:
        ids, ref = reference(args)
        t, loops = args.t, args.loops
        if args.save_ref:
            np.savez(args.save_ref, ids=ids, logits=ref, t=t, loops=loops)
            print(f"saved reference logits to {args.save_ref}")
            return
    else:
        p.error("need --checkpoint or --ref")
    if not (args.gguf and args.runtime):
        p.error("comparing needs --gguf and --runtime")

    got = runtime(args, ids, t, loops)
    masked = ids == MASK_TOKEN_ID
    cos = (ref * got).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(got, axis=-1) + 1e-12)
    agree = (ref.argmax(-1) == got.argmax(-1))[masked].mean()
    print(f"max |diff| {np.abs(ref - got).max():.4g}   min cosine {cos.min():.5f}   "
          f"argmax agreement on masked bytes {agree:.3f}")
    if cos.min() < 0.99 or agree < 0.9:
        sys.exit("FAIL: runtime and PyTorch disagree")
    print("OK")


if __name__ == "__main__":
    main()
