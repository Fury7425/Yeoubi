# SPDX-License-Identifier: Apache-2.0
"""Run Yeoubi pretraining on Modal (https://modal.com).

The same run as notebooks/kaggle_train.ipynb: it clones the GitHub repo, resumes from the
newest checkpoint in the checkpoint repo, trains on the parts that checkpoint's step
points to, and uploads new checkpoints as they are written. Never run it while a Kaggle
or Colab session is training: both would resume from and write to the same repo.

One-time setup:
    pip install modal
    modal setup
    Dashboard > Secrets > New secret > Hugging Face template, keeping its default name
    "huggingface-secret", holding a token with write access (kept out of shell history).

Measure each GPU's speed first (about 10 minutes each, run side by side, nothing uploaded):
    modal run scripts/modal_train.py --mode bench --gpus L4,A100-40GB,H100 \
        --data-repo NAME/yeoubi-data --ckpt-repo NAME/yeoubi-ckpt

Then train on the cheapest-per-step GPU. --detach keeps it running after the terminal
closes; follow it on modal.com/apps:
    modal run --detach scripts/modal_train.py --mode train --gpu H100 --hours 7 \
        --sec-per-step 0.3 --data-repo NAME/yeoubi-data --ckpt-repo NAME/yeoubi-ckpt

Measured 2026-11-24 (bf16, from step 108,874): L4 ~1.4 s/step, A100-40GB ~0.55,
H100 ~0.30. All three come to about 3,000-3,200 steps per dollar.
"""
import glob
import gzip
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import modal

REPO = "https://github.com/Fury7425/Yeoubi.git"
WORK = "/tmp/yeoubi"
KEEP = 2   # resumable checkpoints kept in the checkpoint repo, as in kaggle_train.ipynb

# USD per hour from modal.com/pricing (November 2026). Only used for cost estimates.
GPU_PRICE = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10,
             "A100-80GB": 2.50, "H100": 3.95}
# Sequences per forward pass. meta.json's step counts fix 16 sequences per optimizer
# step, so the rest is gradient accumulation and the schedule is the same on any GPU.
# 4 is what Kaggle's 16 GB T4s run; the larger values are guesses scaled by memory.
# If the benchmark reports out-of-memory, halve it with --micro-batch.
MICRO_BATCH = {"T4": 4, "L4": 4, "A10": 4, "L40S": 8, "A100-40GB": 8, "A100-80GB": 16,
               "H100": 16}
# Train log lines only; the val line also starts with "step=N".
STEP_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] yeoubi INFO: step=(\d+)  loss=")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch>=2.2", "safetensors>=0.4.3", "wandb", "datasets>=2.19",
                 "huggingface_hub")
)
app = modal.App("yeoubi-train", image=image)


def step_of(path):
    match = re.search(r"step_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def is_resumable(name):
    # The pre-A4 snapshot and final.pt are kept for reference, never resumed from or pruned.
    return name.startswith("step_") and name.endswith(".pt") and "pre_a4" not in name


class CheckpointSync:
    """Uploads finished local checkpoints and prunes old ones (kaggle_train.ipynb cell 4)."""

    def __init__(self, api, repo, local_dir):
        self.api, self.repo, self.local_dir = api, repo, local_dir
        self.sizes = {}
        self.lock = threading.Lock()   # the background loop and the final sync must not overlap

    def sync(self, stable_only=True):
        with self.lock:
            uploaded = False
            for path in sorted(glob.glob(f"{self.local_dir}/*.pt"), key=step_of):
                size = os.path.getsize(path)
                # torch.save writes in place, so a file is only known complete once its
                # size has stopped changing between two polls.
                if stable_only and self.sizes.get(path) != size:
                    self.sizes[path] = size
                    continue
                self.api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path),
                                     repo_id=self.repo)
                os.remove(path)   # only after the upload committed
                uploaded = True
            if not uploaded:
                return
            resumable = sorted(filter(is_resumable, self.api.list_repo_files(self.repo)),
                               key=step_of)
            for old in resumable[:-KEEP]:
                self.api.delete_file(old, self.repo)
            if len(resumable) > KEEP:
                # Deleted files keep counting toward storage until the history is squashed.
                self.api.super_squash_history(self.repo)

    def loop(self):
        while True:
            time.sleep(60)
            try:
                self.sync()
            except Exception as error:   # a network hiccup must not end the syncing
                print("checkpoint sync failed, will retry:", error, flush=True)


def gunzip(data_repo, name, target, seed=None):
    from huggingface_hub import hf_hub_download

    # train.py's loader dispatches on .jsonl / .pt and does not read .gz itself.
    with gzip.open(hf_hub_download(data_repo, name, repo_type="dataset"), "rb") as source:
        lines = source.readlines()   # one part fits in RAM by design
    if seed is not None:
        # Seeded like the notebooks, so every platform reads a part in the same order.
        random.Random(seed).shuffle(lines)
    with open(target, "wb") as output:
        output.writelines(lines)


@app.function(secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600)
def run(mode, gpu, data_repo, ckpt_repo, micro_batch=0, compile_model=False,
        minutes=10.0, deadline=0.0, sec_per_step=0.3):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    for alias in ("HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if not os.environ.get("HF_TOKEN") and os.environ.get(alias):
            os.environ["HF_TOKEN"] = os.environ[alias]
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("the 'huggingface-secret' Modal secret has no HF_TOKEN")
    api = HfApi()

    code = f"{WORK}/Yeoubi"
    if not os.path.isdir(code):
        subprocess.run(["git", "clone", "--depth", "1", REPO, code], check=True)

    newest = max(filter(is_resumable, api.list_repo_files(ckpt_repo)), key=step_of, default="")
    resume = hf_hub_download(ckpt_repo, newest) if newest else ""
    step = step_of(newest) if newest else 0
    with open(hf_hub_download(data_repo, "meta.json", repo_type="dataset"),
              encoding="utf-8") as handle:
        meta = json.load(handle)
    parts, steps_per_part = meta["parts"], meta["steps_per_part"]
    # EPOCHS = 1, as in the notebooks: max_steps drives the LR and A8->A4 schedules.
    max_steps = meta.get("total_steps", parts * steps_per_part)

    # A preempted container restarts from the top, so the time left comes from a fixed
    # deadline rather than from --hours: a restart cannot overspend the budget.
    hours_left = minutes / 60 if mode == "bench" else (deadline - time.time()) / 3600
    if mode == "train" and hours_left < 0.1:
        print("the deadline has passed; nothing to do")
        return {"gpu": gpu, "error": "deadline passed"}
    # Enough parts for the time left at sec_per_step, plus slack. Too few and the loader
    # would re-read the window instead of moving on to new data.
    window = min(parts, math.ceil(hours_left * 3600 / sec_per_step / steps_per_part) + 2)
    first = (step // steps_per_part) % parts   # the loader position is not checkpointed
    window_parts = [(first + index) % parts for index in range(window)]
    snapshot_download(
        data_repo, repo_type="dataset", max_workers=8,
        allow_patterns=[f"parts/part_{p:05d}.jsonl.gz" for p in window_parts] + ["val.jsonl.gz"],
    )
    shutil.rmtree(f"{WORK}/train", ignore_errors=True)
    os.makedirs(f"{WORK}/train")
    os.makedirs(f"{WORK}/val", exist_ok=True)
    for index, part in enumerate(window_parts):
        # Named by position: the loader sorts paths, and a wrapping window keeps its order.
        gunzip(data_repo, f"parts/part_{part:05d}.jsonl.gz",
               f"{WORK}/train/{index:03d}.jsonl", seed=part)
    gunzip(data_repo, "val.jsonl.gz", f"{WORK}/val/val.jsonl")
    print(f"resume={newest or 'none'} step={step}/{max_steps} "
          f"parts {window_parts[0]}..{window_parts[-1]}", flush=True)

    micro = micro_batch or MICRO_BATCH.get(gpu, 4)
    assert 16 % micro == 0, "--micro-batch must divide 16"
    local_ckpt = f"{WORK}/ckpt"
    shutil.rmtree(local_ckpt, ignore_errors=True)
    os.makedirs(local_ckpt)
    # Leave a quarter hour after the stop checkpoint for the last upload.
    runtime = minutes / 60 if mode == "bench" else (deadline - time.time()) / 3600 - 0.25
    command = [
        sys.executable, "train.py",
        "--train_data", f"{WORK}/train/*.jsonl", "--val_data", f"{WORK}/val/*.jsonl",
        "--output_dir", local_ckpt, "--resume_from", resume, "--max_steps", str(max_steps),
        "--max_seq_len", "4096", "--batch_size", str(micro),
        "--grad_accum_steps", str(16 // micro),
        "--hidden_dim", "768", "--n_heads", "12", "--ffn_dim", "2048",
        # The T4 has no usable bf16, so it keeps Kaggle's fp16 + loss scaler. The other
        # cards use bf16, which needs no scaler and so skips no steps on inf gradients.
        "--bf16", "false" if gpu == "T4" else "true",
        "--compile_model", "true" if compile_model else "false",
        "--gradient_checkpointing", "false", "--save_every", "500",
        "--muon_weight_decay", "0.01", "--max_runtime_hours", f"{runtime:.3f}",
        "--num_workers", "2", "--wandb_project", "",
        # validate() reads val_max_batches batches of batch_size, so a bigger batch would
        # score more of val.jsonl than Kaggle does (50 x 4 = 200 sequences) and the
        # numbers would stop being comparable across platforms.
        "--val_max_batches", str(200 // micro),
    ]
    # Its own session, so a stop signal from Modal reaches train.py only as the SIGTERM
    # forwarded below. Sent to the whole group, it arrived as SIGINT and interrupted
    # the stop checkpoint with a KeyboardInterrupt.
    process = subprocess.Popen(command, cwd=code, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})

    def forward(signum, frame):
        # Modal interrupts a container it preempts. train.py saves step_*_preempted.pt
        # on SIGTERM, which only reaches it if passed on from here.
        process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, forward)
        except ValueError:   # not on the main thread; preemption then loses up to 500 steps
            pass

    syncer = None
    if mode == "train":
        syncer = CheckpointSync(api, ckpt_repo, local_ckpt)
        threading.Thread(target=syncer.loop, daemon=True).start()

    timings, tail = [], []
    for line in process.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-20:]
        match = STEP_LINE.search(line)
        if match:
            stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").timestamp()
            timings.append((stamp, int(match[2])))
    exit_code = process.wait()

    if syncer:
        syncer.sync(stable_only=False)   # train.py has exited, so every file is complete
        if exit_code:
            raise RuntimeError(f"train.py exited with {exit_code}")
        return {"gpu": gpu}

    # The first line after a resume averages a partial window, so timing starts at the second.
    points = timings[1:]
    if len(points) < 2:
        reason = "".join(tail[-8:]) if exit_code else "too few step lines; raise --minutes"
        return {"gpu": gpu, "micro_batch": micro, "error": reason}
    seconds = (points[-1][0] - points[0][0]) / (points[-1][1] - points[0][1])
    return {"gpu": gpu, "micro_batch": micro, "sec_per_step": seconds}


@app.local_entrypoint()
def main(mode: str = "bench", gpu: str = "A100-40GB", gpus: str = "L4,A100-40GB,H100",
         hours: float = 4.0, minutes: float = 10.0, micro_batch: int = 0,
         compile_model: bool = False, sec_per_step: float = 0.3,
         data_repo: str = "", ckpt_repo: str = ""):
    assert data_repo and ckpt_repo, "pass --data-repo and --ckpt-repo"
    if mode == "bench":
        calls = []
        for name in gpus.split(","):
            function = run.with_options(gpu=name, timeout=int((minutes + 25) * 60))
            calls.append((name, function.spawn("bench", name, data_repo, ckpt_repo,
                                               micro_batch, compile_model, minutes)))
        print(f"{'GPU':<11}{'batch':>6}{'s/step':>8}{'$/h':>7}{'steps/$':>9}{'steps/$30':>11}")
        for name, call in calls:
            try:
                result = call.get()
            except Exception as error:
                result = {"error": str(error)}
            if "sec_per_step" not in result:
                print(f"{name:<11} failed: {result.get('error', '').strip()}")
                continue
            price = GPU_PRICE.get(name, float("nan"))
            per_dollar = 3600 / result["sec_per_step"] / price
            print(f"{name:<11}{result['micro_batch']:>6}{result['sec_per_step']:>8.2f}"
                  f"{price:>7.2f}{per_dollar:>9.0f}{30 * per_dollar:>11.0f}")
        print("Setup (image, downloads) is not in s/step; it costs a few minutes per run.")
    elif mode == "train":
        assert 0 < hours <= 23, "Modal caps a function at 24 h, and setup needs some of it"
        price = GPU_PRICE.get(gpu)
        if price:
            print(f"{gpu} for {hours:g} h is about ${hours * price:.2f} of GPU time, "
                  "plus a little CPU and memory.")
        run.with_options(gpu=gpu, timeout=int(hours * 3600 + 1800)).remote(
            "train", gpu, data_repo, ckpt_repo, micro_batch, compile_model,
            minutes, time.time() + hours * 3600, sec_per_step)
    else:
        raise SystemExit("--mode is bench or train")
