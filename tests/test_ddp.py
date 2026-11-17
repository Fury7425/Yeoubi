# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Two-process DDP smoke test of train() on CPU (gloo backend).

Covers what a single-process run cannot: ranks drawing different RDT loop
counts (find_unused_parameters), per-rank file sharding, rank-0-only saves,
and every rank agreeing to stop on the runtime budget.
"""

import json
import os
import socket
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed with gloo is not available",
)


def _worker(rank, world, port, data_dir, out_dir, runtime_hours):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
        WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
        USE_LIBUV="0",   # Windows builds of torch ship TCPStore without libuv
    )
    from yeoubi.train import TrainConfig, train

    cfg = TrainConfig(
        train_data=os.path.join(data_dir, "train_*.jsonl"),
        val_data=os.path.join(data_dir, "none_*.jsonl"),
        output_dir=out_dir, device="cpu", max_steps=3, warmup_steps=1,
        batch_size=2, grad_accum_steps=2, max_seq_len=64,
        hidden_dim=64, n_heads=4, ffn_dim=128,
        block_causal_min_size=16, block_causal_max_size=32,
        compile_model=False, gradient_checkpointing=False, async_checkpoint=False,
        save_every=1000, num_workers=0, wandb_project="", length_bucket_window=0,
        max_runtime_hours=runtime_hours,
    )
    try:
        train(cfg)
    finally:
        dist.destroy_process_group()


def _run(tmp_path, runtime_hours):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "ckpt"
    data_dir.mkdir()
    out_dir.mkdir()
    for shard in range(2):   # one file per rank
        with open(data_dir / f"train_{shard}.jsonl", "w", encoding="utf-8") as handle:
            for i in range(8):
                text = f"Shard {shard} document {i}. " * 6
                handle.write(json.dumps({"text": text}) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    context = mp.start_processes(
        _worker, args=(2, port, str(data_dir), str(out_dir), runtime_hours),
        nprocs=2, join=False, start_method="spawn",
    )
    # A rank left waiting in an all-reduce never returns, so bound the wait
    # instead of letting the suite hang.
    deadline = time.monotonic() + 300
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail("DDP training did not finish: a rank is likely blocked in a collective")
    return sorted(os.listdir(out_dir))


def test_ddp_trains_to_completion_and_only_rank0_saves(tmp_path):
    # Three steps put the A8 -> A4 switch at step 2, so its snapshot is
    # expected too; one copy of each means only rank 0 wrote.
    assert _run(tmp_path, runtime_hours=0.0) == ["final.pt", "step_2_pre_a4.pt"]


def test_ddp_runtime_budget_stops_every_rank(tmp_path):
    # A budget that has already passed when the first optimizer step ends.
    assert _run(tmp_path, runtime_hours=1e-9) == ["step_1_preempted.pt"]
